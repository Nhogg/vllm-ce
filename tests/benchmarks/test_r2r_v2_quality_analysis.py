"""Tests for strict R2R v2 matrix validation and paired comparisons."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import analyze_r2r_v2_quality as analysis  # noqa: E402

MODEL = "Llama-3.2-1B-Instruct"
TASK = "qasper"


def _fixture(
    root: Path,
    model: str = MODEL,
    task: str = TASK,
    capacities: tuple[int, ...] = (16,),
) -> Path:
    cell = root / f"r2r_v2_{analysis.MODEL_TAGS[model]}_{task}"
    cell.mkdir()
    stem = cell.name
    baseline = {
        "model": model,
        "task": task,
        "mode": "baseline",
        "eviction_policy": "none",
        "block_size": 16,
        "num_prompts": 2,
        "metric_name": analysis.EXPECTED_METRICS[task],
        "qa_f1": 0.5,
        "prompt_ids": ["a", "b"],
        "f1s": [0.4, 0.6],
    }
    (cell / f"{stem}_base.json").write_text(json.dumps(baseline))
    rows = []
    for variant, scores in (("v1", [0.2, 0.6]), ("v2", [0.4, 0.8])):
        for capacity in capacities:
            run = {
                "model": model,
                "task": task,
                "mode": "band",
                "eviction_policy": "v_redundancy",
                "metric_name": analysis.EXPECTED_METRICS[task],
                "prompt_ids": ["a", "b"],
                "f1s": scores,
                "num_prompts": 2,
                "qa_f1": sum(scores) / 2,
                "capacity": capacity,
                "block_size": 16,
                "watermark": 0.75,
                "low_watermark_blocks": int(0.75 * capacity),
                "score_sampled_layers": analysis.EXPECTED_LAYERS[model],
                "candidate_expansion_factor": 4.0,
                "r2r_cover_depth": 2,
                "r2r_query_aggregation": "max",
                "r2r_relevance_signal": "key_anchor",
                "query_tail_tokens": 32,
                "r2r_persistent_floor": variant == "v2",
                "r2r_dispersion_kappa": 2.0 if variant == "v2" else None,
                "throughput_tok_s": 10,
                "mean_ttft_ms": 100,
                "mean_tpot_ms": 10,
                "steady_blocks": 12,
            }
            path = cell / f"{stem}_v_redundancy_{variant}_cap{capacity}.json"
            path.write_text(json.dumps(run))
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "eviction_policy": "v_redundancy",
                    "r2r_variant": variant,
                    "capacity": capacity,
                    "budget_tokens": capacity * 16,
                    "n": 2,
                    "metric_name": analysis.EXPECTED_METRICS[task],
                    "qa_f1": sum(scores) / 2,
                    "block_size": 16,
                    "watermark": 0.75,
                    "score_sampled_layers": analysis.EXPECTED_LAYERS[model],
                    "candidate_expansion_factor": 4.0,
                    "r2r_cover_depth": 2,
                    "r2r_query_aggregation": "max",
                    "r2r_relevance_signal": "key_anchor",
                    "query_tail_tokens": 32,
                    "block_prototype_mode": "mean",
                    "r2r_persistent_floor": str(variant == "v2"),
                    "r2r_dispersion_kappa": "2.0" if variant == "v2" else "",
                    "throughput_tok_s": 10,
                    "mean_ttft_ms": 100,
                    "mean_tpot_ms": 10,
                    "steady_blocks": 12,
                }
            )
    with (cell / "capband_accuracy.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return cell


def test_paired_delta_matches_prompt_ids_not_position():
    v1 = {"prompt_ids": ["a", "b"], "f1s": [0.2, 0.6]}
    v2 = {"prompt_ids": ["b", "a"], "f1s": [0.8, 0.4]}
    n, center, low, high = analysis.paired_delta(v1, v2)
    assert n == 2
    assert center == pytest.approx(0.2)
    assert low == pytest.approx(0.2)
    assert high == pytest.approx(0.2)


def test_paired_delta_rejects_unmatched_prompts():
    v1 = {"prompt_ids": ["a"], "f1s": [0.2]}
    v2 = {"prompt_ids": ["b"], "f1s": [0.8]}
    with pytest.raises(ValueError, match="prompt sets differ"):
        analysis.paired_delta(v1, v2)


def test_load_and_compare_complete_cell(tmp_path):
    _fixture(tmp_path)
    rows, baselines = analysis.load_matrix(
        tmp_path, models=(MODEL,), tasks=(TASK,), capacities=(16,)
    )
    assert len(rows) == 2
    assert baselines[(MODEL, TASK)]["qa_f1"] == 0.5
    comparisons = analysis.paired_rows(tmp_path, rows, baselines, (16,))
    assert len(comparisons) == 1
    assert comparisons[0]["score_delta_v2_minus_v1"] == pytest.approx(0.2)
    output = tmp_path / "plot.png"
    analysis.plot_metric(rows, baselines, "qa_f1", output)
    assert output.is_file()
    delta_output = tmp_path / "delta.png"
    analysis.plot_paired_delta(comparisons, delta_output)
    assert delta_output.is_file()


def test_load_rejects_incomplete_grid(tmp_path):
    _fixture(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        analysis.load_matrix(
            tmp_path, models=(MODEL,), tasks=(TASK, "hotpotqa"), capacities=(16,)
        )


def test_load_rejects_missing_budget_arm(tmp_path):
    cell = _fixture(tmp_path, capacities=(16, 32))
    csv_path = cell / "capband_accuracy.csv"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if not (row["r2r_variant"] == "v2" and row["capacity"] == "32")
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="incomplete grid"):
        analysis.load_matrix(
            tmp_path, models=(MODEL,), tasks=(TASK,), capacities=(16, 32)
        )


def test_load_rejects_mislabeled_v2_guard(tmp_path):
    cell = _fixture(tmp_path)
    path = cell / "capband_accuracy.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["r2r_persistent_floor"] = "False"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="floor setting inconsistent"):
        analysis.load_matrix(tmp_path, models=(MODEL,), tasks=(TASK,), capacities=(16,))


def test_load_rejects_wrong_dispersion_threshold(tmp_path):
    cell = _fixture(tmp_path)
    path = cell / "capband_accuracy.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["r2r_dispersion_kappa"] = "3.0"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="unexpected dispersion kappa"):
        analysis.load_matrix(tmp_path, models=(MODEL,), tasks=(TASK,), capacities=(16,))


def test_full_fifteen_cell_matrix_passes_provenance_and_pairing(tmp_path):
    for model in analysis.MODELS:
        for task in analysis.TASKS:
            _fixture(tmp_path, model, task, analysis.CAPACITIES)
    rows, baselines = analysis.load_matrix(tmp_path)
    assert len(baselines) == 15
    assert len(rows) == 150
    comparisons = analysis.paired_rows(tmp_path, rows, baselines, analysis.CAPACITIES)
    assert len(comparisons) == 75
    assert {row["score_delta_v2_minus_v1"] for row in comparisons} == {0.2}
    analysis.plot_paired_delta(comparisons, tmp_path / "full_delta.png")
    assert (tmp_path / "full_delta.png").is_file()


def test_check_only_is_read_only_on_full_matrix(tmp_path, monkeypatch, capsys):
    for model in analysis.MODELS:
        for task in analysis.TASKS:
            _fixture(tmp_path, model, task, analysis.CAPACITIES)
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_r2r_v2_quality.py", "--root", str(tmp_path), "--check-only"],
    )
    analysis.main()
    assert "PASS: 15 model/task cells, 150 arms" in capsys.readouterr().out
    assert not (tmp_path / "charts").exists()


def test_pairing_rejects_raw_prompt_mismatch(tmp_path):
    cell = _fixture(tmp_path)
    path = cell / f"{cell.name}_v_redundancy_v2_cap16.json"
    run = json.loads(path.read_text())
    run["prompt_ids"] = ["a", "other"]
    path.write_text(json.dumps(run))
    rows, baselines = analysis.load_matrix(
        tmp_path, models=(MODEL,), tasks=(TASK,), capacities=(16,)
    )
    with pytest.raises(ValueError, match="raw/baseline prompt IDs mismatch"):
        analysis.paired_rows(tmp_path, rows, baselines, (16,))


def test_pairing_rejects_raw_configuration_mismatch(tmp_path):
    cell = _fixture(tmp_path)
    path = cell / f"{cell.name}_v_redundancy_v2_cap16.json"
    run = json.loads(path.read_text())
    run["r2r_dispersion_kappa"] = 3.0
    path.write_text(json.dumps(run))
    rows, baselines = analysis.load_matrix(
        tmp_path, models=(MODEL,), tasks=(TASK,), capacities=(16,)
    )
    with pytest.raises(ValueError, match="raw dispersion kappa mismatch"):
        analysis.paired_rows(tmp_path, rows, baselines, (16,))
