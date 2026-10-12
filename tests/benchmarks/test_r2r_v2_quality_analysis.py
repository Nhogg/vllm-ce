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


def _fixture(root: Path) -> Path:
    cell = root / "r2r_v2_1b_qasper"
    cell.mkdir()
    stem = cell.name
    baseline = {
        "model": MODEL,
        "task": TASK,
        "qa_f1": 0.5,
        "prompt_ids": ["a", "b"],
        "f1s": [0.4, 0.6],
    }
    (cell / f"{stem}_base.json").write_text(json.dumps(baseline))
    rows = []
    for variant, scores in (("v1", [0.2, 0.6]), ("v2", [0.4, 0.8])):
        run = {
            "model": MODEL,
            "task": TASK,
            "prompt_ids": ["a", "b"],
            "f1s": scores,
        }
        path = cell / f"{stem}_v_redundancy_{variant}_cap16.json"
        path.write_text(json.dumps(run))
        rows.append(
            {
                "model": MODEL,
                "task": TASK,
                "eviction_policy": "v_redundancy",
                "r2r_variant": variant,
                "capacity": 16,
                "n": 2,
                "metric_name": "qa_f1",
                "qa_f1": sum(scores) / 2,
                "r2r_persistent_floor": str(variant == "v2"),
                "r2r_dispersion_kappa": "3.0" if variant == "v2" else "",
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
    comparisons = analysis.paired_rows(tmp_path, rows, (16,))
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
