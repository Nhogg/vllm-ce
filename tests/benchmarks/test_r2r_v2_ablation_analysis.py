"""Tests for matched four-arm R2R v2 ablation analysis."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import analyze_r2r_v2_ablation as ablation  # noqa: E402
import analyze_r2r_v2_quality as quality  # noqa: E402

MODEL = "Llama-3.2-1B-Instruct"
TASK = "qasper"


def _four_arms(root: Path) -> Path:
    cell = root / f"r2r_v2_{quality.MODEL_TAGS[MODEL]}_{TASK}"
    cell.mkdir()
    baseline = {
        "model": MODEL,
        "task": TASK,
        "mode": "baseline",
        "eviction_policy": "none",
        "block_size": 16,
        "num_prompts": 2,
        "metric_name": "qa_f1",
        "qa_f1": 0.5,
        "prompt_ids": ["a", "b"],
        "f1s": [0.4, 0.6],
    }
    (cell / f"{cell.name}_base.json").write_text(json.dumps(baseline))
    csv_path = cell / "capband_accuracy.csv"
    rows = []
    for variant, scores in (
        ("v1", [0.2, 0.6]),
        ("floor", [0.3, 0.7]),
        ("dispersion", [0.25, 0.65]),
        ("v2", [0.4, 0.8]),
    ):
        floor = variant in ("floor", "v2")
        kappa = 2.0 if variant in ("dispersion", "v2") else None
        row = {
            "model": MODEL,
            "task": TASK,
            "eviction_policy": "v_redundancy",
            "r2r_variant": variant,
            "capacity": 16,
            "budget_tokens": 256,
            "n": 2,
            "metric_name": "qa_f1",
            "qa_f1": sum(scores) / len(scores),
            "block_size": 16,
            "watermark": 0.75,
            "score_sampled_layers": quality.EXPECTED_LAYERS[MODEL],
            "candidate_expansion_factor": 4.0,
            "r2r_cover_depth": 2,
            "r2r_query_aggregation": "max",
            "r2r_relevance_signal": "key_anchor",
            "query_tail_tokens": 32,
            "block_prototype_mode": "mean",
            "r2r_persistent_floor": str(floor),
            "r2r_dispersion_kappa": "" if kappa is None else str(kappa),
            "throughput_tok_s": 10,
            "mean_ttft_ms": 100,
            "mean_tpot_ms": 10,
            "steady_blocks": 12,
        }
        rows.append(row)
        run = {
            **baseline,
            "mode": "band",
            "eviction_policy": "v_redundancy",
            "prompt_ids": ["a", "b"],
            "f1s": scores,
            "qa_f1": sum(scores) / len(scores),
            "capacity": 16,
            "watermark": 0.75,
            "low_watermark_blocks": 12,
            "score_sampled_layers": quality.EXPECTED_LAYERS[MODEL],
            "candidate_expansion_factor": 4.0,
            "r2r_cover_depth": 2,
            "r2r_query_aggregation": "max",
            "r2r_relevance_signal": "key_anchor",
            "query_tail_tokens": 32,
            "r2r_persistent_floor": floor,
            "r2r_dispersion_kappa": kappa,
            "throughput_tok_s": 10,
            "mean_ttft_ms": 100,
            "mean_tpot_ms": 10,
            "steady_blocks": 12,
        }
        path = cell / f"{cell.name}_v_redundancy_{variant}_cap16.json"
        path.write_text(json.dumps(run))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return cell


def test_four_arm_effects_and_interaction(tmp_path):
    _four_arms(tmp_path)
    rows, baselines = quality.load_matrix(
        tmp_path,
        models=(MODEL,),
        tasks=(TASK,),
        capacities=(16,),
        variants=ablation.VARIANTS,
    )
    effects = ablation.effect_rows(tmp_path, rows, baselines, (16,))
    assert len(effects) == 4
    by_effect = {row["effect"]: row for row in effects}
    assert by_effect["floor_minus_v1"]["score_delta"] == pytest.approx(0.1)
    assert by_effect["dispersion_minus_v1"]["score_delta"] == pytest.approx(0.05)
    assert by_effect["v2_minus_v1"]["score_delta"] == pytest.approx(0.2)
    assert by_effect["interaction"]["score_delta"] == pytest.approx(0.05)
    ablation.plot_effects(effects, tmp_path / "effects.png")
    assert (tmp_path / "effects.png").is_file()


def test_four_arm_rejects_raw_floor_mismatch(tmp_path):
    cell = _four_arms(tmp_path)
    path = cell / f"{cell.name}_v_redundancy_floor_cap16.json"
    run = json.loads(path.read_text())
    run["r2r_persistent_floor"] = False
    path.write_text(json.dumps(run))
    rows, baselines = quality.load_matrix(
        tmp_path,
        models=(MODEL,),
        tasks=(TASK,),
        capacities=(16,),
        variants=ablation.VARIANTS,
    )
    with pytest.raises(ValueError, match="raw r2r_persistent_floor mismatch"):
        ablation.effect_rows(tmp_path, rows, baselines, (16,))
