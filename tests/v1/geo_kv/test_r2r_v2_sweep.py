# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the R2R v2 capacity-sweep wiring."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))
import benchmark_kv_capacity_band_vllm as capband  # noqa: E402


def _args(*flags: str):
    return capband.build_parser().parse_args(["--model", "unused", *flags])


def test_legacy_sweep_keeps_one_arm():
    assert capband._sweep_variants(_args(), "v_redundancy") == [""]


def test_v2_sweep_separates_baseline_and_guards():
    args = _args(
        "--candidate-expansion-factor", "4", "--r2r-variants", "v1,v2"
    )
    assert capband._sweep_variants(args, "v_redundancy") == ["v1", "v2"]
    assert capband._sweep_variants(args, "recency") == [""]


@pytest.mark.parametrize(
    "flags",
    [
        ("--r2r-variants", "v1,v2"),
        (
            "--candidate-expansion-factor", "4", "--r2r-variants", "v1,v2",
            "--r2r-persistent-floor",
        ),
        ("--candidate-expansion-factor", "4", "--r2r-variants", "v1,v1"),
        ("--candidate-expansion-factor", "4", "--r2r-variants", "unknown"),
    ],
)
def test_invalid_variant_configuration_rejected(flags):
    with pytest.raises(ValueError):
        capband._sweep_variants(_args(*flags), "v_redundancy")


def test_sweep_writes_distinct_v1_v2_rows(monkeypatch, tmp_path):
    args = _args(
        "--sweep", "--capacities", "16,64",
        "--candidate-expansion-factor", "4", "--r2r-variants", "v1,v2",
        "--output-dir", str(tmp_path),
    )
    calls = []

    def fake_spawn(_args, mode, capacity, _dump, **kwargs):
        calls.append((mode, capacity, kwargs))
        if mode == "baseline":
            return {
                "prompt_ids": ["one"], "f1s": [1.0], "qa_f1": 1.0,
                "num_prompts": 1, "num_gpu_blocks": 999,
            }
        return {
            "prompt_ids": ["one"], "f1s": [0.5], "qa_f1": 0.5,
            "decode_evictions": 1, "steady_blocks": capacity,
            "peak_blocks": capacity, "mean_out_len": 2,
            "r2r_persistent_floor": "--r2r-persistent-floor" in
            kwargs["extra_flags"],
            "r2r_dispersion_kappa": (
                3.0 if "--r2r-dispersion-kappa" in kwargs["extra_flags"]
                else None
            ),
        }

    monkeypatch.setattr(capband, "_spawn", fake_spawn)
    capband.run_sweep(args)
    assert len(calls) == 5
    with (tmp_path / "capband_accuracy.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["r2r_variant"], row["capacity"]) for row in rows] == [
        ("v1", "16"), ("v1", "64"), ("v2", "16"), ("v2", "64")
    ]
    assert [row["r2r_persistent_floor"] for row in rows] == [
        "False", "False", "True", "True"
    ]
