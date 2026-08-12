# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the layer-subset calibrator (plan Phase 1C).

Covers the engine-independent behavior: candidate construction is architecture
agnostic and in-range, disabled is a true no-op, agreement is accumulated per
subset via the scorer-owned select closure, a subset identical to all layers
scores a perfect match, the summary ranks best-first, and a JSON report
round-trips to disk.
"""

import json

import torch

from vllm.v1.geo_kv.layer_calibrator import (
    LayerCalibrator,
    _jaccard,
    default_layer_subset_candidates,
)


def test_candidates_are_in_range_and_sorted():
    for num_layers in (2, 3, 8, 32, 80):
        cands = default_layer_subset_candidates(num_layers)
        assert cands, num_layers
        for name, idxs in cands.items():
            assert idxs == sorted(set(idxs)), (num_layers, name)
            assert all(0 <= i < num_layers for i in idxs), (num_layers, name)
        # Sizes cover the plan's 1 / 2 / 4-layer probes (4-layer probes only
        # exist once there are at least 4 distinct layers to spread across).
        sizes = {len(v) for v in cands.values()}
        assert 1 in sizes and 2 in sizes
        if num_layers >= 4:
            assert 4 in sizes


def test_too_few_layers_yields_no_candidates():
    assert default_layer_subset_candidates(1) == {}
    assert default_layer_subset_candidates(0) == {}


def test_jaccard_matches_hand_values():
    a = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    b = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    assert _jaccard(a, b) == 1 / 3
    assert _jaccard(a, a) == 1.0
    z = torch.zeros(4, dtype=torch.bool)
    assert _jaccard(z, z) == 1.0  # both empty -> perfect agreement


def test_disabled_is_a_noop():
    cal = LayerCalibrator(enabled=False, num_layers=8)
    cal.observe([0, 1, 2], torch.zeros(4, dtype=torch.bool), lambda rows: None)
    assert cal.summary()["subsets"] == []
    cal.dump()  # must not raise


def _select_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Tiny reference selector: mark the ``k`` highest-scoring blocks evicted."""
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if k > 0:
        mask[torch.topk(scores, k).indices] = True
    return mask


def test_perfect_agreement_when_subset_equals_all_layers():
    num_layers = 4
    num_blocks = 6
    cal = LayerCalibrator(enabled=True, num_layers=num_layers)
    torch.manual_seed(0)
    stacked = torch.rand(num_layers, num_blocks)
    row_layer_idxs = [0, 1, 2, 3]
    ref_scores = stacked.mean(dim=0)
    ref_mask = _select_topk(ref_scores, 2)

    def aggregate_and_select(rows: list[int]) -> torch.Tensor:
        idx = torch.tensor(rows, dtype=torch.long)
        return _select_topk(stacked[idx].mean(dim=0), 2)

    for _ in range(3):
        cal.observe(row_layer_idxs, ref_mask, aggregate_and_select)

    summ = cal.summary()
    assert summ["fires"] == 3
    by_name = {r["subset"]: r for r in summ["subsets"]}
    # A candidate spanning all 4 rows reproduces the all-layer selection exactly.
    assert by_name["even4"]["mean_jaccard"] == 1.0
    assert by_name["even4"]["fires"] == 3
    # Every candidate produced a score in [0, 1].
    assert all(0.0 <= r["mean_jaccard"] <= 1.0 for r in summ["subsets"])


def test_summary_is_ranked_best_first():
    cal = LayerCalibrator(enabled=True, num_layers=8)
    stacked = torch.zeros(8, 5)
    # Make one layer's score dominate so single-layer subsets diverge.
    stacked[4] = torch.tensor([9.0, 0.0, 0.0, 0.0, 0.0])
    row_layer_idxs = list(range(8))
    ref_mask = _select_topk(stacked.mean(dim=0), 1)

    def aggregate_and_select(rows: list[int]) -> torch.Tensor:
        idx = torch.tensor(rows, dtype=torch.long)
        return _select_topk(stacked[idx].mean(dim=0), 1)

    cal.observe(row_layer_idxs, ref_mask, aggregate_and_select)
    ranked = [r["mean_jaccard"] for r in cal.summary()["subsets"]]
    assert ranked == sorted(ranked, reverse=True)


def test_missing_layers_skip_that_subset_only():
    # Only two rows scored this fire; subsets referencing absent layers are
    # skipped, but subsets fully covered still accumulate.
    cal = LayerCalibrator(enabled=True, num_layers=8)
    stacked = torch.rand(2, 4)
    row_layer_idxs = [1, 6]  # matches the 'endpoints2' default for L=8
    ref_mask = _select_topk(stacked.mean(dim=0), 1)

    def aggregate_and_select(rows: list[int]) -> torch.Tensor:
        idx = torch.tensor(rows, dtype=torch.long)
        return _select_topk(stacked[idx].mean(dim=0), 1)

    cal.observe(row_layer_idxs, ref_mask, aggregate_and_select)
    by_name = {r["subset"]: r for r in cal.summary()["subsets"]}
    assert "endpoints2" in by_name  # both of its layers (1, 6) were present
    # A subset needing a layer we did not score (e.g. mid1=layer 4) is absent.
    assert "mid1" not in by_name


def test_dump_writes_json_report(tmp_path):
    log_path = str(tmp_path / "run.log")
    cal = LayerCalibrator(enabled=True, num_layers=4, log_path=log_path)
    stacked = torch.rand(4, 5)
    ref_mask = _select_topk(stacked.mean(dim=0), 2)

    def aggregate_and_select(rows: list[int]) -> torch.Tensor:
        idx = torch.tensor(rows, dtype=torch.long)
        return _select_topk(stacked[idx].mean(dim=0), 2)

    cal.observe([0, 1, 2, 3], ref_mask, aggregate_and_select)
    cal.dump()
    report = tmp_path / "run.log.layer_calibration.json"
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["fires"] == 1
    assert data["num_layers"] == 4
    assert data["subsets"]


def test_dump_without_fires_writes_nothing(tmp_path):
    log_path = str(tmp_path / "run.log")
    cal = LayerCalibrator(enabled=True, num_layers=4, log_path=log_path)
    cal.dump()
    assert not (tmp_path / "run.log.layer_calibration.json").exists()


def test_calibrate_layers_reproduces_reference_with_full_subset():
    """End-to-end: EvictionPolicy._calibrate_layers re-runs the real aggregate +
    finalize + select pipeline, so a subset covering every scored row must match
    the reference mask exactly (Jaccard 1.0)."""
    from vllm.v1.geo_kv.config import GeoKVConfig
    from vllm.v1.geo_kv.eviction_policy import EvictionPolicy, select_evicted_blocks

    num_layers, num_blocks = 4, 8
    torch.manual_seed(1)
    stacked = torch.rand(num_layers, num_blocks)

    # Build a bare EvictionPolicy without touching the GPU-only constructor; the
    # calibration path only needs config, the calibrator, and the per-fire stash.
    pol = EvictionPolicy.__new__(EvictionPolicy)
    pol.config = GeoKVConfig(
        experiment_mode="geo_uniform",
        eviction_policy="v_redundancy",
        score_sampled_layers="all",
    )
    pol.layer_calibrator = LayerCalibrator(enabled=True, num_layers=num_layers)
    pol._cal_stacked = stacked
    pol._cal_vnorm_stacked = None
    pol._cal_row_layer_idxs = [0, 1, 2, 3]

    count = num_blocks

    def select(scores, vnorm):
        return select_evicted_blocks(scores, count, 0.4, "v_redundancy", 0, 0)

    # Reference: finalize + select over all rows (mean aggregation is the default
    # "mean" set in config for this policy path).
    from vllm.v1.geo_kv.eviction_policy import aggregate_layer_scores

    ref_scores = aggregate_layer_scores(
        stacked, pol.config.block_score_aggregation, pol.config.topk_frac
    )
    ref_scores = pol._finalize_scores(ref_scores, None, False)
    ref_mask = select(ref_scores, None)

    pol._calibrate_layers(ref_mask, select)
    summ = pol.layer_calibrator.summary()
    by_name = {r["subset"]: r for r in summ["subsets"]}
    # A subset spanning all four rows reproduces the reference selection exactly.
    assert by_name["even4"]["mean_jaccard"] == 1.0
    assert summ["fires"] == 1
