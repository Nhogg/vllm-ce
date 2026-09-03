# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the mask-only KV block-eviction policy (Option A).

Covers the pure, engine-independent pieces: block selection determinism and the
matched-count invariant across policies, cross-layer score aggregation, config
validation, and an end-to-end check that duplicated V blocks are the ones the
v_redundancy policy drops first.
"""

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.geo_kv.config import GeoKVConfig, resolve_indices
from vllm.v1.geo_kv.eviction_policy import (
    aggregate_layer_scores,
    select_evicted_blocks,
    select_evicted_by_coverage,
    select_evicted_r2r,
    select_evicted_to_budget,
)
from vllm.v1.geo_kv.scoring import (
    block_anchors,
    block_joint_redundancy,
    block_joint_redundancy_greedy,
    block_joint_similarity,
    block_key_anchor_relevance,
    block_multi_prototype_redundancy,
    block_pair_residual_cost,
    block_prototypes,
    block_query_attention_mass,
    block_value_l2,
    positional_cosh_weights,
    query_direction_coherence,
    refine_with_query_relevance,
    zscore,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec


def test_select_rate_zero_is_all_false():
    scores = torch.rand(8)
    mask = select_evicted_blocks(scores, 8, 0.0, "v_redundancy", 0, 0)
    assert mask.dtype == torch.bool
    assert not bool(mask.any())


def test_query_direction_coherence_detects_rotation_per_head():
    stable = torch.tensor([[[1.0, 0.0]], [[2.0, 0.0]]])
    cancelling = torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]])

    torch.testing.assert_close(
        query_direction_coherence(stable), torch.tensor(1.0)
    )
    torch.testing.assert_close(
        query_direction_coherence(cancelling), torch.tensor(0.0)
    )


def test_runtime_index_resolution_rejects_wrong_architecture_subset():
    assert resolve_indices("3,11,20,28", 28) == [3, 11, 20]
    with pytest.raises(ValueError, match=r"out-of-range indices \[28\]"):
        resolve_indices("3,11,20,28", 28, strict=True)


def test_block_query_attention_mass_is_causal_and_gqa_aware():
    # Two KV heads, two query heads per KV head. The final query points at the
    # first block; a very large future key is causally invisible to query 0.
    q = torch.tensor(
        [
            [[1.0, 0.0]] * 4,
            [[0.0, 1.0]] * 4,
        ]
    )
    k = torch.zeros(2, 2, 2, 2)
    k[0, :, :, 1] = 2.0
    k[1, 1, :, 0] = 100.0
    mass = block_query_attention_mass(q, k, torch.tensor([2, 2]), scale=1.0)
    assert mass.shape == (2,)
    assert torch.isclose(mass.sum(), torch.tensor(1.0), atol=1e-6)
    assert mass[0] > mass[1]


def test_block_query_attention_mass_supports_peak_window_aggregation():
    queries = torch.tensor([[[8.0, 0.0]], [[0.0, 8.0]]])
    keys = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])
    valid = torch.tensor([1, 1])

    mean = block_query_attention_mass(
        queries, keys, valid, scale=1.0, query_aggregation="mean"
    )
    peak = block_query_attention_mass(
        queries, keys, valid, scale=1.0, query_aggregation="max"
    )

    assert torch.all(peak >= mean)
    assert bool(torch.any(peak > mean))


def test_block_key_anchor_relevance_masks_partial_blocks_and_supports_gqa():
    queries = torch.tensor([[[1.0, 0.0]] * 4, [[1.0, 0.0]] * 4])
    keys = torch.zeros(2, 2, 2, 2)
    keys[0, 0, :, 0] = 2.0
    keys[0, 1, :, 0] = -100.0  # padding in the one-token partial block
    keys[1, :, :, 0] = 1.0

    relevance = block_key_anchor_relevance(
        queries, keys, torch.tensor([1, 2]), scale=1.0
    )

    torch.testing.assert_close(relevance, torch.tensor([2.0, 1.0]))


def test_block_key_anchor_relevance_uses_peak_query_not_mean():
    queries = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    keys = torch.tensor([[[[4.0, 0.0]]], [[[0.0, 3.0]]]])

    relevance = block_key_anchor_relevance(
        queries, keys, torch.tensor([1, 1]), scale=1.0
    )

    torch.testing.assert_close(relevance, torch.tensor([4.0, 3.0]))


def test_block_key_anchor_relevance_mean_collapses_query_window():
    queries = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    keys = torch.tensor([[[[4.0, 0.0]]], [[[0.0, 2.0]]]])

    relevance = block_key_anchor_relevance(
        queries,
        keys,
        torch.tensor([1, 1]),
        scale=1.0,
        query_aggregation="mean",
    )

    torch.testing.assert_close(relevance, torch.tensor([2.0, 1.0]))


def test_block_key_anchor_relevance_rejects_incompatible_gqa_shapes():
    with pytest.raises(ValueError, match="Hq % Hkv"):
        block_key_anchor_relevance(
            torch.zeros(1, 3, 2),
            torch.zeros(2, 2, 2, 2),
            torch.tensor([2, 2]),
        )


def test_block_pair_residual_cost_uses_magnitude_and_absolute_cosine():
    anchors = torch.tensor(
        [
            [[1.0, 0.0]],
            [[-1.0, 0.0]],
            [[0.0, 2.0]],
        ]
    )

    cost = block_pair_residual_cost(anchors)

    assert torch.isinf(cost.diagonal()).all()
    torch.testing.assert_close(cost[0, 1], torch.tensor(0.0))
    torch.testing.assert_close(cost[1, 0], torch.tensor(0.0))
    torch.testing.assert_close(cost[2, 0], torch.tensor(2.0))


def test_block_pair_residual_cost_is_conservative_across_heads():
    anchors = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 3.0]],
            [[1.0, 0.0], [3.0, 0.0]],
        ]
    )

    cost = block_pair_residual_cost(anchors)

    # Head 0 is perfectly covered, but head 1 loses its full magnitude.
    torch.testing.assert_close(cost[0, 1], torch.tensor(3.0))


def test_query_relevance_refinement_is_inert_weighted_and_tie_only():
    scores = torch.tensor([0.1, 0.8, 0.8, 1.2])
    relevance = torch.tensor([0.0, 0.9, 0.1, 1.0])
    assert refine_with_query_relevance(scores, relevance, None) is scores

    weighted = refine_with_query_relevance(scores, relevance, 0.5)
    assert weighted[1] < weighted[2]  # relevant member of the tie is protected

    tied = refine_with_query_relevance(scores, relevance, None, tiebreak_only=True)
    assert tied[1] < tied[2]
    # The perturbation does not cross either neighboring non-tied score.
    assert tied[0] < tied[1] < tied[3]


def test_r2r_selects_least_relevant_from_redundancy_shortlist():
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 9.0])
    relevance = torch.tensor([0.9, 0.8, 0.1, 0.2, 0.0, 0.3, 0.4, 1.0])

    mask = select_evicted_r2r(
        scores,
        relevance,
        num_blocks=8,
        budget=6,
        warmup_pages=0,
        candidate_expansion_factor=2.0,
    )

    assert torch.nonzero(mask).flatten().tolist() == [2, 3]


def test_r2r_factor_one_matches_pure_redundancy():
    scores = torch.tensor([0.2, 0.1, 0.4, 0.3, 0.8, 0.7, 0.6, 9.0])
    relevance = torch.tensor([0.0, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 1.0])
    actual = select_evicted_r2r(
        scores,
        relevance,
        num_blocks=8,
        budget=5,
        warmup_pages=1,
        candidate_expansion_factor=1.0,
    )

    assert torch.nonzero(actual).flatten().tolist() == [1, 2, 3]


def test_r2r_large_factor_becomes_pure_relevance_and_keeps_protections():
    scores = torch.arange(10, dtype=torch.float32)
    relevance = torch.tensor([0.0, 0.1, 0.9, 0.8, 0.7, 0.6, 0.2, 0.3, 0.4, 0.0])

    mask = select_evicted_r2r(
        scores,
        relevance,
        num_blocks=10,
        budget=7,
        warmup_pages=1,
        candidate_expansion_factor=100.0,
        tail_protect_frac=0.2,
    )

    assert torch.nonzero(mask).flatten().tolist() == [1, 5, 6]
    assert int(mask.sum()) == 3
    assert not bool(mask[0])
    assert not bool(mask[8:].any())


def test_r2r_twin_guard_keeps_one_duplicate_and_backfills():
    scores = torch.tensor([0.0, 0.0, 0.1, 0.2, 0.9, 1.0])
    relevance = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.9, 1.0])
    similarity = torch.eye(6)
    similarity[0, 1] = similarity[1, 0] = 0.99
    similarity[2, 3] = similarity[3, 2] = 0.5

    unguarded = select_evicted_r2r(
        scores,
        relevance,
        num_blocks=6,
        budget=4,
        warmup_pages=0,
        candidate_expansion_factor=2.0,
    )
    guarded = select_evicted_r2r(
        scores,
        relevance,
        num_blocks=6,
        budget=4,
        warmup_pages=0,
        candidate_expansion_factor=2.0,
        similarity=similarity,
    )

    assert torch.nonzero(unguarded).flatten().tolist() == [0, 1]
    assert torch.nonzero(guarded).flatten().tolist() == [0, 2]
    assert int(guarded.sum()) == 2


def test_r2r_twin_guard_falls_back_to_preserve_exact_budget():
    scores = torch.tensor([0.0, 0.0, 0.2, 1.0])
    relevance = torch.tensor([0.0, 0.1, 0.9, 1.0])
    similarity = torch.eye(4)
    similarity[0, 1] = similarity[1, 0] = 0.99

    mask = select_evicted_r2r(
        scores,
        relevance,
        num_blocks=4,
        budget=2,
        warmup_pages=0,
        candidate_expansion_factor=1.0,
        similarity=similarity,
    )

    assert torch.nonzero(mask).flatten().tolist() == [0, 1]
    assert int(mask.sum()) == 2


def test_r2r_cover_depth_zero_disables_guard():
    scores = torch.tensor([0.0, 0.0, 0.1, 0.2, 0.9, 1.0])
    relevance = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.9, 1.0])
    similarity = torch.eye(6)
    similarity[0, 1] = similarity[1, 0] = 0.99

    mask = select_evicted_r2r(
        scores,
        relevance,
        num_blocks=6,
        budget=4,
        warmup_pages=0,
        candidate_expansion_factor=2.0,
        similarity=similarity,
        cover_depth=0,
    )

    assert torch.nonzero(mask).flatten().tolist() == [0, 1]


def test_r2r_second_cover_allows_eviction_on_later_pass():
    scores = torch.tensor([0.0, 0.1, 0.2, 0.9, 1.0])
    relevance = torch.tensor([0.0, 0.1, 0.2, 0.9, 1.0])
    similarity = torch.eye(5)
    similarity[0, 1] = similarity[1, 0] = 0.99
    similarity[1, 3] = similarity[3, 1] = 0.8
    similarity[2, 0] = similarity[0, 2] = 0.7

    depth_one_stats: dict[str, int] = {}
    depth_one = select_evicted_r2r(
        scores,
        relevance,
        num_blocks=5,
        budget=2,
        warmup_pages=0,
        candidate_expansion_factor=1.0,
        similarity=similarity,
        cover_depth=1,
        selection_stats=depth_one_stats,
    )
    depth_two_stats: dict[str, int] = {}
    depth_two = select_evicted_r2r(
        scores,
        relevance,
        num_blocks=5,
        budget=2,
        warmup_pages=0,
        candidate_expansion_factor=1.0,
        similarity=similarity,
        cover_depth=2,
        selection_stats=depth_two_stats,
    )

    assert int(depth_one.sum()) == 3
    assert int(depth_two.sum()) == 3
    assert depth_one_stats == {
        "candidate_blocks": 3,
        "eligible_blocks": 4,
        "cover_1": 1,
        "backfill": 2,
    }
    assert depth_two_stats == {
        "candidate_blocks": 3,
        "eligible_blocks": 4,
        "cover_1": 1,
        "cover_2": 1,
        "backfill": 1,
    }


def test_r2r_cover_ranking_treats_anti_alignment_as_coverage():
    scores = torch.tensor([0.0, 0.1, 0.2, 1.0])
    relevance = torch.tensor([0.0, 0.1, 0.2, 1.0])
    similarity = torch.eye(4)
    similarity[0, 1] = similarity[1, 0] = -0.99
    similarity[0, 2] = similarity[2, 0] = 0.2
    similarity[2, 3] = similarity[3, 2] = 0.5

    mask = select_evicted_r2r(
        scores,
        relevance,
        num_blocks=4,
        budget=2,
        warmup_pages=0,
        candidate_expansion_factor=1.5,
        similarity=similarity,
        cover_depth=1,
    )

    assert int(mask.sum()) == 2
    assert not (bool(mask[0]) and bool(mask[1]))


def test_r2r_config_enables_query_capture_and_validates_combinations():
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "prefill_evict_frac": 0.5,
            "candidate_expansion_factor": 2.0,
        }
    )
    assert cfg.query_relevance_enabled
    assert cfg.candidate_expansion_factor == 2.0
    assert cfg.r2r_cover_depth == 2
    assert cfg.r2r_query_aggregation == "max"

    key_anchor_cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "prefill_evict_frac": 0.5,
            "candidate_expansion_factor": 2.0,
            "r2r_relevance_signal": "key_anchor",
        }
    )
    assert key_anchor_cfg.r2r_relevance_signal == "key_anchor"

    mean_cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "prefill_evict_frac": 0.5,
            "candidate_expansion_factor": 2.0,
            "r2r_query_aggregation": "mean",
        }
    )
    assert mean_cfg.r2r_query_aggregation == "mean"

    for factor in (0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="candidate_expansion_factor"):
            GeoKVConfig.from_dict(
                {
                    "experiment_mode": "geo_uniform",
                    "prefill_evict_frac": 0.5,
                    "candidate_expansion_factor": factor,
                }
            )
    with pytest.raises(ValueError, match="budget-based eviction"):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "candidate_expansion_factor": 2.0,
            }
        )
    with pytest.raises(ValueError, match="score refinements"):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "prefill_evict_frac": 0.5,
                "candidate_expansion_factor": 2.0,
                "query_alignment_weight": 0.5,
            }
        )
    with pytest.raises(ValueError, match="physical_reclaim"):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "decode_evict_budget": 64,
                "candidate_expansion_factor": 2.0,
            }
        )
    with pytest.raises(ValueError, match="r2r_relevance_signal"):
        GeoKVConfig.from_dict({"r2r_relevance_signal": "key_anchor"})
    with pytest.raises(ValueError, match="r2r_query_aggregation"):
        GeoKVConfig.from_dict({"r2r_query_aggregation": "mean"})
    with pytest.raises(ValueError, match="r2r_query_aggregation"):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "prefill_evict_frac": 0.5,
                "candidate_expansion_factor": 2.0,
                "r2r_query_aggregation": "median",
            }
        )
    with pytest.raises(ValueError, match="r2r_relevance_signal"):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "prefill_evict_frac": 0.5,
                "candidate_expansion_factor": 2.0,
                "r2r_relevance_signal": "unknown",
            }
        )
    for depth in (-1, 1.5, True):
        with pytest.raises(ValueError, match="r2r_cover_depth"):
            GeoKVConfig.from_dict(
                {
                    "experiment_mode": "geo_uniform",
                    "prefill_evict_frac": 0.5,
                    "candidate_expansion_factor": 2.0,
                    "r2r_cover_depth": depth,
                }
            )
    with pytest.raises(ValueError, match="r2r_cover_depth"):
        GeoKVConfig.from_dict({"r2r_cover_depth": 0})


def test_query_relevance_config_is_opt_in_and_validated():
    assert not GeoKVConfig().query_relevance_enabled
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "query_alignment_weight": 0.5,
            "query_tail_tokens": 16,
        }
    )
    assert cfg.query_relevance_enabled
    assert cfg.query_tail_tokens == 16
    with pytest.raises(ValueError, match="query_alignment_weight"):
        GeoKVConfig.from_dict({"query_alignment_weight": -0.1})
    with pytest.raises(ValueError, match="eviction_policy='v_redundancy'"):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "eviction_policy": "value_l2",
                "enable_query_tiebreak": True,
            }
        )


def test_query_relevance_hard_protection_config_is_inert_and_composes_with_cosh():
    assert GeoKVConfig().query_relevance_protect_quantile is None
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "physical_reclaim": True,
            "decode_evict_budget": 64,
            "query_relevance_protect_quantile": 0.9,
            "positional_cosh_alpha": 1.5,
        }
    )
    assert cfg.query_relevance_enabled
    assert cfg.query_relevance_protect_quantile == 0.9
    assert cfg.positional_cosh_alpha == 1.5

    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError, match="query_relevance_protect_quantile"):
            GeoKVConfig.from_dict(
                {
                    "experiment_mode": "geo_uniform",
                    "physical_reclaim": True,
                    "decode_evict_budget": 64,
                    "query_relevance_protect_quantile": bad,
                }
            )


def test_query_relevance_hard_protection_rejects_incompatible_paths():
    common = {
        "experiment_mode": "geo_uniform",
        "physical_reclaim": True,
        "decode_evict_budget": 64,
        "query_relevance_protect_quantile": 0.9,
    }
    with pytest.raises(ValueError, match="eviction_policy='v_redundancy'"):
        GeoKVConfig.from_dict({**common, "eviction_policy": "value_l2"})
    with pytest.raises(ValueError, match="redundancy_mode='pairwise'"):
        GeoKVConfig.from_dict({**common, "redundancy_mode": "coverage"})
    with pytest.raises(ValueError, match="incremental_decode_scoring"):
        GeoKVConfig.from_dict({**common, "incremental_decode_scoring": True})
    with pytest.raises(ValueError, match="budget-based eviction"):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "query_relevance_protect_quantile": 0.9,
            }
        )


def test_query_relevance_hard_protection_shields_high_relevance_blocks():
    scores = torch.tensor([0.99, 0.9, 0.8, 0.7, 0.6, 0.0])
    relevance = torch.tensor([0.99, 0.9, 0.1, 0.2, 0.3, 1.0])
    mask = select_evicted_to_budget(
        scores,
        num_blocks=6,
        budget=4,
        query_relevance=relevance,
        query_relevance_protect_quantile=0.6,
        warmup_pages=0,
    )
    assert torch.nonzero(mask).flatten().tolist() == [2, 3]


def test_query_relevance_hard_protection_relaxes_lowest_relevance_exactly():
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.0])
    relevance = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 9.0])
    mask = select_evicted_to_budget(
        scores,
        num_blocks=6,
        budget=2,
        query_relevance=relevance,
        query_relevance_protect_quantile=0.0,
        warmup_pages=0,
    )
    assert torch.nonzero(mask).flatten().tolist() == [1, 2, 3, 4]
    assert not bool(mask[0])
    assert not bool(mask[5])


def test_query_relevance_hard_protection_relaxation_ties_use_block_order():
    scores = torch.ones(6)
    relevance = torch.tensor([9.0, 1.0, 1.0, 2.0, 3.0, 10.0])
    mask = select_evicted_to_budget(
        scores,
        num_blocks=6,
        budget=4,
        query_relevance=relevance,
        query_relevance_protect_quantile=0.0,
        warmup_pages=0,
    )
    assert torch.nonzero(mask).flatten().tolist() == [1, 2]


def test_query_relevance_hard_protection_none_is_exactly_inert():
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.0])
    relevance = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 1.0])
    base = select_evicted_to_budget(
        scores, num_blocks=6, budget=3, warmup_pages=0
    )
    disabled = select_evicted_to_budget(
        scores,
        num_blocks=6,
        budget=3,
        warmup_pages=0,
        query_relevance=relevance,
        query_relevance_protect_quantile=None,
    )
    torch.testing.assert_close(base, disabled, rtol=0, atol=0)


def test_select_rate_one_keeps_sink_and_last():
    n, warmup = 10, 2
    mask = select_evicted_blocks(torch.zeros(n), n, 1.0, "recency", warmup, 0)
    # Leading sink blocks and the final (decode-anchor) block are always kept.
    assert not bool(mask[:warmup].any())
    assert not bool(mask[n - 1])
    # Everything strictly between is evicted at rate 1.0.
    assert bool(mask[warmup : n - 1].all())


def test_select_never_evicts_last_block():
    n = 6
    for rate in (0.5, 0.9, 1.0):
        for policy in ("v_redundancy", "recency", "random"):
            mask = select_evicted_blocks(torch.rand(n), n, rate, policy, 0, 0)
            assert not bool(mask[n - 1]), (policy, rate)


def test_select_v_redundancy_picks_highest_scores():
    # Evictable window is [0, n-1); block 3 has the top score, block 1 next.
    scores = torch.tensor([0.1, 0.8, 0.2, 0.9, 0.3, 0.0])
    n = scores.numel()
    mask = select_evicted_blocks(
        scores, n, rate=0.5, policy="v_redundancy", warmup_pages=0, seed=0
    )
    # evictable = [0..4], k = round(0.5 * 5) = 2 -> the two highest: idx 3, 1.
    assert bool(mask[3]) and bool(mask[1])
    assert int(mask.sum()) == 2


def test_select_recency_picks_oldest():
    n = 8
    mask = select_evicted_blocks(torch.zeros(n), n, 0.5, "recency", 1, 0)
    # evictable = [1..6], k = round(0.5 * 6) = 3 -> oldest three: 1, 2, 3.
    assert [i for i in range(n) if bool(mask[i])] == [1, 2, 3]


def test_select_random_is_deterministic_and_matched_count():
    n = 20
    a = select_evicted_blocks(torch.rand(n), n, 0.4, "random", 0, seed=7)
    b = select_evicted_blocks(torch.rand(n), n, 0.4, "random", 0, seed=7)
    assert bool((a == b).all())  # same seed -> same mask
    c = select_evicted_blocks(torch.rand(n), n, 0.4, "random", 0, seed=8)
    # Matched count: every policy drops the same number at a given rate.
    v = select_evicted_blocks(torch.rand(n), n, 0.4, "v_redundancy", 0, 0)
    r = select_evicted_blocks(torch.rand(n), n, 0.4, "recency", 0, 0)
    assert int(a.sum()) == int(c.sum()) == int(v.sum()) == int(r.sum())


def test_budget_noop_when_under_budget():
    # num_blocks <= budget -> nothing to evict, regardless of policy.
    for policy in ("v_redundancy", "recency", "random"):
        mask = select_evicted_to_budget(
            torch.rand(5), num_blocks=5, budget=8, warmup_pages=0, policy=policy
        )
        assert not bool(mask.any()), policy


def test_budget_v_redundancy_drops_highest_scores_to_budget():
    # 8 blocks down to budget 5 -> drop k = 3 highest-score evictable blocks.
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.0])
    mask = select_evicted_to_budget(
        scores, num_blocks=8, budget=5, warmup_pages=0, policy="v_redundancy"
    )
    # evictable = [0..6] (block 7 is the kept anchor); top-3 scores: idx 1,3,5.
    assert int(mask.sum()) == 3
    assert bool(mask[1]) and bool(mask[3]) and bool(mask[5])
    assert not bool(mask[7])  # anchor kept


def test_scored_selection_stable_tie_breaks_by_lowest_index():
    # Phase 1A: with tied top scores, the stable sort evicts the LOWEST block
    # index first. Blocks 1, 3, 5 all share the max score; budget cut of 2 must
    # pick the two lowest-index ties (1, 3), deterministically.
    scores = torch.tensor([0.1, 0.9, 0.2, 0.9, 0.3, 0.9, 0.0, 0.0])
    mask = select_evicted_to_budget(
        scores, num_blocks=8, budget=6, warmup_pages=0, policy="v_redundancy"
    )
    assert int(mask.sum()) == 2
    assert bool(mask[1]) and bool(mask[3])
    assert not bool(mask[5])  # the third tie is left in place (lowest-index wins)
    assert not bool(mask[7])  # anchor kept


def test_scored_selection_is_deterministic_across_repeats():
    # The stable tie rule makes selection reproducible run-to-run even with ties.
    scores = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    first = select_evicted_to_budget(
        scores, num_blocks=6, budget=3, warmup_pages=1, policy="v_redundancy"
    )
    for _ in range(4):
        again = select_evicted_to_budget(
            scores, num_blocks=6, budget=3, warmup_pages=1, policy="v_redundancy"
        )
        assert bool((again == first).all())
    # evictable = [1..4]; k = 6 - 3 = 3, all tied -> lowest three: 1, 2, 3.
    assert [i for i in range(6) if bool(first[i])] == [1, 2, 3]


def test_norm_protect_shields_high_value_blocks():
    # Phase 2: the most-redundant blocks (highest score) are also the highest
    # value-norm, so protecting the top norm-quantile forces the cut onto the
    # next-most-redundant *low-norm* blocks instead.
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.0])
    value_norm = torch.tensor([9.0, 8.0, 1.0, 1.0, 1.0, 1.0])
    # Without protection, k=2 would drop the top-2 scores: blocks 0 and 1.
    plain = select_evicted_to_budget(
        scores, num_blocks=6, budget=4, warmup_pages=0, policy="v_redundancy"
    )
    assert [i for i in range(6) if bool(plain[i])] == [0, 1]
    # Protect the top ~40% by norm (blocks 0, 1) -> cut falls on 2, 3.
    prot = select_evicted_to_budget(
        scores,
        num_blocks=6,
        budget=4,
        warmup_pages=0,
        policy="v_redundancy",
        value_norm=value_norm,
        value_norm_protect_quantile=0.6,
    )
    assert int(prot.sum()) == 2
    assert [i for i in range(6) if bool(prot[i])] == [2, 3]
    assert not bool(prot[0]) and not bool(prot[1])  # high-norm blocks kept


def test_norm_protect_relaxes_lowest_norm_to_keep_budget_exact():
    # If protection would leave too few candidates to reach the budget, the
    # LOWEST-norm protected blocks give up protection first so the budget stays
    # exact. Here every evictable block is above the q=0 threshold (all protected),
    # but k=3 must still be evicted: the three lowest-norm blocks are released.
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.0])
    value_norm = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 9.0])
    mask = select_evicted_to_budget(
        scores,
        num_blocks=6,
        budget=2,  # keep 2 -> evict k = 6 - 2 = 4? anchor kept -> window [0..4]
        warmup_pages=0,
        policy="v_redundancy",
        value_norm=value_norm,
        value_norm_protect_quantile=0.0,  # protect all evictable blocks
    )
    # Evictable window is [0..4]; k = num_blocks - budget = 4. With all protected
    # and relaxation by ascending norm, the 4 lowest-norm evictable blocks are
    # released, i.e. all but the highest-norm block 0 -> evict {1,2,3,4}.
    assert [i for i in range(6) if bool(mask[i])] == [1, 2, 3, 4]
    assert not bool(mask[0])  # highest-norm evictable block survives
    assert not bool(mask[5])  # anchor always kept


def test_norm_protect_quantile_zero_matches_value_l2_ordering():
    # q=0 protects everything then relaxes by ascending norm to exactly k, so the
    # evicted set is the k lowest-norm evictable blocks -- the value_l2 cut.
    scores = torch.rand(8)  # redundancy ordering is irrelevant at q=0
    value_norm = torch.tensor([7.0, 1.0, 5.0, 2.0, 6.0, 3.0, 4.0, 8.0])
    mask = select_evicted_to_budget(
        scores,
        num_blocks=8,
        budget=5,  # k = 3
        warmup_pages=0,
        policy="v_redundancy",
        value_norm=value_norm,
        value_norm_protect_quantile=0.0,
    )
    # Evictable window [0..6]; 3 lowest norms there are blocks 1 (1.0), 3 (2.0),
    # 5 (3.0). Block 7 is the anchor (excluded regardless of its high norm).
    assert [i for i in range(8) if bool(mask[i])] == [1, 3, 5]


def test_norm_protect_none_is_byte_identical_to_pinned_path():
    # value_norm_protect_quantile=None must not perturb the pinned selection.
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.0])
    value_norm = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    base = select_evicted_to_budget(
        scores, num_blocks=8, budget=5, warmup_pages=0, policy="v_redundancy"
    )
    off = select_evicted_to_budget(
        scores,
        num_blocks=8,
        budget=5,
        warmup_pages=0,
        policy="v_redundancy",
        value_norm=value_norm,
        value_norm_protect_quantile=None,
    )
    assert bool((base == off).all())


def test_norm_protect_never_relaxes_sink_anchor_or_tail():
    # Sink / anchor / tail sit OUTSIDE the evictable window, so even q=0 (protect
    # everything, then relax to budget) must never evict them. Give the protected
    # ends the LOWEST value norms -- if they were in the relaxable set they would
    # be the first to go. n=10, warmup=2 sink, tail_protect_frac shields the tail.
    n = 10
    scores = torch.arange(n, dtype=torch.float32) / n  # arbitrary redundancy
    value_norm = torch.ones(n)
    value_norm[:2] = 0.01  # sink blocks: lowest norm (tempting to relax)
    value_norm[-2:] = 0.01  # anchor + tail: lowest norm
    mask = select_evicted_to_budget(
        scores,
        num_blocks=n,
        budget=4,  # aggressive cut so relaxation is forced
        warmup_pages=2,
        policy="v_redundancy",
        tail_protect_frac=0.15,  # shields ~ceil(0.15*10)=2 tail blocks
        value_norm=value_norm,
        value_norm_protect_quantile=0.0,  # protect all, relax by norm to budget
    )
    assert not bool(mask[:2].any())  # sink never evicted
    assert not bool(mask[n - 1])  # anchor never evicted
    assert not bool(mask[n - 2])  # tail-protected block never evicted
    # The realized cut must still be exact regardless of the protected ends.
    assert int(mask.sum()) == n - 4


def test_budget_recency_drops_oldest_to_budget_and_protects_sink():
    # StreamingLLM contrast: oldest-first, but the sink (warmup) and the final
    # anchor are always kept. 10 blocks -> budget 6 => drop k = 4.
    n, warmup = 10, 2
    mask = select_evicted_to_budget(
        torch.zeros(n), num_blocks=n, budget=6, warmup_pages=warmup, policy="recency"
    )
    # evictable = [2..8]; oldest four: 2, 3, 4, 5.
    assert [i for i in range(n) if bool(mask[i])] == [2, 3, 4, 5]
    assert not bool(mask[:warmup].any())  # sink kept
    assert not bool(mask[n - 1])  # anchor kept


def test_budget_random_is_seed_reproducible_and_matched_count():
    n = 24
    a = select_evicted_to_budget(
        torch.rand(n), num_blocks=n, budget=16, warmup_pages=0, policy="random", seed=7
    )
    b = select_evicted_to_budget(
        torch.rand(n), num_blocks=n, budget=16, warmup_pages=0, policy="random", seed=7
    )
    assert bool((a == b).all())  # same seed -> same mask
    # Every policy drops exactly num_blocks - budget = 8.
    v = select_evicted_to_budget(
        torch.rand(n), num_blocks=n, budget=16, warmup_pages=0, policy="v_redundancy"
    )
    r = select_evicted_to_budget(
        torch.zeros(n), num_blocks=n, budget=16, warmup_pages=0, policy="recency"
    )
    assert int(a.sum()) == int(v.sum()) == int(r.sum()) == 8


def test_budget_recency_matches_rate_path_at_equal_k():
    # The budget path and the rate path must select the SAME oldest blocks when
    # asked to drop the same count -- both delegate to _choose_evicted.
    n, warmup = 12, 1
    # Rate 0.5 over evictable [1..10] (10 blocks) -> k = 5.
    rate_mask = select_evicted_blocks(torch.zeros(n), n, 0.5, "recency", warmup, 0)
    k = int(rate_mask.sum())
    budget_mask = select_evicted_to_budget(
        torch.zeros(n),
        num_blocks=n,
        budget=n - k,
        warmup_pages=warmup,
        policy="recency",
    )
    assert bool((rate_mask == budget_mask).all())


def test_config_capacity_band_allows_recency_policy():
    # The band now honors any policy (recency == StreamingLLM at matched memory),
    # so this must validate rather than raise.
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "eviction_policy": "recency",
            "decode_evict_budget": 64,
            "decode_evict_watermark": 0.75,
        }
    )
    assert cfg.eviction_policy == "recency"
    assert cfg.decode_evict_budget == 64


@pytest.mark.parametrize("policy", ["recency", "random", "value_l2"])
@pytest.mark.parametrize("knob", ["decode_evict_frac", "prefill_evict_frac"])
def test_config_frac_band_allows_baseline_policies(policy, knob):
    # The fraction-of-prompt path (Stage D) must honor the matched-memory
    # baselines too, exactly like the budget path -- select_evicted_to_budget
    # dispatches on eviction_policy, so the config must not restrict it.
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "eviction_policy": policy,
            knob: 1.0,
        }
    )
    assert cfg.eviction_policy == policy
    assert getattr(cfg, knob) == 1.0


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_config_frac_band_rejects_out_of_range(bad):
    with pytest.raises(ValueError, match="must be in"):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "eviction_policy": "recency",
                "decode_evict_frac": bad,
            }
        )


def test_config_frac_band_requires_active_eviction():
    with pytest.raises(ValueError, match="active_eviction"):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "score_only",
                "decode_evict_frac": 0.5,
            }
        )


def test_aggregate_layer_scores_modes():
    stacked = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])  # (layers, blk)
    assert torch.allclose(
        aggregate_layer_scores(stacked, "mean", 0.1), torch.tensor([2.0, 3.0])
    )
    assert torch.allclose(
        aggregate_layer_scores(stacked, "max", 0.1), torch.tensor([4.0, 5.0])
    )
    # topk_mean with frac -> 1 layer -> equals max here.
    assert torch.allclose(
        aggregate_layer_scores(stacked, "topk_mean", 0.1),
        torch.tensor([4.0, 5.0]),
    )


def test_config_active_eviction_and_validation():
    cfg = GeoKVConfig.from_dict({"experiment_mode": "geo_uniform"})
    assert cfg.active_eviction
    assert cfg.eviction_policy == "v_redundancy"
    # score_only is not an eviction mode.
    assert not GeoKVConfig.from_dict({"experiment_mode": "score_only"}).active_eviction


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_config_rejects_bad_eviction_rate(rate):
    with pytest.raises(ValueError):
        GeoKVConfig.from_dict({"experiment_mode": "geo_uniform", "eviction_rate": rate})


def test_config_rejects_bad_eviction_policy():
    with pytest.raises(ValueError):
        GeoKVConfig.from_dict(
            {"experiment_mode": "geo_uniform", "eviction_policy": "bogus"}
        )


def test_duplicated_v_blocks_are_evicted_first():
    # Four blocks; blocks 1 and 2 are identical (max mutual cosine), block 0 and
    # 3 are orthogonal to everything. Block 3 is the always-kept final block.
    vecs = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # block 0 (unique)
            [0.0, 1.0, 0.0, 0.0],  # block 1 (A)
            [0.0, 1.0, 0.0, 0.0],  # block 2 (A duplicate)
            [0.0, 0.0, 1.0, 0.0],  # block 3 (unique, final -> kept)
        ]
    )
    B, S, H, D = 4, 2, 1, 4
    v = vecs.view(B, 1, H, D).expand(B, S, H, D).contiguous()
    valid = torch.full((B,), S, dtype=torch.long)
    scores = block_joint_redundancy(block_anchors(v, valid))  # (4,)
    # Duplicates should carry the highest redundancy.
    assert scores[1] > scores[0] and scores[2] > scores[0]

    # Drop exactly one evictable block; it must be one of the duplicates.
    mask = select_evicted_blocks(
        scores, B, rate=0.34, policy="v_redundancy", warmup_pages=0, seed=0
    )
    assert int(mask.sum()) == 1
    evicted = [i for i in range(B) if bool(mask[i])]
    assert evicted[0] in (1, 2)


def test_block_value_l2_masks_partial_final_block():
    # value_l2 score = norm(V, p=2, dim=-1).mean(heads).sum(valid tokens).
    # With one head and each block's vectors set to a fixed magnitude m along a
    # single axis, per-token norm == m, so score == m * valid_len. Block 2's last
    # token is padding (valid_len=1), so masking must exclude it.
    B, S, H, D = 3, 2, 1, 4
    v = torch.zeros(B, S, H, D)
    v[0, :, 0, 0] = 0.5  # block 0: two valid tokens, m=0.5 -> 1.0
    v[1, :, 0, 0] = 2.0  # block 1: two valid tokens, m=2.0 -> 4.0
    v[2, :, 0, 0] = 3.0  # block 2: one valid token,  m=3.0 -> 3.0 (mask 2nd)
    valid = torch.tensor([2, 2, 1], dtype=torch.long)
    score = block_value_l2(v, valid)
    assert torch.allclose(score, torch.tensor([1.0, 4.0, 3.0]))


def test_block_value_l2_rare_signal_reductions_mask_padding():
    v = torch.zeros(2, 3, 2, 2)
    # Block 0 has a rare strong token in one head; block 1 has distributed
    # moderate signal. Its third token is padding and must not affect maxima.
    v[0, 1, 1, 0] = 10.0
    v[1, :2, :, 0] = 3.0
    v[1, 2, :, 0] = 100.0
    valid = torch.tensor([3, 2])

    assert torch.allclose(block_value_l2(v, valid, "sum"), torch.tensor([5.0, 6.0]))
    assert torch.allclose(
        block_value_l2(v, valid, "max_token"), torch.tensor([5.0, 3.0])
    )
    assert torch.allclose(
        block_value_l2(v, valid, "max_token_head"), torch.tensor([10.0, 3.0])
    )


def test_value_l2_block_reduction_config_is_opt_in_and_validated():
    assert GeoKVConfig().value_l2_block_reduction == "sum"
    for reduction in ("sum", "max_token", "max_token_head"):
        cfg = GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "eviction_policy": "value_l2",
                "value_l2_block_reduction": reduction,
            }
        )
        assert cfg.value_l2_block_reduction == reduction
    with pytest.raises(ValueError, match="value_l2_block_reduction"):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "eviction_policy": "v_redundancy",
                "value_l2_block_reduction": "max_token",
            }
        )
    with pytest.raises(ValueError, match="value_l2_block_reduction"):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "eviction_policy": "value_l2",
                "value_l2_block_reduction": "bogus",
            }
        )


def test_budget_value_l2_drops_lowest_norm_blocks_and_protects_ends():
    # Paged-Eviction baseline on the budget path: keep the highest value-L2-norm
    # blocks, drop the lowest. Sink (warmup) and the final anchor are protected.
    # 6 blocks -> budget 4 => drop k=2, from the evictable interior [1..4].
    B, S, H, D = 6, 2, 1, 4
    mags = torch.tensor([9.0, 0.9, 0.1, 0.8, 0.2, 9.0])  # per-block magnitude
    v = torch.zeros(B, S, H, D)
    for b in range(B):
        v[b, :, 0, 0] = mags[b]
    valid = torch.full((B,), S, dtype=torch.long)
    # Runner negates the norm so "higher == more droppable" (lowest norm first).
    scores = -block_value_l2(v, valid)
    mask = select_evicted_to_budget(
        scores, num_blocks=B, budget=4, warmup_pages=1, policy="value_l2"
    )
    # Lowest-norm interior blocks are 2 (0.1) and 4 (0.2).
    assert [i for i in range(B) if bool(mask[i])] == [2, 4]
    assert not bool(mask[0])  # sink kept
    assert not bool(mask[B - 1])  # anchor kept


def test_tail_protect_shields_last_blocks_and_stays_matched():
    # Tail-protect must (a) never evict the protected tail, and (b) still drop
    # exactly num_blocks-budget blocks (memory matched). 12 blocks -> budget 8 =>
    # k=4. With warmup=1 the base window is [1..10] (block 11 = anchor). Scores
    # make the *tail* blocks look most droppable, so without protection they'd be
    # chosen; protection must force selection into the earlier interior instead.
    n = 12
    scores = torch.zeros(n)
    scores[8], scores[9], scores[10] = 3.0, 2.0, 1.0  # tail looks droppable
    scores[2], scores[3], scores[4], scores[5] = 0.9, 0.8, 0.7, 0.6
    mask = select_evicted_to_budget(
        scores, num_blocks=n, budget=8, warmup_pages=1,
        policy="v_redundancy", tail_protect_frac=0.34,  # ceil(.34*12)=5 -> [7..10]
    )
    assert int(mask.sum()) == 4  # matched: still evicts to budget
    protected = {7, 8, 9, 10}
    assert not any(bool(mask[i]) for i in protected)  # tail shielded
    assert not bool(mask[0])  # sink kept
    assert not bool(mask[n - 1])  # anchor kept
    # The four highest-scoring *unprotected* interior blocks (2,3,4,5) are chosen.
    assert [i for i in range(n) if bool(mask[i])] == [2, 3, 4, 5]


def test_tail_protect_none_is_identical_to_pinned():
    # frac=None and frac=0.0 must reproduce the pinned v_redundancy selection
    # exactly (the fix is inert unless the flag is set).
    n = 16
    scores = torch.rand(n)
    base = select_evicted_to_budget(
        scores, num_blocks=n, budget=10, warmup_pages=1, policy="v_redundancy"
    )
    for frac in (None, 0.0):
        got = select_evicted_to_budget(
            scores, num_blocks=n, budget=10, warmup_pages=1,
            policy="v_redundancy", tail_protect_frac=frac,
        )
        assert bool((got == base).all()), frac


def test_tail_protect_clamped_when_budget_needs_the_tail():
    # If the protected tail would leave too few evictable blocks to reach budget,
    # protection is clamped so budget still wins (memory matched is invariant).
    # 6 blocks -> budget 2 => k=4; interior [1..4] has only 4 blocks, so a large
    # tail_protect_frac must be clamped to 0 to still drop all 4.
    n = 6
    mask = select_evicted_to_budget(
        torch.zeros(n), num_blocks=n, budget=2, warmup_pages=1,
        policy="recency", tail_protect_frac=0.9,
    )
    assert int(mask.sum()) == 4  # budget reached despite the aggressive request


def test_config_accepts_and_rejects_tail_protect_frac():
    ok = GeoKVConfig.from_dict(
        {"experiment_mode": "geo_uniform", "query_tail_protect_frac": 0.25}
    )
    assert ok.query_tail_protect_frac == 0.25
    for bad in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            GeoKVConfig.from_dict(
                {"experiment_mode": "geo_uniform", "query_tail_protect_frac": bad}
            )


# --- Positional cosh (sech-bump) re-weighting -------------------------------


def test_cosh_weights_peak_in_middle_and_decay_to_ends():
    w = positional_cosh_weights(9, alpha=1.5, device=torch.device("cpu"))
    assert w.shape == (9,)
    # Symmetric U-curve: middle == 1.0 (sech(0)), ends smallest and equal.
    mid = w.shape[0] // 2
    assert torch.isclose(w[mid], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(w[0], w[-1], atol=1e-6)
    assert w[0] < w[mid] and w[-1] < w[mid]
    # Monotone non-increasing from the middle out to each end.
    left = w[: mid + 1]
    assert bool((left[1:] >= left[:-1] - 1e-6).all())
    # sech is in (0, 1]; ends equal 1/cosh(alpha).
    assert bool((w > 0).all()) and bool((w <= 1.0 + 1e-6).all())
    assert torch.isclose(w[0], 1.0 / torch.cosh(torch.tensor(1.5)), atol=1e-6)


def test_cosh_weights_alpha_zero_and_singleton_are_ones():
    dev = torch.device("cpu")
    assert bool((positional_cosh_weights(10, 0.0, dev) == 1.0).all())
    assert bool((positional_cosh_weights(1, 5.0, dev) == 1.0).all())  # x=0 -> 1
    assert positional_cosh_weights(0, 5.0, dev).shape == (0,)


def test_cosh_larger_alpha_protects_ends_more():
    dev = torch.device("cpu")
    soft = positional_cosh_weights(11, 1.0, dev)
    hard = positional_cosh_weights(11, 3.0, dev)
    # Steeper alpha pushes the ends further down (more end protection) while the
    # middle stays pinned at 1.0.
    assert hard[0] < soft[0]
    assert torch.isclose(hard[5], soft[5], atol=1e-6)  # shared middle


def test_config_accepts_and_rejects_positional_cosh_alpha():
    for good in (0.0, 0.5, 2.0):
        cfg = GeoKVConfig.from_dict(
            {"experiment_mode": "geo_uniform", "positional_cosh_alpha": good}
        )
        assert cfg.positional_cosh_alpha == good
    with pytest.raises(ValueError):
        GeoKVConfig.from_dict(
            {"experiment_mode": "geo_uniform", "positional_cosh_alpha": -0.1}
        )


def _cosh_rescore(raw: torch.Tensor, alpha: float) -> torch.Tensor:
    """Mirror the scorer's shift-then-multiply so tests exercise the real math."""
    w = positional_cosh_weights(raw.shape[0], alpha, raw.device, raw.dtype)
    return (raw - raw.min()) * w


def test_cosh_reweight_pulls_eviction_off_the_ends():
    # Raw droppability that looks *highest at the two ends* (a mild U-shape): a
    # plain argsort would evict the end blocks first. The cosh re-weight (ends
    # shielded, middle inflated) must move the top-k selection inward.
    raw = torch.tensor(
        [0.9, 0.8, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2, 0.3, 0.8, 0.9]
    )  # peaks at indices 0 and 10
    # Without re-weight: the top-2 droppable are the very ends.
    assert set(torch.argsort(raw, descending=True)[:2].tolist()) == {0, 10}
    # With a strong sech bump the ends are suppressed; the top-k moves inward
    # (away from both end blocks 0 and 10).
    rescored = _cosh_rescore(raw, alpha=3.0)
    top4 = set(torch.argsort(rescored, descending=True)[:4].tolist())
    assert 0 not in top4 and 10 not in top4


def test_cosh_alpha_zero_preserves_ranking():
    # alpha=0 (weights all 1.0) must leave the *ordering* untouched vs the raw
    # score: the shift is a monotone affine, so argsort is identical. This is the
    # byte-identical-when-off guarantee at the selection level.
    raw = torch.tensor([0.2, -0.5, 0.9, 0.1, 0.4, -0.3, 0.7])
    rescored = _cosh_rescore(raw, alpha=0.0)
    assert bool(
        (
            torch.argsort(rescored, descending=True)
            == torch.argsort(raw, descending=True)
        ).all()
    )


def _make_manager(num_blocks: int = 32, block_size: int = 16):
    """Build a real FullAttentionManager over a real BlockPool (CPU-only)."""
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )
    pool = BlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=True,
        hash_block_size=block_size,
    )
    mgr = FullAttentionManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    return mgr, pool


def _own_blocks(mgr, pool, req_id, n, cached_idxs=()):
    """Give ``req_id`` ``n`` freshly-allocated blocks (ref_cnt==1 each).

    Blocks at ``cached_idxs`` get a non-None hash so free_blocks_at routes them
    through the cached (append) branch; the rest stay uncached (prepend branch).
    """
    blocks = pool.get_new_blocks(n)
    for i in cached_idxs:
        # free_blocks_at only checks ``block_hash is None``; set the backing
        # field directly to avoid building a real BlockHashWithGroupId.
        blocks[i]._block_hash = object()
    mgr.req_to_blocks[req_id] = blocks
    return blocks


def test_free_blocks_at_nulls_interior_and_returns_count():
    mgr, pool = _make_manager()
    _own_blocks(mgr, pool, "r0", 6, cached_idxs=(1,))
    free_before = pool.get_num_free_blocks()

    freed = mgr.free_blocks_at("r0", [1, 3])

    assert freed == 2
    assert mgr.req_to_blocks["r0"][1].is_null
    assert mgr.req_to_blocks["r0"][3].is_null
    assert not mgr.req_to_blocks["r0"][0].is_null
    assert not mgr.req_to_blocks["r0"][2].is_null
    # Freed backing returned to the pool; length preserved (null substitution).
    assert pool.get_num_free_blocks() == free_before + 2
    assert len(mgr.req_to_blocks["r0"]) == 6


def test_free_blocks_at_protects_tail_and_out_of_bounds():
    mgr, pool = _make_manager()
    _own_blocks(mgr, pool, "r0", 4)
    free_before = pool.get_num_free_blocks()

    # index 3 == tail (n-1); 99 and -1 are OOB -> all skipped.
    freed = mgr.free_blocks_at("r0", [3, 99, -1])

    assert freed == 0
    assert not any(b.is_null for b in mgr.req_to_blocks["r0"])
    assert pool.get_num_free_blocks() == free_before


def test_free_blocks_at_skips_already_null():
    mgr, pool = _make_manager()
    _own_blocks(mgr, pool, "r0", 5)

    assert mgr.free_blocks_at("r0", [1]) == 1
    assert mgr.free_blocks_at("r0", [1]) == 0  # second attempt is a no-op


def test_free_blocks_at_unknown_request_is_noop():
    mgr, pool = _make_manager()
    free_before = pool.get_num_free_blocks()

    assert mgr.free_blocks_at("ghost", [0, 1]) == 0
    assert "ghost" not in mgr.req_to_blocks  # .get() must not resurrect an entry
    assert pool.get_num_free_blocks() == free_before


def test_free_blocks_at_uncached_freed_first_for_reuse():
    mgr, pool = _make_manager()
    blocks = _own_blocks(mgr, pool, "r0", 8, cached_idxs=(1,))
    uncached = blocks[4]  # block_hash is None -> prepend branch

    freed = mgr.free_blocks_at("r0", [1, 4])  # 1 cached (append), 4 scratch (prepend)

    assert freed == 2
    # prepend=True puts the scratch block at the front of the free queue,
    # so the next allocation hands it back first.
    assert pool.get_new_blocks(1)[0] is uncached


# -- Greedy (iterative) redundancy: protect the surviving twin ----------------


def _dup_pair_anchors() -> torch.Tensor:
    """Anchors where blocks 2 & 3 are a near-duplicate pair carrying the ONLY
    copy of one topic, and blocks 4/5/6 are a mutually-similar filler cluster.
    Block 0 is the sink, block 7 the decode anchor (both protected downstream).
    """
    torch.manual_seed(0)
    H, D = 2, 8
    anc = torch.randn(8, H, D)
    anc[3] = anc[2] + 0.01 * torch.randn(H, D)  # topic-X duplicate pair
    anc[5] = anc[4] + 0.05 * torch.randn(H, D)  # filler cluster
    anc[6] = anc[4] + 0.05 * torch.randn(H, D)
    return anc


def test_pairwise_double_drops_duplicate_pair():
    # Baseline failure mode: pairwise redundancy scores BOTH twins maximally,
    # so a budget cut of 3 drops both 2 and 3 -- destroying topic X entirely.
    anc = _dup_pair_anchors()
    scores = block_joint_redundancy(anc)
    mask = select_evicted_to_budget(
        scores, num_blocks=8, budget=5, warmup_pages=1, policy="v_redundancy"
    )
    evicted = {i for i in range(8) if bool(mask[i])}
    assert int(mask.sum()) == 3
    assert 2 in evicted and 3 in evicted  # both twins gone (the bug)


def test_greedy_protects_surviving_twin_at_matched_memory():
    # The fix: greedy peels one twin, then its partner is no longer redundant
    # and is protected. Same budget (drops 3), but topic X survives.
    anc = _dup_pair_anchors()
    protect = torch.zeros(8, dtype=torch.bool)
    protect[0] = True  # sink
    protect[7] = True  # anchor
    scores = block_joint_redundancy_greedy(anc, protect=protect)
    mask = select_evicted_to_budget(
        scores, num_blocks=8, budget=5, warmup_pages=1, policy="v_redundancy"
    )
    evicted = {i for i in range(8) if bool(mask[i])}
    assert int(mask.sum()) == 3  # matched memory: still drops 3
    assert not (2 in evicted and 3 in evicted)  # at least one twin kept


def test_greedy_never_peels_protected_blocks():
    anc = _dup_pair_anchors()
    protect = torch.zeros(8, dtype=torch.bool)
    protect[0] = True
    protect[7] = True
    scores = block_joint_redundancy_greedy(anc, protect=protect)
    # Protected blocks get the lowest (-inf) score -> never selected first.
    assert scores[0] == float("-inf")
    assert scores[7] == float("-inf")
    # Every evictable block gets a finite, strictly-ordered peel score.
    finite = scores[1:7]
    assert torch.isfinite(finite).all()
    assert len(set(finite.tolist())) == 6  # distinct eviction ranks


def test_greedy_matches_pairwise_when_no_duplicates():
    # With mutually-orthogonal blocks (no redundancy structure), greedy and
    # pairwise pick the same first block -- greedy only diverges on duplicates.
    torch.manual_seed(1)
    anc = torch.randn(6, 2, 8)
    anc = anc / anc.reshape(6, -1).norm(dim=1).view(6, 1, 1)
    protect = torch.zeros(6, dtype=torch.bool)
    protect[5] = True
    pw = block_joint_redundancy(anc)
    gr = block_joint_redundancy_greedy(anc, protect=protect)
    pw_local = pw.clone()
    pw_local[5] = float("-inf")
    assert int(pw_local.argmax()) == int(gr.argmax())


def test_global_coverage_keeps_one_representative_per_duplicate_cluster():
    similarity = torch.eye(6)
    similarity[1, 2] = similarity[2, 1] = 0.99
    similarity[3, 4] = similarity[4, 3] = 0.98

    mask = select_evicted_by_coverage(
        similarity, num_blocks=6, budget=4, warmup_pages=1
    )

    assert torch.nonzero(mask).flatten().tolist() == [2, 4]
    assert not bool(mask[0])  # sink
    assert not bool(mask[5])  # anchor


def test_global_coverage_from_anchors_avoids_pairwise_twin_annihilation():
    anchors = _dup_pair_anchors()
    similarity = block_joint_similarity(anchors)
    coverage_mask = select_evicted_by_coverage(
        similarity, num_blocks=8, budget=5, warmup_pages=1
    )

    evicted = set(torch.nonzero(coverage_mask).flatten().tolist())
    assert int(coverage_mask.sum()) == 3
    assert not ({2, 3} <= evicted)


def test_global_coverage_honors_tail_and_exact_budget():
    similarity = torch.eye(8)
    mask = select_evicted_by_coverage(
        similarity,
        num_blocks=8,
        budget=5,
        warmup_pages=1,
        tail_protect_frac=0.25,
    )

    assert int(mask.sum()) == 3
    assert not bool(mask[0])
    assert not bool(mask[5:].any())


# -- Value-norm blend ---------------------------------------------------------


def test_zscore_standardizes_and_handles_degenerate():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    z = zscore(x)
    assert abs(float(z.mean())) < 1e-5
    assert abs(float(z.std()) - 1.0) < 1e-5
    # Constant vector -> all zeros (no signal to amplify).
    assert bool((zscore(torch.full((4,), 7.0)) == 0).all())


def test_value_blend_protects_high_value_block():
    # Equal redundancy across blocks; block 0 has a much larger value norm.
    # The blend must make block 0 the LEAST droppable.
    red = torch.tensor([2.0, 2.0, 2.0, 2.0])
    vnorm = torch.tensor([10.0, 1.0, 1.0, 1.0])
    blended = zscore(red) - 1.0 * zscore(vnorm)
    assert int(blended.argmin()) == 0


def test_value_and_query_weights_standardize_each_raw_signal_once():
    red = torch.tensor([-1.0, 0.0, 0.5, 3.0])
    vnorm = torch.tensor([8.0, 1.0, 4.0, 2.0])
    relevance = torch.tensor([0.1, 0.7, 0.0, 0.2])
    beta = 0.6
    query_weight = 0.4
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "value_blend_beta": beta,
            "query_alignment_weight": query_weight,
        }
    )
    policy = object.__new__(EvictionPolicy)
    policy.config = cfg

    actual = policy._finalize_scores(red, vnorm, False, relevance)
    expected = (
        zscore(red) - beta * zscore(vnorm) - query_weight * zscore(relevance)
    )
    legacy = zscore(zscore(red) - beta * zscore(vnorm)) - query_weight * zscore(
        relevance
    )

    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, legacy)


def test_config_accepts_and_rejects_redundancy_mode_and_blend():
    for mode in ("pairwise", "greedy", "coverage"):
        cfg = GeoKVConfig.from_dict(
            {"experiment_mode": "geo_uniform", "redundancy_mode": mode}
        )
        assert cfg.redundancy_mode == mode
    for beta in (0.0, 0.5, 2.0):
        cfg = GeoKVConfig.from_dict(
            {"experiment_mode": "geo_uniform", "value_blend_beta": beta}
        )
        assert cfg.value_blend_beta == beta
    with pytest.raises(ValueError):
        GeoKVConfig.from_dict(
            {"experiment_mode": "geo_uniform", "redundancy_mode": "bogus"}
        )
    with pytest.raises(ValueError):
        GeoKVConfig.from_dict(
            {"experiment_mode": "geo_uniform", "value_blend_beta": -0.5}
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"eviction_policy": "recency"},
        {"block_prototype_mode": "quarters"},
        {"value_blend_beta": 0.5},
        {"value_norm_protect_quantile": 0.9},
        {"query_alignment_weight": 1.0},
        {"positional_cosh_alpha": 1.0},
        {"decode_evict_budget": 64, "physical_reclaim": False},
    ],
)
def test_coverage_mode_rejects_incompatible_paths(extra):
    raw = {
        "experiment_mode": "geo_uniform",
        "redundancy_mode": "coverage",
    }
    raw.update(extra)
    with pytest.raises(ValueError, match="coverage"):
        GeoKVConfig.from_dict(raw)


def test_mean_prototype_is_identical_to_block_anchor():
    torch.manual_seed(4)
    v = torch.randn(3, 8, 2, 4)
    valid = torch.tensor([8, 5, 1])
    prototypes, mask = block_prototypes(v, valid, "mean")
    assert prototypes.shape == (3, 1, 2, 4)
    assert bool(mask.all())
    assert torch.equal(prototypes[:, 0], block_anchors(v, valid))
    assert torch.equal(
        block_multi_prototype_redundancy(prototypes, mask),
        block_joint_redundancy(block_anchors(v, valid)),
    )


def test_quarter_prototypes_ignore_empty_partial_groups():
    v = torch.zeros(2, 8, 1, 2)
    v[0, :2] = torch.tensor([1.0, 0.0])
    v[0, 2:4] = torch.tensor([0.0, 1.0])
    v[0, 4:6] = torch.tensor([-1.0, 0.0])
    v[0, 6:8] = torch.tensor([0.0, -1.0])
    v[1, 0] = torch.tensor([1.0, 0.0])
    prototypes, mask = block_prototypes(v, torch.tensor([8, 1]), "quarters")
    assert prototypes.shape == (2, 4, 1, 2)
    assert mask.tolist() == [[True, True, True, True], [True, False, False, False]]


def test_multi_prototype_preserves_unique_subspan_hidden_by_mean():
    # Blocks 0 and 1 have the same mean, so the pinned one-anchor scorer calls
    # block 0 redundant. Block 0 nevertheless contains a unique +Y subspan.
    # Multi-prototype scoring must give it lower droppability than block 1, whose
    # two prototypes are both covered by other blocks.
    p = torch.tensor(
        [
            [[[1.0, 0.0]], [[0.0, 1.0]]],   # block 0: +X, unique +Y
            [[[1.0, 0.0]], [[1.0, 0.0]]],   # block 1: +X, +X
            [[[1.0, 0.0]], [[1.0, 0.0]]],   # block 2 covers block 1
        ]
    )
    valid = torch.ones(3, 2, dtype=torch.bool)
    scores = block_multi_prototype_redundancy(p, valid)
    assert scores[0] < scores[1]
    assert scores[0] < scores[2]


def test_mean_top2_norm_selects_valid_strongest_tokens():
    v = torch.zeros(2, 4, 1, 2)
    v[0, :, 0, 0] = torch.tensor([1.0, 5.0, 3.0, 2.0])
    v[1, 0, 0, 0] = 7.0
    p, valid = block_prototypes(v, torch.tensor([4, 1]), "mean_top2_norm")
    assert p.shape == (2, 3, 1, 2)
    assert valid.tolist() == [[True, True, True], [True, True, False]]
    assert p[0, 1:, 0, 0].tolist() == [5.0, 3.0]


def test_config_accepts_prototype_modes_and_rejects_greedy_combo():
    for mode in ("mean", "quarters", "mean_top2_norm"):
        cfg = GeoKVConfig.from_dict(
            {"experiment_mode": "geo_uniform", "block_prototype_mode": mode}
        )
        assert cfg.block_prototype_mode == mode
    with pytest.raises(ValueError, match="block_prototype_mode"):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "block_prototype_mode": "bogus",
            }
        )
    with pytest.raises(ValueError, match="redundancy_mode='pairwise'"):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "block_prototype_mode": "quarters",
                "redundancy_mode": "greedy",
            }
        )


def test_greedy_mode_accepts_unrefined_v_redundancy():
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "redundancy_mode": "greedy",
        }
    )
    assert cfg.redundancy_mode == "greedy"


@pytest.mark.parametrize(
    "extra",
    [
        {"eviction_policy": "recency"},
        {"value_blend_beta": 0.5},
        {"value_norm_protect_quantile": 0.9},
        {"query_alignment_weight": 1.0},
        {"enable_query_tiebreak": True},
        {"positional_cosh_alpha": 1.0},
        {"calibrate_layer_subsets": True},
    ],
)
def test_greedy_mode_rejects_incompatible_paths(extra):
    raw = {
        "experiment_mode": "geo_uniform",
        "redundancy_mode": "greedy",
    }
    raw.update(extra)
    with pytest.raises(ValueError, match="greedy"):
        GeoKVConfig.from_dict(raw)


def test_config_accepts_and_rejects_value_norm_protect_quantile():
    for q in (0.0, 0.5, 0.95):
        cfg = GeoKVConfig.from_dict(
            {"experiment_mode": "geo_uniform", "value_norm_protect_quantile": q}
        )
        assert cfg.value_norm_protect_quantile == q
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            GeoKVConfig.from_dict(
                {"experiment_mode": "geo_uniform", "value_norm_protect_quantile": bad}
            )


def test_default_config_is_inert():
    cfg = GeoKVConfig()
    assert cfg.redundancy_mode == "pairwise"
    assert cfg.value_blend_beta is None
    assert cfg.value_norm_protect_quantile is None
    assert cfg.block_prototype_mode == "mean"
    assert not cfg.incremental_decode_scoring


def test_incremental_decode_scoring_accepts_exact_supported_path():
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "eviction_policy": "v_redundancy",
            "physical_reclaim": True,
            "decode_evict_budget": 32,
            "incremental_decode_scoring": True,
        }
    )
    assert cfg.incremental_decode_scoring


@pytest.mark.parametrize(
    "override",
    [
        {"physical_reclaim": False},
        {"incremental_decode_scoring": True, "decode_evict_budget": None},
        {"eviction_policy": "recency"},
        {"redundancy_mode": "greedy"},
        {"block_prototype_mode": "quarters"},
        {"value_blend_beta": 0.5},
        {"value_norm_protect_quantile": 0.9},
        {"query_alignment_weight": 1.0},
    ],
)
def test_incremental_decode_scoring_rejects_inexact_paths(override):
    raw = {
        "experiment_mode": "geo_uniform",
        "eviction_policy": "v_redundancy",
        "physical_reclaim": True,
        "decode_evict_budget": 32,
        "incremental_decode_scoring": True,
    }
    raw.update(override)
    with pytest.raises(ValueError, match="incremental_decode_scoring"):
        GeoKVConfig.from_dict(raw)
