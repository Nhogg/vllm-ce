# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for decode-time, budget-triggered KV block eviction.

Covers the pure, engine-independent pieces added for the decode path: the
budget-based block selector (:func:`select_evicted_to_budget`) and the config
validation for the new decode knobs. The prefill path and its selector
(:func:`select_evicted_blocks`) are exercised by ``test_eviction_policy.py`` and
are deliberately untouched here.
"""

import pytest
import torch

from vllm.v1.geo_kv.config import GeoKVConfig
from vllm.v1.geo_kv.eviction_policy import (
    capacity_band,
    select_evicted_to_budget,
)
from vllm.v1.geo_kv.scoring import block_anchors, block_joint_redundancy


def test_budget_noop_when_within_budget():
    scores = torch.rand(5)
    for budget in (5, 6, 100):
        mask = select_evicted_to_budget(scores, 5, budget, warmup_pages=0)
        assert mask.dtype == torch.bool
        assert not bool(mask.any())  # nothing evicted when count <= budget


def test_budget_evicts_down_to_target():
    n = 10
    # Distinct scores so selection is unambiguous.
    scores = torch.arange(n, dtype=torch.float32)
    mask = select_evicted_to_budget(scores, n, budget=6, warmup_pages=0)
    # count - budget = 4 interior blocks dropped; kept count == budget.
    assert int(mask.sum()) == 4
    assert n - int(mask.sum()) == 6


def test_budget_keeps_sink_and_final():
    n, warmup, budget = 8, 2, 3
    # Make the sink and final blocks look maximally droppable; they must still
    # be kept (protected), so eviction comes only from the interior.
    scores = torch.zeros(n)
    scores[0] = 100.0
    scores[1] = 100.0
    scores[n - 1] = 100.0
    mask = select_evicted_to_budget(scores, n, budget, warmup_pages=warmup)
    assert not bool(mask[:warmup].any())  # sink kept
    assert not bool(mask[n - 1])  # final kept
    # Evictable interior is [2, 7) = 5 blocks; need to drop count-budget = 5.
    assert int(mask.sum()) == 5
    assert bool(mask[warmup : n - 1].all())


def test_budget_drops_highest_redundancy_first():
    # Evictable window [1, n-1); block 3 highest score, then block 5.
    scores = torch.tensor([0.0, 0.1, 0.2, 0.9, 0.3, 0.8, 0.4, 0.0])
    n = scores.numel()
    # Drop exactly 2 (budget = n-2 = 6).
    mask = select_evicted_to_budget(scores, n, budget=n - 2, warmup_pages=0)
    assert int(mask.sum()) == 2
    assert bool(mask[3]) and bool(mask[5])


def test_budget_clamped_by_evictable_count():
    # Budget smaller than sink+final can ever reach: evict all interior only.
    n, warmup = 6, 2
    mask = select_evicted_to_budget(torch.zeros(n), n, budget=1, warmup_pages=warmup)
    # Evictable interior = [2, 5) = 3 blocks; cannot drop the protected 3.
    assert int(mask.sum()) == 3
    assert bool(mask[2]) and bool(mask[3]) and bool(mask[4])
    assert not bool(mask[0]) and not bool(mask[1]) and not bool(mask[n - 1])


def test_budget_tiny_request_noop():
    # n=2 with warmup=1: block 0 is sink, block 1 is final -> no interior at all,
    # so even an aggressive budget evicts nothing.
    mask = select_evicted_to_budget(torch.zeros(2), 2, budget=1, warmup_pages=1)
    assert not bool(mask.any())


def test_budget_evicts_duplicated_v_blocks_first():
    # End-to-end with the real scoring: blocks 1 and 2 are identical (max mutual
    # cosine); block 3 is the always-kept final block. With budget = n-1 = 3 we
    # drop exactly one, and it must be one of the duplicates.
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
    scores = block_joint_redundancy(block_anchors(v, valid))
    mask = select_evicted_to_budget(scores, B, budget=B - 1, warmup_pages=0)
    assert int(mask.sum()) == 1
    evicted = [i for i in range(B) if bool(mask[i])]
    assert evicted[0] in (1, 2)


# -- config validation --------------------------------------------------------
def test_config_decode_budget_valid():
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "eviction_policy": "v_redundancy",
            "decode_evict_budget": 32,
        }
    )
    assert cfg.decode_evict_budget == 32
    assert cfg.decode_evict_interval == 1


def test_config_decode_budget_none_is_default():
    cfg = GeoKVConfig.from_dict({"experiment_mode": "geo_uniform"})
    assert cfg.decode_evict_budget is None  # decode eviction off by default


@pytest.mark.parametrize("budget", [0, -1])
def test_config_rejects_bad_decode_budget(budget):
    with pytest.raises(ValueError):
        GeoKVConfig.from_dict(
            {"experiment_mode": "geo_uniform", "decode_evict_budget": budget}
        )


def test_config_rejects_bad_decode_interval():
    with pytest.raises(ValueError):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "decode_evict_budget": 16,
                "decode_evict_interval": 0,
            }
        )


def test_config_decode_budget_allows_recency_policy():
    # The capacity band now honors any eviction_policy: recency (oldest-first)
    # is StreamingLLM at matched memory, the baseline contrast for v_redundancy.
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "eviction_policy": "recency",
            "decode_evict_budget": 16,
        }
    )
    assert cfg.eviction_policy == "recency"
    assert cfg.decode_evict_budget == 16


def test_config_decode_budget_requires_active_eviction():
    # score_only is not an active_eviction mode.
    with pytest.raises(ValueError):
        GeoKVConfig.from_dict(
            {"experiment_mode": "score_only", "decode_evict_budget": 16}
        )


# -- capacity band (watermark) ------------------------------------------------
def test_config_watermark_default():
    cfg = GeoKVConfig.from_dict(
        {"experiment_mode": "geo_uniform", "decode_evict_budget": 32}
    )
    assert cfg.decode_evict_watermark == 0.75


@pytest.mark.parametrize("wm", [0.0, 1.0, -0.1, 1.5])
def test_config_rejects_bad_watermark(wm):
    with pytest.raises(ValueError):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "decode_evict_budget": 32,
                "decode_evict_watermark": wm,
            }
        )


def test_capacity_band_default_ratio():
    cfg = GeoKVConfig.from_dict(
        {"experiment_mode": "geo_uniform", "decode_evict_budget": 32}
    )
    capacity, low = capacity_band(cfg)
    assert capacity == 32
    assert low == 24  # floor(0.75 * 32)


def test_capacity_band_custom_ratio_floor():
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "decode_evict_budget": 10,
            "decode_evict_watermark": 0.55,
        }
    )
    capacity, low = capacity_band(cfg)
    assert capacity == 10
    assert low == 5  # floor(0.55 * 10) = 5


def test_capacity_band_clamps_to_at_least_one():
    # A tiny capacity with a small ratio would floor to 0; clamp keeps low >= 1.
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "decode_evict_budget": 2,
            "decode_evict_watermark": 0.25,
        }
    )
    capacity, low = capacity_band(cfg)
    assert capacity == 2
    assert low == 1  # floor(0.5)=0 -> clamped to 1, and < capacity


def test_capacity_band_low_below_capacity():
    # A high ratio must still leave a nonzero hysteresis gap (low < capacity).
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "decode_evict_budget": 4,
            "decode_evict_watermark": 0.99,
        }
    )
    capacity, low = capacity_band(cfg)
    assert low == capacity - 1  # floor(3.96)=3, already < 4; stays 3


def test_evict_to_watermark_reaches_target():
    # End-of-prefill / decode selector drops count down to the low watermark.
    n = 20
    scores = torch.arange(n, dtype=torch.float32)
    _capacity, low = capacity_band(
        GeoKVConfig.from_dict(
            {"experiment_mode": "geo_uniform", "decode_evict_budget": n}
        )
    )
    mask = select_evicted_to_budget(scores, n, low, warmup_pages=0)
    assert n - int(mask.sum()) == low  # kept == watermark
    assert low == 15  # floor(0.75 * 20)
