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
from vllm.v1.geo_kv.config import GeoKVConfig
from vllm.v1.geo_kv.eviction_policy import (
    aggregate_layer_scores,
    select_evicted_blocks,
)
from vllm.v1.geo_kv.scoring import block_anchors, block_joint_redundancy
from vllm.v1.kv_cache_interface import FullAttentionSpec


def test_select_rate_zero_is_all_false():
    scores = torch.rand(8)
    mask = select_evicted_blocks(scores, 8, 0.0, "v_redundancy", 0, 0)
    assert mask.dtype == torch.bool
    assert not bool(mask.any())


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
