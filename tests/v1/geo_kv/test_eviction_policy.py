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
    select_evicted_to_budget,
)
from vllm.v1.geo_kv.scoring import (
    block_anchors,
    block_joint_redundancy,
    block_value_l2,
)
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
