# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifecycle tests for the capacity-band admission state machine.

The pure block *selector* (:func:`select_evicted_to_budget`) is covered by
``test_decode_eviction.py``. This file covers the piece nothing else does: the
**admission state machine** driven by a real :class:`EvictionPolicy` instance,
end to end, on CPU with faked engine tensors (no GPU, no model).

What the state machine must do:

* **Prefill gate** (:meth:`EvictionPolicy.on_step` -> ``_evict_request``): a
  request under capacity ``C`` is left intact; once it reaches ``C`` at end of
  prefill it is evicted down to the low watermark ``floor(watermark * C)``,
  keeping the sink + final blocks.
* **Decode gate** (:meth:`on_decode_step` -> ``_evict_decode_request``): a
  request that regrows to ``C`` during decode is evicted again to the watermark;
  the step interval throttles how often this runs.
* **Hysteresis**: after eviction fires, no re-eviction until the compaction has
  landed (the store row is cleared) AND the request has regrown back to ``C``.
  This is the fill(C) -> evict(0.75C) -> regrow(C) -> evict(0.75C) cycle.
* **In-flight guard** (physical reclaim): while a compaction is pending (store
  row still set), decode eviction is a no-op so it never scores a stale row.
* Synthetic / warmup requests and still-prefilling requests are skipped.

The engine surfaces the policy touches are faked faithfully: the paged KV cache
``(num_gpu_blocks, 2, block_size, H, D)``, ``block_tables.num_blocks.np`` shaped
``[group, req]`` and ``block_tables.block_tables[group].gpu`` shaped
``[req, max_blocks]``, and the ``InputBatch`` fields the two entry points read.
``_apply_compaction`` mirrors what the model runner does when a compaction lands
(repack survivors, add ``len(freed) * block_size`` evicted tokens, zero the mask
row) so the hysteresis tests exercise the real re-arm path.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.geo_kv.config import GeoKVConfig
from vllm.v1.geo_kv.eviction_policy import EvictionPolicy, capacity_band, frac_band
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

# Small, fixed geometry so evicted counts are hand-checkable.
BLOCK_SIZE = 4
H, D = 1, 4
NUM_GPU_BLOCKS = 512
MAX_REQS = 8
MAX_BLOCKS = 64
CAP = 8  # decode_evict_budget

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Fakes for the engine tensors the policy reads.
# ---------------------------------------------------------------------------
def _make_block_tables() -> SimpleNamespace:
    """A stand-in for the runner's ``BlockTables`` (single geo group)."""
    return SimpleNamespace(
        num_blocks=SimpleNamespace(np=np.zeros((1, MAX_REQS), dtype=np.int32)),
        block_tables=[
            SimpleNamespace(gpu=torch.zeros(MAX_REQS, MAX_BLOCKS, dtype=torch.long))
        ],
    )


def _make_policy(
    *,
    capacity: int = CAP,
    watermark: float = 0.75,
    warmup_pages: int = 1,
    physical: bool = True,
    interval: int = 1,
) -> EvictionPolicy:
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "eviction_policy": "v_redundancy",
            "score_sampled_layers": "all",
            "block_score_aggregation": "mean",
            "warmup_pages": warmup_pages,
            "decode_evict_budget": capacity,
            "decode_evict_watermark": watermark,
            "decode_evict_interval": interval,
            "physical_reclaim": physical,
        }
    )
    kv = {"l0": torch.randn(NUM_GPU_BLOCKS, 2, BLOCK_SIZE, H, D)}
    groups = [
        KVCacheGroupSpec(
            layer_names=["l0"],
            kv_cache_spec=FullAttentionSpec(
                block_size=BLOCK_SIZE, num_kv_heads=H, head_size=D, dtype=torch.float32
            ),
        )
    ]
    evicted_store = torch.zeros(MAX_REQS, MAX_BLOCKS, dtype=torch.bool)
    num_evicted_tokens_np = np.zeros(MAX_REQS, dtype=np.int32)
    return EvictionPolicy(
        cfg,
        kv,
        groups,
        evicted_store,
        BLOCK_SIZE,
        torch.device("cpu"),
        num_evicted_tokens_np,
    )


def _make_frac_policy(
    *,
    prefill_evict_frac: float | None = None,
    decode_evict_frac: float | None = None,
    watermark: float = 0.75,
    warmup_pages: int = 1,
    physical: bool = True,
    interval: int = 1,
) -> EvictionPolicy:
    """A policy on the decoupled fraction-of-prompt path (no decode_evict_budget)."""
    raw: dict = {
        "experiment_mode": "geo_uniform",
        "eviction_policy": "v_redundancy",
        "score_sampled_layers": "all",
        "block_score_aggregation": "mean",
        "warmup_pages": warmup_pages,
        "decode_evict_watermark": watermark,
        "decode_evict_interval": interval,
        "physical_reclaim": physical,
    }
    if prefill_evict_frac is not None:
        raw["prefill_evict_frac"] = prefill_evict_frac
    if decode_evict_frac is not None:
        raw["decode_evict_frac"] = decode_evict_frac
    cfg = GeoKVConfig.from_dict(raw)
    kv = {"l0": torch.randn(NUM_GPU_BLOCKS, 2, BLOCK_SIZE, H, D)}
    groups = [
        KVCacheGroupSpec(
            layer_names=["l0"],
            kv_cache_spec=FullAttentionSpec(
                block_size=BLOCK_SIZE, num_kv_heads=H, head_size=D, dtype=torch.float32
            ),
        )
    ]
    evicted_store = torch.zeros(MAX_REQS, MAX_BLOCKS, dtype=torch.bool)
    num_evicted_tokens_np = np.zeros(MAX_REQS, dtype=np.int32)
    return EvictionPolicy(
        cfg,
        kv,
        groups,
        evicted_store,
        BLOCK_SIZE,
        torch.device("cpu"),
        num_evicted_tokens_np,
    )


# ---------------------------------------------------------------------------
# Row helpers (block ids are reserved per-request so they never collide).
# ---------------------------------------------------------------------------
def _reserved(req_index: int) -> list[int]:
    base = req_index * MAX_BLOCKS
    return list(range(base, base + MAX_BLOCKS))


def _place(bt: SimpleNamespace, req_index: int, count: int) -> None:
    ids = _reserved(req_index)[:count]
    bt.num_blocks.np[0, req_index] = count
    bt.block_tables[0].gpu[req_index, :count] = torch.tensor(ids, dtype=torch.long)


def _count(bt: SimpleNamespace, req_index: int) -> int:
    return int(bt.num_blocks.np[0, req_index])


def _store_true(policy: EvictionPolicy, req_index: int) -> int:
    return int(policy.evicted_store[req_index].sum())


def _apply_compaction(
    policy: EvictionPolicy,
    bt: SimpleNamespace,
    req_index: int,
    freed: list[int],
) -> None:
    """Mirror the model runner landing a compaction for one request.

    Repack survivors into the row (from the request's reserved id pool so a later
    regrow appends fresh ids), add the whole-block evicted-token offset, and zero
    the mask row -- the exact three effects that re-arm the decode trigger.
    """
    c = _count(bt, req_index)
    freed_set = set(freed)
    survivors = c - len(freed_set)
    # Reuse the reserved pool: survivors occupy the first `survivors` slots; the
    # tail of the pool is available for regrowth.
    _place(bt, req_index, survivors)
    policy.num_evicted_tokens_np[req_index] += len(freed_set) * BLOCK_SIZE
    policy.evicted_store[req_index].zero_()


def _regrow(bt: SimpleNamespace, req_index: int, to_count: int) -> None:
    """Grow a decoding request's block row back up to ``to_count`` blocks."""
    _place(bt, req_index, to_count)


# ---------------------------------------------------------------------------
# InputBatch fakes for the two entry points.
# ---------------------------------------------------------------------------
def _prefill_batch(req_ids, req_indices, prompt_lens, finished) -> SimpleNamespace:
    """A batch where each request is prefilling; ``finished[i]`` controls whether
    this step completes its prefill (computed + scheduled >= prompt_len)."""
    n = len(req_ids)
    # A finished prefill has computed+scheduled >= prompt_len this step; an
    # unfinished one schedules a single token so it stays short of prompt_len.
    scheduled = np.array([prompt_lens[i] if finished[i] else 1 for i in range(n)])
    computed = np.zeros(n, dtype=np.int64)
    return SimpleNamespace(
        num_reqs=n,
        req_ids=list(req_ids),
        is_prefilling_np=np.ones(n, dtype=bool),
        idx_mapping_np=np.array(req_indices, dtype=np.int64),
        num_computed_prefill_tokens_np=computed,
        num_scheduled_tokens=scheduled,
        prefill_len_np=np.array(prompt_lens, dtype=np.int64),
    )


def _decode_batch(req_ids, req_indices, seq_lens, prefilling=None) -> SimpleNamespace:
    n = len(req_ids)
    if prefilling is None:
        prefilling = [False] * n
    # seq_len = num_computed_tokens + num_scheduled_tokens; put it all in computed.
    return SimpleNamespace(
        num_reqs=n,
        req_ids=list(req_ids),
        is_prefilling_np=np.array(prefilling, dtype=bool),
        idx_mapping_np=np.array(req_indices, dtype=np.int64),
        num_computed_tokens_np=np.array(seq_lens, dtype=np.int64),
        num_scheduled_tokens=np.ones(n, dtype=np.int64),
    )


# ===========================================================================
# capacity_band watermark math
# ===========================================================================
@pytest.mark.parametrize(
    "wm,expected_low",
    # floor(wm*C) clamped to [1, C-1]: 0.1*8=0.8->floor 0->clamp 1;
    # 0.99*8=7.92->floor 7 (already <= C-1).
    [(0.75, 6), (0.5, 4), (0.875, 7), (0.1, 1), (0.99, 7)],
)
def test_capacity_band_watermark_math(wm, expected_low):
    cfg = GeoKVConfig.from_dict(
        {
            "experiment_mode": "geo_uniform",
            "decode_evict_budget": CAP,
            "decode_evict_watermark": wm,
        }
    )
    capacity, low = capacity_band(cfg)
    assert capacity == CAP
    assert low == expected_low


@pytest.mark.parametrize("wm", [0.0, 1.0, -0.1, 1.5])
def test_watermark_open_interval_rejected(wm):
    """Config guards the open interval (0, 1); the clamp is only for valid wm."""
    with pytest.raises(ValueError):
        GeoKVConfig.from_dict(
            {
                "experiment_mode": "geo_uniform",
                "decode_evict_budget": CAP,
                "decode_evict_watermark": wm,
            }
        )


# ===========================================================================
# Prefill gate: on_step -> _evict_request
# ===========================================================================
def test_prefill_below_capacity_no_evict():
    policy = _make_policy()
    bt = _make_block_tables()
    _place(bt, 0, CAP - 1)  # under budget
    batch = _prefill_batch(["r0"], [0], [(CAP - 1) * BLOCK_SIZE], [True])

    freed = policy.on_step(batch, bt)

    assert freed == {}
    assert _store_true(policy, 0) == 0  # nothing masked


def test_prefill_at_capacity_evicts_to_watermark():
    policy = _make_policy()
    bt = _make_block_tables()
    _place(bt, 0, CAP)
    batch = _prefill_batch(["r0"], [0], [CAP * BLOCK_SIZE], [True])

    freed = policy.on_step(batch, bt)

    _, low = capacity_band(policy.config)
    assert set(freed) == {"r0"}
    assert len(freed["r0"]) == CAP - low  # evicted down to the watermark
    assert _store_true(policy, 0) == CAP - low


def test_prefill_above_capacity_evicts_to_watermark():
    policy = _make_policy()
    bt = _make_block_tables()
    _place(bt, 0, 2 * CAP)  # well over budget
    batch = _prefill_batch(["r0"], [0], [2 * CAP * BLOCK_SIZE], [True])

    freed = policy.on_step(batch, bt)

    _, low = capacity_band(policy.config)
    # Drains all the way to the watermark, not just by one band.
    assert len(freed["r0"]) == 2 * CAP - low


def test_prefill_keeps_sink_and_final():
    warmup = 1
    policy = _make_policy(warmup_pages=warmup)
    bt = _make_block_tables()
    _place(bt, 0, CAP)
    batch = _prefill_batch(["r0"], [0], [CAP * BLOCK_SIZE], [True])

    policy.on_step(batch, bt)

    mask = policy.evicted_store[0, :CAP]
    assert not bool(mask[:warmup].any())  # sink kept
    assert not bool(mask[CAP - 1])  # final (decode anchor) kept


def test_prefill_mask_only_sets_store_but_frees_nothing():
    policy = _make_policy(physical=False)
    bt = _make_block_tables()
    _place(bt, 0, CAP)
    batch = _prefill_batch(["r0"], [0], [CAP * BLOCK_SIZE], [True])

    freed = policy.on_step(batch, bt)

    _, low = capacity_band(policy.config)
    assert freed == {}  # mask-only never reports physical frees
    assert _store_true(policy, 0) == CAP - low  # but the mask is still written


def test_prefill_skips_synthetic_requests():
    policy = _make_policy()
    bt = _make_block_tables()
    _place(bt, 0, CAP)
    # "req_" and "_warmup_" are synthetic prefixes -> never evicted.
    batch = _prefill_batch(["req_0"], [0], [CAP * BLOCK_SIZE], [True])

    freed = policy.on_step(batch, bt)

    assert freed == {}
    assert _store_true(policy, 0) == 0


def test_prefill_only_finished_requests():
    policy = _make_policy()
    bt = _make_block_tables()
    _place(bt, 0, CAP)
    # Not finished this step: computed + scheduled < prompt_len.
    batch = _prefill_batch(["r0"], [0], [CAP * BLOCK_SIZE], [False])

    freed = policy.on_step(batch, bt)

    assert freed == {}
    assert _store_true(policy, 0) == 0


# ===========================================================================
# Decode gate: on_decode_step -> _evict_decode_request
# ===========================================================================
def test_decode_below_capacity_no_evict():
    policy = _make_policy()
    bt = _make_block_tables()
    _place(bt, 0, CAP - 1)
    batch = _decode_batch(["r0"], [0], [(CAP - 1) * BLOCK_SIZE])

    freed = policy.on_decode_step(batch, bt)

    assert freed == {}
    assert policy._num_decode_evictions == 0


def test_decode_at_capacity_evicts_and_counts():
    policy = _make_policy()
    bt = _make_block_tables()
    _place(bt, 0, CAP)
    batch = _decode_batch(["r0"], [0], [CAP * BLOCK_SIZE])

    freed = policy.on_decode_step(batch, bt)

    _, low = capacity_band(policy.config)
    assert set(freed) == {"r0"}
    assert len(freed["r0"]) == CAP - low
    assert policy._num_decode_evictions == 1


def test_decode_in_flight_guard_blocks_refire():
    """A pending compaction (store row set) must suppress decode eviction."""
    policy = _make_policy()
    bt = _make_block_tables()
    _place(bt, 0, CAP)
    # Simulate a compaction already staged but not yet applied: mask row is set.
    policy.evicted_store[0, :CAP][2] = True

    batch = _decode_batch(["r0"], [0], [CAP * BLOCK_SIZE])
    freed = policy.on_decode_step(batch, bt)

    assert freed == {}  # scored nothing; waiting for the row to clear
    assert policy._num_decode_evictions == 0


def test_decode_interval_throttle():
    policy = _make_policy(interval=2)
    bt = _make_block_tables()
    _place(bt, 0, CAP)
    batch = _decode_batch(["r0"], [0], [CAP * BLOCK_SIZE])

    # Step 1: _decode_step -> 1, 1 % 2 != 0 -> throttled.
    assert policy.on_decode_step(batch, bt) == {}
    assert policy._num_decode_evictions == 0
    # Step 2: _decode_step -> 2, 2 % 2 == 0 -> fires.
    freed = policy.on_decode_step(batch, bt)
    assert set(freed) == {"r0"}
    assert policy._num_decode_evictions == 1


def test_decode_skips_prefilling_and_synthetic():
    policy = _make_policy()
    bt = _make_block_tables()
    _place(bt, 0, CAP)
    _place(bt, 1, CAP)
    # r0 is still prefilling; req_1 is synthetic -> both skipped.
    batch = _decode_batch(
        ["r0", "req_1"],
        [0, 1],
        [CAP * BLOCK_SIZE, CAP * BLOCK_SIZE],
        prefilling=[True, False],
    )

    freed = policy.on_decode_step(batch, bt)

    assert freed == {}
    assert policy._num_decode_evictions == 0


def test_decode_gates_on_block_count_not_seq_len():
    """With compacted tokens present, the trigger is block count == C, and the
    storage basis (seq_len - num_evicted) is used for scoring without error."""
    policy = _make_policy()
    bt = _make_block_tables()
    _place(bt, 0, CAP)
    policy.num_evicted_tokens_np[0] = 5 * BLOCK_SIZE  # previously compacted tokens
    # seq_len far exceeds C*block_size, but only CAP blocks are physically held.
    batch = _decode_batch(["r0"], [0], [(CAP + 5) * BLOCK_SIZE])

    freed = policy.on_decode_step(batch, bt)

    assert set(freed) == {"r0"}  # fires on the CAP physical blocks
    assert policy._num_decode_evictions == 1


# ===========================================================================
# Full hysteresis cycle: prefill(C) -> evict -> compact -> regrow(C) -> evict
# ===========================================================================
def test_hysteresis_full_cycle():
    policy = _make_policy()
    bt = _make_block_tables()
    _, low = capacity_band(policy.config)

    # 1) Prefill reaches C -> evict to watermark.
    _place(bt, 0, CAP)
    freed = policy.on_step(_prefill_batch(["r0"], [0], [CAP * BLOCK_SIZE], [True]), bt)
    assert len(freed["r0"]) == CAP - low
    assert _store_true(policy, 0) == CAP - low  # mask set, compaction pending

    # 2) Compaction lands: row repacked to `low`, tokens offset, mask cleared.
    _apply_compaction(policy, bt, 0, freed["r0"])
    assert _count(bt, 0) == low
    assert _store_true(policy, 0) == 0
    assert int(policy.num_evicted_tokens_np[0]) == (CAP - low) * BLOCK_SIZE

    # 3) Decode below C after eviction -> no-op (hysteresis floor).
    seq_at_low = (CAP + (CAP - low)) * BLOCK_SIZE  # uncompacted seq keeps growing
    assert policy.on_decode_step(_decode_batch(["r0"], [0], [seq_at_low]), bt) == {}
    assert policy._num_decode_evictions == 0

    # 4) Regrow back to C during decode -> eviction fires again.
    _regrow(bt, 0, CAP)
    seq_at_cap = seq_at_low + (CAP - low) * BLOCK_SIZE
    freed2 = policy.on_decode_step(_decode_batch(["r0"], [0], [seq_at_cap]), bt)
    assert set(freed2) == {"r0"}
    assert len(freed2["r0"]) == CAP - low
    assert policy._num_decode_evictions == 1

    # 5) Second compaction lands and re-arms cleanly.
    _apply_compaction(policy, bt, 0, freed2["r0"])
    assert _count(bt, 0) == low
    assert _store_true(policy, 0) == 0


def test_no_refire_until_compaction_applied():
    """Between decode eviction and its compaction, repeated steps are no-ops."""
    policy = _make_policy()
    bt = _make_block_tables()
    _place(bt, 0, CAP)

    freed = policy.on_decode_step(_decode_batch(["r0"], [0], [CAP * BLOCK_SIZE]), bt)
    assert set(freed) == {"r0"}
    assert policy._num_decode_evictions == 1

    # Store row still set (compaction not applied) -> further steps do nothing,
    # even though the block count is still at capacity.
    for _ in range(3):
        assert (
            policy.on_decode_step(_decode_batch(["r0"], [0], [CAP * BLOCK_SIZE]), bt)
            == {}
        )
    assert policy._num_decode_evictions == 1  # never double-counted


def test_two_requests_independent_bands():
    """Concurrent requests are gated independently by their own block counts."""
    policy = _make_policy()
    bt = _make_block_tables()
    _place(bt, 0, CAP)  # at capacity -> should evict
    _place(bt, 1, CAP - 2)  # under capacity -> untouched

    freed = policy.on_decode_step(
        _decode_batch(["r0", "r1"], [0, 1], [CAP * BLOCK_SIZE, (CAP - 2) * BLOCK_SIZE]),
        bt,
    )

    assert set(freed) == {"r0"}
    assert _store_true(policy, 1) == 0
    assert policy._num_decode_evictions == 1


# ===========================================================================
# Decoupled fraction-of-prompt path: independent admission vs decode targets.
# P is the prompt block count; the band is resolved per-request at admission.
# ===========================================================================
P = 8  # prompt block count used across the decoupled-gate tests


def _admit(policy: EvictionPolicy, bt: SimpleNamespace, req_index: int, p: int):
    """Place a p-block prompt and run its end-of-prefill admission step."""
    _place(bt, req_index, p)
    return policy.on_step(
        _prefill_batch([f"r{req_index}"], [req_index], [p * BLOCK_SIZE], [True]), bt
    )


def test_frac_resolution_records_decode_band():
    """Decode-only admission records the per-request (cap, low) from d and P."""
    policy = _make_frac_policy(decode_evict_frac=0.5, watermark=0.75)
    bt = _make_block_tables()
    freed = _admit(policy, bt, 0, P)

    cap, low = frac_band(0.5, 0.75, P)  # ceil(0.5*8)=4, ceil(0.75*4)=3
    assert (cap, low) == (4, 3)
    assert freed == {}  # decode-only admits the full prompt (no admission evict)
    assert _store_true(policy, 0) == 0
    assert int(policy._decode_cap_np[0]) == cap
    assert int(policy._decode_low_np[0]) == low


def test_prefill_only_evicts_to_keep_and_decode_is_noop():
    """prefill_only evicts admission to ceil(f*P); decode stays disarmed."""
    policy = _make_frac_policy(prefill_evict_frac=0.5, warmup_pages=1)
    bt = _make_block_tables()
    freed = _admit(policy, bt, 0, P)

    keep = math.ceil(0.5 * P)  # 4
    assert set(freed) == {"r0"}
    assert len(freed["r0"]) == P - keep  # evicted down to keep
    assert _store_true(policy, 0) == P - keep
    # No decode band recorded -> decode is a complete no-op even at capacity.
    assert int(policy._decode_cap_np[0]) == 0
    _apply_compaction(policy, bt, 0, freed["r0"])
    _regrow(bt, 0, P)
    assert policy.on_decode_step(_decode_batch(["r0"], [0], [P * BLOCK_SIZE]), bt) == {}
    assert policy._num_decode_evictions == 0


def test_prefill_only_frac_one_is_inert():
    """f == 1.0 retains the whole prompt: nothing evicted (baseline-identical)."""
    policy = _make_frac_policy(prefill_evict_frac=1.0)
    bt = _make_block_tables()
    freed = _admit(policy, bt, 0, P)

    assert freed == {}
    assert _store_true(policy, 0) == 0


def test_decode_only_admits_full_then_caps_during_decode():
    """decode_only keeps the full prompt at admission, then caps in decode."""
    policy = _make_frac_policy(decode_evict_frac=0.5, watermark=0.75)
    bt = _make_block_tables()
    freed_admit = _admit(policy, bt, 0, P)
    assert freed_admit == {}  # full prompt admitted

    cap, low = frac_band(0.5, 0.75, P)  # (4, 3)
    # First decode step: count == P (8) >= cap (4) -> evict down to low (3).
    freed = policy.on_decode_step(_decode_batch(["r0"], [0], [P * BLOCK_SIZE]), bt)
    assert set(freed) == {"r0"}
    assert len(freed["r0"]) == P - low
    assert policy._num_decode_evictions == 1


def test_decode_only_below_cap_no_evict():
    """A decode-only request that never reaches its cap is left intact."""
    policy = _make_frac_policy(decode_evict_frac=0.5)
    bt = _make_block_tables()
    _admit(policy, bt, 0, P)
    cap, _ = frac_band(0.5, 0.75, P)  # 4
    # Shrink the physical row below cap (as if it were a short request): no evict.
    _place(bt, 0, cap - 1)
    freed = policy.on_decode_step(
        _decode_batch(["r0"], [0], [(cap - 1) * BLOCK_SIZE]), bt
    )
    assert freed == {}
    assert policy._num_decode_evictions == 0


def test_combined_independent_admission_and_decode_targets():
    """Strict admission + lenient decode: the two targets differ and both fire."""
    a, d, w = 0.375, 0.9, 0.75
    policy = _make_frac_policy(
        prefill_evict_frac=a, decode_evict_frac=d, watermark=w, warmup_pages=1
    )
    bt = _make_block_tables()

    keep = math.ceil(a * P)  # 3
    cap, low = frac_band(d, w, P)  # ceil(0.9*8)=8, ceil(0.75*8)=6

    # Independence: admission target (keep=3) differs from decode floor (low=6).
    assert keep != low
    assert (cap, low) == (8, 6)

    # 1) Admission evicts to the strict keep.
    freed = _admit(policy, bt, 0, P)
    assert len(freed["r0"]) == P - keep
    assert int(policy._decode_cap_np[0]) == cap
    assert int(policy._decode_low_np[0]) == low

    # 2) Compaction lands (row -> keep). Decode below cap -> no-op.
    _apply_compaction(policy, bt, 0, freed["r0"])
    assert _count(bt, 0) == keep
    seq = (P + (P - keep)) * BLOCK_SIZE
    assert policy.on_decode_step(_decode_batch(["r0"], [0], [seq]), bt) == {}

    # 3) Regrow to the lenient decode cap -> decode eviction fires to its floor.
    _regrow(bt, 0, cap)
    freed2 = policy.on_decode_step(
        _decode_batch(["r0"], [0], [seq + (cap - keep) * BLOCK_SIZE]), bt
    )
    assert set(freed2) == {"r0"}
    assert len(freed2["r0"]) == cap - low
    assert policy._num_decode_evictions == 1


def test_reset_request_clears_decode_band():
    """A recycled slot must not inherit a stale decode band."""
    policy = _make_frac_policy(decode_evict_frac=0.5)
    bt = _make_block_tables()
    _admit(policy, bt, 0, P)
    assert int(policy._decode_cap_np[0]) > 0

    policy.reset_request(0)
    assert int(policy._decode_cap_np[0]) == 0
    assert int(policy._decode_low_np[0]) == 0


def test_frac_path_takes_precedence_over_legacy_budget():
    """When a frac knob is set the legacy decode_evict_budget is ignored."""
    # decode_evict_frac resolves per-request; a huge legacy budget must not leak.
    policy = _make_frac_policy(decode_evict_frac=0.5, watermark=0.75)
    assert policy.config.decode_evict_budget is None
    bt = _make_block_tables()
    _admit(policy, bt, 0, P)
    # The band came from the fraction (cap=4), not any global budget.
    assert int(policy._decode_cap_np[0]) == frac_band(0.5, 0.75, P)[0]
