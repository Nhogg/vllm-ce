# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V-redundancy block-eviction policy (Option A).

Owned by the V2 GPU model runner and constructed only when an ``active_eviction``
experiment mode is selected. Mirrors :class:`PrefillScorer`: it reads a request's
V vectors from the paged KV cache, scores every whole block by joint (all-heads)
V redundancy aggregated across a layer band, decides which blocks to drop, and
writes a per-request boolean eviction mask into a runner-owned store that the
FlexAttention backend consults to hide the flagged *logical* blocks.

Two eviction regimes share this scoring:

* **Rate mode** (``eviction_rate``): at end of prefill, drop a fixed fraction of
  evictable prompt blocks. Mask-only unless ``physical_reclaim`` is set.
* **Capacity band** (``decode_evict_budget`` = ``C`` with
  ``decode_evict_watermark``): let the cache fill to ``C``, then evict the
  most-redundant interior blocks down to the low watermark ``floor(watermark*C)``.
  Enforced at end of prefill *and* during decode as the request regrows to ``C``
  (hysteresis). Under ``physical_reclaim`` the evicted blocks are compacted and
  their GPU memory reclaimed; the mask covers the one-step latency window.

When ``physical_reclaim`` is False the module never mutates the KV cache
(mask-only), the faithful in-engine analog of the HF reference screen, which
likewise only masked evicted positions.

IMPORTANT (Option A): the physical eviction unit in vLLM is a whole block across
all heads/layers. layer/kv_head are *scoring* dimensions only.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.geo_kv.config import GeoKVConfig, resolve_indices
from vllm.v1.geo_kv.scoring import (
    block_anchors,
    block_joint_redundancy,
    block_valid_lens,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

# Synthetic request-id prefixes used by kernel warmup / dummy runs; never real
# serving requests, so never evicted. Kept in sync with prefill_scorer.py.
_SYNTHETIC_REQ_PREFIXES = ("_warmup_", "req_")


def _choose_evicted(
    evictable_t: torch.Tensor,
    scores: torch.Tensor,
    lo: int,
    hi: int,
    k: int,
    policy: str,
    seed: int,
) -> torch.Tensor:
    """Pick ``k`` blocks to evict from ``evictable_t`` under ``policy``.

    Shared block-selection core for both the rate path
    (:func:`select_evicted_blocks`) and the budget/capacity-band path
    (:func:`select_evicted_to_budget`), so every policy behaves identically no
    matter which trigger fired.

    Args:
        evictable_t: ``(n,)`` long tensor of evictable block indices (the
            interior slice, sink and anchor already excluded).
        scores: ``(num_blocks,)`` per-block droppability; only the ``[lo:hi]``
            slice is consulted, and only by ``v_redundancy``.
        lo: Start of the evictable slice (== ``warmup_pages``).
        hi: End of the evictable slice (exclusive; == ``num_blocks - 1``).
        k: Number of blocks to choose (already clamped to ``0 < k <= n``).
        policy: ``"v_redundancy"``, ``"recency"``, or ``"random"``.
        seed: Seed for the ``"random"`` policy (reproducible).

    Returns:
        ``(k,)`` long tensor: the chosen block indices.
    """
    if policy == "v_redundancy":
        local = scores.detach().to(device="cpu", dtype=torch.float32)[lo:hi]
        order = torch.argsort(local, descending=True)
        return evictable_t[order[:k]]
    if policy == "recency":
        return evictable_t[:k]  # oldest k blocks (StreamingLLM contrast)
    if policy == "random":
        gen = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(evictable_t.shape[0], generator=gen)
        return evictable_t[perm[:k]]
    raise ValueError(f"unknown eviction policy: {policy!r}")


def select_evicted_blocks(
    scores: torch.Tensor,
    num_blocks: int,
    rate: float,
    policy: str,
    warmup_pages: int,
    seed: int,
) -> torch.Tensor:
    """Choose which whole blocks to hide from attention for one request.

    The leading ``warmup_pages`` blocks (attention sink) and the final block
    (the decode anchor) are always kept; the rest are *evictable*. Every policy
    drops the SAME count ``k = round(rate * num_evictable)`` so the accuracy
    comparison is matched across policies. Runs entirely on CPU and returns a CPU
    bool tensor (the caller copies it to the store's device).

    Args:
        scores: ``(num_blocks,)`` per-block droppability, higher = more
            redundant / more droppable. Only used by ``v_redundancy``; may be
            all-zero for the other policies.
        num_blocks: Number of logical blocks the request occupies.
        rate: Fraction of evictable blocks to drop, in ``[0, 1]``.
        policy: ``"v_redundancy"``, ``"recency"``, or ``"random"``.
        warmup_pages: Number of leading blocks to always keep (sink).
        seed: Seed for the ``"random"`` policy (reproducible).

    Returns:
        ``(num_blocks,)`` bool tensor; True = evict (hide from attention).
    """
    evict = torch.zeros(num_blocks, dtype=torch.bool)
    lo = max(int(warmup_pages), 0)
    hi = num_blocks - 1  # always keep the final block (decode anchor)
    evictable = list(range(lo, hi))
    n = len(evictable)
    if n <= 0 or rate <= 0.0:
        return evict
    k = int(round(rate * n))
    if k <= 0:
        return evict
    k = min(k, n)

    evictable_t = torch.tensor(evictable, dtype=torch.long)
    chosen = _choose_evicted(evictable_t, scores, lo, hi, k, policy, seed)
    evict[chosen] = True
    return evict


def select_evicted_to_budget(
    scores: torch.Tensor,
    num_blocks: int,
    budget: int,
    warmup_pages: int,
    policy: str = "v_redundancy",
    seed: int = 0,
) -> torch.Tensor:
    """Evict interior blocks down to a per-request budget under ``policy``.

    The budget analog of :func:`select_evicted_blocks`: instead of a fixed rate,
    it drops exactly enough *evictable* blocks so that ``num_blocks`` falls to
    ``budget``. Same keep-rules: the leading ``warmup_pages`` blocks (attention
    sink) and the final block (decode anchor) are never evicted. Which blocks are
    dropped is chosen by ``policy`` (``v_redundancy`` scores highest-redundancy
    first; ``recency`` drops oldest first == StreamingLLM; ``random`` is seeded).
    Used by the capacity-band prefill and decode triggers. Runs on CPU and
    returns a CPU bool tensor.

    Args:
        scores: ``(num_blocks,)`` per-block droppability (higher = more
            redundant / more droppable). Only consulted by ``v_redundancy``; may
            be all-zero for the other policies.
        num_blocks: Current logical block count for the request.
        budget: Target maximum number of blocks to keep. No-op when
            ``num_blocks <= budget``.
        warmup_pages: Number of leading blocks to always keep (sink).
        policy: ``"v_redundancy"``, ``"recency"``, or ``"random"``.
        seed: Seed for the ``"random"`` policy (reproducible).

    Returns:
        ``(num_blocks,)`` bool tensor; True = evict (hide from attention).
    """
    evict = torch.zeros(num_blocks, dtype=torch.bool)
    if num_blocks <= budget:
        return evict
    lo = max(int(warmup_pages), 0)
    hi = num_blocks - 1  # always keep the final block (decode anchor)
    evictable = list(range(lo, hi))
    n = len(evictable)
    if n <= 0:
        return evict
    # Drop just enough to reach budget, but never more than the evictable set.
    k = min(num_blocks - budget, n)
    if k <= 0:
        return evict
    evictable_t = torch.tensor(evictable, dtype=torch.long)
    chosen = _choose_evicted(evictable_t, scores, lo, hi, k, policy, seed)
    evict[chosen] = True
    return evict


def capacity_band(cfg: GeoKVConfig) -> tuple[int, int]:
    """Return the ``(capacity, low_watermark)`` block band for the given config.

    ``capacity`` (``C``) is the per-request block cap; eviction fires only when a
    request reaches it. ``low_watermark`` is ``floor(decode_evict_watermark * C)``
    (clamped to ``[1, C - 1]``), the target the eviction drains down to. The gap
    between the two is the hysteresis band that bounds how often the O(C^2)
    scoring + compaction runs.

    Args:
        cfg: The geo_kv config; ``decode_evict_budget`` must be set.

    Returns:
        ``(capacity, low_watermark)`` in whole blocks.
    """
    assert cfg.decode_evict_budget is not None
    capacity = int(cfg.decode_evict_budget)
    low = int(cfg.decode_evict_watermark * capacity)
    low = max(1, min(low, capacity - 1))
    return capacity, low


def frac_band(frac: float, watermark: float, prompt_blocks: int) -> tuple[int, int]:
    """Resolve a per-request ``(capacity, low_watermark)`` band from a fraction.

    The fraction-of-prompt analog of :func:`capacity_band`: the capacity is
    ``ceil(frac * prompt_blocks)`` and the low watermark is
    ``ceil(watermark * capacity)`` (clamped to ``[1, capacity - 1]`` when the
    capacity leaves room to evict). Resolving from each request's own prompt
    block count keeps a single fraction comparable across models / tasks /
    prompt lengths.

    Args:
        frac: Target capacity as a fraction of prompt blocks, in ``(0, 1]``.
        watermark: Low-watermark fraction of the capacity, in ``(0, 1)``.
        prompt_blocks: The request's prompt block count ``P``.

    Returns:
        ``(capacity, low_watermark)`` in whole blocks.
    """
    capacity = max(1, math.ceil(frac * prompt_blocks))
    if capacity < 2:
        return capacity, capacity
    low = math.ceil(watermark * capacity)
    low = max(1, min(low, capacity - 1))
    return capacity, low


def aggregate_layer_scores(
    stacked: torch.Tensor, aggregation: str, topk_frac: float
) -> torch.Tensor:
    """Reduce ``(num_layers, num_blocks)`` per-layer scores to ``(num_blocks,)``.

    This is the cross-layer aggregation of per-layer joint-V redundancy into one
    whole-block droppability score (Option A: one decision per block across all
    layers). ``mean`` matches the HF reference screen's band aggregation.

    Args:
        stacked: ``(num_layers, num_blocks)`` per-layer per-block scores.
        aggregation: one of :data:`config.BLOCK_SCORE_AGGREGATIONS`.
        topk_frac: fraction used by the ``topk_mean`` aggregation.

    Returns:
        ``(num_blocks,)`` aggregated scores.
    """
    x = stacked.to(torch.float32)
    if aggregation == "mean":
        return x.mean(dim=0)
    if aggregation == "max":
        return x.max(dim=0).values
    if aggregation == "percentile_90":
        return torch.quantile(x, 0.9, dim=0)
    if aggregation == "topk_mean":
        num_layers = x.shape[0]
        k = max(1, int(round(topk_frac * num_layers)))
        return x.topk(k, dim=0).values.mean(dim=0)
    raise ValueError(f"unknown block_score_aggregation: {aggregation!r}")


class EvictionPolicy:
    """Writes per-request V-redundancy block-eviction decisions.

    Fires at end of prefill (rate or capacity-band) and, for the capacity band,
    during decode as requests regrow to capacity. Emits mask writes and, under
    ``physical_reclaim``, freed logical block indices for the scheduler to
    compact.
    """

    def __init__(
        self,
        config: GeoKVConfig,
        kv_caches_by_layer: dict[str, torch.Tensor],
        kv_cache_groups: list[Any],
        evicted_store: torch.Tensor,
        block_size: int,
        device: torch.device,
        num_evicted_tokens_np: Any = None,
    ) -> None:
        self.config = config
        self.kv = kv_caches_by_layer
        self.evicted_store = evicted_store
        self.block_size = block_size
        self.device = device
        # CPU mirror of per-request compacted-out token counts (owned by the
        # runner's RequestState). Decode scoring subtracts this to recover the
        # storage length from the uncompacted seq_len. None => never compacted.
        self.num_evicted_tokens_np = num_evicted_tokens_np

        # Ordered attention-layer metadata: (layer_idx, layer_name, group_id).
        # Built identically to PrefillScorer so the sampled-layer band lines up.
        self.layers: list[tuple[int, str, int]] = []
        layer_idx = 0
        for g, group in enumerate(kv_cache_groups):
            if not isinstance(group.kv_cache_spec, AttentionSpec):
                continue  # score attention layers only (skip e.g. Mamba)
            for name in group.layer_names:
                if name in self.kv:
                    self.layers.append((layer_idx, name, g))
                layer_idx += 1
        self.num_layers = layer_idx
        self.sampled_layer_idxs = set(
            resolve_indices(config.score_sampled_layers, self.num_layers)
        )
        self._num_requests_evicted = 0
        # Decode-time eviction counters/throttle (mask-only).
        self._decode_step = 0
        self._num_decode_evictions = 0
        # Decoupled per-end targets (fraction-of-prompt path). Resolved at each
        # request's admission from its prompt block count P and read back during
        # decode, so admission and decode can use independent capacities. 0 means
        # "unset" (a request slot that has not been admitted under this path).
        max_num_reqs = int(self.evicted_store.shape[0])
        self._decode_cap_np = np.zeros(max_num_reqs, dtype=np.int32)
        self._decode_low_np = np.zeros(max_num_reqs, dtype=np.int32)
        logger.info(
            "[geo_kv] EvictionPolicy ready: %d attn layers, band=%s, "
            "policy=%s, rate=%s, agg=%s, warmup_pages=%s, capacity=%s, "
            "watermark=%s, prefill_frac=%s, decode_frac=%s, "
            "physical_reclaim=%s",
            self.num_layers,
            config.score_sampled_layers,
            config.eviction_policy,
            config.eviction_rate,
            config.block_score_aggregation,
            config.warmup_pages,
            config.decode_evict_budget,
            config.decode_evict_watermark,
            config.prefill_evict_frac,
            config.decode_evict_frac,
            config.physical_reclaim,
        )

    def reset_request(self, req_index: int) -> None:
        """Clear a request slot's decoupled per-end band (recycled-slot safety).

        Admission always re-resolves the band before a request decodes, so this
        is belt-and-suspenders: it guarantees a reused slot never inherits a
        stale decode capacity if admission is skipped for any reason.

        Args:
            req_index: Persistent request-state index being (re)assigned.
        """
        self._decode_cap_np[req_index] = 0
        self._decode_low_np[req_index] = 0

    # -- per-step entry point ----------------------------------------------
    def on_step(
        self, input_batch: InputBatch, block_tables: Any
    ) -> dict[str, list[int]]:
        """Decide eviction for any request whose prefill completed this step.

        Returns:
            Mapping {req_id: freed_logical_block_indices} for prefills that
            completed this step. Populated only when config.physical_reclaim is
            set.
        """
        finished = self._finished_prefill_requests(input_batch)
        freed: dict[str, list[int]] = {}
        for batch_i, req_index, prompt_len in finished:
            evicted = self._evict_request(req_index, prompt_len, block_tables)
            if evicted:
                freed[input_batch.req_ids[batch_i]] = evicted
        self._num_requests_evicted += len(finished)
        return freed

    def _finished_prefill_requests(
        self, input_batch: InputBatch
    ) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        for i in range(input_batch.num_reqs):
            if not bool(input_batch.is_prefilling_np[i]):
                continue
            if input_batch.req_ids[i].startswith(_SYNTHETIC_REQ_PREFIXES):
                continue  # skip kernel-warmup / dummy requests
            computed = int(input_batch.num_computed_prefill_tokens_np[i])
            scheduled = int(input_batch.num_scheduled_tokens[i])
            prompt_len = int(input_batch.prefill_len_np[i])
            if computed + scheduled >= prompt_len:
                req_index = int(input_batch.idx_mapping_np[i])
                out.append((i, req_index, prompt_len))
        return out

    # -- per-request eviction decision -------------------------------------
    def _prompt_block_count(self, req_index: int, block_tables: Any) -> int:
        """Cheap read of a request's block count from the first sampled layer.

        The physical eviction unit is a whole block shared across layers (Option
        A), so every sampled layer reports the same count; reading one avoids the
        O(C^2) scoring scan when no admission eviction will fire (decode-only).
        """
        for layer_idx, _layer_name, group_id in self.layers:
            if layer_idx not in self.sampled_layer_idxs:
                continue
            return int(block_tables.num_blocks.np[group_id, req_index])
        return 0

    def _score_request_blocks(
        self, req_index: int, valid_len: int, block_tables: Any
    ) -> torch.Tensor | None:
        """Score a request's blocks by V-redundancy across the sampled layers.

        Args:
            req_index: Persistent request-state index.
            valid_len: Sequence length used to compute per-block valid token
                counts (prompt length at admission; storage length at decode).
            block_tables: The runner's block tables.

        Returns:
            ``(count,)`` aggregated droppability scores, or ``None`` when the
            request holds fewer than two blocks (nothing to evict).
        """
        cfg = self.config
        per_layer: list[torch.Tensor] = []
        for layer_idx, layer_name, group_id in self.layers:
            if layer_idx not in self.sampled_layer_idxs:
                continue
            c = int(block_tables.num_blocks.np[group_id, req_index])
            if c < 2:
                continue
            block_ids = (
                block_tables.block_tables[group_id].gpu[req_index, :c].to(torch.long)
            )
            v = self.kv[layer_name][block_ids, 1]  # (c, block_size, H, D)
            valid = block_valid_lens(valid_len, c, v.shape[1], v.device)
            per_layer.append(block_joint_redundancy(block_anchors(v, valid)))
        if not per_layer:
            return None
        stacked = torch.stack(per_layer, dim=0)  # (num_sampled, count)
        return aggregate_layer_scores(
            stacked, cfg.block_score_aggregation, cfg.topk_frac
        )

    def _evict_request(
        self, req_index: int, prompt_len: int, block_tables: Any
    ) -> list[int]:
        cfg = self.config
        # Decoupled fraction-of-prompt path (takes precedence over the coupled
        # budget path). ``count`` is this request's prompt block count P. Record
        # its independent decode band now, from a cheap block-count read, so
        # decode can enforce a capacity that differs from the admission target;
        # ``0`` means decode eviction is off for this slot. Decode-only skips the
        # expensive admission scoring entirely.
        frac_active = (
            cfg.prefill_evict_frac is not None or cfg.decode_evict_frac is not None
        )
        if frac_active:
            count = self._prompt_block_count(req_index, block_tables)
            if cfg.decode_evict_frac is not None and count >= 2:
                cap, low = frac_band(
                    cfg.decode_evict_frac, cfg.decode_evict_watermark, count
                )
                self._decode_cap_np[req_index] = cap
                self._decode_low_np[req_index] = low
            else:
                self._decode_cap_np[req_index] = 0
                self._decode_low_np[req_index] = 0
            # Admission eviction only when prefill_evict_frac is set; decode-only
            # admits the full prompt (evict nothing now). keep = ceil(f * P);
            # f == 1.0 (or a sub-2-block prompt) is inert.
            if cfg.prefill_evict_frac is None or count < 2:
                return []
            keep = max(1, math.ceil(cfg.prefill_evict_frac * count))
            if keep >= count:
                return []
            scores = self._score_request_blocks(req_index, prompt_len, block_tables)
            if scores is None:
                return []
            mask = select_evicted_to_budget(
                scores,
                count,
                keep,
                int(cfg.warmup_pages or 0),
                cfg.eviction_policy,
                cfg.eviction_seed,
            )
            self.evicted_store[req_index, : scores.shape[0]].copy_(
                mask.to(self.evicted_store.device)
            )
            if not cfg.physical_reclaim:
                return []
            return torch.nonzero(mask).flatten().tolist()

        # Legacy coupled paths (rate / decode_evict_budget). Both share the same
        # end-of-prefill scoring over the full prompt.
        scores = self._score_request_blocks(req_index, prompt_len, block_tables)
        if scores is None:
            return []
        count = int(scores.shape[0])
        if cfg.decode_evict_budget is not None:
            # Capacity-band mode: prefill fills the cache to the full prompt
            # (full-quality prefill). Only once it has reached capacity C do we
            # evict the most-redundant interior blocks down to the low watermark
            # (0.75 C). A sub-capacity prompt is under budget and left intact;
            # decode grows it to C before the first eviction fires. Same trigger
            # (reach C) / target (0.75 C) as the decode path.
            capacity, low = capacity_band(cfg)
            if count < capacity:
                return []
            mask = select_evicted_to_budget(
                scores,
                count,
                low,
                int(cfg.warmup_pages or 0),
                cfg.eviction_policy,
                cfg.eviction_seed,
            )
        else:
            mask = select_evicted_blocks(
                scores,
                count,
                cfg.eviction_rate or 0.0,
                cfg.eviction_policy,
                int(cfg.warmup_pages or 0),
                cfg.eviction_seed,
            )
        self.evicted_store[req_index, :count].copy_(mask.to(self.evicted_store.device))
        if not cfg.physical_reclaim:
            return []
        # The store row was all-False before this write, so every True here is
        # newly evicted. These are logical block indices the scheduler will
        # physically free
        return torch.nonzero(mask).flatten().tolist()

    # -- decode-time eviction (capacity band, physical reclaim) ------------
    def on_decode_step(
        self, input_batch: InputBatch, block_tables: Any
    ) -> dict[str, list[int]]:
        """Enforce the per-request capacity band during decode.

        The decode analog of :meth:`on_step`: for every request currently in
        decode that has regrown to capacity ``C`` (``decode_evict_budget``),
        score its blocks by V-redundancy and evict the most-redundant interior
        blocks down to the low watermark ``floor(watermark * C)``. Under physical
        reclaim the evicted logical block indices are returned for the scheduler
        to compact and free; the mask hides them for the one-step latency window
        until the compaction lands. A no-op unless ``decode_evict_budget`` is set.

        Returns:
            Mapping ``{req_id: freed_logical_block_indices}`` for requests
            evicted this step (physical reclaim only; empty under mask-only).
        """
        cfg = self.config
        # Decoupled frac path takes precedence; else the coupled budget path.
        frac_decode = cfg.decode_evict_frac is not None
        if not frac_decode and cfg.decode_evict_budget is None:
            return {}
        self._decode_step += 1
        if self._decode_step % max(int(cfg.decode_evict_interval), 1) != 0:
            return {}
        # Global band for the legacy budget path; per-request bands (resolved at
        # admission) for the frac path.
        cap_g, low_g = (0, 0) if frac_decode else capacity_band(cfg)
        freed: dict[str, list[int]] = {}
        for i in range(input_batch.num_reqs):
            if bool(input_batch.is_prefilling_np[i]):
                continue  # decode requests only
            if input_batch.req_ids[i].startswith(_SYNTHETIC_REQ_PREFIXES):
                continue
            req_index = int(input_batch.idx_mapping_np[i])
            if frac_decode:
                capacity = int(self._decode_cap_np[req_index])
                low = int(self._decode_low_np[req_index])
                if capacity < 2:
                    continue  # decode eviction not armed for this request
            else:
                capacity, low = cap_g, low_g
            # Current total sequence length after this step's tokens (RoPE /
            # uncompacted positions); storage length subtracts evicted tokens.
            seq_len = int(input_batch.num_computed_tokens_np[i]) + int(
                input_batch.num_scheduled_tokens[i]
            )
            evicted = self._evict_decode_request(
                req_index, seq_len, block_tables, capacity, low
            )
            if evicted:
                freed[input_batch.req_ids[i]] = evicted
        return freed

    def _evict_decode_request(
        self,
        req_index: int,
        seq_len: int,
        block_tables: Any,
        capacity: int,
        low: int,
    ) -> list[int]:
        """Evict one decoding request down to the low watermark at capacity.

        Mirrors :meth:`_evict_request`'s scoring but on the *storage* basis: the
        block table row is already compacted, so validity uses the storage length
        ``seq_len - num_evicted_tokens`` (not the uncompacted ``seq_len``), and
        the block indices are storage-relative — exactly what the scheduler's
        packed survivor row expects.

        Re-eviction safety (physical reclaim): a non-empty mask row means a prior
        compaction (prefill's fill-to-watermark or an earlier decode eviction) is
        still in flight — the block table has not yet been repacked. Firing then
        would score a stale row and hand the scheduler indices into a list it has
        already compacted, corrupting it. So skip until the compaction lands (the
        model runner zeroes the row on apply). Growth back to ``C`` after that is
        what re-arms the trigger — this is the hysteresis band.

        Returns:
            Logical (storage) block indices to physically free, or ``[]`` when
            nothing is evicted or under mask-only mode (which frees nothing).
        """
        cfg = self.config
        physical = cfg.physical_reclaim
        if physical and bool(self.evicted_store[req_index].any()):
            return []  # compaction in flight; wait for the row to clear
        # Hysteresis: fire only once the request has regrown to capacity C.
        # Read the block count cheaply first to skip scoring when under C.
        count = self._prompt_block_count(req_index, block_tables)
        if count < capacity or count < 2:
            return []
        num_evicted = 0
        if self.num_evicted_tokens_np is not None:
            num_evicted = int(self.num_evicted_tokens_np[req_index])
        storage_len = seq_len - num_evicted
        scores = self._score_request_blocks(req_index, storage_len, block_tables)
        if scores is None:
            return []
        if physical:
            # Clean basis (row cleared before we got here): the selected mask is
            # the full eviction set. Set the mask (hides blocks for the one-step
            # window) and report the indices for compaction.
            mask = select_evicted_to_budget(
                scores,
                count,
                low,
                int(cfg.warmup_pages or 0),
                cfg.eviction_policy,
                cfg.eviction_seed,
            )
            if not bool(mask.any()):
                return []
            self._num_decode_evictions += 1
            self.evicted_store[req_index, :count].copy_(
                mask.to(self.evicted_store.device)
            )
            return torch.nonzero(mask).flatten().tolist()
        # Mask-only A/B mode (no compaction ever lands to clear the row): never
        # re-keep an already-hidden block. For v_redundancy, force its score high
        # so scoring re-selects it; recency/random pick by index (oldest-k is a
        # prefix, so previously-hidden oldest blocks stay chosen). The OR-merge
        # then guarantees the store only ever gains True and stays at the
        # watermark for every policy.
        prev = self.evicted_store[req_index, :count].to("cpu")
        if bool(prev.any()):
            scores = scores.detach().to(device="cpu", dtype=torch.float32).clone()
            scores[prev] = float("inf")
        mask = select_evicted_to_budget(
            scores,
            count,
            low,
            int(cfg.warmup_pages or 0),
            cfg.eviction_policy,
            cfg.eviction_seed,
        )
        merged = mask | prev
        if bool((merged & ~prev).any()):
            self._num_decode_evictions += 1
        self.evicted_store[req_index, :count].copy_(
            merged.to(self.evicted_store.device)
        )
        return []

    def close(self) -> None:
        logger.info(
            "[geo_kv] EvictionPolicy closed: evicted %d requests "
            "(prefill), %d decode-eviction events",
            self._num_requests_evicted,
            self._num_decode_evictions,
        )
