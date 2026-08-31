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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.geo_kv.config import GeoKVConfig, resolve_indices
from vllm.v1.geo_kv.incremental_scorer import IncrementalRedundancyState
from vllm.v1.geo_kv.layer_calibrator import LayerCalibrator
from vllm.v1.geo_kv.scorer_profiler import ScorerProfiler
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
    block_valid_lens,
    block_value_l2,
    positional_cosh_weights,
    refine_with_query_relevance,
    zscore,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

# Reserved request-id prefix used by kernel warmup; never a real serving request,
# so never evicted. Dummy runs are excluded explicitly by the model runner.
_KERNEL_WARMUP_REQ_PREFIX = "_warmup_"


def _choose_evicted(
    evictable_t: torch.Tensor,
    scores: torch.Tensor,
    lo: int,
    hi: int,
    k: int,
    policy: str,
    seed: int,
    protect_mask: torch.Tensor | None = None,
    protect_order: torch.Tensor | None = None,
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
        protect_mask: optional ``(num_blocks,)`` bool of blocks to shield from
            eviction (Phase 2 norm protection). Only the ``[lo:hi]`` slice is
            consulted, and only for the scored ``v_redundancy`` path. Protected
            blocks are pushed below every candidate so they are never chosen --
            unless too few candidates remain to reach ``k``, in which case
            protection is relaxed (see ``protect_order``) to keep the budget exact.
        protect_order: optional ``(num_blocks,)`` float used to order relaxation
            when the protect set leaves fewer than ``k`` candidates: protected
            blocks are un-protected in ascending ``protect_order`` (lowest first,
            ties by block index) until exactly ``k`` candidates are available.
            The aggregated value-L2 norm, so the least-valuable protected blocks
            give up protection first.

    Returns:
        ``(k,)`` long tensor: the chosen block indices.
    """
    if policy in ("v_redundancy", "value_l2"):
        # Both are scored policies: highest droppability first. v_redundancy
        # scores by V-redundancy; value_l2 passes a negated value-L2 norm (so
        # lowest-norm == highest droppability), computed in _score_request_blocks.
        #
        # Phase 1A: rank on the score's own device and move only the k chosen
        # indices to host (the old path copied the whole score vector to CPU and
        # fully sorted it). ``stable=True`` fixes the tie rule: equal scores keep
        # ascending window order, so the LOWER block index is evicted first --
        # deterministic and independent of the sort backend. For tie-free scores
        # this selects exactly the same set as the previous unstable argsort.
        local = scores.detach()[lo:hi].to(torch.float32)
        if protect_mask is not None:
            # Phase 2: shield high-value-norm blocks, but keep the budget exact by
            # relaxing the lowest-norm protections when there are too few other
            # candidates to reach k. Sink / anchor / tail are already outside
            # [lo:hi], so they are never in this window and never relaxed.
            prot = protect_mask[lo:hi].to(device=local.device, dtype=torch.bool)
            num_prot = int(prot.sum())
            avail = local.shape[0] - num_prot
            if avail < k and num_prot > 0:
                need = k - avail
                if protect_order is not None:
                    order_w = protect_order[lo:hi].to(
                        device=local.device, dtype=torch.float32
                    )
                else:
                    order_w = torch.zeros_like(local)
                prot = prot.clone()
                prot_idx = torch.nonzero(prot).flatten()
                relax_rank = torch.argsort(
                    order_w[prot_idx], descending=False, stable=True
                )
                prot[prot_idx[relax_rank[:need]]] = False
            # Push still-protected blocks below every candidate. The relaxation
            # above guarantees at least k finite (unprotected) scores remain.
            local = local.masked_fill(prot, float("-inf"))
        order = torch.argsort(local, descending=True, stable=True)
        # ``evictable_t`` is a CPU long tensor; index it with a CPU order (the
        # ``.cpu()`` is a no-op when the scores were already on CPU).
        return evictable_t[order[:k].cpu()]
    if policy == "recency":
        return evictable_t[:k]  # oldest k blocks (StreamingLLM contrast)
    if policy == "random":
        gen = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(evictable_t.shape[0], generator=gen)
        return evictable_t[perm[:k]]
    raise ValueError(f"unknown eviction policy: {policy!r}")


def _value_norm_protect_mask(
    num_blocks: int,
    lo: int,
    hi: int,
    policy: str,
    value_norm: torch.Tensor | None,
    quantile: float | None,
) -> torch.Tensor | None:
    """Build the Phase-2 high-value-norm protection mask, or ``None`` if off.

    Protects every evictable block whose aggregated value-L2 norm is at or above
    the per-request ``quantile`` of the evictable window ``[lo:hi]``. Sink /
    anchor / tail blocks sit outside that window and are handled by the caller, so
    they are never marked here.

    Args:
        num_blocks: Current logical block count (mask length).
        lo: Start of the evictable window (== ``warmup_pages``).
        hi: End of the evictable window (exclusive; excludes the anchor / tail).
        policy: Eviction policy; protection applies only to ``v_redundancy``.
        value_norm: ``(num_blocks,)`` aggregated value-L2 norm, or ``None``.
        quantile: Protection quantile in ``[0, 1)``, or ``None`` (off).

    Returns:
        ``(num_blocks,)`` bool mask (True == protect), or ``None`` when
        protection is disabled or inapplicable.
    """
    if (
        quantile is None
        or value_norm is None
        or policy != "v_redundancy"
        or hi - lo <= 0
    ):
        return None
    win = value_norm[lo:hi].to(dtype=torch.float32)
    if win.numel() == 0:
        return None
    # ``quantile`` interpolates linearly, so q == 0 -> min (protect all, then the
    # budget relaxation peels the lowest-norm blocks back to exactly k -> behaves
    # like a value-L2 cut); q near 1 -> only the top-norm blocks are protected
    # (near-pure V-redundancy).
    thr = torch.quantile(win, float(quantile))
    mask = torch.zeros(num_blocks, dtype=torch.bool)
    mask[lo:hi] = (win >= thr).to(device="cpu")
    return mask


def _query_relevance_protect_mask(
    num_blocks: int,
    lo: int,
    hi: int,
    policy: str,
    query_relevance: torch.Tensor | None,
    quantile: float | None,
) -> torch.Tensor | None:
    """Build a high-query-relevance protection mask, or ``None`` if off.

    The threshold is computed only over the current evictable window, after
    sink, anchor, and optional tail protection have narrowed it. The selector
    uses the same relevance values to deterministically relax the least-relevant
    protected blocks when necessary to preserve an exact budget.

    Args:
        num_blocks: Current logical block count (mask length).
        lo: Start of the evictable window.
        hi: End of the evictable window (exclusive).
        policy: Eviction policy; only ``v_redundancy`` is supported.
        query_relevance: ``(num_blocks,)`` aggregated causal QK attention mass.
        quantile: Protection quantile in ``[0, 1)``, or ``None`` when disabled.

    Returns:
        ``(num_blocks,)`` CPU bool mask, or ``None`` when disabled.

    Raises:
        ValueError: If protection is enabled without matching relevance values.
    """
    if quantile is None:
        return None
    if policy != "v_redundancy":
        raise ValueError(
            "query relevance protection requires policy='v_redundancy'"
        )
    if query_relevance is None:
        raise ValueError(
            "query relevance protection requires per-block query relevance"
        )
    if query_relevance.shape != (num_blocks,):
        raise ValueError("query relevance must match num_blocks")
    if hi - lo <= 0:
        return None
    win = query_relevance[lo:hi].to(dtype=torch.float32)
    if win.numel() == 0:
        return None
    threshold = torch.quantile(win, float(quantile))
    mask = torch.zeros(num_blocks, dtype=torch.bool)
    mask[lo:hi] = (win >= threshold).to(device="cpu")
    return mask


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
    tail_protect_frac: float | None = None,
    value_norm: torch.Tensor | None = None,
    value_norm_protect_quantile: float | None = None,
    query_relevance: torch.Tensor | None = None,
    query_relevance_protect_quantile: float | None = None,
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
        tail_protect_frac: When set (>0), also protect the last
            ``ceil(tail_protect_frac * num_blocks)`` blocks from eviction (a
            recency floor over the prompt tail, where the LongBench question
            lives). Clamped so enough interior blocks remain to still reach
            ``budget`` -- memory stays matched. None/0.0 restores the pinned
            v_redundancy window (no tail guard beyond the single anchor).
        value_norm: optional ``(num_blocks,)`` aggregated value-L2 norm per block,
            required when ``value_norm_protect_quantile`` is set. Used both to
            build the protection threshold and to order relaxation.
        value_norm_protect_quantile: Phase 2 norm-constrained redundancy. When set
            (and ``value_norm`` given), protect blocks whose value-L2 norm is at or
            above this per-request quantile so V-eviction chooses among the
            remaining lower-norm blocks. Budget stays exact: if too few candidates
            remain, the lowest-norm protections are relaxed until ``k`` are
            available. Applies only to the scored ``v_redundancy`` path; sink /
            anchor / tail protection always take priority. None == off.
        query_relevance: Optional ``(num_blocks,)`` causal query-attention mass.
        query_relevance_protect_quantile: Protect blocks at or above this
            per-request relevance quantile. If necessary, relax protection in
            ascending relevance order to keep the exact budget. None == off.

    Returns:
        ``(num_blocks,)`` bool tensor; True = evict (hide from attention).
    """
    evict = torch.zeros(num_blocks, dtype=torch.bool)
    if num_blocks <= budget:
        return evict
    lo = max(int(warmup_pages), 0)
    hi = num_blocks - 1  # always keep the final block (decode anchor)
    # Drop just enough to reach budget (bounded by the base evictable window).
    k = min(num_blocks - budget, max(hi - lo, 0))
    if k <= 0:
        return evict
    # Tail-protect (recency floor): pull ``hi`` in to shield the last M blocks.
    # Clamp M so at least k evictable blocks remain -- we must still reach budget,
    # so the guard never shrinks the window below what the budget cut requires.
    if tail_protect_frac:
        m = math.ceil(tail_protect_frac * num_blocks)
        m = max(0, min(m, (hi - lo) - k))
        hi -= m
    evictable = list(range(lo, hi))
    n = len(evictable)
    if n <= 0:
        return evict
    k = min(k, n)
    if k <= 0:
        return evict
    protect_mask = _value_norm_protect_mask(
        num_blocks, lo, hi, policy, value_norm, value_norm_protect_quantile
    )
    query_protect_mask = _query_relevance_protect_mask(
        num_blocks,
        lo,
        hi,
        policy,
        query_relevance,
        query_relevance_protect_quantile,
    )
    if protect_mask is not None and query_protect_mask is not None:
        raise ValueError(
            "value-norm and query-relevance hard protections cannot be combined"
        )
    if query_protect_mask is not None:
        protect_mask = query_protect_mask
        protect_order = query_relevance
    else:
        protect_order = value_norm if protect_mask is not None else None
    evictable_t = torch.tensor(evictable, dtype=torch.long)
    chosen = _choose_evicted(
        evictable_t,
        scores,
        lo,
        hi,
        k,
        policy,
        seed,
        protect_mask=protect_mask,
        protect_order=protect_order,
    )
    evict[chosen] = True
    return evict


def select_evicted_r2r(
    scores: torch.Tensor,
    query_relevance: torch.Tensor,
    num_blocks: int,
    budget: int,
    warmup_pages: int,
    candidate_expansion_factor: float,
    tail_protect_frac: float | None = None,
    similarity: torch.Tensor | None = None,
    cover_cost: torch.Tensor | None = None,
    cover_depth: int = 2,
    selection_stats: dict[str, int] | None = None,
) -> torch.Tensor:
    """Evict least-relevant blocks from a redundancy-ranked shortlist.

    Stage 1 selects up to ``round(factor * k)`` of the lowest projection-
    residual-cost blocks, where ``k`` is the exact number required by the
    budget. Stage 2
    evicts the ``k`` least query-relevant candidates. When ``similarity`` is
    or ``cover_cost`` is provided, a top-``cover_depth`` guard avoids evicting
    all available covers of a candidate when another candidate can satisfy the
    budget. Legacy similarity input ranks by absolute cosine; production R2R
    ranks the same projection-residual pair costs used by Stage 1.
    Sink, final-anchor, and clamped tail protections match
    :func:`select_evicted_to_budget`.
    """
    if scores.shape != (num_blocks,) or query_relevance.shape != (num_blocks,):
        raise ValueError("R2R scores and query relevance must match num_blocks")
    if selection_stats is not None:
        selection_stats.clear()
    if candidate_expansion_factor < 1.0:
        raise ValueError("R2R candidate expansion factor must be >= 1")
    if (
        not isinstance(cover_depth, int)
        or isinstance(cover_depth, bool)
        or cover_depth < 0
    ):
        raise ValueError("R2R cover depth must be an integer >= 0")
    evict = torch.zeros(num_blocks, dtype=torch.bool)
    if num_blocks <= budget:
        return evict
    lo = max(int(warmup_pages), 0)
    hi = num_blocks - 1
    k = min(num_blocks - budget, max(hi - lo, 0))
    if k <= 0:
        return evict
    if tail_protect_frac:
        tail = math.ceil(tail_protect_frac * num_blocks)
        tail = max(0, min(tail, (hi - lo) - k))
        hi -= tail
    window_size = hi - lo
    if window_size <= 0:
        return evict
    k = min(k, window_size)
    candidate_count = min(
        max(int(round(candidate_expansion_factor * k)), k),
        window_size,
    )

    residual_cost = scores.detach()[lo:hi].to(torch.float32)
    candidate_order = torch.argsort(residual_cost, descending=False, stable=True)
    candidates = candidate_order[:candidate_count] + lo
    relevance = query_relevance.detach().to(
        device=candidates.device, dtype=torch.float32
    )
    relevance_order = torch.argsort(
        relevance[candidates], descending=False, stable=True
    )
    ranked_candidates = candidates[relevance_order]
    if similarity is None and cover_cost is None:
        chosen = ranked_candidates[:k].to(device="cpu")
        if selection_stats is not None:
            selection_stats["unguarded"] = k
    elif cover_depth == 0:
        chosen = ranked_candidates[:k].to(device="cpu")
        if selection_stats is not None:
            selection_stats["unguarded"] = k
    else:
        cover_matrix = cover_cost if cover_cost is not None else similarity
        assert cover_matrix is not None
        if cover_matrix.shape != (num_blocks, num_blocks):
            raise ValueError("R2R cover matrix must be square and match num_blocks")
        depth = min(int(cover_depth), max(num_blocks - 1, 0))
        # Only candidate rows are needed. Keep the O(n*b*B) gather and cover
        # ranking on GPU, then transfer the small decision table once. The guard
        # is order-dependent within each pass, so a Python loop is appropriate
        # on CPU; doing bool(tensor) here would synchronize the GPU per candidate.
        candidate_scores = cover_matrix.detach().to(
            device=ranked_candidates.device, dtype=torch.float32
        ).index_select(0, ranked_candidates)
        if cover_cost is None:
            candidate_scores = candidate_scores.abs()
        rows = torch.arange(
            ranked_candidates.numel(), device=ranked_candidates.device
        )
        candidate_scores[rows, ranked_candidates] = (
            float("inf") if cover_cost is not None else float("-inf")
        )
        ranked_covers = torch.argsort(
            candidate_scores,
            dim=1,
            descending=cover_cost is None,
            stable=True,
        )[:, :depth]
        decision_table = torch.cat(
            (ranked_candidates[:, None], ranked_covers), dim=1
        ).to(device="cpu")
        decision_rows = decision_table.tolist()

        chosen_list: list[int] = []
        chosen_set: set[int] = set()
        for cover_rank in range(depth):
            for decision in decision_rows:
                candidate = decision[0]
                if candidate in chosen_set:
                    continue
                if decision[cover_rank + 1] not in chosen_set:
                    chosen_list.append(candidate)
                    chosen_set.add(candidate)
                    if selection_stats is not None:
                        key = f"cover_{cover_rank + 1}"
                        selection_stats[key] = selection_stats.get(key, 0) + 1
                    if len(chosen_list) == k:
                        break
            if len(chosen_list) == k:
                break
        # Final unguarded pass preserves exact matched memory.
        if len(chosen_list) < k:
            for decision in decision_rows:
                candidate = decision[0]
                if candidate not in chosen_set:
                    chosen_list.append(candidate)
                    chosen_set.add(candidate)
                    if selection_stats is not None:
                        selection_stats["backfill"] = (
                            selection_stats.get("backfill", 0) + 1
                        )
                    if len(chosen_list) == k:
                        break
        chosen = torch.tensor(chosen_list, dtype=torch.long, device="cpu")
    evict[chosen] = True
    return evict


def select_evicted_by_coverage(
    similarity: torch.Tensor,
    num_blocks: int,
    budget: int,
    warmup_pages: int,
    tail_protect_frac: float | None = None,
    importance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct a global kept set by greedy facility-location coverage.

    Maximizes ``sum_b importance[b] * max_{s in kept} similarity[b, s]`` at the
    actual keep budget. Sink, final anchor, and configured tail blocks seed the
    kept set before additions. Selection happens once on the similarity matrix
    aggregated across layers, unlike the older greedy mode's independent
    per-layer ordinal peels.
    """
    evict = torch.zeros(num_blocks, dtype=torch.bool)
    if num_blocks <= budget:
        return evict
    if similarity.shape != (num_blocks, num_blocks):
        raise ValueError("coverage similarity must be square and match num_blocks")
    lo = max(int(warmup_pages), 0)
    hi = num_blocks - 1
    k = min(num_blocks - budget, max(hi - lo, 0))
    if k <= 0:
        return evict
    if tail_protect_frac:
        tail = math.ceil(tail_protect_frac * num_blocks)
        tail = max(0, min(tail, (hi - lo) - k))
        hi -= tail
    keep_target = num_blocks - k
    protected = np.zeros(num_blocks, dtype=np.bool_)
    protected[:lo] = True
    protected[hi:] = True
    kept = protected.copy()

    sim = similarity.detach().to(device="cpu", dtype=torch.float32).numpy()
    if importance is None:
        weight = np.ones(num_blocks, dtype=np.float32)
    else:
        if importance.shape != (num_blocks,):
            raise ValueError("coverage importance must match num_blocks")
        weight = importance.detach().to(device="cpu", dtype=torch.float32).numpy()
        if np.any(weight < 0):
            raise ValueError("coverage importance must be non-negative")

    protected_idx = np.flatnonzero(protected)
    # The final anchor is always protected, so this is non-empty for B >= 2.
    coverage = sim[:, protected_idx].max(axis=1)
    while int(kept.sum()) < keep_target:
        gain = (np.maximum(sim - coverage[:, None], 0.0) * weight[:, None]).sum(
            axis=0
        )
        gain[kept] = -np.inf
        chosen = int(np.argmax(gain))  # first index gives deterministic tie-break
        kept[chosen] = True
        coverage = np.maximum(coverage, sim[:, chosen])
    evict.copy_(torch.from_numpy(~kept))
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
        query_capturer: Any = None,
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
        self.query_capturer = query_capturer

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
            resolve_indices(
                config.score_sampled_layers, self.num_layers, strict=True
            )
        )
        # Opt-in scorer profiler (plan Phase 1B). A no-op unless enable_tracing is
        # set; when off it adds no timing and no device synchronization.
        self.profiler = ScorerProfiler(
            enabled=bool(config.enable_tracing),
            device=device,
            log_path=config.log_path,
        )
        # Opt-in layer-subset calibrator (plan Phase 1C). Only meaningful while
        # scoring all layers (there must be all rows to sub-select from); a no-op
        # otherwise. Requires enable_tracing so it never runs on serving paths.
        self.layer_calibrator = LayerCalibrator(
            enabled=bool(
                config.calibrate_layer_subsets
                and config.enable_tracing
                and config.score_sampled_layers == "all"
            ),
            num_layers=self.num_layers,
            log_path=config.log_path,
        )
        # Per-fire scratch for the calibrator (set by scoring, read by the
        # selection site). Only populated when the calibrator is enabled.
        self._cal_stacked: torch.Tensor | None = None
        self._cal_vnorm_stacked: torch.Tensor | None = None
        self._cal_row_layer_idxs: list[int] = []
        # Per-fire global similarity matrix for coverage or R2R cover selection.
        self._coverage_similarity: torch.Tensor | None = None
        self._r2r_cover_cost: torch.Tensor | None = None
        # Raw causal QK attention mass for hard relevance protection. It remains
        # separate from finalized scores so positional cosh can compose without
        # changing which high-relevance blocks the hard constraint protects.
        self._query_relevance: torch.Tensor | None = None
        self._num_requests_evicted = 0
        self._r2r_selection_counts: dict[str, int] = {}
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
        # Phase 6: lazy, bounded metadata for requests that have actually fired
        # the scorer.  State tensors are pooled-anchor/similarity copies and do
        # not alias paged KV.  A generation advances only at an authoritative
        # lifecycle boundary (compaction landing or request-slot reset).
        self._incremental_states: dict[
            int, dict[int, IncrementalRedundancyState]
        ] = {}
        self._incremental_pending_masks: dict[int, torch.Tensor] = {}
        self._incremental_generation_np = np.zeros(max_num_reqs, dtype=np.int64)
        logger.info(
            "[geo_kv] EvictionPolicy ready: %d attn layers, band=%s, "
            "policy=%s, rate=%s, agg=%s, warmup_pages=%s, capacity=%s, "
            "watermark=%s, prefill_frac=%s, decode_frac=%s, "
            "decode_blocks_per_step=%s, physical_reclaim=%s",
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
            config.decode_evict_blocks_per_step,
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
        self._incremental_generation_np[req_index] += 1
        self._incremental_states.pop(req_index, None)
        self._incremental_pending_masks.pop(req_index, None)

    def _stage_incremental_eviction(
        self, req_index: int, eviction_mask: torch.Tensor
    ) -> None:
        """Remember the old-row mask until its physical compaction lands."""
        if not self.config.incremental_decode_scoring:
            return
        self._incremental_pending_masks[req_index] = (
            eviction_mask.detach().to(device="cpu", dtype=torch.bool).clone()
        )

    def on_compaction_applied(self, req_index: int, survivor_count: int) -> None:
        """Acknowledge the runner's authoritative physical row replacement.

        Cached state is pruned only here, after the scheduler snapshot passed the
        runner's row-length assertion.  Missing/mismatched pending state is
        invalidated conservatively; it is never guessed from reusable physical
        block ids.
        """
        if not self.config.incremental_decode_scoring:
            return
        generation = int(self._incremental_generation_np[req_index]) + 1
        self._incremental_generation_np[req_index] = generation
        pending = self._incremental_pending_masks.pop(req_index, None)
        states = self._incremental_states.get(req_index)
        if pending is None or not states:
            self._incremental_states.pop(req_index, None)
            return
        for state in states.values():
            state.retain(pending, generation)
            if state.num_blocks != survivor_count:
                self._incremental_states.pop(req_index, None)
                return

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
            if input_batch.req_ids[i].startswith(_KERNEL_WARMUP_REQ_PREFIX):
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

        Thin wrapper over :meth:`_score_request_blocks_with_vnorm` that returns
        only the droppability scores (the value-norm is discarded).

        Args:
            req_index: Persistent request-state index.
            valid_len: Sequence length used to compute per-block valid token
                counts (prompt length at admission; storage length at decode).
            block_tables: The runner's block tables.

        Returns:
            ``(count,)`` aggregated droppability scores, or ``None`` when the
            request holds fewer than two blocks (nothing to evict).
        """
        scores, _ = self._score_request_blocks_with_vnorm(
            req_index, valid_len, block_tables
        )
        return scores

    def _score_request_blocks_with_vnorm(
        self, req_index: int, valid_len: int, block_tables: Any
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Score a request's blocks and, when needed, its per-block value norm.

        Args:
            req_index: Persistent request-state index.
            valid_len: Sequence length used to compute per-block valid token
                counts (prompt length at admission; storage length at decode).
            block_tables: The runner's block tables.

        Returns:
            ``(scores, value_norm)``. ``scores`` is the ``(count,)`` aggregated
            droppability, or ``None`` when the request holds fewer than two
            blocks. ``value_norm`` is the matching ``(count,)`` aggregated
            positive value-L2 norm (higher == more valuable), populated only when
            the blend or the Phase-2 norm-protection quantile is active; ``None``
            otherwise.
        """
        cfg = self.config
        self._coverage_similarity = None
        self._r2r_cover_cost = None
        self._query_relevance = None
        value_l2 = cfg.eviction_policy == "value_l2"
        greedy = (not value_l2) and cfg.redundancy_mode == "greedy"
        coverage_mode = (not value_l2) and cfg.redundancy_mode == "coverage"
        r2r_mode = (not value_l2) and cfg.candidate_expansion_factor is not None
        blend_beta = None if value_l2 else cfg.value_blend_beta
        # Phase 2 norm protection also needs the per-block value-L2 norm; compute
        # it whenever either consumer is active (both only apply to v_redundancy).
        want_vnorm = (not value_l2) and (
            bool(blend_beta) or cfg.value_norm_protect_quantile is not None
        )
        want_query = (not value_l2) and cfg.query_relevance_enabled
        warmup = int(cfg.warmup_pages or 0)
        per_layer: list[torch.Tensor] = []
        per_layer_vnorm: list[torch.Tensor] = []  # populated when want_vnorm
        per_layer_query: list[torch.Tensor] = []  # populated when available
        per_layer_similarity: list[torch.Tensor] = []
        per_layer_cover_cost: list[torch.Tensor] = []
        query_window_len: int | None = None
        scored_layer_idxs: list[int] = []  # global layer idx per per_layer row
        prof = self.profiler
        num_scored_layers = 0
        block_count = 0
        incremental = cfg.incremental_decode_scoring
        generation = int(self._incremental_generation_np[req_index])
        for layer_idx, layer_name, group_id in self.layers:
            if layer_idx not in self.sampled_layer_idxs:
                continue
            c = int(block_tables.num_blocks.np[group_id, req_index])
            if c < 2:
                continue
            valid_lens_cpu = tuple(
                max(0, min(self.block_size, valid_len - i * self.block_size))
                for i in range(c)
            )
            incremental_state: IncrementalRedundancyState | None = None
            changed_positions: list[int] | None = None
            changed_positions_tensor: torch.Tensor | None = None
            if incremental:
                request_states = self._incremental_states.setdefault(req_index, {})
                incremental_state = request_states.setdefault(
                    layer_idx, IncrementalRedundancyState.empty(generation)
                )
                changed_positions = incremental_state.changed_positions(
                    valid_lens_cpu, generation
                )
                changed_positions_tensor = torch.tensor(
                    changed_positions, dtype=torch.long, device=self.device
                )
            with prof.section("gather"):
                if incremental_state is not None:
                    assert changed_positions is not None
                    assert changed_positions_tensor is not None
                    changed_block_ids = (
                        block_tables.block_tables[group_id]
                        .gpu[req_index, :c]
                        .index_select(0, changed_positions_tensor)
                        .to(torch.long)
                    )
                    v = self.kv[layer_name][changed_block_ids, 1]
                    block_ids = None
                else:
                    block_ids = (
                        block_tables.block_tables[group_id]
                        .gpu[req_index, :c]
                        .to(torch.long)
                    )
                    v = self.kv[layer_name][
                        block_ids, 1
                    ]  # (c, block_size, H, D)
                queries = (
                    self.query_capturer.get(req_index, layer_idx)
                    if want_query and self.query_capturer is not None
                    else None
                )
                k = (
                    self.kv[layer_name][block_ids, 0]
                    if queries is not None and block_ids is not None
                    else None
                )
            num_scored_layers += 1
            block_count = c
            scored_layer_idxs.append(layer_idx)
            valid = block_valid_lens(valid_len, c, self.block_size, self.device)
            if value_l2:
                # Paged-Eviction baseline: drop lowest value-L2-norm blocks.
                # Negate so higher == more droppable, matching the scored-policy
                # convention consumed by argsort(descending) in _choose_evicted.
                with prof.section("anchor_norm"):
                    per_layer.append(
                        -block_value_l2(v, valid, cfg.value_l2_block_reduction)
                    )
                continue
            with prof.section("anchor_norm"):
                if cfg.block_prototype_mode == "mean":
                    if incremental_state is not None:
                        assert changed_positions is not None
                        assert changed_positions_tensor is not None
                        if changed_positions:
                            anchors = block_anchors(
                                v, valid.index_select(0, changed_positions_tensor)
                            )
                        else:
                            anchors = torch.empty(
                                (
                                    0,
                                    *incremental_state.normalized_anchors.shape[1:],
                                ),
                                dtype=torch.float32,
                                device=self.device,
                            )
                    else:
                        anchors = block_anchors(v, valid)
                    prototypes = prototype_valid = None
                else:
                    prototypes, prototype_valid = block_prototypes(
                        v, valid, cfg.block_prototype_mode
                    )
                    anchors = None
                if want_vnorm:
                    per_layer_vnorm.append(block_value_l2(v, valid))
            if queries is not None:
                assert k is not None
                query_window_len = int(queries.shape[0])
                with prof.section("query_attention"):
                    relevance_fn = (
                        block_key_anchor_relevance
                        if cfg.r2r_relevance_signal == "key_anchor"
                        else block_query_attention_mass
                    )
                    per_layer_query.append(
                        relevance_fn(
                            queries,
                            k,
                            valid,
                            self.query_capturer.scale(layer_idx),
                        )
                    )
            with prof.section("similarity"):
                if coverage_mode:
                    assert anchors is not None
                    similarity = block_joint_similarity(anchors)
                    per_layer_similarity.append(similarity)
                    diagonal = torch.eye(c, dtype=torch.bool, device=v.device)
                    per_layer.append(
                        similarity.masked_fill(diagonal, float("-inf"))
                        .max(dim=1)
                        .values
                    )
                elif r2r_mode:
                    assert anchors is not None
                    pair_cost = block_pair_residual_cost(anchors)
                    per_layer_cover_cost.append(pair_cost)
                    per_layer.append(pair_cost.amin(dim=1))
                elif greedy:
                    assert anchors is not None
                    # Keep sink + anchor in the comparison set (a block that
                    # duplicates them is genuinely redundant) but never peel them.
                    protect = torch.zeros(c, dtype=torch.bool, device=v.device)
                    if warmup > 0:
                        protect[: min(warmup, c)] = True
                    protect[c - 1] = True  # decode anchor
                    per_layer.append(
                        block_joint_redundancy_greedy(anchors, protect=protect)
                    )
                else:
                    if anchors is not None:
                        if incremental_state is not None:
                            assert changed_positions is not None
                            per_layer.append(
                                incremental_state.update(
                                    valid_lens_cpu,
                                    generation,
                                    changed_positions,
                                    anchors,
                                    changed_positions_tensor,
                                )
                            )
                            prof.record_incremental_update(
                                total_blocks=c,
                                recomputed_blocks=len(changed_positions),
                            )
                        else:
                            per_layer.append(block_joint_redundancy(anchors))
                    else:
                        assert prototypes is not None
                        assert prototype_valid is not None
                        per_layer.append(
                            block_multi_prototype_redundancy(
                                prototypes, prototype_valid
                            )
                        )
        if not per_layer:
            return None, None
        with prof.section("aggregate"):
            stacked = torch.stack(per_layer, dim=0)  # (num_sampled, count)
            scores = aggregate_layer_scores(
                stacked, cfg.block_score_aggregation, cfg.topk_frac
            )
            # Aggregate the positive value-L2 norm on the same footing as the
            # scores (same layer aggregation / topk_frac), for the blend and/or
            # Phase-2 norm protection. Higher == more valuable.
            vnorm: torch.Tensor | None = None
            if want_vnorm and per_layer_vnorm:
                vnorm = aggregate_layer_scores(
                    torch.stack(per_layer_vnorm, dim=0),
                    cfg.block_score_aggregation,
                    cfg.topk_frac,
                )
            query_relevance: torch.Tensor | None = None
            if per_layer_query:
                query_relevance = torch.stack(per_layer_query, dim=0).mean(dim=0)
            if per_layer_similarity:
                self._coverage_similarity = aggregate_layer_scores(
                    torch.stack(per_layer_similarity, dim=0),
                    cfg.block_score_aggregation,
                    cfg.topk_frac,
                )
            if per_layer_cover_cost:
                self._r2r_cover_cost = aggregate_layer_scores(
                    torch.stack(per_layer_cover_cost, dim=0),
                    cfg.block_score_aggregation,
                    cfg.topk_frac,
                )
                # Keep candidate scores and guard covers consistent: choose the
                # cheapest cover only after aggregating its cost across layers.
                scores = self._r2r_cover_cost.amin(dim=1)
        if (
            cfg.query_relevance_protect_quantile is not None
            and query_relevance is None
        ):
            raise RuntimeError(
                "query relevance protection is enabled but no query relevance "
                "was captured"
            )
        self._query_relevance = query_relevance
        prof.record_fire(block_count, num_scored_layers)
        if query_window_len is not None:
            prof.record_query_window(query_window_len)
        # Stash the raw per-layer stacks for the Phase-1C calibrator (all-layer
        # only). Kept as detached references so the calibrator can re-aggregate
        # arbitrary subsets against the reference selection computed downstream.
        if self.layer_calibrator.enabled:
            self._cal_stacked = stacked
            self._cal_vnorm_stacked = (
                torch.stack(per_layer_vnorm, dim=0) if per_layer_vnorm else None
            )
            self._cal_row_layer_idxs = list(scored_layer_idxs)
        scores = self._finalize_scores(
            scores, vnorm, value_l2, query_relevance=query_relevance
        )
        return scores, vnorm

    def _finalize_scores(
        self,
        scores: torch.Tensor,
        vnorm: torch.Tensor | None,
        value_l2: bool,
        query_relevance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply value/query refinements and positional weighting to scores.

        Shared by the main scoring path and the Phase-1C calibrator so a layer
        subset's aggregated scores go through exactly the same post-aggregation
        pipeline as the all-layer reference before selection.

        Args:
            scores: ``(count,)`` aggregated redundancy droppability.
            vnorm: ``(count,)`` aggregated value-L2 norm, or ``None``.
            value_l2: Whether the active policy is the value_l2 baseline (which
                skips both re-weights and keeps its raw negated norm).
            query_relevance: ``(count,)`` query-attention mass, or ``None`` when
                query capture did not produce a relevance signal.

        Returns:
            ``(count,)`` finalized droppability scores.
        """
        cfg = self.config
        blend_beta = None if value_l2 else cfg.value_blend_beta
        query_weight = None if value_l2 else cfg.query_alignment_weight
        blend_active = bool(blend_beta and vnorm is not None)
        query_weight_active = bool(query_weight and query_relevance is not None)
        # Weighted refinements share one linear combination of independently
        # standardized raw signals. In particular, do not z-score the result of
        # the value blend again before adding query relevance: that changes the
        # configured value/query ratio according to the blend's request-specific
        # variance. A high value norm or query relevance lowers droppability.
        if blend_active or query_weight_active:
            scores = zscore(scores)
            if blend_active:
                assert vnorm is not None
                assert blend_beta is not None
                scores = scores - float(blend_beta) * zscore(vnorm)
            if query_weight_active:
                assert query_relevance is not None
                assert query_weight is not None
                scores = scores - float(query_weight) * zscore(query_relevance)
        # Weighted mode already incorporated relevance above. As in the prior
        # helper behavior, it takes precedence over tie-breaking.
        if not value_l2 and not query_weight_active:
            scores = refine_with_query_relevance(
                scores,
                query_relevance,
                None if blend_active else query_weight,
                cfg.enable_query_tiebreak,
            )
        # Positional cosh (sech-bump) re-weight: pull v_redundancy eviction
        # toward the middle and away from the ends. Only for the scored thesis
        # signal (value_l2 keeps its raw negated norm); inert when alpha is
        # None/0. Droppability can be negative (cosine), so shift to >= 0 before
        # multiplying by the positive weight -- otherwise the weight would flip
        # the ranking on negative scores. The shift is a monotone per-request
        # affine, so ordering is unchanged when the weights are uniform.
        alpha = cfg.positional_cosh_alpha
        if not value_l2 and alpha and scores.shape[0] >= 2:
            w = positional_cosh_weights(
                scores.shape[0], float(alpha), scores.device, scores.dtype
            )
            shifted = scores - scores.min()
            scores = shifted * w
        return scores

    def _calibrate_layers(
        self,
        reference_mask: torch.Tensor,
        select_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    ) -> None:
        """Score fixed layer subsets and record their agreement with the fire.

        Uses the stacks stashed by :meth:`_score_request_blocks_with_vnorm` on
        this fire to re-aggregate + finalize each candidate subset, applies the
        caller's own selector, then feeds the resulting mask to the calibrator so
        it can accumulate selected-block agreement (Jaccard) vs the all-layer
        ``reference_mask``. Only invoked when the calibrator is enabled.

        Passing the selector as a closure keeps this path selector-agnostic: the
        legacy rate path, the capacity-band path, and the frac path each supply
        the same budget selector they used for the reference, so a subset mask is
        always compared against a like-for-like reference.

        Args:
            reference_mask: The ``(count,)`` all-layer eviction mask.
            select_fn: Maps ``(finalized_scores, vnorm)`` to a ``(count,)`` bool
                eviction mask using the caller's selection policy/budget.
        """
        cfg = self.config
        stacked = self._cal_stacked
        vnorm_stacked = self._cal_vnorm_stacked
        if stacked is None:
            return
        value_l2 = cfg.eviction_policy == "value_l2"

        def aggregate_and_select(rows: list[int]) -> torch.Tensor:
            idx = torch.tensor(rows, dtype=torch.long, device=stacked.device)
            scores = aggregate_layer_scores(
                stacked[idx], cfg.block_score_aggregation, cfg.topk_frac
            )
            vnorm = None
            if vnorm_stacked is not None:
                vnorm = aggregate_layer_scores(
                    vnorm_stacked[idx], cfg.block_score_aggregation, cfg.topk_frac
                )
            scores = self._finalize_scores(scores, vnorm, value_l2)
            return select_fn(scores, vnorm)

        self.layer_calibrator.observe(
            self._cal_row_layer_idxs, reference_mask, aggregate_and_select
        )

    def _select_budget_mask(
        self,
        scores: torch.Tensor,
        count: int,
        budget: int,
        value_norm: torch.Tensor | None,
    ) -> torch.Tensor:
        """Dispatch a budget cut to pairwise ranking or global coverage."""
        cfg = self.config
        if cfg.candidate_expansion_factor is not None:
            relevance = self._query_relevance
            if relevance is None:
                raise RuntimeError("R2R selection missing query relevance")
            fire_stats: dict[str, int] = {}
            mask = select_evicted_r2r(
                scores,
                relevance,
                count,
                budget,
                int(cfg.warmup_pages or 0),
                cfg.candidate_expansion_factor,
                tail_protect_frac=cfg.query_tail_protect_frac,
                cover_cost=self._r2r_cover_cost,
                cover_depth=cfg.r2r_cover_depth,
                selection_stats=fire_stats,
            )
            for key, value in fire_stats.items():
                self._r2r_selection_counts[key] = (
                    self._r2r_selection_counts.get(key, 0) + value
                )
            return mask
        if cfg.redundancy_mode == "coverage":
            similarity = self._coverage_similarity
            if similarity is None:
                raise RuntimeError("coverage selection missing aggregated similarity")
            return select_evicted_by_coverage(
                similarity,
                count,
                budget,
                int(cfg.warmup_pages or 0),
                cfg.query_tail_protect_frac,
            )
        return select_evicted_to_budget(
            scores,
            count,
            budget,
            int(cfg.warmup_pages or 0),
            cfg.eviction_policy,
            cfg.eviction_seed,
            cfg.query_tail_protect_frac,
            value_norm=value_norm,
            value_norm_protect_quantile=cfg.value_norm_protect_quantile,
            query_relevance=self._query_relevance,
            query_relevance_protect_quantile=(
                cfg.query_relevance_protect_quantile
            ),
        )

    def _select_rate_mask(
        self,
        scores: torch.Tensor,
        count: int,
        rate: float,
    ) -> torch.Tensor:
        """Dispatch the legacy matched-rate cut, preserving its exact count."""
        cfg = self.config
        if cfg.redundancy_mode == "coverage":
            lo = max(int(cfg.warmup_pages or 0), 0)
            evictable = max(count - 1 - lo, 0)
            num_evict = min(int(round(rate * evictable)), evictable)
            similarity = self._coverage_similarity
            if similarity is None:
                raise RuntimeError("coverage selection missing aggregated similarity")
            return select_evicted_by_coverage(
                similarity,
                count,
                count - num_evict,
                lo,
            )
        return select_evicted_blocks(
            scores,
            count,
            rate,
            cfg.eviction_policy,
            int(cfg.warmup_pages or 0),
            cfg.eviction_seed,
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
            scores, vnorm = self._score_request_blocks_with_vnorm(
                req_index, prompt_len, block_tables
            )
            if scores is None:
                return []

            def _select_to_budget(
                s: torch.Tensor, vn: torch.Tensor | None
            ) -> torch.Tensor:
                return self._select_budget_mask(s, count, keep, vn)

            with self.profiler.section("selection"):
                mask = _select_to_budget(scores, vnorm)
            if self.layer_calibrator.enabled:
                self._calibrate_layers(mask, _select_to_budget)
            with self.profiler.section("transfer"):
                self.evicted_store[req_index, : scores.shape[0]].copy_(
                    mask.to(self.evicted_store.device)
                )
            if not cfg.physical_reclaim:
                return []
            self._stage_incremental_eviction(req_index, mask)
            return torch.nonzero(mask).flatten().tolist()

        # Legacy coupled paths (rate / decode_evict_budget). Both share the same
        # end-of-prefill scoring over the full prompt.
        scores, vnorm = self._score_request_blocks_with_vnorm(
            req_index, prompt_len, block_tables
        )
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

            def _select(s: torch.Tensor, vn: torch.Tensor | None) -> torch.Tensor:
                return self._select_budget_mask(s, count, low, vn)
        else:

            def _select(s: torch.Tensor, vn: torch.Tensor | None) -> torch.Tensor:
                return self._select_rate_mask(s, count, cfg.eviction_rate or 0.0)

        with self.profiler.section("selection"):
            mask = _select(scores, vnorm)
        if self.layer_calibrator.enabled:
            self._calibrate_layers(mask, _select)
        self.evicted_store[req_index, :count].copy_(mask.to(self.evicted_store.device))
        if not cfg.physical_reclaim:
            return []
        self._stage_incremental_eviction(req_index, mask)
        # The store row was all-False before this write, so every True here is
        # newly evicted. These are logical block indices the scheduler will
        # physically free
        return torch.nonzero(mask).flatten().tolist()

    # -- decode-time eviction (capacity band, physical reclaim) ------------
    def on_decode_step(
        self,
        input_batch: InputBatch,
        block_tables: Any,
        cache_free_fraction: float | None = None,
    ) -> dict[str, list[int]]:
        """Enforce the per-request capacity band during decode.

        The decode analog of :meth:`on_step`: for every request currently in
        decode that has regrown to capacity ``C`` (``decode_evict_budget``),
        score its blocks by V-redundancy and evict the most-redundant interior
        blocks down to the low watermark ``floor(watermark * C)``. Under physical
        reclaim the evicted logical block indices are returned for the scheduler
        to compact and free; the mask hides them for the one-step latency window
        until the compaction lands. A no-op unless ``decode_evict_budget`` is set.

        Args:
            input_batch: The current decode batch.
            block_tables: The model runner's block tables.
            cache_free_fraction: Global KV-pool free fraction for this step
                (scheduler-side signal). Only consulted when
                ``decode_evict_pressure_watermark`` is set, to gate the drip on
                cache pressure. ``None`` is treated as "pool full" (fire), so a
                caller that omits it keeps the unconditional drip behavior.

        Returns:
            Mapping ``{req_id: freed_logical_block_indices}`` for requests
            evicted this step (physical reclaim only; empty under mask-only).
        """
        cfg = self.config
        # Three decode paths, mutually exclusive (enforced in config.validate):
        # fixed-rate drip > per-request frac band > coupled global budget band.
        drip = cfg.decode_evict_blocks_per_step
        frac_decode = cfg.decode_evict_frac is not None
        if drip is None and not frac_decode and cfg.decode_evict_budget is None:
            return {}
        # Pressure gate (drip only): fire only once the global pool is >= watermark
        # full. Validation guarantees the watermark coexists only with the drip, so
        # this never affects the frac / budget paths. A missing signal -> full ->
        # fire (preserves unconditional behavior for callers that omit it).
        if cfg.decode_evict_pressure_watermark is not None:
            free = cache_free_fraction if cache_free_fraction is not None else 0.0
            if (1.0 - free) < cfg.decode_evict_pressure_watermark:
                return {}
        self._decode_step += 1
        if self._decode_step % max(int(cfg.decode_evict_interval), 1) != 0:
            return {}
        # Global band for the legacy budget path only; the drip and frac paths do
        # not use it (capacity_band asserts decode_evict_budget is not None).
        cap_g, low_g = (
            capacity_band(cfg) if (drip is None and not frac_decode) else (0, 0)
        )
        freed: dict[str, list[int]] = {}
        for i in range(input_batch.num_reqs):
            if bool(input_batch.is_prefilling_np[i]):
                continue  # decode requests only
            if input_batch.req_ids[i].startswith(_KERNEL_WARMUP_REQ_PREFIX):
                continue
            req_index = int(input_batch.idx_mapping_np[i])
            # Current total sequence length after this step's tokens (RoPE /
            # uncompacted positions); storage length subtracts evicted tokens.
            seq_len = int(input_batch.num_computed_tokens_np[i]) + int(
                input_batch.num_scheduled_tokens[i]
            )
            if drip is not None:
                # Drip reads no per-request band; it targets N below live count.
                evicted = self._evict_decode_request(
                    req_index, seq_len, block_tables, 0, 0, blocks_per_step=drip
                )
            else:
                if frac_decode:
                    capacity = int(self._decode_cap_np[req_index])
                    low = int(self._decode_low_np[req_index])
                    if capacity < 2:
                        continue  # decode eviction not armed for this request
                else:
                    capacity, low = cap_g, low_g
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
        blocks_per_step: int | None = None,
    ) -> list[int]:
        """Evict one decoding request during decode.

        Two triggers share this body (only the trigger + target budget differ;
        scoring, the physical/mask-only branches, and the in-flight guard are
        identical):

        - Watermark band (``blocks_per_step is None``): fire only once the request
          has regrown to ``capacity`` C, then evict down to the low watermark
          ``low``. This is the hysteresis band.
        - Fixed-rate drip (``blocks_per_step=N``): no capacity gate — evict exactly
          N interior blocks whenever any remain (target budget ``count - N``),
          draining monotonically toward the floor (sink + anchor + tail).

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
        model runner zeroes the row on apply). For the drip this guard is what
        halves the effective cadence to ~N blocks per 2 steps.

        Returns:
            Logical (storage) block indices to physically free, or ``[]`` when
            nothing is evicted or under mask-only mode (which frees nothing).
        """
        cfg = self.config
        physical = cfg.physical_reclaim
        if physical and bool(self.evicted_store[req_index].any()):
            return []  # compaction in flight; wait for the row to clear
        # Read the block count cheaply first to decide whether to score at all.
        count = self._prompt_block_count(req_index, block_tables)
        warmup = int(cfg.warmup_pages or 0)
        if blocks_per_step is not None:
            # Drip: no capacity gate; target N below current. Skip scoring once
            # only sink + anchor remain (select would evict nothing anyway).
            if count <= warmup + 1 or count < 2:
                return []
            budget = max(count - int(blocks_per_step), 0)
        else:
            # Hysteresis: fire only once the request has regrown to capacity C.
            if count < capacity or count < 2:
                return []
            budget = low
        num_evicted = 0
        if self.num_evicted_tokens_np is not None:
            num_evicted = int(self.num_evicted_tokens_np[req_index])
        storage_len = seq_len - num_evicted
        scores, vnorm = self._score_request_blocks_with_vnorm(
            req_index, storage_len, block_tables
        )
        if scores is None:
            return []
        if physical:
            # Clean basis (row cleared before we got here): the selected mask is
            # the full eviction set. Set the mask (hides blocks for the one-step
            # window) and report the indices for compaction.
            with self.profiler.section("selection"):
                mask = self._select_budget_mask(scores, count, budget, vnorm)
            if not bool(mask.any()):
                return []
            self._num_decode_evictions += 1
            self.evicted_store[req_index, :count].copy_(
                mask.to(self.evicted_store.device)
            )
            self._stage_incremental_eviction(req_index, mask)
            return torch.nonzero(mask).flatten().tolist()
        # Mask-only A/B mode (no compaction ever lands to clear the row): never
        # re-keep an already-hidden block. For scored policies (v_redundancy,
        # value_l2), force its score to +inf so scoring re-selects it;
        # recency/random pick by index (oldest-k is a prefix, so previously-hidden
        # oldest blocks stay chosen). The OR-merge then guarantees the store only
        # ever gains True for every policy.
        prev = self.evicted_store[req_index, :count].to("cpu")
        if blocks_per_step is not None:
            # The selector's budget describes the total number of kept blocks,
            # while a mask-only drip promises N *new* hidden blocks per fire.
            # Include the blocks hidden by prior fires in the total eviction
            # target; otherwise +inf re-selection consumes the entire top-N and
            # the drip stalls after its first fire.
            budget = max(
                count - int(blocks_per_step) - int(prev.sum().item()),
                0,
            )
        if bool(prev.any()):
            scores = scores.detach().to(device="cpu", dtype=torch.float32).clone()
            scores[prev] = float("inf")
        # Phase-2 norm protection is intentionally not applied on this mask-only
        # diagnostic path: the +inf forced re-selection of already-hidden blocks
        # relies on the store only ever gaining True and holding the watermark, and
        # protecting a prev-hidden block would fight that monotone invariant. Norm
        # protection is a physical-reclaim feature (prefill + physical decode).
        with self.profiler.section("selection"):
            mask = select_evicted_to_budget(
                scores,
                count,
                budget,
                warmup,
                cfg.eviction_policy,
                cfg.eviction_seed,
                cfg.query_tail_protect_frac,
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
            "(prefill), %d decode-eviction events, R2R reasons=%s",
            self._num_requests_evicted,
            self._num_decode_evictions,
            self._r2r_selection_counts,
        )
        # Flush the Phase-1B scorer profile (no-op unless enable_tracing).
        self.profiler.dump()
        # Flush the Phase-1C layer-subset calibration (no-op unless enabled).
        self.layer_calibrator.dump()
