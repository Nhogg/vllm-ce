# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-of-prefill block-eviction policy (Option A, mask-only / Milestone 1).

Owned by the V2 GPU model runner and constructed only when an ``active_eviction``
experiment mode is selected. Mirrors :class:`PrefillScorer`: for each request
that finishes its prefill on a step, it reads the request's V vectors from the
paged KV cache, scores every whole block by joint (all-heads) V redundancy
aggregated across a layer band, decides which blocks to drop, and writes a
per-request boolean eviction mask into a runner-owned store. The FlexAttention
backend reads that store and hides the flagged *logical* blocks from attention.

This module never frees or mutates the KV cache (Milestone 1 is mask-only); it
is the faithful in-engine analog of the HF reference screen, which likewise only
masked evicted positions. Physical reclamation is a later milestone.

IMPORTANT (Option A): the physical eviction unit in vLLM is a whole block across
all heads/layers. layer/kv_head are *scoring* dimensions only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    if policy == "v_redundancy":
        local = scores.detach().to(device="cpu", dtype=torch.float32)[lo:hi]
        order = torch.argsort(local, descending=True)
        chosen = evictable_t[order[:k]]
    elif policy == "recency":
        chosen = evictable_t[:k]  # oldest k blocks (StreamingLLM contrast)
    elif policy == "random":
        gen = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(n, generator=gen)
        chosen = evictable_t[perm[:k]]
    else:
        raise ValueError(f"unknown eviction policy: {policy!r}")

    evict[chosen] = True
    return evict


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
    """Writes per-request block-eviction masks at end-of-prefill (mask-only)."""

    def __init__(
        self,
        config: GeoKVConfig,
        kv_caches_by_layer: dict[str, torch.Tensor],
        kv_cache_groups: list[Any],
        evicted_store: torch.Tensor,
        block_size: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.kv = kv_caches_by_layer
        self.evicted_store = evicted_store
        self.block_size = block_size
        self.device = device

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
        logger.info(
            "[geo_kv] EvictionPolicy ready: %d attn layers, band=%s, "
            "policy=%s, rate=%s, agg=%s, warmup_pages=%s",
            self.num_layers,
            config.score_sampled_layers,
            config.eviction_policy,
            config.eviction_rate,
            config.block_score_aggregation,
            config.warmup_pages,
        )

    # -- per-step entry point ----------------------------------------------
    def on_step(self, input_batch: InputBatch, block_tables: Any) -> None:
        """Decide eviction for any request whose prefill completed this step."""
        finished = self._finished_prefill_requests(input_batch)
        for _batch_i, req_index, prompt_len in finished:
            self._evict_request(req_index, prompt_len, block_tables)
        self._num_requests_evicted += len(finished)

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
    def _evict_request(
        self, req_index: int, prompt_len: int, block_tables: Any
    ) -> None:
        cfg = self.config
        per_layer: list[torch.Tensor] = []
        count = 0
        for layer_idx, layer_name, group_id in self.layers:
            if layer_idx not in self.sampled_layer_idxs:
                continue
            c = int(block_tables.num_blocks.np[group_id, req_index])
            if c < 2:
                continue
            count = c
            block_ids = (
                block_tables.block_tables[group_id].gpu[req_index, :c].to(torch.long)
            )
            v = self.kv[layer_name][block_ids, 1]  # (c, block_size, H, D)
            valid = block_valid_lens(prompt_len, c, v.shape[1], v.device)
            per_layer.append(block_joint_redundancy(block_anchors(v, valid)))
        if count < 2 or not per_layer:
            return

        stacked = torch.stack(per_layer, dim=0)  # (num_sampled, count)
        scores = aggregate_layer_scores(
            stacked, cfg.block_score_aggregation, cfg.topk_frac
        )
        mask = select_evicted_blocks(
            scores,
            count,
            cfg.eviction_rate or 0.0,
            cfg.eviction_policy,
            int(cfg.warmup_pages or 0),
            cfg.eviction_seed,
        )
        self.evicted_store[req_index, :count].copy_(mask.to(self.evicted_store.device))

    def close(self) -> None:
        logger.info(
            "[geo_kv] EvictionPolicy closed: evicted %d requests",
            self._num_requests_evicted,
        )
