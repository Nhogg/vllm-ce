"""PagedEviction for active KV-cache block pruning.

Adapted from :
    arXiv:2509.04377 - PagedEviction: Structured Block-wise KV Cache
    Pruning for Efficient Large Language Model Inference

Paper behavior:
    token_score = ||V_i||_2 / (||K_i||_2 + eps)
    block_score = mean(token_score for tokens in block)
    evict the lowest-scoring full page/block when over budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PagedEvictionDecision:
    request_id: str
    victim_logical_block_idx: int | None
    victim_score: float | None
    num_blocks_before: int
    budget_blocks: int

    @property
    def should_evict(self) -> bool:
        return self.victim_logical_block_idx is not None


@dataclass
class PagedEvictionState:
    """State owned by the KV cache manager.

    Scores are keyed by request_id then logical block index.

    Key by logical block index because PagedEviction removes blocks from a
    request's active block table. Physical block ids can be reused after
    being freed.
    """

    block_scores: dict[str, dict[int, float]] = field(default_factory=dict)
    num_evictions: dict[str, int] = field(default_factory=dict)
    prefill_evicted_request_ids: set[str] = field(default_factory=set)

    def reset_request(self, request_id: str) -> None:
        self.block_scores.pop(request_id, None)
        self.num_evictions.pop(request_id, None)
        self.prefill_evicted_request_ids.discard(request_id)

    def reset_all(self) -> None:
        self.block_scores.clear()
        self.num_evictions.clear()
        self.prefill_evicted_request_ids.clear()

    def set_request_block_scores(
        self,
        request_id: str,
        scores_by_logical_block_idx: dict[int, float],
    ) -> None:
        self.block_scores[request_id] = dict(scores_by_logical_block_idx)

    def update_request_block_score(
        self,
        request_id: str,
        logical_block_idx: int,
        score: float,
    ) -> None:
        self.block_scores.setdefault(request_id, {})[logical_block_idx] = score

    def get_score(
        self,
        request_id: str,
        logical_block_idx: int,
        default: float = 0.0,
    ) -> float:
        return self.block_scores.get(request_id, {}).get(logical_block_idx, default)

    def record_eviction(self, request_id: str) -> None:
        self.num_evictions[request_id] = self.num_evictions.get(request_id, 0) + 1

    def record_prefill_eviction(self, request_id: str) -> None:
        self.prefill_evicted_request_ids.add(request_id)

    def has_prefill_eviction(self, request_id: str) -> bool:
        return request_id in self.prefill_evicted_request_ids

    def remove_request_block_score(
        self,
        request_id: str,
        logical_block_idx: int,
    ) -> None:
        scores = self.block_scores.get(request_id)
        if scores is not None:
            scores.pop(logical_block_idx, None)


class PagedEvictionScorer:
    """PyTorch scorer for paper's K/V norm method"""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def token_scores(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Return one score per token.

        Expected shape:
            k, v: [num_tokens, ...]
        """
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {k.shape=} {v.shape=}")

        k_flat = k.float().reshape(k.shape[0], -1)
        v_flat = v.float().reshape(v.shape[0], -1)

        k_norm = torch.linalg.vector_norm(k_flat, dim=-1)
        v_norm = torch.linalg.vector_norm(v_flat, dim=-1)

        return v_norm / (k_norm + self.eps)

    def block_scores(
        self, k_blocks: torch.Tensor, v_blocks: torch.Tensor
    ) -> torch.Tensor:
        """Return one score per KV block.

        Expected shape:
            k_blocks, v_blocks: [num_blocks, block_size, ...]
        """
        if k_blocks.shape != v_blocks.shape:
            raise ValueError(
                f"K/V block shape mismatch: {k_blocks.shape=} {v_blocks.shape=}"
            )

        if k_blocks.ndim < 3:
            raise ValueError(
                "Expected [num_blocks, block_size, ...] K/V tensors, got "
                f"{k_blocks.shape=}"
            )

        num_blocks, block_size = k_blocks.shape[:2]

        k_flat = k_blocks.float().reshape(num_blocks, block_size, -1)
        v_flat = v_blocks.float().reshape(num_blocks, block_size, -1)

        k_norm = torch.linalg.vector_norm(k_flat, dim=-1)
        v_norm = torch.linalg.vector_norm(v_flat, dim=-1)

        token_scores = v_norm / (k_norm + self.eps)

        return token_scores.mean(dim=1)

    def scores_by_logical_blocks(
        self,
        k_blocks: torch.Tensor,
        v_blocks: torch.Tensor,
        logical_block_indices: list[int],
    ) -> dict[int, float]:
        scores = self.block_scores(k_blocks, v_blocks)
        if scores.numel() != len(logical_block_indices):
            raise ValueError(
                f"Got {scores.numel()} scores for "
                f"{len(logical_block_indices)} logical block indices."
            )

        return {
            logical_idx: float(score.item())
            for logical_idx, score in zip(logical_block_indices, scores)
        }


class PagedEvictionPolicy:
    """Select the lowest-scoring active logical block."""

    def __init__(
        self,
        cache_budget_tokens: int,
        block_size: int,
    ) -> None:
        if cache_budget_tokens <= 0:
            raise ValueError("cache_budget_tokens must be positive")
        if cache_budget_tokens < block_size:
            raise ValueError("cache_budget_tokens must be at least one block")
        self.cache_budget_tokens = cache_budget_tokens
        self.block_size = block_size
        self.budget_blocks = cache_budget_tokens // block_size

    def should_run_decode_eviction(self, num_computed_tokens: int) -> bool:
        """Run only when the newest page/block becomes full."""
        return num_computed_tokens > 0 and num_computed_tokens % self.block_size == 0

    def select_victim_logical_block(
        self,
        *,
        request_id: str,
        logical_block_indices: list[int],
        scores_by_logical_block_idx: dict[int, float],
        num_computed_tokens: int,
        num_blocks_before: int | None = None,
    ) -> PagedEvictionDecision:
        blocks_before = (
            len(logical_block_indices)
            if num_blocks_before is None
            else num_blocks_before
        )
        if not self.should_run_decode_eviction(num_computed_tokens):
            return PagedEvictionDecision(
                request_id=request_id,
                victim_logical_block_idx=None,
                victim_score=None,
                num_blocks_before=blocks_before,
                budget_blocks=self.budget_blocks,
            )

        if blocks_before <= self.budget_blocks:
            return PagedEvictionDecision(
                request_id=request_id,
                victim_logical_block_idx=None,
                victim_score=None,
                num_blocks_before=blocks_before,
                budget_blocks=self.budget_blocks,
            )

        if not logical_block_indices:
            return PagedEvictionDecision(
                request_id=request_id,
                victim_logical_block_idx=None,
                victim_score=None,
                num_blocks_before=blocks_before,
                budget_blocks=self.budget_blocks,
            )

        # Evict the lowest-scoring page/block. Missing scores must not look
        # maximally evictable: score collection can legitimately skip blocks
        # whose KV layout is unsupported by the scorer, and treating those as
        # zero repeatedly selects arbitrary pages instead of the paper's
        # K/V-norm victim.
        victim_idx = min(
            logical_block_indices,
            key=lambda idx: scores_by_logical_block_idx.get(idx, float("inf")),
        )
        victim_score = scores_by_logical_block_idx.get(victim_idx)
        if victim_score is None:
            return PagedEvictionDecision(
                request_id=request_id,
                victim_logical_block_idx=None,
                victim_score=None,
                num_blocks_before=blocks_before,
                budget_blocks=self.budget_blocks,
            )

        return PagedEvictionDecision(
            request_id=request_id,
            victim_logical_block_idx=victim_idx,
            victim_score=victim_score,
            num_blocks_before=blocks_before,
            budget_blocks=self.budget_blocks,
        )
