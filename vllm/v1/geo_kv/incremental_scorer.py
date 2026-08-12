# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generation-safe incremental state for pairwise block redundancy.

The ordinary scorer rebuilds every block anchor and the full ``B x B`` cosine
matrix on each decode eviction fire.  Under physical reclaim, however, the next
row is exactly the previous survivor row plus newly appended tail blocks.  This
module retains *copied* anchors and similarities for those survivors and only
computes rows whose contents changed (normally the formerly-partial tail and the
new tail block).

No tensor here aliases the paged KV cache.  A compaction acknowledgement prunes
the state using the already-selected logical mask and advances its generation;
request-slot reset drops it completely.  Those two lifecycle boundaries prevent
freed or subsequently reused physical slots from being mistaken for live data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IncrementalRedundancyState:
    """Cached anchors and cosine matrix for one request and one layer."""

    generation: int
    valid_lens: tuple[int, ...]
    normalized_anchors: torch.Tensor
    similarity: torch.Tensor

    @classmethod
    def empty(cls, generation: int) -> IncrementalRedundancyState:
        return cls(
            generation=generation,
            valid_lens=(),
            normalized_anchors=torch.empty(0),
            similarity=torch.empty(0),
        )

    @property
    def num_blocks(self) -> int:
        return len(self.valid_lens)

    @property
    def capacity(self) -> int:
        return int(self.normalized_anchors.shape[0])

    def changed_positions(
        self,
        valid_lens: tuple[int, ...],
        generation: int,
    ) -> list[int]:
        """Return rows whose anchors must be read from the live KV cache.

        Normal growth preserves the cached prefix.  A generation mismatch or an
        unexplained row shrink is treated conservatively as a full rebuild.
        """
        count = len(valid_lens)
        if generation != self.generation or count < self.num_blocks:
            return list(range(count))
        changed = [
            i
            for i in range(min(count, self.num_blocks))
            if valid_lens[i] != self.valid_lens[i]
        ]
        changed.extend(range(self.num_blocks, count))
        return changed

    def update(
        self,
        valid_lens: tuple[int, ...],
        generation: int,
        changed_positions: list[int],
        changed_anchors: torch.Tensor,
        changed_positions_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update changed rows and return exact pairwise max-cosine scores."""
        count = len(valid_lens)
        if count < 2:
            raise ValueError("incremental redundancy requires at least two blocks")
        if changed_anchors.shape[0] != len(changed_positions):
            raise ValueError("changed anchor count does not match changed positions")

        device = changed_anchors.device
        can_reuse = generation == self.generation and count >= self.num_blocks
        anchor_shape = changed_anchors.shape[1:]
        if self.capacity < count or not can_reuse:
            capacity = max(count, self.capacity if can_reuse else 0)
            normalized_anchors = torch.empty(
                (capacity, *anchor_shape), dtype=torch.float32, device=device
            )
            similarity = torch.empty(
                (capacity, capacity), dtype=torch.float32, device=device
            )
            if can_reuse and self.num_blocks:
                old_count = self.num_blocks
                normalized_anchors[:old_count].copy_(
                    self.normalized_anchors[:old_count]
                )
                similarity[:old_count, :old_count].copy_(
                    self.similarity[:old_count, :old_count]
                )
        else:
            normalized_anchors = self.normalized_anchors
            similarity = self.similarity
        active_similarity = similarity[:count, :count]
        if changed_positions:
            changed = changed_positions_tensor
            if changed is None:
                changed = torch.tensor(
                    changed_positions, dtype=torch.long, device=device
                )
            changed_normalized = torch.nn.functional.normalize(
                changed_anchors.reshape(len(changed_positions), -1), dim=-1
            ).reshape(len(changed_positions), *anchor_shape)
            normalized_anchors[changed] = changed_normalized
            flat = normalized_anchors[:count].reshape(count, -1)
            changed_similarity = flat[changed] @ flat.t()
            active_similarity[changed, :] = changed_similarity
            active_similarity[:, changed] = changed_similarity.t()
        active_similarity.fill_diagonal_(float("-inf"))

        # These are independent tensors produced by pooling/matmul; retaining
        # them cannot keep a reclaimed paged-KV allocation live.
        self.generation = generation
        self.valid_lens = valid_lens
        self.normalized_anchors = normalized_anchors.detach()
        self.similarity = similarity.detach()
        return active_similarity.max(dim=1).values

    def retain(self, eviction_mask: torch.Tensor, generation: int) -> None:
        """Apply an acknowledged physical compaction to cached logical rows."""
        mask = eviction_mask.to(device="cpu", dtype=torch.bool).flatten()
        if mask.numel() != self.num_blocks:
            # ``eviction_mask`` describes the old row, so its length must match.
            # Drop
            # rather than risk carrying state across a lifecycle desync.
            self.generation = generation
            self.valid_lens = ()
            self.normalized_anchors = torch.empty(0)
            self.similarity = torch.empty(0)
            return
        survivor = torch.nonzero(~mask).flatten()
        survivor_dev = survivor.to(device=self.normalized_anchors.device)
        survivor_count = int(survivor.numel())
        compact_anchors = self.normalized_anchors[: self.num_blocks].index_select(
            0, survivor_dev
        )
        compact_similarity = self.similarity[
            : self.num_blocks, : self.num_blocks
        ].index_select(0, survivor_dev).index_select(1, survivor_dev)
        self.valid_lens = tuple(self.valid_lens[i] for i in survivor.tolist())
        # Keep the existing allocation so ordinary regrowth can update its tail
        # in place. Compaction is outside the scorer timing path, and the storage
        # remains bounded by this request's largest previously scored row.
        self.normalized_anchors[:survivor_count].copy_(compact_anchors)
        self.similarity[:survivor_count, :survivor_count].copy_(compact_similarity)
        self.generation = generation
