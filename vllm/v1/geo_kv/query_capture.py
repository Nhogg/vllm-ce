# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in post-RoPE query capture for query-aware GeoKV scoring."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


class QueryCapturer:
    """Capture each request's recent query window on sampled layers.

    The model runner calls :meth:`begin_step` before the forward. Attention-layer
    hooks then call :meth:`capture`; gathered queries land in a fixed-size device
    buffer, avoiding retention of model activations. The buffer is read
    synchronously by eviction scoring immediately after the same forward.

    Scheduler chunks are rolled into a window of at most ``tail_tokens`` until
    an eviction decision resets it. When prefix caching skips part of that
    window, the capturer uses the final suffix that actually executes on this
    runner. Those queries can still attend every cached key, so their QK mass is
    exact; unavailable cached queries are simply not included.
    """

    def __init__(
        self,
        layer_specs: Sequence[tuple[int, int, int, float]],
        max_num_reqs: int,
        tail_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not layer_specs:
            raise ValueError("query capture requires at least one sampled layer")
        if len(layer_specs) > 8:
            raise ValueError("query capture supports at most 8 sampled layers")
        if max_num_reqs < 1 or tail_tokens < 1:
            raise ValueError("max_num_reqs and tail_tokens must be positive")
        heads = {spec[1] for spec in layer_specs}
        dims = {spec[2] for spec in layer_specs}
        if len(heads) != 1 or len(dims) != 1:
            raise ValueError("query capture requires uniform query heads and head dims")

        self.layer_to_row = {
            layer_idx: row for row, (layer_idx, _, _, _) in enumerate(layer_specs)
        }
        self.scales = {layer_idx: scale for layer_idx, _, _, scale in layer_specs}
        self.tail_tokens = int(tail_tokens)
        self.max_num_reqs = int(max_num_reqs)
        self.num_heads = next(iter(heads))
        self.head_dim = next(iter(dims))
        self.buffer = torch.empty(
            len(layer_specs),
            max_num_reqs,
            tail_tokens,
            self.num_heads,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        self._capture_indices = torch.empty(0, dtype=torch.long, device=device)
        # (req_index, gathered_start, gathered_end, old_fill, new_fill)
        self._step_entries: list[tuple[int, int, int, int, int]] = []
        self._request_lengths: dict[int, int] = {}
        self._window_filled = np.zeros(max_num_reqs, dtype=np.int32)
        self._captured_rows: set[int] = set()
        self._stashed: dict[str, tuple[torch.Tensor, int]] = {}

    def reset_request(self, req_index: int) -> None:
        """Clear rolling query state when a request slot is reused."""
        self.reset_window(req_index)

    def reset_window(self, req_index: int) -> None:
        """Start a fresh query window after a permanent eviction decision."""
        self._window_filled[int(req_index)] = 0
        self._request_lengths.pop(int(req_index), None)

    def stash_request(self, req_id: str, req_index: int) -> None:
        """Preserve a rolling tail across scheduler preemption/re-add."""
        idx = int(req_index)
        filled = int(self._window_filled[idx])
        if filled:
            self._stashed[req_id] = (
                self.buffer[:, idx, :filled].clone(),
                filled,
            )

    def restore_request(self, req_id: str, req_index: int) -> bool:
        """Restore stashed rolling state into a newly assigned request slot."""
        saved = self._stashed.pop(req_id, None)
        if saved is None:
            self.reset_request(req_index)
            return False
        values, filled = saved
        idx = int(req_index)
        self.buffer[:, idx, :filled].copy_(values)
        self._window_filled[idx] = filled
        return True

    def drop_request(self, req_id: str) -> None:
        """Discard state for a genuinely finished (not preempted) request."""
        self._stashed.pop(req_id, None)

    def begin_step(
        self,
        req_indices: np.ndarray,
        query_start_loc: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        num_computed_prefill_tokens: np.ndarray,
        prefill_lens: np.ndarray,
        is_prefilling: np.ndarray,
        *,
        dummy: bool = False,
    ) -> None:
        """Prepare packed-token gather indices for the next model forward."""
        self._step_entries.clear()
        self._request_lengths.clear()
        self._captured_rows.clear()
        if dummy:
            self._capture_indices = self._capture_indices[:0]
            return

        source: list[int] = []
        gathered_start = 0
        for i, req_index_raw in enumerate(req_indices):
            scheduled = int(num_scheduled_tokens[i])
            if scheduled <= 0:
                continue
            req_index = int(req_index_raw)
            old_fill = int(self._window_filled[req_index])
            take = min(self.tail_tokens, scheduled)
            new_fill = min(self.tail_tokens, old_fill + take)
            start = int(query_start_loc[i])
            source.extend(range(start + scheduled - take, start + scheduled))
            gathered_end = gathered_start + take
            self._step_entries.append(
                (
                    req_index,
                    gathered_start,
                    gathered_end,
                    old_fill,
                    new_fill,
                )
            )
            self._request_lengths[req_index] = new_fill
            self._window_filled[req_index] = new_fill
            gathered_start = gathered_end

        if gathered_start > self.max_num_reqs * self.tail_tokens:
            raise RuntimeError(
                f"query capture needs {gathered_start} temporary slots, "
                f"limit is {self.max_num_reqs * self.tail_tokens}"
            )
        self._capture_indices = torch.tensor(
            source, dtype=torch.long, device=self.buffer.device
        )

    def capture(self, layer_idx: int, query: torch.Tensor) -> None:
        """Gather the prepared per-request query tails for one sampled layer."""
        row = self.layer_to_row.get(int(layer_idx))
        if row is None or self._capture_indices.numel() == 0:
            return
        q = query.view(-1, self.num_heads, self.head_dim)
        gathered = q.index_select(0, self._capture_indices)
        for req_index, start, end, old_fill, new_fill in self._step_entries:
            values = gathered[start:end]
            take = end - start
            if take >= self.tail_tokens:
                self.buffer[row, req_index].copy_(values[-self.tail_tokens :])
                continue
            keep = min(old_fill, self.tail_tokens - take)
            if keep:
                prior = self.buffer[row, req_index, old_fill - keep : old_fill].clone()
                self.buffer[row, req_index, :keep].copy_(prior)
            self.buffer[row, req_index, keep:new_fill].copy_(values)
        self._captured_rows.add(row)

    def get(self, req_index: int, layer_idx: int) -> torch.Tensor | None:
        """Return ``(W,Hq,D)`` captured queries, or ``None`` if unavailable."""
        row = self.layer_to_row.get(int(layer_idx))
        length = self._request_lengths.get(int(req_index))
        if row is None or length is None or row not in self._captured_rows:
            return None
        return self.buffer[row, int(req_index), :length]

    def scale(self, layer_idx: int) -> float:
        return float(self.scales[layer_idx])
