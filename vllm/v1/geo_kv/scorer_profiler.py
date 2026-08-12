# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in scorer profiler for the GeoKV eviction path (plan Phase 1B).

Breaks a single eviction "fire" (scoring + selection + transfer) into named
phases and records per-phase wall time plus per-fire block/layer counts, so the
scorer's cost can be attributed to block-table gather, anchor/norm computation,
similarity, layer aggregation, selection, and host transfer.

Disabled by default. When disabled, :meth:`ScorerProfiler.section` yields a
shared null context and adds **no** timing and **no** device synchronization to
the eviction path -- normal runs are byte-for-byte unaffected. When enabled it
synchronizes the CUDA device around each timed section (accurate GPU timing is
impossible without it), so it is a diagnostic mode, not something to leave on in
production.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch

logger = init_logger(__name__)

# Canonical phase names, in the pipeline order the plan (Phase 1B) asks to
# measure. "compaction" is scheduler-owned (outside EvictionPolicy) and is left
# for the caller to record if it ever wires it; it is listed here only so the
# dumped report keeps a stable, documented column order.
PHASES = (
    "gather",  # block-table gather of the KV rows for a layer
    "anchor_norm",  # block anchors + value-L2 norm
    "query_attention",  # exact causal QK mass for query-aware scoring
    "similarity",  # joint-redundancy (pairwise / greedy)
    "aggregate",  # cross-layer score aggregation
    "selection",  # _choose_evicted / select_evicted_*
    "transfer",  # host<->device copies (mask write, prev read)
    "compaction",  # scheduler compaction (recorded by the caller, if at all)
)


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Return the ``q`` percentile (0..100) of an already-sorted list.

    Uses nearest-rank so there is no numpy dependency and no interpolation
    surprises on tiny sample counts.
    """
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = max(
        0, min(len(sorted_vals) - 1, round((q / 100.0) * (len(sorted_vals) - 1)))
    )
    return sorted_vals[rank]


class ScorerProfiler:
    """Accumulates per-phase timings for the eviction scorer.

    Args:
        enabled: Master switch. When False every method is a cheap no-op and
            :meth:`section` never touches the CUDA device.
        device: The scorer's device. CUDA devices are synchronized around each
            timed section; non-CUDA devices are timed with the wall clock only.
        log_path: Optional path; :meth:`dump` writes the JSON report next to it
            (``<log_path>.scorer_profile.json``). When None, :meth:`dump` only
            logs a summary line.
    """

    def __init__(
        self,
        enabled: bool,
        device: torch.device | None = None,
        log_path: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.log_path = log_path
        self._is_cuda = bool(device is not None and device.type == "cuda")
        self._device = device
        # Per-phase list of elapsed seconds; one entry per timed invocation.
        self._samples: dict[str, list[float]] = defaultdict(list)
        # Per-fire (blocks_scored, layers_scored) so cost can be normalized.
        self._fires: list[tuple[int, int]] = []
        # One observed prompt/decode query-window length per scoring fire.
        self._query_windows: list[int] = []
        # One (total rows, recomputed rows) sample per incremental layer update.
        self._incremental_updates: list[tuple[int, int]] = []
        # Per-fire scorer-only detail. Selection/transfer happen after
        # record_fire(), so this intentionally covers gather through aggregate.
        self._scoring_fires: list[dict] = []
        self._phase_offsets = {name: 0 for name in PHASES}
        self._incremental_offset = 0
        # A single reusable null context so the disabled path allocates nothing.
        self._null_ctx = contextlib.nullcontext()

    def section(self, name: str):
        """Time a named phase. Returns a context manager.

        When disabled, returns a shared no-op context (no sync, no allocation).
        When enabled, synchronizes CUDA before and after the block so the elapsed
        time reflects completed device work.
        """
        if not self.enabled:
            return self._null_ctx
        return self._timed(name)

    @contextlib.contextmanager
    def _timed(self, name: str):
        if self._is_cuda:
            import torch

            torch.cuda.synchronize(self._device)
            t0 = time.perf_counter()
            try:
                yield
            finally:
                torch.cuda.synchronize(self._device)
                self._samples[name].append(time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._samples[name].append(time.perf_counter() - t0)

    def record_fire(self, num_blocks: int, num_layers: int) -> None:
        """Record that one eviction fire scored ``num_blocks`` over ``num_layers``."""
        if not self.enabled:
            return
        self._fires.append((int(num_blocks), int(num_layers)))
        phase_totals: dict[str, float] = {}
        for name in (
            "gather",
            "anchor_norm",
            "query_attention",
            "similarity",
            "aggregate",
        ):
            values = self._samples.get(name, [])
            start = self._phase_offsets[name]
            if len(values) > start:
                phase_totals[name] = sum(values[start:])
            self._phase_offsets[name] = len(values)
        updates = self._incremental_updates[self._incremental_offset :]
        self._incremental_offset = len(self._incremental_updates)
        detail: dict = {
            "num_blocks": int(num_blocks),
            "num_layers": int(num_layers),
            "phases_s": phase_totals,
        }
        if updates:
            detail["incremental_total_rows"] = sum(total for total, _ in updates)
            detail["incremental_recomputed_rows"] = sum(
                recomputed for _, recomputed in updates
            )
        self._scoring_fires.append(detail)

    def record_query_window(self, num_queries: int) -> None:
        """Record the observed query suffix length used by one scoring fire."""
        if not self.enabled:
            return
        self._query_windows.append(int(num_queries))

    def record_incremental_update(
        self, total_blocks: int, recomputed_blocks: int
    ) -> None:
        """Record how many block rows an incremental layer update rebuilt."""
        if not self.enabled:
            return
        self._incremental_updates.append(
            (int(total_blocks), int(recomputed_blocks))
        )

    def summary(self) -> dict:
        """Compute the aggregate report (median/p95/total per phase)."""
        out: dict = {"fires": len(self._fires), "phases": {}}
        for name in PHASES:
            vals = self._samples.get(name)
            if not vals:
                continue
            sv = sorted(vals)
            total = sum(sv)
            out["phases"][name] = {
                "count": len(sv),
                "total_s": total,
                "median_ms": _percentile(sv, 50) * 1e3,
                "p95_ms": _percentile(sv, 95) * 1e3,
                "mean_ms": (total / len(sv)) * 1e3,
            }
        if self._fires:
            blocks = sorted(b for b, _ in self._fires)
            layers = sorted(ll for _, ll in self._fires)
            out["blocks_per_fire"] = {
                "median": _percentile([float(b) for b in blocks], 50),
                "p95": _percentile([float(b) for b in blocks], 95),
                "total": sum(blocks),
            }
            out["layers_per_fire"] = {
                "median": _percentile([float(ll) for ll in layers], 50),
            }
        if self._query_windows:
            windows = sorted(float(value) for value in self._query_windows)
            out["queries_per_fire"] = {
                "median": _percentile(windows, 50),
                "p95": _percentile(windows, 95),
                "min": windows[0],
                "max": windows[-1],
            }
        if self._incremental_updates:
            totals = [total for total, _ in self._incremental_updates]
            recomputed = sorted(
                float(changed) for _, changed in self._incremental_updates
            )
            total_rows = sum(totals)
            changed_rows = int(sum(recomputed))
            out["incremental_updates"] = {
                "count": len(self._incremental_updates),
                "total_block_rows": total_rows,
                "recomputed_block_rows": changed_rows,
                "reused_fraction": (
                    1.0 - changed_rows / total_rows if total_rows else 0.0
                ),
                "recomputed_median": _percentile(recomputed, 50),
                "recomputed_p95": _percentile(recomputed, 95),
            }
        if self._scoring_fires:
            out["scoring_fires"] = self._scoring_fires
        return out

    def dump(self) -> None:
        """Log the summary and, if ``log_path`` is set, write the JSON report."""
        if not self.enabled or not self._fires:
            return
        report = self.summary()
        logger.info("[geo_kv] scorer profile: %s", json.dumps(report))
        if self.log_path:
            path = f"{self.log_path}.scorer_profile.json"
            try:
                with open(path, "w") as fh:
                    json.dump(report, fh, indent=2)
                logger.info("[geo_kv] scorer profile written to %s", path)
            except OSError as e:  # never let profiling break a run
                logger.warning("[geo_kv] could not write scorer profile: %s", e)
