# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-subset calibration for the GeoKV scorer (plan Phase 1C).

Measures how well a small fixed subset of attention layers reproduces the
*selected-block set* of full all-layer scoring. One all-layer scoring pass yields
the agreement for every candidate subset at once, so the expensive part (a full
eval per subset) is only needed for the winners.

Enabled only when ``score_sampled_layers == "all"`` (there must be all layers to
sub-select from) and ``calibrate_layer_subsets`` is set. Disabled by default; the
:meth:`LayerCalibrator.observe` call is a cheap no-op otherwise and never runs on
the normal serving path.

The downstream task metric and per-subset scorer latency are obtained separately
by running arms with ``score_sampled_layers`` set to a chosen subset -- this
module only supplies the cheap agreement signal that says which subsets are worth
running.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch

logger = init_logger(__name__)


def default_layer_subset_candidates(num_layers: int) -> dict[str, list[int]]:
    """Canonical fixed subsets of size 1, 2, 4 spanning early/mid/late layers.

    Positions are chosen as fractions of depth so the same policy is architecture
    agnostic (the plan requires a fixed subset per architecture, not per task).
    Returns an empty dict when there are too few layers to form distinct subsets.

    Args:
        num_layers: Total attention layers available to score.

    Returns:
        Mapping of subset name -> sorted unique layer indices in ``[0, L)``.
    """
    if num_layers < 2:
        return {}

    def at(frac: float) -> int:
        return max(0, min(num_layers - 1, round(frac * (num_layers - 1))))

    raw: dict[str, list[int]] = {
        "early1": [at(0.1)],
        "mid1": [at(0.5)],
        "late1": [at(0.9)],
        "endpoints2": [at(0.1), at(0.9)],
        "mid2": [at(0.4), at(0.6)],
        "spread4": [at(0.1), at(0.37), at(0.63), at(0.9)],
        "even4": [at(0.0), at(0.33), at(0.66), at(0.99)],
    }
    # De-duplicate indices within each subset and drop degenerate ones (a subset
    # that collapsed to fewer distinct layers than intended is still valid, but a
    # single-layer collapse of a multi-layer candidate is dropped to avoid dupes).
    out: dict[str, list[int]] = {}
    for name, idxs in raw.items():
        uniq = sorted(set(i for i in idxs if 0 <= i < num_layers))
        if uniq:
            out[name] = uniq
    return out


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Jaccard overlap of two boolean eviction masks (1.0 when both empty)."""
    inter = int((a & b).sum())
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    return inter / union


class LayerCalibrator:
    """Accumulates selected-block agreement of layer subsets vs all-layer scoring.

    Args:
        enabled: Master switch. When False, :meth:`observe` is a no-op.
        num_layers: Total attention layers (used to build default candidates).
        log_path: Optional path; :meth:`dump` writes
            ``<log_path>.layer_calibration.json`` when set.
        candidates: Optional explicit ``{name: [layer_idx, ...]}``. Defaults to
            :func:`default_layer_subset_candidates`.
    """

    def __init__(
        self,
        enabled: bool,
        num_layers: int,
        log_path: str | None = None,
        candidates: dict[str, list[int]] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.log_path = log_path
        self.num_layers = int(num_layers)
        self.candidates = (
            candidates
            if candidates is not None
            else default_layer_subset_candidates(num_layers)
        )
        # subset name -> (sum_jaccard, count)
        self._agree: dict[str, list[float]] = {n: [0.0, 0.0] for n in self.candidates}
        self._fires = 0

    def observe(
        self,
        row_layer_idxs: list[int],
        reference_mask: torch.Tensor,
        aggregate_and_select: Callable[[list[int]], torch.Tensor],
    ) -> None:
        """Record one fire's subset agreement against the all-layer selection.

        Args:
            row_layer_idxs: Global attention-layer index for each scored row of
                this fire (i.e. ``row_layer_idxs[r]`` is the layer that produced
                stacked row ``r``). Must be all layers for a meaningful result.
            reference_mask: ``(num_blocks,)`` bool eviction mask chosen by the
                full all-layer aggregation (the scorer's real decision).
            aggregate_and_select: Maps a list of stacked row indices to the
                ``(num_blocks,)`` bool eviction mask that scoring only those rows
                would produce, applying the identical layer aggregation, blend,
                re-weight, windowing, budget, and policy as the reference. Owned
                by the scorer so all selection policy stays there and the subset
                masks are directly comparable.
        """
        if not self.enabled or not self.candidates:
            return
        # Map global layer idx -> stacked row so a subset can pull its rows.
        row_of = {gi: r for r, gi in enumerate(row_layer_idxs)}
        self._fires += 1
        for name, layer_idxs in self.candidates.items():
            rows = [row_of[g] for g in layer_idxs if g in row_of]
            if not rows:
                continue
            sub_mask = aggregate_and_select(rows)
            acc = self._agree[name]
            acc[0] += _jaccard(reference_mask, sub_mask)
            acc[1] += 1.0

    def summary(self) -> dict:
        """Mean selected-block agreement per candidate subset, best first."""
        rows = []
        for name, (s, c) in self._agree.items():
            if c <= 0:
                continue
            rows.append(
                {
                    "subset": name,
                    "layers": self.candidates[name],
                    "size": len(self.candidates[name]),
                    "mean_jaccard": s / c,
                    "fires": int(c),
                }
            )
        rows.sort(key=lambda r: r["mean_jaccard"], reverse=True)
        return {"fires": self._fires, "num_layers": self.num_layers, "subsets": rows}

    def dump(self) -> None:
        """Log the ranking and, when ``log_path`` is set, write the JSON report."""
        if not self.enabled or self._fires == 0:
            return
        report = self.summary()
        logger.info("[geo_kv] layer calibration: %s", json.dumps(report))
        if self.log_path:
            path = f"{self.log_path}.layer_calibration.json"
            try:
                with open(path, "w") as fh:
                    json.dump(report, fh, indent=2)
                logger.info("[geo_kv] layer calibration written to %s", path)
            except OSError as e:
                logger.warning("[geo_kv] could not write layer calibration: %s", e)
