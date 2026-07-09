# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output writers for the score_only prefill experiment (Option A).

Writes ``per_head_scores.csv`` (one row per request/layer/kv_head) and
``summary.json`` into the experiment output directory. Rows are flushed
incrementally so partial results survive an abrupt exit.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
from typing import Any

CSV_FIELDS = [
    "run_id",
    "request_id",
    "prompt_id",
    "policy",
    "model",
    "dataset",
    "phase",
    "layer",
    "kv_head",
    "num_blocks_scored",
    "mean_redundancy",
    "max_redundancy",
    "p90_redundancy",
    "std_redundancy",
    "M_total",
    "block_size",
    "physical_eviction_unit",
    "scoring_unit",
]

# Token-level diagnostic schema (no block pooling): num_tokens_scored replaces
# num_blocks_scored; no memory-budget column. mean_* columns are K cosine
# redundancy (kept for back-compat with compare_geo_runs); v_* are the same on
# the V tensors; {key,value}_norm_* are RAW-vector magnitude stats (the signal
# cosine discards).
TOKEN_CSV_FIELDS = [
    "run_id",
    "request_id",
    "prompt_id",
    "policy",
    "model",
    "dataset",
    "phase",
    "layer",
    "kv_head",
    "num_tokens_scored",
    "mean_redundancy",
    "max_redundancy",
    "p90_redundancy",
    "std_redundancy",
    "v_mean_redundancy",
    "v_max_redundancy",
    "v_p90_redundancy",
    "v_std_redundancy",
    "key_norm_mean",
    "key_norm_std",
    "key_norm_p90",
    "key_norm_cov",
    "value_norm_mean",
    "value_norm_std",
    "value_norm_p90",
    "value_norm_cov",
    "block_size",
    "physical_eviction_unit",
    "scoring_unit",
]

# Normalization-variant schema: per (normalization, scoring_unit, layer,
# kv_head) redundancy. Lets one run compare raw vs centered/whitened cosine for
# both block and token units.
NORM_CSV_FIELDS = [
    "run_id",
    "request_id",
    "prompt_id",
    "policy",
    "model",
    "dataset",
    "phase",
    "normalization",
    "scoring_unit",
    "layer",
    "kv_head",
    "num_units_scored",
    "mean_redundancy",
    "max_redundancy",
    "p90_redundancy",
    "std_redundancy",
    "block_size",
    "physical_eviction_unit",
]

# Per-block positional schema: one row per (request, layer, block) with the
# block's joint-V (all-heads) redundancy at that layer. Kept per-layer (not
# averaged) because the V signal is layer-dependent and flips sign by mid-stack,
# so a flat layer average cancels it. Feeds three offline analyses: (A) per-layer
# across-page spread = where V-redundancy discriminates pages, (B) the
# positional-vs-content control, (C) top-X% droppable-set overlap. num_blocks
# lets the analysis compute relative position.
BLOCKPOS_CSV_FIELDS = [
    "run_id",
    "request_id",
    "prompt_id",
    "policy",
    "model",
    "dataset",
    "phase",
    "layer",
    "block_index",
    "num_blocks",
    "joint_v_block_redundancy",
    "block_size",
    "physical_eviction_unit",
]

# Joint all-heads (per-layer) schema: redundancy/norm of the concatenated
# per-token footprint that whole-block eviction actually acts on. One row per
# (request, layer); no kv_head dimension.
JOINT_CSV_FIELDS = [
    "run_id",
    "request_id",
    "prompt_id",
    "policy",
    "model",
    "dataset",
    "phase",
    "layer",
    "num_tokens_scored",
    "joint_redundancy_mean",
    "joint_redundancy_std",
    "joint_redundancy_p90",
    "joint_norm_mean",
    "joint_norm_std",
    "joint_norm_cov",
    "joint_v_redundancy_mean",
    "joint_v_redundancy_std",
    "joint_v_redundancy_p90",
    "joint_v_norm_mean",
    "joint_v_norm_std",
    "joint_v_norm_cov",
    "block_size",
    "physical_eviction_unit",
    "scoring_unit",
]


class PrefillScoreWriter:
    """Incrementally appends score rows to a CSV file."""

    def __init__(
        self,
        output_dir: str,
        filename: str = "per_head_scores.csv",
        fields: list[str] | None = None,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, filename)
        # Persistent handle held for the writer's lifetime (incremental flush),
        # so a context manager doesn't apply here.
        self._fh = open(self.csv_path, "w", newline="")  # noqa: SIM115
        self._writer = csv.DictWriter(self._fh, fieldnames=fields or CSV_FIELDS)
        self._writer.writeheader()
        self._fh.flush()

    def append_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._writer.writerows(rows)
        self._fh.flush()

    def close(self) -> None:
        # Best-effort close on teardown; ignore any error.
        with contextlib.suppress(Exception):
            self._fh.close()


def write_summary(output_dir: str, summary: dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
