# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-of-prefill geometric scorer (Option A, score_only mode).

Owned by the V2 GPU model runner. For each request that finishes its prefill on
a given step, it enumerates the request's retained physical blocks, reads the K
vectors from the paged KV cache, and computes per-(layer, KV head) geometric
redundancy statistics (see ``scoring.py``). Results are written to
``per_head_scores.csv`` + ``summary.json``.

IMPORTANT (Option A): the physical eviction unit in vLLM is a whole block across
all heads/layers. layer/kv_head here are *scoring* dimensions only. This module
never evicts or mutates the KV cache; it is read-only and gated behind
``experiment_mode=score_only``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.v1.geo_kv.config import GeoKVConfig, resolve_indices
from vllm.v1.geo_kv.scoring import (
    block_anchors,
    block_joint_redundancy,
    block_valid_lens,
    compute_prefill_head_scores,
    compute_redundancy_variants,
    compute_token_diagnostics,
)
from vllm.v1.geo_kv.writers import (
    BLOCKPOS_CSV_FIELDS,
    CSV_FIELDS,
    JOINT_CSV_FIELDS,
    NORM_CSV_FIELDS,
    TOKEN_CSV_FIELDS,
    PrefillScoreWriter,
    write_summary,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

PHYSICAL_EVICTION_UNIT = "whole_vllm_block"
SCORING_UNIT = "layer_kv_head"

# Synthetic request-id prefixes used by vLLM's kernel warmup (warmup_kernels,
# req_id="_warmup_{i}_") and dummy runs (InputBatch.make_dummy, "req_{i}_...").
# These are not real serving requests and must never be scored.
_SYNTHETIC_REQ_PREFIXES = ("_warmup_", "req_")
SUMMARY_NOTE = (
    "vLLM does not support physical per-head page eviction; these are geometric "
    "score distributions, not physical per-head cache sizes."
)

# Row-bucket name -> (csv filename, field list). Writers are created lazily the
# first time a bucket produces rows, so disabled diagnostics leave no files.
_WRITER_SPECS: dict[str, tuple[str, list[str]]] = {
    "block": ("per_head_scores.csv", CSV_FIELDS),
    "token": ("token_level_scores.csv", TOKEN_CSV_FIELDS),
    "joint": ("joint_layer_scores.csv", JOINT_CSV_FIELDS),
    "norm": ("normalization_scores.csv", NORM_CSV_FIELDS),
    "blockpos": ("block_positional_scores.csv", BLOCKPOS_CSV_FIELDS),
}


class PrefillScorer:
    """Reads K after prefill and writes per-head redundancy statistics."""

    def __init__(
        self,
        config: GeoKVConfig,
        kv_caches_by_layer: dict[str, torch.Tensor],
        kv_cache_groups: list[Any],
        model_name: str,
        device: torch.device,
    ) -> None:
        self.config = config
        self.kv = kv_caches_by_layer
        self.device = device
        self.model_name = os.path.basename(model_name.rstrip("/")) or model_name

        # Ordered attention-layer metadata: (layer_idx, layer_name, group_id).
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

        self.output_dir = config.output_dir or os.path.join(
            "results", "geo_prefill_distribution", config.run_id or "run"
        )
        self._writers: dict[str, PrefillScoreWriter] = {}
        self._num_requests_scored = 0
        self._block_size: int | None = None
        logger.info(
            "[geo_kv] PrefillScorer ready: %d attn layers, sampled_layers=%s, "
            "kv_heads=%s, output_dir=%s",
            self.num_layers,
            config.score_sampled_layers,
            config.score_sampled_kv_heads,
            self.output_dir,
        )

    # -- per-step entry point ----------------------------------------------
    def on_step(self, input_batch: InputBatch, block_tables: Any) -> None:
        """Score any request whose prefill completed on this step."""
        finished = self._finished_prefill_requests(input_batch)
        if not finished:
            return

        buckets: dict[str, list[dict[str, Any]]] = {k: [] for k in _WRITER_SPECS}
        for batch_i, req_index, prompt_len in finished:
            req_id = input_batch.req_ids[batch_i]
            out = self._score_request(req_id, req_index, prompt_len, block_tables)
            for name, rows in out.items():
                buckets[name].extend(rows)

        for name, rows in buckets.items():
            self._flush(name, rows)
        self._num_requests_scored += len(finished)
        self._write_summary()

    def _flush(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        writer = self._writers.get(name)
        if writer is None:
            filename, fields = _WRITER_SPECS[name]
            writer = PrefillScoreWriter(self.output_dir, filename, fields)
            self._writers[name] = writer
        writer.append_rows(rows)

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

    # -- per-request scoring -----------------------------------------------
    def _score_request(
        self,
        req_id: str,
        req_index: int,
        prompt_len: int,
        block_tables: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {k: [] for k in _WRITER_SPECS}
        cfg = self.config
        modes = cfg.resolved_norm_variants()
        for layer_idx, layer_name, group_id in self.layers:
            if layer_idx not in self.sampled_layer_idxs:
                continue
            count = int(block_tables.num_blocks.np[group_id, req_index])
            if count < 2:
                continue
            block_ids = (
                block_tables.block_tables[group_id]
                .gpu[req_index, :count]
                .to(torch.long)
            )
            kv = self.kv[layer_name]  # (num_blocks, 2, block_size, H, D)
            k = kv[block_ids, 0]  # (count, block_size, H, D)
            block_size = k.shape[1]
            num_heads = k.shape[2]
            d = k.shape[3]
            if self._block_size is None:
                self._block_size = block_size
            heads = resolve_indices(cfg.score_sampled_kv_heads, num_heads)

            valid = block_valid_lens(prompt_len, count, block_size, k.device)
            stats = compute_prefill_head_scores(k, valid)
            if stats is not None:
                for h in heads:
                    out["block"].append(
                        self._row(req_id, layer_idx, h, stats, block_size)
                    )

            k_tok = None
            if cfg.score_token_level or cfg.score_norm_variants:
                k_tok = k.reshape(count * block_size, num_heads, d)[:prompt_len]

            if cfg.score_token_level:
                v = kv[block_ids, 1]  # (count, block_size, H, D)
                v_tok = v.reshape(count * block_size, num_heads, d)[:prompt_len]
                diag = compute_token_diagnostics(
                    k_tok, v_tok, sample_cap=cfg.token_sample_cap
                )
                if diag is not None:
                    for h in heads:
                        out["token"].append(
                            self._token_row(req_id, layer_idx, h, diag, block_size)
                        )
                    out["joint"].append(
                        self._joint_row(req_id, layer_idx, diag, block_size)
                    )

            if cfg.score_block_positional:
                # Per-(block) joint-V redundancy from this layer's V block
                # anchors, emitted per-layer (not averaged) so the offline
                # analysis can pick the layer band / measure across-page spread.
                v_bp = kv[block_ids, 1]  # (count, block_size, H, D)
                br = block_joint_redundancy(block_anchors(v_bp, valid))  # (count,)
                br_cpu = br.detach().to(torch.float32).cpu()
                for bi in range(count):
                    out["blockpos"].append(
                        self._blockpos_row(
                            req_id, layer_idx, bi, count,
                            float(br_cpu[bi]), block_size,
                        )
                    )

            if cfg.score_norm_variants:
                # Same cosine redundancy under raw/center/whiten, for block
                # anchors and per-token K -> normalization_scores.csv.
                anchors = block_anchors(k, valid)
                block_var = compute_redundancy_variants(anchors, modes)
                token_var = compute_redundancy_variants(
                    k_tok, modes, sample_cap=cfg.token_sample_cap
                )
                for unit, var in (("block", block_var), ("token", token_var)):
                    if var is None:
                        continue
                    for mode in modes:
                        for h in heads:
                            out["norm"].append(
                                self._norm_row(
                                    req_id, layer_idx, h, mode, unit, var, block_size
                                )
                            )

        return out

    def _row(
        self,
        req_id: str,
        layer_idx: int,
        head: int,
        stats: dict[str, Any],
        block_size: int,
    ) -> dict[str, Any]:
        return {
            "run_id": self.config.run_id or "",
            "request_id": req_id,
            "prompt_id": req_id,
            "policy": self.config.experiment_mode,
            "model": self.model_name,
            "dataset": self.config.dataset or "unknown",
            "phase": "prefill",
            "layer": layer_idx,
            "kv_head": head,
            "num_blocks_scored": int(stats["num_blocks"]),
            "mean_redundancy": float(stats["mean"][head]),
            "max_redundancy": float(stats["max"][head]),
            "p90_redundancy": float(stats["p90"][head]),
            "std_redundancy": float(stats["std"][head]),
            "M_total": self.config.total_page_cap
            if self.config.total_page_cap is not None
            else "",
            "block_size": block_size,
            "physical_eviction_unit": PHYSICAL_EVICTION_UNIT,
            "scoring_unit": SCORING_UNIT,
        }

    def _token_row(
        self,
        req_id: str,
        layer_idx: int,
        head: int,
        diag: dict[str, Any],
        block_size: int,
    ) -> dict[str, Any]:
        k_red, v_red = diag["k_red"], diag["v_red"]
        k_norm, v_norm = diag["k_norm"], diag["v_norm"]
        return {
            "run_id": self.config.run_id or "",
            "request_id": req_id,
            "prompt_id": req_id,
            "policy": self.config.experiment_mode,
            "model": self.model_name,
            "dataset": self.config.dataset or "unknown",
            "phase": "prefill",
            "layer": layer_idx,
            "kv_head": head,
            "num_tokens_scored": int(diag["num_tokens"]),
            "mean_redundancy": float(k_red["mean"][head]),
            "max_redundancy": float(k_red["max"][head]),
            "p90_redundancy": float(k_red["p90"][head]),
            "std_redundancy": float(k_red["std"][head]),
            "v_mean_redundancy": float(v_red["mean"][head]),
            "v_max_redundancy": float(v_red["max"][head]),
            "v_p90_redundancy": float(v_red["p90"][head]),
            "v_std_redundancy": float(v_red["std"][head]),
            "key_norm_mean": float(k_norm["mean"][head]),
            "key_norm_std": float(k_norm["std"][head]),
            "key_norm_p90": float(k_norm["p90"][head]),
            "key_norm_cov": float(k_norm["cov"][head]),
            "value_norm_mean": float(v_norm["mean"][head]),
            "value_norm_std": float(v_norm["std"][head]),
            "value_norm_p90": float(v_norm["p90"][head]),
            "value_norm_cov": float(v_norm["cov"][head]),
            "block_size": block_size,
            "physical_eviction_unit": PHYSICAL_EVICTION_UNIT,
            "scoring_unit": "token",
        }

    def _joint_row(
        self,
        req_id: str,
        layer_idx: int,
        diag: dict[str, Any],
        block_size: int,
    ) -> dict[str, Any]:
        return {
            "run_id": self.config.run_id or "",
            "request_id": req_id,
            "prompt_id": req_id,
            "policy": self.config.experiment_mode,
            "model": self.model_name,
            "dataset": self.config.dataset or "unknown",
            "phase": "prefill",
            "layer": layer_idx,
            "num_tokens_scored": int(diag["num_tokens"]),
            "joint_redundancy_mean": float(diag["joint_red_mean"]),
            "joint_redundancy_std": float(diag["joint_red_std"]),
            "joint_redundancy_p90": float(diag["joint_red_p90"]),
            "joint_norm_mean": float(diag["joint_norm_mean"]),
            "joint_norm_std": float(diag["joint_norm_std"]),
            "joint_norm_cov": float(diag["joint_norm_cov"]),
            "joint_v_redundancy_mean": float(diag["joint_v_red_mean"]),
            "joint_v_redundancy_std": float(diag["joint_v_red_std"]),
            "joint_v_redundancy_p90": float(diag["joint_v_red_p90"]),
            "joint_v_norm_mean": float(diag["joint_v_norm_mean"]),
            "joint_v_norm_std": float(diag["joint_v_norm_std"]),
            "joint_v_norm_cov": float(diag["joint_v_norm_cov"]),
            "block_size": block_size,
            "physical_eviction_unit": PHYSICAL_EVICTION_UNIT,
            "scoring_unit": "joint_all_heads",
        }

    def _norm_row(
        self,
        req_id: str,
        layer_idx: int,
        head: int,
        mode: str,
        unit: str,
        var: dict[str, Any],
        block_size: int,
    ) -> dict[str, Any]:
        s = var[mode]
        return {
            "run_id": self.config.run_id or "",
            "request_id": req_id,
            "prompt_id": req_id,
            "policy": self.config.experiment_mode,
            "model": self.model_name,
            "dataset": self.config.dataset or "unknown",
            "phase": "prefill",
            "normalization": mode,
            "scoring_unit": unit,
            "layer": layer_idx,
            "kv_head": head,
            "num_units_scored": int(var["num_units"]),
            "mean_redundancy": float(s["mean"][head]),
            "max_redundancy": float(s["max"][head]),
            "p90_redundancy": float(s["p90"][head]),
            "std_redundancy": float(s["std"][head]),
            "block_size": block_size,
            "physical_eviction_unit": PHYSICAL_EVICTION_UNIT,
        }

    def _blockpos_row(
        self,
        req_id: str,
        layer_idx: int,
        block_index: int,
        num_blocks: int,
        redundancy: float,
        block_size: int,
    ) -> dict[str, Any]:
        return {
            "run_id": self.config.run_id or "",
            "request_id": req_id,
            "prompt_id": req_id,
            "policy": self.config.experiment_mode,
            "model": self.model_name,
            "dataset": self.config.dataset or "unknown",
            "phase": "prefill",
            "layer": layer_idx,
            "block_index": block_index,
            "num_blocks": num_blocks,
            "joint_v_block_redundancy": redundancy,
            "block_size": block_size,
            "physical_eviction_unit": PHYSICAL_EVICTION_UNIT,
        }

    # -- summary / teardown -------------------------------------------------
    def _write_summary(self) -> None:
        write_summary(
            self.output_dir,
            {
                "run_id": self.config.run_id,
                "model": self.model_name,
                "dataset": self.config.dataset or "unknown",
                "policy": self.config.experiment_mode,
                "physical_eviction_unit": PHYSICAL_EVICTION_UNIT,
                "scoring_unit": SCORING_UNIT,
                "note": SUMMARY_NOTE,
                "num_requests": self._num_requests_scored,
                "num_attn_layers": self.num_layers,
                "sampled_layers": self.config.score_sampled_layers,
                "sampled_kv_heads": self.config.score_sampled_kv_heads,
                "block_size": self._block_size,
                "M_total": self.config.total_page_cap,
            },
        )

    def close(self) -> None:
        if self._num_requests_scored > 0:
            self._write_summary()
        for writer in self._writers.values():
            writer.close()
        logger.info(
            "[geo_kv] PrefillScorer closed: scored %d requests -> %s",
            self._num_requests_scored,
            self.output_dir,
        )
