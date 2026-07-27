# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for geometric KV-cache eviction experiments (Option A).

This is *experimental* validation scaffolding. All behavior is gated behind an
explicit ``additional_config["geo_kv"]`` block; when absent (the default) vLLM
behaves exactly as upstream.

Option A recap: vLLM cannot physically free individual heads or layers. A block
is monolithic across heads, and a uniform model shares one block table across
all layers. So per-layer/per-head signals are *scoring* dimensions only; the
physical eviction unit is always a whole vLLM block. See ``plan.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Key under ``VllmConfig.additional_config`` that holds our experiment block.
ADDITIONAL_CONFIG_KEY = "geo_kv"

EXPERIMENT_MODES = (
    "none",
    "score_only",
    "geo_emergent",
    "geo_uniform",
    "geo_frozen_emergent",
    "attention_budget",
    "proxy_attention_budget",
)
HEAD_REDUNDANCY_STATS = ("max", "topk_mean", "percentile_90")
BLOCK_SCORE_AGGREGATIONS = ("mean", "max", "topk_mean", "percentile_90")
OVERFLOW_EVICTION_MODES = ("repeated_single",)
# Milestone-1 (mask-only) eviction policies. v_redundancy is the thesis;
# recency (drop-oldest) and random are the matched-count baselines.
EVICTION_POLICIES = ("v_redundancy", "recency", "random")
# Mirror of scoring.NORM_VARIANTS, duplicated so config stays import-light
# (no torch). Kept in sync by the norm_variants validation test.
NORM_VARIANT_CHOICES = ("raw", "center_request", "whiten_request")


@dataclass
class GeoKVConfig:
    """Resolved geometric-KV experiment configuration.

    Defaults are the inert (disabled) settings, so constructing this object from
    an empty/absent config never changes vLLM behavior.
    """

    experiment_mode: str = "none"
    enable_tracing: bool = False
    log_path: str | None = None

    # Global memory target, expressed in whole vLLM physical blocks (M_total).
    total_page_cap: int | None = None
    # Admission threshold (tau). Only meaningful for score variants (see plan).
    admission_threshold: float | None = None
    # Warmup pages (W) kept before any scoring/eviction is considered.
    warmup_pages: int | None = None

    # Scoring dimensions: "all", "last", or comma-separated indices e.g. "0,8,16".
    score_sampled_layers: str = "all"
    score_sampled_kv_heads: str = "all"

    # Diagnostic: also score redundancy at the TOKEN level (no block pooling),
    # writing token_level_scores.csv. Off by default; only used in score_only.
    score_token_level: bool = False
    # Cap on tokens per request used for token-level scoring (bounds O(N^2)).
    token_sample_cap: int = 4096

    # Diagnostic: also score redundancy under several pre-cosine normalizations
    # (raw / center_request / whiten_request) for both block and token units,
    # writing normalization_scores.csv. Tests whether the redundancy fingerprint
    # is just a per-head mean cone. Off by default; only used in score_only.
    score_norm_variants: bool = False
    # Comma-separated subset of scoring.NORM_VARIANTS to evaluate.
    norm_variants: str = "raw,center_request,whiten_request"

    # Diagnostic: emit per-(request, block) joint-V block-anchor redundancy
    # (mean over layers), writing block_positional_scores.csv. Feeds the
    # positional-vs-content control. Off by default; only used in score_only.
    score_block_positional: bool = False

    # How per-head redundancy is reduced across blocks, and how per-(layer,head)
    # scores are aggregated into one whole-block score.
    head_redundancy_stat: str = "percentile_90"
    block_score_aggregation: str = "percentile_90"
    # Fraction used by the "topk_mean" statistic/aggregation.
    topk_frac: float = 0.1

    # Freeze prefill-derived layer/head weights during decode (frozen_emergent).
    freeze_after_prefill: bool = False
    overflow_eviction_mode: str = "repeated_single"

    # -- Milestone 1 (mask-only) eviction knobs -----------------------------
    # Per-request fraction of *evictable* prompt blocks to hide from attention.
    # The leading warmup_pages (sink) and the final block (decode anchor) are
    # always kept. The direct analog of the HF screen's --rates sweep; None/0.0
    # means no block is hidden (the path stays inert). Active only for the
    # ``active_eviction`` modes (e.g. geo_uniform).
    eviction_rate: float | None = None
    # Which blocks to drop at that rate: geometric V-redundancy (the thesis),
    # recency (drop-oldest, the StreamingLLM contrast), or random (seeded).
    eviction_policy: str = "v_redundancy"
    # Seed for the "random" policy so a run is reproducible.
    eviction_seed: int = 0
    # When True, evicted blocks are physically freed back into the block pool
    # reclaiming GPU memory. Requires an active_eviction mode. When False,
    # behavior is mask-only
    physical_reclaim: bool = False

    # -- Capacity-budget eviction with hysteresis (physical reclaim) --------
    # Per-request capacity C on KV blocks, enforced with a low-watermark band.
    # The cache is allowed to fill to C, then V-redundancy scoring physically
    # frees the most-redundant interior blocks down to floor(watermark * C).
    # This governs BOTH ends: at end of prefill (fill to C, evict to the
    # watermark) and during decode (regrow to C, evict again). None disables it
    # entirely (pure prefill-rate behavior, unchanged). The leading
    # warmup_pages (sink) and the final block are always kept.
    decode_evict_budget: int | None = None
    # Low-watermark ratio: eviction targets floor(decode_evict_watermark *
    # decode_evict_budget) blocks. The hysteresis gap (C down to watermark*C)
    # bounds how often the O(C^2) scoring + compaction runs during decode.
    decode_evict_watermark: float = 0.75
    # Re-check the capacity only every N decode steps (throttle). A request only
    # crosses a block boundary every block_size steps, so 1 is already cheap;
    # larger values reduce scoring frequency further.
    decode_evict_interval: int = 1

    # -- Decoupled per-end targets, expressed as a fraction of prompt blocks --
    # These decouple the two eviction ends so admission and decode can be tuned
    # independently (e.g. strict admission + lenient decode). They are resolved
    # per-request from that request's prompt block count P, so a single fraction
    # is comparable across models / tasks / prompt lengths. Both are additive and
    # independent: set either, both, or neither. When set they take precedence
    # over the coupled ``decode_evict_budget`` path.
    #
    # ``prefill_evict_frac``: at end of prefill, retain ceil(f * P) prompt blocks
    # (V-redundancy drops the rest). None disables admission eviction. f == 1.0 is
    # inert (retain all).
    prefill_evict_frac: float | None = None
    # ``decode_evict_frac``: the decode capacity is C = ceil(f * P); when a
    # decoding request regrows to C, the most-redundant interior blocks are
    # evicted down to ceil(decode_evict_watermark * C). None disables decode
    # eviction. Reuses ``decode_evict_watermark`` (band floor) and
    # ``decode_evict_interval`` (throttle).
    decode_evict_frac: float | None = None

    # Output location for score_only artifacts (CSV/JSON). Optional.
    output_dir: str | None = None
    run_id: str | None = None
    # Free-form dataset label recorded in the CSV (e.g. "synthetic_fixed").
    dataset: str | None = None

    # Optional query-alignment tie-break knobs (used by later phases only).
    query_ema_beta: float | None = None
    query_alignment_weight: float | None = None
    enable_query_tiebreak: bool = False

    @property
    def enabled(self) -> bool:
        """True when any experimental behavior should activate."""
        return self.experiment_mode != "none"

    @property
    def is_score_only(self) -> bool:
        return self.experiment_mode == "score_only"

    def resolved_norm_variants(self) -> list[str]:
        """Parse ``norm_variants`` into a de-duplicated, order-preserving list."""
        seen: dict[str, None] = {}
        for p in self.norm_variants.split(","):
            p = p.strip()
            if p:
                seen.setdefault(p, None)
        return list(seen)

    @property
    def active_eviction(self) -> bool:
        """True for policies that physically evict blocks (Phase 7+)."""
        return self.experiment_mode in (
            "geo_emergent",
            "geo_uniform",
            "geo_frozen_emergent",
            "attention_budget",
            "proxy_attention_budget",
        )

    # -- construction -------------------------------------------------------
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GeoKVConfig:
        """Build and validate from a plain dict (the additional_config block)."""
        valid = {f.name for f in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(
                f"Unknown geo_kv config keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid)}"
            )
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> GeoKVConfig:
        """Extract the geo_kv block from ``vllm_config.additional_config``.

        Returns an inert (disabled) config when the block is absent, so callers
        can always rely on ``.enabled``.
        """
        additional = getattr(vllm_config, "additional_config", None) or {}
        if not isinstance(additional, dict):
            return cls()
        raw = additional.get(ADDITIONAL_CONFIG_KEY)
        if not raw:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(
                f"additional_config['{ADDITIONAL_CONFIG_KEY}'] must be a dict, "
                f"got {type(raw).__name__}"
            )
        return cls.from_dict(raw)

    # -- validation ---------------------------------------------------------
    def validate(self) -> None:
        """Validate all fields, raising ValueError with actionable messages."""
        _check_choice("experiment_mode", self.experiment_mode, EXPERIMENT_MODES)
        _check_choice(
            "head_redundancy_stat", self.head_redundancy_stat, HEAD_REDUNDANCY_STATS
        )
        _check_choice(
            "block_score_aggregation",
            self.block_score_aggregation,
            BLOCK_SCORE_AGGREGATIONS,
        )
        _check_choice(
            "overflow_eviction_mode",
            self.overflow_eviction_mode,
            OVERFLOW_EVICTION_MODES,
        )
        _check_choice("eviction_policy", self.eviction_policy, EVICTION_POLICIES)
        _check_index_spec("score_sampled_layers", self.score_sampled_layers)
        _check_index_spec("score_sampled_kv_heads", self.score_sampled_kv_heads)

        if self.eviction_rate is not None and not (0.0 <= self.eviction_rate <= 1.0):
            raise ValueError("geo_kv.eviction_rate must be in [0, 1]")

        if self.physical_reclaim and not self.active_eviction:
            raise ValueError(
                "geo_kv.physical_reclaim=True requires an active_eviction "
                "experiment_mode"
            )

        if self.total_page_cap is not None and self.total_page_cap <= 0:
            raise ValueError("geo_kv.total_page_cap must be a positive integer")
        if self.warmup_pages is not None and self.warmup_pages < 0:
            raise ValueError("geo_kv.warmup_pages must be >= 0")
        if self.decode_evict_budget is not None:
            if self.decode_evict_budget < 1:
                raise ValueError("geo_kv.decode_evict_budget must be >= 1")
            if self.decode_evict_interval < 1:
                raise ValueError("geo_kv.decode_evict_interval must be >= 1")
            if not (0.0 < self.decode_evict_watermark < 1.0):
                raise ValueError(
                    "geo_kv.decode_evict_watermark must be in (0, 1) "
                    "(the low-watermark fraction of decode_evict_budget)"
                )
            # eviction_policy selects WHICH blocks the band drops:
            # v_redundancy (scored, the thesis), recency (oldest-first ==
            # StreamingLLM baseline at matched memory), or random. All three are
            # honored by select_evicted_to_budget, so no policy restriction here.
            if not self.active_eviction:
                raise ValueError(
                    "geo_kv.decode_evict_budget requires an active_eviction "
                    "experiment_mode"
                )
        # Decoupled fraction-of-prompt targets. Each is independent; when either
        # is set it uses the same policy/mode requirements as the budget path.
        for _name, _frac in (
            ("prefill_evict_frac", self.prefill_evict_frac),
            ("decode_evict_frac", self.decode_evict_frac),
        ):
            if _frac is None:
                continue
            if not (0.0 < _frac <= 1.0):
                raise ValueError(f"geo_kv.{_name} must be in (0, 1]")
            if self.eviction_policy != "v_redundancy":
                raise ValueError(
                    f"geo_kv.{_name} currently requires "
                    "eviction_policy='v_redundancy' (the core thesis signal)"
                )
            if not self.active_eviction:
                raise ValueError(
                    f"geo_kv.{_name} requires an active_eviction experiment_mode"
                )
        if self.decode_evict_frac is not None and self.decode_evict_interval < 1:
            raise ValueError("geo_kv.decode_evict_interval must be >= 1")
        if self.decode_evict_frac is not None and not (
            0.0 < self.decode_evict_watermark < 1.0
        ):
            raise ValueError(
                "geo_kv.decode_evict_watermark must be in (0, 1) "
                "(the low-watermark fraction of the decode capacity)"
            )
        if not (0.0 < self.topk_frac <= 1.0):
            raise ValueError("geo_kv.topk_frac must be in (0, 1]")
        if self.token_sample_cap < 2:
            raise ValueError("geo_kv.token_sample_cap must be >= 2")
        for v in self.resolved_norm_variants():
            if v not in NORM_VARIANT_CHOICES:
                raise ValueError(
                    f"geo_kv.norm_variants entry {v!r} is invalid; choose from "
                    f"{list(NORM_VARIANT_CHOICES)}"
                )
        if self.enable_tracing and not self.log_path:
            raise ValueError(
                "geo_kv.enable_tracing=True requires geo_kv.log_path to be set"
            )

    # -- display ------------------------------------------------------------
    def summary(self) -> str:
        """Compact one-line description for logging."""
        parts = [f"mode={self.experiment_mode}"]
        if self.total_page_cap is not None:
            parts.append(f"M_total={self.total_page_cap}blk")
        parts.append(f"layers={self.score_sampled_layers}")
        parts.append(f"kv_heads={self.score_sampled_kv_heads}")
        parts.append(f"head_stat={self.head_redundancy_stat}")
        parts.append(f"block_agg={self.block_score_aggregation}")
        parts.append(f"freeze_after_prefill={self.freeze_after_prefill}")
        parts.append(f"tracing={self.enable_tracing}")
        if self.log_path:
            parts.append(f"log={self.log_path}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(
            f"geo_kv.{name}={value!r} is invalid; choose one of {list(choices)}"
        )


def _check_index_spec(name: str, spec: str) -> None:
    """Validate an index spec: 'all', 'last', or comma-separated ints."""
    if spec in ("all", "last"):
        return
    parts = [p.strip() for p in spec.split(",") if p.strip() != ""]
    ok = bool(parts)
    for p in parts:
        if not (p.lstrip("-").isdigit() and int(p) >= 0):
            ok = False
            break
    if not ok:
        raise ValueError(
            f"geo_kv.{name}={spec!r} is invalid; use 'all', 'last', or a "
            f"comma-separated list of non-negative ints e.g. '0,8,16,24,31'"
        )


def resolve_indices(spec: str, n: int) -> list[int]:
    """Resolve a validated index spec into concrete indices in ``[0, n)``.

    Args:
        spec: "all", "last", or a comma-separated list of non-negative ints.
        n: The size of the dimension being sampled.

    Returns:
        Sorted, de-duplicated indices; out-of-range entries are dropped.
    """
    if n <= 0:
        return []
    if spec == "all":
        return list(range(n))
    if spec == "last":
        return [n - 1]
    idxs = {int(p.strip()) for p in spec.split(",") if p.strip() != ""}
    return sorted(i for i in idxs if 0 <= i < n)
