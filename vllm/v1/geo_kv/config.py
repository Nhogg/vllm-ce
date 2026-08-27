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
# How v_redundancy ranks blocks: pairwise (score-once, the pinned default) or
# greedy (iterative peel that protects the surviving twin of a duplicate pair).
REDUNDANCY_MODES = ("pairwise", "greedy", "coverage")
# How a whole KV block is represented before pairwise V-redundancy scoring.
# ``mean`` is the pinned one-anchor scorer. The experimental multi-prototype
# modes retain within-block structure while preserving whole-block eviction.
BLOCK_PROTOTYPE_MODES = ("mean", "quarters", "mean_top2_norm")
# How the value-L2 baseline collapses valid token/head norms into one physical
# block importance. ``sum`` is the pinned get_block_score reproduction. The max
# variants are experimental whole-block approximations to PagedEviction's much
# finer token/head selection: one rare strong cell can protect its whole block.
VALUE_L2_BLOCK_REDUCTIONS = ("sum", "max_token", "max_token_head")
OVERFLOW_EVICTION_MODES = ("repeated_single",)
# Milestone-1 (mask-only) eviction policies. v_redundancy is the thesis;
# recency (drop-oldest) and random are the matched-count baselines; value_l2 is
# the Paged-Eviction baseline (drop lowest value-L2-norm blocks), ported into the
# V1 capacity-band path for a matched-memory contrast against v_redundancy.
EVICTION_POLICIES = ("v_redundancy", "recency", "random", "value_l2")
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

    # Diagnostic (plan Phase 1C): while scoring with ALL layers, also measure how
    # well fixed 1/2/4-layer subsets reproduce the all-layer selected-block set
    # (mean Jaccard), so a cheap universal subset can be picked without a full
    # eval per subset. Off by default; requires score_sampled_layers=="all" and
    # enable_tracing (writes <log_path>.layer_calibration.json). Adds a small
    # per-fire overhead, so it is a calibration mode, not for production.
    calibrate_layer_subsets: bool = False

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

    # How the v_redundancy score ranks blocks:
    #   "pairwise" -- each block's max cosine to ANY other block, scored once
    #     (the pinned thesis signal). Near-duplicate blocks BOTH score maximally
    #     redundant, so a budget cut can drop both and lose the duplicated info.
    #   "greedy" -- peel the most-redundant block one at a time, recomputing each
    #     survivor's redundancy against the SURVIVING set only, so once one twin
    #     is dropped its partner is protected (set-cover-correct). Encoded as an
    #     eviction-order score consumed by the same argsort budget selector.
    #   "coverage" -- aggregate similarity across layers first, then greedily
    #     construct one global kept set that maximizes whole-request coverage at
    #     the actual keep budget. This fixes the old greedy mode's independent
    #     per-layer ordinal aggregation and duplicate-annihilation failure.
    # Default "pairwise" == byte-identical to the pinned scorer. Applies only to
    # eviction_policy="v_redundancy".
    redundancy_mode: str = "pairwise"
    # Representation used by pairwise V-redundancy:
    #   mean -- current valid-token mean (one prototype; pinned behavior).
    #   quarters -- four contiguous sub-block means.
    #   mean_top2_norm -- the block mean plus its two highest-value-norm tokens.
    # Multi-prototype modes make a block droppable only when every prototype is
    # covered by another block, preventing a rare token from being cancelled by
    # the block mean. They do not change the physical eviction unit. Experimental;
    # default "mean" is byte-identical to the pinned scorer.
    block_prototype_mode: str = "mean"
    # Optional value-norm blend for v_redundancy: final droppability =
    # zscore(redundancy) - value_blend_beta * zscore(value_L2_norm). Redundancy
    # wins on span-finding tasks (multifieldqa); value-L2 wins on distributed-
    # information tasks (summarization, dense QA). Both signals are standardized
    # per request so beta is scale-free. A high-value-norm block becomes LESS
    # droppable. None/0.0 == off (pure redundancy, byte-identical). Applies only
    # to eviction_policy="v_redundancy".
    value_blend_beta: float | None = None

    # Norm-constrained redundancy (Phase 2): protect blocks whose aggregated
    # value-L2 norm is at or above the ``value_norm_protect_quantile`` quantile
    # (per request), then evict the highest-redundancy blocks from the REMAINING
    # candidates. A hard protection constraint rather than the soft linear
    # ``value_blend_beta``: value-L2 wins on distributed-information tasks
    # (summarization, dense QA) while V-redundancy wins on span-finding
    # (multifieldqa), and treating high value-norm as a keep-guard aims to capture
    # both. Budget stays exact -- if too few candidates remain to reach the budget,
    # protection is relaxed in ascending value-norm order (lowest-norm protected
    # blocks first) until exactly ``k`` candidates are available. Sink / anchor /
    # tail protection always take priority and are never relaxed. Quantile in
    # [0, 1): 0.0 protects every block (fully relaxed back to budget), values near
    # 1 protect only the top-norm blocks. None == off (byte-identical to the pinned
    # scorer). Applies only to eviction_policy="v_redundancy".
    value_norm_protect_quantile: float | None = None

    # Within-block reduction for eviction_policy="value_l2". ``sum`` exactly
    # preserves the existing baseline. The max modes retain rare high-norm
    # token or token/head signal while the physical eviction unit stays a whole
    # shared vLLM block. Non-default values are rejected for other policies.
    value_l2_block_reduction: str = "sum"

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

    # Cache copied block anchors and their pairwise cosine matrix across physical
    # decode compactions.  On the next fire, only changed rows (normally the
    # formerly-partial tail plus newly appended blocks) are read and multiplied.
    # This is an exact optimization of the pinned pairwise/mean V-redundancy
    # scorer, not a new policy.  Generation changes at compaction acknowledgement
    # and request-slot reuse make physical-block reuse safe.  Off by default.
    incremental_decode_scoring: bool = False

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

    # -- Fixed-rate decode "drip" ------------------------------------------
    # ``decode_evict_blocks_per_step``: instead of a watermark band, evict
    # exactly N whole interior blocks on each decode-eviction fire (budget =
    # count - N via select_evicted_to_budget). N is a drain *rate*, not a
    # steady-state size: with no capacity gate the request drains monotonically
    # toward the floor (warmup_pages sink + 1 anchor, plus any tail-protect).
    # Distinct from and MUTUALLY EXCLUSIVE with the two watermark-band decode
    # paths (decode_evict_frac / decode_evict_budget); COMPOSABLE with
    # prefill_evict_frac (admission). Works off the live block count, never the
    # per-request _decode_cap_np band. Reuses decode_evict_interval (throttle).
    # Under physical_reclaim the in-flight compaction guard halves the effective
    # cadence to ~N blocks per 2 steps (still exactly N per fire). None disables.
    decode_evict_blocks_per_step: int | None = None

    # -- Pressure gate on the fixed-rate drip ------------------------------
    # ``decode_evict_pressure_watermark``: gate the drip on GLOBAL cache pressure
    # instead of firing on a fixed schedule. A FILL threshold in (0, 1): the drip
    # only fires on a decode step when the global KV pool is >= this fraction full
    # (used_fraction >= watermark). Below the threshold on_decode_step is a no-op
    # for every request, so short / low-pressure generations pay nothing. This is
    # a MODIFIER on the drip, not its own mechanism: it REQUIRES
    # decode_evict_blocks_per_step (the N it gates). The global free fraction is a
    # scheduler-side signal plumbed to the worker via SchedulerOutput. None
    # disables the gate -> the drip fires unconditionally (legacy behavior).
    decode_evict_pressure_watermark: float | None = None

    # Output location for score_only artifacts (CSV/JSON). Optional.
    output_dir: str | None = None
    run_id: str | None = None
    # Free-form dataset label recorded in the CSV (e.g. "synthetic_fixed").
    dataset: str | None = None

    # Actual query-to-key relevance refinement. When enabled, the engine
    # captures up to the final ``query_tail_tokens`` observed post-RoPE queries
    # on the sampled scoring layers and lowers droppability for blocks receiving
    # high causal QK attention mass. With prefix caching, the observed uncached
    # suffix is used against the complete visible K cache. A positive weight uses
    # per-request z-scored signals; tiebreak mode only separates equal redundancy
    # scores. Both are inert by default and apply only to v_redundancy.
    query_ema_beta: float | None = None
    query_alignment_weight: float | None = None
    enable_query_tiebreak: bool = False
    # Protect blocks at or above this per-request query-relevance quantile, then
    # rank the remaining blocks by the finalized redundancy score. If the hard
    # guard leaves too few candidates for the exact budget, relax protected
    # blocks in ascending relevance order. Unlike the linear query blend, this
    # leaves the redundancy/cosh ranking unchanged among unprotected blocks.
    # None is inert. Applies only to budget-based pairwise V-redundancy.
    query_relevance_protect_quantile: float | None = None
    query_tail_tokens: int = 32
    # Tail-protect (recency floor) for the budget path: never V-evict the last
    # ``ceil(query_tail_protect_frac * num_blocks)`` blocks of a request. The
    # question sits at the prompt tail in every LongBench template, so query-
    # agnostic V-redundancy can drop question-relevant blocks at end-of-prefill;
    # protecting the tail grafts StreamingLLM's recency guard onto v_redundancy.
    # None/0.0 == off (pinned v_redundancy behavior, byte-identical). Memory stays
    # matched: the protected count is clamped so exactly enough interior blocks are
    # still dropped to reach budget.
    query_tail_protect_frac: float | None = None
    # Positional cosh (sech-bump) re-weighting of v_redundancy droppability.
    # Multiply each block's droppability by ``1/cosh(alpha * x)`` where ``x``
    # maps the block's position over the evictable span to ``[-1, 1]`` (ends at
    # +/-1, middle at 0). ``sech`` peaks at the middle and decays toward the
    # ends, so a larger multiplier lands on interior blocks: eviction is pulled
    # toward the middle and away from the sink/tail, a smooth analog of the
    # hard ``query_tail_protect_frac`` guard. Applied only to ``v_redundancy``
    # (the scored thesis signal); ignored by recency/random and by value_l2.
    # None/0.0 == off (byte-identical to the pinned scorer). Must be >= 0.
    positional_cosh_alpha: float | None = None

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

    @property
    def query_relevance_enabled(self) -> bool:
        return (
            bool(self.query_alignment_weight)
            or self.enable_query_tiebreak
            or self.query_relevance_protect_quantile is not None
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
        _check_choice(
            "value_l2_block_reduction",
            self.value_l2_block_reduction,
            VALUE_L2_BLOCK_REDUCTIONS,
        )
        if (
            self.value_l2_block_reduction != "sum"
            and self.eviction_policy != "value_l2"
        ):
            raise ValueError(
                "geo_kv.value_l2_block_reduction requires "
                "eviction_policy='value_l2' when non-default"
            )
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
        # eviction_policy selects WHICH blocks the band drops: v_redundancy
        # (scored, the thesis), recency (oldest-first == StreamingLLM baseline at
        # matched memory), random, or value_l2 (the Paged-Eviction contrast). All
        # are honored by select_evicted_to_budget, so no policy restriction here
        # -- exactly as on the coupled decode_evict_budget path above.
        for _name, _frac in (
            ("prefill_evict_frac", self.prefill_evict_frac),
            ("decode_evict_frac", self.decode_evict_frac),
        ):
            if _frac is None:
                continue
            if not (0.0 < _frac <= 1.0):
                raise ValueError(f"geo_kv.{_name} must be in (0, 1]")
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
        # Fixed-rate decode drip: exactly one decode mechanism may be active.
        if self.decode_evict_blocks_per_step is not None:
            if self.decode_evict_blocks_per_step < 1:
                raise ValueError("geo_kv.decode_evict_blocks_per_step must be >= 1")
            if (
                self.decode_evict_frac is not None
                or self.decode_evict_budget is not None
            ):
                raise ValueError(
                    "geo_kv.decode_evict_blocks_per_step is mutually exclusive "
                    "with decode_evict_frac and decode_evict_budget "
                    "(pick one decode eviction mechanism)"
                )
            if self.eviction_policy != "v_redundancy":
                raise ValueError(
                    "geo_kv.decode_evict_blocks_per_step currently requires "
                    "eviction_policy='v_redundancy' (the core thesis signal)"
                )
            if not self.active_eviction:
                raise ValueError(
                    "geo_kv.decode_evict_blocks_per_step requires an "
                    "active_eviction experiment_mode"
                )
            if self.decode_evict_interval < 1:
                raise ValueError("geo_kv.decode_evict_interval must be >= 1")
        # Pressure gate on the drip: a FILL threshold that decides WHEN the drip
        # may fire; it is not a standalone mechanism, so it requires the drip N.
        if self.decode_evict_pressure_watermark is not None:
            if not (0.0 < self.decode_evict_pressure_watermark < 1.0):
                raise ValueError(
                    "geo_kv.decode_evict_pressure_watermark must be in (0, 1) "
                    "(the global cache FILL fraction at which the drip fires)"
                )
            if self.decode_evict_blocks_per_step is None:
                raise ValueError(
                    "geo_kv.decode_evict_pressure_watermark requires "
                    "decode_evict_blocks_per_step (it gates the fixed-rate drip; "
                    "it is not a standalone eviction mechanism)"
                )
        if self.query_tail_protect_frac is not None and not (
            0.0 <= self.query_tail_protect_frac < 1.0
        ):
            raise ValueError(
                "geo_kv.query_tail_protect_frac must be in [0, 1) "
                "(fraction of trailing blocks protected from V-eviction)"
            )
        if self.query_alignment_weight is not None and self.query_alignment_weight < 0:
            raise ValueError("geo_kv.query_alignment_weight must be >= 0")
        if self.query_relevance_protect_quantile is not None and not (
            0.0 <= self.query_relevance_protect_quantile < 1.0
        ):
            raise ValueError(
                "geo_kv.query_relevance_protect_quantile must be in [0, 1)"
            )
        if self.query_tail_tokens < 1:
            raise ValueError("geo_kv.query_tail_tokens must be >= 1")
        if self.query_relevance_enabled:
            if self.eviction_policy != "v_redundancy":
                raise ValueError(
                    "geo_kv query relevance requires "
                    "eviction_policy='v_redundancy'"
                )
            if not self.active_eviction:
                raise ValueError(
                    "geo_kv query relevance requires an active_eviction "
                    "experiment_mode"
                )
            if self.calibrate_layer_subsets:
                raise ValueError(
                    "geo_kv query relevance is incompatible with "
                    "calibrate_layer_subsets; calibrate the base layer subset "
                    "first, then evaluate query relevance on that fixed subset"
                )
        if self.query_relevance_protect_quantile is not None:
            budget_selection_active = (
                self.prefill_evict_frac is not None
                or self.decode_evict_frac is not None
                or self.decode_evict_budget is not None
                or self.decode_evict_blocks_per_step is not None
            )
            if not budget_selection_active:
                raise ValueError(
                    "geo_kv.query_relevance_protect_quantile requires a "
                    "budget-based eviction mechanism"
                )
            if self.redundancy_mode != "pairwise":
                raise ValueError(
                    "geo_kv.query_relevance_protect_quantile requires "
                    "redundancy_mode='pairwise'"
                )
            decode_active = (
                self.decode_evict_frac is not None
                or self.decode_evict_budget is not None
                or self.decode_evict_blocks_per_step is not None
            )
            if decode_active and not self.physical_reclaim:
                raise ValueError(
                    "geo_kv.query_relevance_protect_quantile requires "
                    "physical_reclaim when decode eviction is active"
                )
            if self.value_norm_protect_quantile is not None:
                raise ValueError(
                    "geo_kv query-relevance and value-norm hard protections "
                    "cannot be combined"
                )
        if self.positional_cosh_alpha is not None and self.positional_cosh_alpha < 0.0:
            raise ValueError(
                "geo_kv.positional_cosh_alpha must be >= 0 "
                "(sech-bump steepness; 0 == off)"
            )
        _check_choice("redundancy_mode", self.redundancy_mode, REDUNDANCY_MODES)
        _check_choice(
            "block_prototype_mode",
            self.block_prototype_mode,
            BLOCK_PROTOTYPE_MODES,
        )
        if self.redundancy_mode == "greedy":
            if self.eviction_policy != "v_redundancy":
                raise ValueError(
                    "geo_kv.redundancy_mode='greedy' requires "
                    "eviction_policy='v_redundancy'"
                )
            if self.block_prototype_mode != "mean":
                raise ValueError(
                    "geo_kv.block_prototype_mode currently requires "
                    "redundancy_mode='pairwise' unless it is 'mean'"
                )
            if (
                self.value_blend_beta
                or self.value_norm_protect_quantile is not None
                or self.query_relevance_enabled
                or self.calibrate_layer_subsets
                or bool(self.positional_cosh_alpha)
            ):
                raise ValueError(
                    "geo_kv.redundancy_mode='greedy' is incompatible with "
                    "value/query refinements and layer calibration"
                )
        if self.redundancy_mode == "coverage":
            decode_active = (
                self.decode_evict_budget is not None
                or self.decode_evict_frac is not None
                or self.decode_evict_blocks_per_step is not None
            )
            if self.eviction_policy != "v_redundancy":
                raise ValueError(
                    "geo_kv.redundancy_mode='coverage' requires "
                    "eviction_policy='v_redundancy'"
                )
            if self.block_prototype_mode != "mean":
                raise ValueError(
                    "geo_kv.redundancy_mode='coverage' currently requires "
                    "block_prototype_mode='mean'"
                )
            if (
                self.value_blend_beta
                or self.value_norm_protect_quantile is not None
                or self.query_relevance_enabled
                or self.calibrate_layer_subsets
                or bool(self.positional_cosh_alpha)
            ):
                raise ValueError(
                    "geo_kv.redundancy_mode='coverage' is incompatible with "
                    "value/query refinements and layer calibration"
                )
            if decode_active and not self.physical_reclaim:
                raise ValueError(
                    "geo_kv.redundancy_mode='coverage' requires physical_reclaim "
                    "when decode eviction is active"
                )
        if self.value_blend_beta is not None and self.value_blend_beta < 0.0:
            raise ValueError(
                "geo_kv.value_blend_beta must be >= 0 "
                "(value-norm blend weight; 0/None == off)"
            )
        if self.value_norm_protect_quantile is not None and not (
            0.0 <= self.value_norm_protect_quantile < 1.0
        ):
            raise ValueError(
                "geo_kv.value_norm_protect_quantile must be in [0, 1) "
                "(value-norm protection quantile; None == off)"
            )
        if self.incremental_decode_scoring:
            decode_active = (
                self.decode_evict_budget is not None
                or self.decode_evict_frac is not None
                or self.decode_evict_blocks_per_step is not None
            )
            if not self.physical_reclaim or not decode_active:
                raise ValueError(
                    "geo_kv.incremental_decode_scoring requires physical_reclaim "
                    "and an active decode eviction mechanism"
                )
            if (
                self.eviction_policy != "v_redundancy"
                or self.redundancy_mode != "pairwise"
                or self.block_prototype_mode != "mean"
            ):
                raise ValueError(
                    "geo_kv.incremental_decode_scoring currently requires "
                    "pairwise mean-prototype eviction_policy='v_redundancy'"
                )
            if (
                self.value_blend_beta
                or self.value_norm_protect_quantile is not None
                or self.query_relevance_enabled
            ):
                raise ValueError(
                    "geo_kv.incremental_decode_scoring is incompatible with "
                    "value-norm and query-relevance refinements"
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


def resolve_indices(spec: str, n: int, *, strict: bool = False) -> list[int]:
    """Resolve a validated index spec into concrete indices in ``[0, n)``.

    Args:
        spec: "all", "last", or a comma-separated list of non-negative ints.
        n: The size of the dimension being sampled.
        strict: Raise when an explicit index is outside ``[0, n)`` instead of
            silently dropping it. Runtime scorers should use this so a layer
            subset calibrated for a different architecture cannot quietly turn
            into a smaller subset.

    Returns:
        Sorted, de-duplicated indices. Out-of-range entries are dropped unless
        ``strict`` is true.
    """
    if n <= 0:
        return []
    if spec == "all":
        return list(range(n))
    if spec == "last":
        return [n - 1]
    idxs = {int(p.strip()) for p in spec.split(",") if p.strip() != ""}
    invalid = sorted(i for i in idxs if not 0 <= i < n)
    if strict and invalid:
        raise ValueError(
            f"geo_kv index spec {spec!r} contains out-of-range indices "
            f"{invalid} for a dimension of size {n}"
        )
    return sorted(i for i in idxs if 0 <= i < n)
