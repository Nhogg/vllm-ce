#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GeoKV capacity-band decode-eviction smoke test (in-engine, vLLM).

The capacity band (``decode_evict_budget`` = ``C`` with ``decode_evict_watermark``)
lets a request's KV cache fill to ``C`` blocks, then evicts the most-redundant
interior blocks (V-redundancy) down to ``floor(watermark * C)``. This is enforced
at end of prefill AND during decode as the request regrows to ``C`` (hysteresis).
Under ``physical_reclaim`` the evicted blocks are compacted and their GPU memory
returned to the pool.

This driver is the end-to-end acceptance smoke for that scheme. Unlike the payoff
benchmark (``benchmark_kv_reclaim_capacity_vllm.py``, which measures concurrency),
this runs a SINGLE long-generation request per prompt and checks the mechanism
actually fires and stays correct:

  1. **Decode eviction fires.** The engine's live ``EvictionPolicy`` counter
     ``_num_decode_evictions`` must be > 0 (read via ``collective_rpc``): decode
     genuinely reached capacity and evicted, not just prefill.
  2. **Occupancy stays capped.** Steady-state blocks-in-use for a request must sit
     near ``C`` (within a block of C, not growing unbounded with generation).
  3. **No engine assert trips.** A completed run implies the M2c row-size assert
     (``model_runner`` compaction consistency) held every step; any desync raises.
  4. **Coherent output.** Generated text is non-empty and not degenerate (checked
     via QA-F1 sanity + a repetition guard), so eviction did not wreck the model.
  5. **Inert control.** ``--capacity huge`` (never reached) must be TOKEN-IDENTICAL
     to a no-geo baseline: the band path is a no-op below capacity.

Runs the engine IN-PROCESS (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) so a background
thread can poll the block pool and ``collective_rpc`` can read the policy counter.
Each engine config runs in its own subprocess (``--single``) for a clean GPU.

Run inside a PBS GPU job (torch is broken on the login node): see
``benchmarks/geo_capacity_band_vllm.pbs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

# Reuse prompt construction, the block-pool/scheduler resolvers, the occupancy
# sampler, and the scoring metric (single source of truth across geo drivers).
# score_prediction dispatches per task (qa_f1 for QA, ROUGE-L for summarization).
from benchmark_kv_eviction_accuracy import DATASET2METRIC, score_prediction
from benchmark_kv_eviction_accuracy_vllm import build_prompts
from benchmark_kv_reclaim_capacity_vllm import (
    _occupancy_sampler,
    _resolve_block_pool,
    _resolve_scheduler,
)

# A capacity so large no real prompt+generation reaches it => the band path is a
# guaranteed no-op (used by the inert control run).
HUGE_CAPACITY = 1_000_000

R2R_VARIANTS = {
    "v1": (False, None),
    "floor": (True, None),
    "dispersion": (False, 3.0),
    "v2": (True, 3.0),
}


def _sweep_variants(args, policy: str) -> list[str]:
    """Return R2R variants for a policy, preserving legacy sweep behavior."""
    if policy != "v_redundancy" or not args.r2r_variants:
        return [""]
    variants = [name.strip() for name in args.r2r_variants.split(",")]
    if not variants or any(name not in R2R_VARIANTS for name in variants):
        raise ValueError(f"--r2r-variants must use {sorted(R2R_VARIANTS)}")
    if len(set(variants)) != len(variants):
        raise ValueError("--r2r-variants cannot contain duplicates")
    if args.candidate_expansion_factor is None:
        raise ValueError("--r2r-variants requires --candidate-expansion-factor")
    if args.r2r_persistent_floor or args.r2r_dispersion_kappa is not None:
        raise ValueError("--r2r-variants cannot combine with direct v2 flags")
    return variants


def _read_decode_evictions(llm) -> int:
    """Read the live EvictionPolicy's decode-eviction event counter.

    Uses ``collective_rpc`` with a callable, which ``run_method`` invokes as
    ``fn(worker)`` (partial-binds the worker). Returns -1 if the policy is absent
    (e.g. a no-geo baseline) so callers can distinguish "not configured" from 0.
    """

    def _get(worker) -> int:
        policy = getattr(worker.model_runner, "geo_eviction_policy", None)
        if policy is None:
            return -1
        return int(getattr(policy, "_num_decode_evictions", 0))

    try:
        core = llm.llm_engine.engine_core.engine_core
        results = core.model_executor.collective_rpc(_get)
        return int(results[0])
    except Exception as e:  # pragma: no cover - config guard
        print(f"[capacity-band] WARN could not read decode-eviction counter: {e}")
        return -1


def _max_repetition_run(text: str) -> int:
    """Longest run of a single repeated whitespace token (degeneracy guard).

    A collapsed model spews the same token forever; a healthy one does not. This
    is a cheap proxy that does not need references.
    """
    toks = text.split()
    if not toks:
        return 0
    best = run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def _throughput_stats(
    outs, out_lens: list[int], wall: float
) -> tuple[float, float, float]:
    """Aggregate decode throughput and per-token latency for a batch.

    Uses each RequestOutput's engine-side ``metrics`` (RequestStateStats) when
    present -- ``first_token_ts``/``last_token_ts`` give honest server timings
    that exclude client overhead. TPOT (time-per-output-token) is the mean over
    requests of decode_time / (gen_tokens - 1); TTFT is the mean first-token
    latency. Throughput is total generated tokens / batch wall time (the batch
    ran concurrently, so wall time is the right denominator for tok/s).

    Falls back to wall-clock throughput and 0.0 latencies if metrics are absent
    (e.g. an engine build that does not populate them), so the columns are always
    present and numeric.

    Args:
        outs: RequestOutput list from ``llm.generate``.
        out_lens: Generated token count per request (index-aligned with outs).
        wall: Batch wall time in seconds.

    Returns:
        (throughput_tok_s, mean_tpot_ms, mean_ttft_ms), all rounded.
    """
    total_gen = sum(out_lens)
    tput = round(total_gen / wall, 1) if wall > 0 else 0.0

    tpots: list[float] = []
    ttfts: list[float] = []
    for o, n in zip(outs, out_lens):
        m = getattr(o, "metrics", None)
        if m is None:
            continue
        ttft = getattr(m, "first_token_latency", None)
        first_ts = getattr(m, "first_token_ts", None)
        last_ts = getattr(m, "last_token_ts", None)
        if ttft is not None and ttft > 0:
            ttfts.append(ttft * 1000.0)
        # decode span covers tokens 2..n; per-token = span / (n - 1).
        if first_ts and last_ts and n > 1 and last_ts > first_ts:
            tpots.append((last_ts - first_ts) / (n - 1) * 1000.0)

    mean_tpot = round(statistics.fmean(tpots), 3) if tpots else 0.0
    mean_ttft = round(statistics.fmean(ttfts), 3) if ttfts else 0.0
    return tput, mean_tpot, mean_ttft


# ---------------------------------------------------------------------------
# Single engine config (one subprocess per config; clean GPU memory)
# ---------------------------------------------------------------------------
def run_single(args) -> None:
    # geo hooks are V2-runner only; force it before importing vllm. In-process
    # engine so the sampler thread + collective_rpc can reach engine internals.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from benchmark_kv_eviction_ablation_vllm import (
        _answer_logprob_metrics,
        _confidence_prompt,
        _parse_verbalized_confidence,
    )
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(tokenizer, args)
    if not prompts:
        raise RuntimeError("no prompts built; check --tasks/--num-prompts")

    geo = args.mode != "baseline"
    capacity = args.capacity
    kwargs: dict[str, Any] = dict(
        model=args.model,
        enforce_eager=True,  # required by the geo_kv eviction guard
        attention_backend="FLEX_ATTENTION",
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_mem_util,
        async_scheduling=False,  # physical_reclaim requires the sync scheduler
        enable_prefix_caching=False,  # physical_reclaim is incompatible
        disable_log_stats=False,
    )
    if geo:
        sampled_layers = getattr(args, "score_sampled_layers", None) or "all"
        geo_cfg = {
            "experiment_mode": "geo_uniform",
            "eviction_policy": args.eviction_policy,
            "eviction_seed": args.eviction_seed,
            "score_sampled_layers": sampled_layers,
            "block_score_aggregation": "mean",
            "block_prototype_mode": getattr(args, "block_prototype_mode", "mean"),
            "value_l2_block_reduction": getattr(
                args, "value_l2_block_reduction", "sum"
            ),
            "warmup_pages": args.warmup_pages,
            "decode_evict_budget": capacity,
            "decode_evict_watermark": args.watermark,
            "decode_evict_interval": args.interval,
            "physical_reclaim": True,
        }
        if args.candidate_expansion_factor is not None:
            geo_cfg.update(
                {
                    "candidate_expansion_factor": args.candidate_expansion_factor,
                    "r2r_cover_depth": args.r2r_cover_depth,
                    "r2r_query_aggregation": args.r2r_query_aggregation,
                    "r2r_relevance_signal": args.r2r_relevance_signal,
                }
            )
            if getattr(args, "r2r_persistent_floor", False):
                geo_cfg["r2r_persistent_floor"] = True
            if getattr(args, "r2r_dispersion_kappa", None) is not None:
                geo_cfg["r2r_dispersion_kappa"] = args.r2r_dispersion_kappa
        # Tail-protect (query-aware recency floor); omit when unset so baseline
        # runs stay byte-identical to the pinned v_redundancy config.
        if getattr(args, "query_tail_protect_frac", None):
            geo_cfg["query_tail_protect_frac"] = args.query_tail_protect_frac
        # Positional cosh (sech-bump) re-weight of v_redundancy droppability;
        # omit when unset/0 so baseline runs stay byte-identical.
        if getattr(args, "positional_cosh_alpha", None):
            geo_cfg["positional_cosh_alpha"] = args.positional_cosh_alpha
        # Greedy (iterative) redundancy peel + value-norm blend; omit when unset
        # so baseline runs stay byte-identical to the pinned pairwise scorer.
        if getattr(args, "redundancy_mode", None):
            geo_cfg["redundancy_mode"] = args.redundancy_mode
        if getattr(args, "value_blend_beta", None):
            geo_cfg["value_blend_beta"] = args.value_blend_beta
        if getattr(args, "value_norm_protect_quantile", None) is not None:
            geo_cfg["value_norm_protect_quantile"] = (
                args.value_norm_protect_quantile
            )
        if getattr(args, "query_relevance_protect_quantile", None) is not None:
            geo_cfg["query_relevance_protect_quantile"] = (
                args.query_relevance_protect_quantile
            )
            geo_cfg["query_tail_tokens"] = args.query_tail_tokens
        if getattr(args, "query_alignment_weight", None):
            geo_cfg["query_alignment_weight"] = args.query_alignment_weight
            geo_cfg["query_tail_tokens"] = args.query_tail_tokens
        if getattr(args, "enable_query_tiebreak", False):
            geo_cfg["enable_query_tiebreak"] = True
            geo_cfg["query_tail_tokens"] = args.query_tail_tokens
        if getattr(args, "incremental_decode_scoring", False):
            geo_cfg["incremental_decode_scoring"] = True
        if getattr(args, "enable_tracing", False):
            geo_cfg["enable_tracing"] = True
            geo_cfg["log_path"] = args.trace_log_path or args.dump_json
        kwargs["additional_config"] = {"geo_kv": geo_cfg}
    print(
        f"[capacity-band] building engine: mode={args.mode} capacity={capacity} "
        f"watermark={args.watermark} policy={args.eviction_policy} geo={geo}"
    )
    llm = LLM(**kwargs)
    block_pool = _resolve_block_pool(llm)
    scheduler = _resolve_scheduler(llm)
    num_gpu_blocks = int(block_pool.num_gpu_blocks)

    reqs = [TokensPrompt(prompt_token_ids=p["token_ids"]) for p in prompts]
    confidence_topk = int(getattr(args, "confidence_logprobs_topk", 0) or 0)
    requested_logprobs = (
        args.confidence_logprobs if args.confidence_eval else confidence_topk
    )
    if requested_logprobs and not 1 <= requested_logprobs <= 20:
        raise ValueError("requested logprobs must be in [1, 20]")
    if args.natural_gen:
        # Accuracy mode: respect EOS and use each task's natural max_gen so qa_f1
        # reflects real answer quality. Eviction still fires because the long
        # prompt already exceeds C at end-of-prefill (fill-to-C -> evict-to-wm).
        sps = [
            SamplingParams(
                temperature=0.0,
                max_tokens=p["max_gen"],
                logprobs=requested_logprobs or None,
            )
            for p in prompts
        ]
    else:
        # Smoke mode: force a long deterministic generation (ignore EOS) so decode
        # reliably regrows to C and the decode-time eviction path is exercised.
        sps = [
            SamplingParams(
                temperature=0.0,
                max_tokens=args.max_gen,
                ignore_eos=True,
                logprobs=requested_logprobs or None,
            )
            for _ in prompts
        ]

    ts: list[float] = []
    used: list[int] = []
    running: list[int] = []
    waiting: list[int] = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=_occupancy_sampler,
        args=(block_pool, scheduler, stop, ts, used, running, waiting, args.sample_ms),
        daemon=True,
    )
    sampler.start()
    t0 = time.perf_counter()
    outs = llm.generate(reqs, sps)
    wall = time.perf_counter() - t0
    stop.set()
    sampler.join(timeout=2.0)

    decode_evictions = _read_decode_evictions(llm)
    texts = [o.outputs[0].text for o in outs]
    out_lens = [len(o.outputs[0].token_ids) for o in outs]
    token_ids = [list(o.outputs[0].token_ids) for o in outs]
    answer_logprobs: list[dict[str, float]] | None = None
    verbalized_confidence: list[float | None] | None = None
    verbalized_confidence_raw: list[str] | None = None
    confidence_wall = 0.0
    if args.confidence_eval:
        answer_logprobs = [
            _answer_logprob_metrics(output, prompt["prompt_id"])
            for prompt, output in zip(prompts, outs)
        ]
        confidence_ids = [
            _confidence_prompt(tokenizer, prompt, answer)
            for prompt, answer in zip(prompts, texts)
        ]
        if any(
            len(ids) + args.confidence_max_tokens > args.max_model_len
            for ids in confidence_ids
        ):
            raise RuntimeError(
                "confidence prompt exceeds max_model_len; raise the model limit"
            )
        confidence_started = time.perf_counter()
        confidence_outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=ids) for ids in confidence_ids],
            SamplingParams(
                temperature=0.0,
                max_tokens=args.confidence_max_tokens,
            ),
        )
        confidence_wall = time.perf_counter() - confidence_started
        verbalized_confidence_raw = [
            output.outputs[0].text.strip() for output in confidence_outputs
        ]
        verbalized_confidence = [
            _parse_verbalized_confidence(text)
            for text in verbalized_confidence_raw
        ]
    decode_logprobs: list[list[dict[str, float]]] | None = None
    if confidence_topk:
        decode_logprobs = []
        for prompt, out in zip(prompts, outs):
            steps = out.outputs[0].logprobs
            if steps is None or len(steps) != len(out.outputs[0].token_ids):
                raise RuntimeError(
                    "missing or misaligned decode logprobs for "
                    f"{prompt['prompt_id']}"
                )
            decode_logprobs.append(
                [
                    {
                        str(token_id): float(item.logprob)
                        for token_id, item in step.items()
                    }
                    for step in steps
                ]
            )
    # Task-aware scoring: qa_f1 for QA tasks, ROUGE-L for summarization. Keyed on
    # each prompt's own task so mixed --tasks runs score each prompt correctly.
    scores = [
        score_prediction(p["task"], t, p["answers"])
        for p, t in zip(prompts, texts)
    ]
    # The task's metric name (uniform when --tasks is a single task; "mixed"
    # otherwise) is recorded so downstream plots label the y-axis correctly.
    metric_names = {DATASET2METRIC.get(p["task"], "qa_f1") for p in prompts}
    metric_name = metric_names.pop() if len(metric_names) == 1 else "mixed"
    max_rep = max((_max_repetition_run(t) for t in texts), default=0)

    # Throughput + per-token latency from the engine's own RequestStateStats
    # (honest server-side timings, not wall-clock guesses). TPOT = decode time /
    # (gen_tokens - 1), averaged over requests; TTFT = first-token latency.
    tput, mean_tpot_ms, mean_ttft_ms = _throughput_stats(outs, out_lens, wall)

    peak = max(used) if used else 0
    tail = used[len(used) // 2 :] if used else []
    steady = int(statistics.median(tail)) if tail else 0

    result = {
        "run_id": args.run_id,
        "mode": args.mode,
        "task": args.tasks,
        "model": os.path.basename(args.model.rstrip("/")),
        "capacity": capacity,
        "watermark": args.watermark,
        "eviction_policy": args.eviction_policy if geo else "none",
        # Phase-3 method knobs, recorded for provenance so a v_redundancy+cosh
        # run is distinguishable on disk from pinned v_redundancy (both share
        # eviction_policy="v_redundancy"). None when off.
        "positional_cosh_alpha": (
            args.positional_cosh_alpha if geo else None
        ),
        "query_tail_protect_frac": (
            args.query_tail_protect_frac if geo else None
        ),
        "redundancy_mode": (
            getattr(args, "redundancy_mode", None) if geo else None
        ),
        "value_blend_beta": (
            getattr(args, "value_blend_beta", None) if geo else None
        ),
        "value_norm_protect_quantile": (
            getattr(args, "value_norm_protect_quantile", None) if geo else None
        ),
        "query_relevance_protect_quantile": (
            getattr(args, "query_relevance_protect_quantile", None)
            if geo
            else None
        ),
        "block_prototype_mode": (
            getattr(args, "block_prototype_mode", "mean") if geo else None
        ),
        "value_l2_block_reduction": (
            getattr(args, "value_l2_block_reduction", "sum") if geo else None
        ),
        "score_sampled_layers": (
            getattr(args, "score_sampled_layers", None) or "all"
            if geo
            else None
        ),
        "query_alignment_weight": (
            getattr(args, "query_alignment_weight", None) if geo else None
        ),
        "query_tail_tokens": (
            getattr(args, "query_tail_tokens", None) if geo else None
        ),
        "incremental_decode_scoring": bool(
            geo and getattr(args, "incremental_decode_scoring", False)
        ),
        "candidate_expansion_factor": (
            args.candidate_expansion_factor if geo else None
        ),
        "r2r_cover_depth": args.r2r_cover_depth if geo else None,
        "r2r_query_aggregation": args.r2r_query_aggregation if geo else None,
        "r2r_relevance_signal": args.r2r_relevance_signal if geo else None,
        "r2r_persistent_floor": (
            bool(getattr(args, "r2r_persistent_floor", False)) if geo else None
        ),
        "r2r_dispersion_kappa": (
            getattr(args, "r2r_dispersion_kappa", None) if geo else None
        ),
        "block_size": args.block_size,
        "low_watermark_blocks": (
            max(1, min(int(args.watermark * capacity), capacity - 1))
            if geo
            else None
        ),
        "gpu_mem_util": args.gpu_mem_util,
        "num_prompts": len(prompts),
        "max_gen": args.max_gen,
        "num_gpu_blocks": num_gpu_blocks,
        "peak_blocks": peak,
        "steady_blocks": steady,
        "decode_evictions": decode_evictions,
        "mean_out_len": round(statistics.fmean(out_lens), 1) if out_lens else 0,
        "min_out_len": min(out_lens) if out_lens else 0,
        # qa_f1/f1s hold the task's canonical metric (qa_f1 for QA, ROUGE-L for
        # summarization); metric_name disambiguates. Column names kept for the
        # sweep aggregator + plot scripts that already read them.
        "qa_f1": round(statistics.fmean(scores), 4) if scores else 0.0,
        "metric_name": metric_name,
        "throughput_tok_s": tput,
        "mean_tpot_ms": mean_tpot_ms,
        "mean_ttft_ms": mean_ttft_ms,
        "max_repetition_run": max_rep,
        "wall_s": round(wall, 2),
        # Per-prompt scores + ids let the sweep aggregate and bootstrap CIs.
        "f1s": [round(x, 4) for x in scores],
        "prompt_ids": [p["prompt_id"] for p in prompts],
        # Token ids drive the inert-control bit-identity check.
        "token_ids": token_ids,
        "prompt_blocks": [p["num_blocks"] for p in prompts],
    }
    if decode_logprobs is not None:
        result["confidence_logprobs_topk"] = confidence_topk
        result["decode_logprobs"] = decode_logprobs
    if args.confidence_eval:
        assert answer_logprobs is not None
        assert verbalized_confidence is not None
        assert verbalized_confidence_raw is not None
        parsed = [value for value in verbalized_confidence if value is not None]
        result.update(
            {
                "confidence_eval": True,
                "confidence_logprobs": args.confidence_logprobs,
                "answer_logprob_metrics": answer_logprobs,
                "verbalized_confidence": verbalized_confidence,
                "verbalized_confidence_raw": verbalized_confidence_raw,
                "verbalized_confidence_parse_rate": (
                    len(parsed) / len(verbalized_confidence)
                    if verbalized_confidence
                    else 0.0
                ),
                "mean_answer_logprob": statistics.fmean(
                    item["mean_logprob"] for item in answer_logprobs
                ),
                "mean_answer_probability": statistics.fmean(
                    item["geometric_mean_probability"]
                    for item in answer_logprobs
                ),
                "mean_verbalized_confidence": (
                    statistics.fmean(parsed) if parsed else None
                ),
                "confidence_pass_wall_s": round(confidence_wall, 3),
            }
        )
    if args.dump_json:
        with open(args.dump_json, "w") as f:
            json.dump(result, f)
    # Human-readable summary (token_ids omitted).
    summary = {
        k: v
        for k, v in result.items()
        if k not in ("token_ids", "decode_logprobs")
    }
    print("[capacity-band] result:", json.dumps(summary, indent=2))
    if getattr(args, "enable_tracing", False):
        try:
            llm.llm_engine.engine_core.shutdown()
            print("[capacity-band] engine shut down; scorer profile flushed")
        except Exception as e:  # best-effort: never fail a completed benchmark
            print(f"[capacity-band] warning: engine shutdown failed: {e}")


# ---------------------------------------------------------------------------
# Orchestration: spawn each engine config as a subprocess, then assert
# ---------------------------------------------------------------------------
def _spawn(
    args,
    mode: str,
    capacity: int,
    dump: str,
    extra_flags: list[str] | None = None,
    label: str | None = None,
    watermark: float | None = None,
    eviction_policy: str | None = None,
) -> dict:
    wm = args.watermark if watermark is None else watermark
    policy = eviction_policy or args.eviction_policy
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single",
        "--mode",
        mode,
        "--capacity",
        str(capacity),
        "--watermark",
        str(wm),
        "--interval",
        str(args.interval),
        "--warmup-pages",
        str(args.warmup_pages),
        "--eviction-policy",
        policy,
        "--eviction-seed",
        str(args.eviction_seed),
        "--model",
        args.model,
        "--longbench-dir",
        args.longbench_dir,
        "--tasks",
        args.tasks,
        "--num-prompts",
        str(args.num_prompts),
        "--max-prompt-len",
        str(args.max_prompt_len),
        "--reserve",
        str(args.reserve),
        "--max-model-len",
        str(args.max_model_len),
        "--max-gen",
        str(args.max_gen),
        "--gpu-mem-util",
        str(args.gpu_mem_util),
        "--block-size",
        str(args.block_size),
        "--run-id",
        f"{args.run_id}_{label or mode}",
        "--dump-json",
        dump,
    ]
    if getattr(args, "query_tail_protect_frac", None):
        cmd += ["--query-tail-protect-frac", str(args.query_tail_protect_frac)]
    if getattr(args, "positional_cosh_alpha", None):
        cmd += ["--positional-cosh-alpha", str(args.positional_cosh_alpha)]
    if getattr(args, "redundancy_mode", None):
        cmd += ["--redundancy-mode", str(args.redundancy_mode)]
    if getattr(args, "value_blend_beta", None):
        cmd += ["--value-blend-beta", str(args.value_blend_beta)]
    if getattr(args, "value_norm_protect_quantile", None) is not None:
        cmd += [
            "--value-norm-protect-quantile",
            str(args.value_norm_protect_quantile),
        ]
    if getattr(args, "query_relevance_protect_quantile", None) is not None:
        cmd += [
            "--query-relevance-protect-quantile",
            str(args.query_relevance_protect_quantile),
            "--query-tail-tokens",
            str(args.query_tail_tokens),
        ]
    if getattr(args, "block_prototype_mode", "mean") != "mean":
        cmd += ["--block-prototype-mode", args.block_prototype_mode]
    if getattr(args, "value_l2_block_reduction", "sum") != "sum":
        cmd += ["--value-l2-block-reduction", args.value_l2_block_reduction]
    if getattr(args, "score_sampled_layers", None):
        cmd += ["--score-sampled-layers", args.score_sampled_layers]
    if getattr(args, "query_alignment_weight", None):
        cmd += [
            "--query-alignment-weight",
            str(args.query_alignment_weight),
            "--query-tail-tokens",
            str(args.query_tail_tokens),
        ]
    if getattr(args, "enable_query_tiebreak", False):
        cmd += ["--enable-query-tiebreak"]
    if getattr(args, "incremental_decode_scoring", False):
        cmd += ["--incremental-decode-scoring"]
    if (
        policy == "v_redundancy"
        and getattr(args, "candidate_expansion_factor", None) is not None
    ):
        cmd += [
            "--candidate-expansion-factor",
            str(args.candidate_expansion_factor),
            "--r2r-cover-depth",
            str(args.r2r_cover_depth),
            "--r2r-query-aggregation",
            args.r2r_query_aggregation,
            "--r2r-relevance-signal",
            args.r2r_relevance_signal,
        ]
        if getattr(args, "r2r_persistent_floor", False):
            cmd += ["--r2r-persistent-floor"]
        if getattr(args, "r2r_dispersion_kappa", None) is not None:
            cmd += ["--r2r-dispersion-kappa", str(args.r2r_dispersion_kappa)]
    if getattr(args, "confidence_eval", False):
        cmd += [
            "--confidence-eval",
            "--confidence-logprobs",
            str(args.confidence_logprobs),
            "--confidence-max-tokens",
            str(args.confidence_max_tokens),
        ]
    if getattr(args, "enable_tracing", False):
        cmd += ["--enable-tracing"]
    if getattr(args, "trace_log_path", None):
        cmd += ["--trace-log-path", args.trace_log_path]
    if extra_flags:
        cmd += extra_flags
    print(
        f"[capacity-band] spawning: {label or mode} policy={policy} "
        f"capacity={capacity} watermark={wm}"
    )
    subprocess.run(cmd, check=True)
    with open(dump) as f:
        return json.load(f)


def run_smoke(args) -> None:
    outdir = args.output_dir or "."
    os.makedirs(outdir, exist_ok=True)

    band = _spawn(
        args, "band", args.capacity, os.path.join(outdir, f"{args.run_id}_band.json")
    )
    inert = _spawn(
        args, "band", HUGE_CAPACITY, os.path.join(outdir, f"{args.run_id}_inert.json")
    )
    base = _spawn(
        args,
        "baseline",
        HUGE_CAPACITY,
        os.path.join(outdir, f"{args.run_id}_base.json"),
    )

    failures: list[str] = []

    # (1) Decode eviction fired.
    if band["decode_evictions"] <= 0:
        failures.append(
            f"decode_evictions={band['decode_evictions']} (expected > 0): decode "
            "never reached capacity; raise --max-gen or lower --capacity"
        )

    # (2) Physical reclaim actually caps memory. The inert run (capacity=huge,
    # same prompts + generation) never evicts, so its peak grows to prompt +
    # max_gen; the band run must use strictly LESS steady memory. Comparing the
    # two runs (rather than an absolute block-count formula) is robust to
    # scheduler/allocator details.
    if band["steady_blocks"] >= inert["steady_blocks"]:
        failures.append(
            f"steady_blocks band={band['steady_blocks']} >= inert="
            f"{inert['steady_blocks']}: physical reclaim did not lower occupancy"
        )

    # (4) Coherent output (non-empty, not collapsed).
    if band["min_out_len"] <= 0:
        failures.append("band produced an empty output")
    if band["max_repetition_run"] >= args.max_rep_threshold:
        failures.append(
            f"band max_repetition_run={band['max_repetition_run']} >= "
            f"{args.max_rep_threshold}: output looks degenerate"
        )

    # (5) Inert control == baseline, token-for-token.
    inert_toks = inert["token_ids"]
    base_toks = base["token_ids"]
    if inert_toks != base_toks:
        n_div = sum(1 for a, b in zip(inert_toks, base_toks) if a != b)
        failures.append(
            f"inert(capacity=huge) diverged from baseline in {n_div}/"
            f"{len(base_toks)} prompts: the band path is NOT a no-op below capacity"
        )

    print("\n" + "=" * 60)
    print(f"CAPACITY-BAND SMOKE  run_id={args.run_id}")
    print(f"  capacity C            = {args.capacity} blocks")
    print(f"  watermark             = {args.watermark} "
          f"(target {int(args.watermark * args.capacity)} blocks)")
    print(f"  decode_evictions      = {band['decode_evictions']}")
    print(f"  steady_blocks (band)  = {band['steady_blocks']} / "
          f"{band['num_gpu_blocks']} gpu")
    print(f"  peak_blocks (band)    = {band['peak_blocks']}")
    print(f"  mean_out_len          = {band['mean_out_len']}")
    print(f"  qa_f1 (band)          = {band['qa_f1']}")
    print(f"  max_repetition_run    = {band['max_repetition_run']}")
    print(f"  inert==baseline       = {inert_toks == base_toks}")
    print("=" * 60)

    if failures:
        print("\nSMOKE FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nSMOKE PASSED: capacity-band decode eviction fires, caps memory, "
          "stays coherent, and is inert below capacity.")


def _paired_delta_ci(
    band_f1: list[float],
    base_f1: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for the paired mean(band - baseline) F1 delta.

    Prompts are paired (same prompt run at both settings), so we resample prompt
    indices with replacement and recompute the mean per-prompt delta. Returns
    (mean_delta, ci_lo, ci_hi). Pure-stdlib (no numpy dependency at import time).
    """
    import random

    n = min(len(band_f1), len(base_f1))
    if n == 0:
        return 0.0, 0.0, 0.0
    deltas = [band_f1[i] - base_f1[i] for i in range(n)]
    mean_delta = statistics.fmean(deltas)
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(statistics.fmean(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return mean_delta, lo, hi


def _align_f1(
    run: dict, base_by_id: dict[str, float]
) -> tuple[list[float], list[float]]:
    """Pair a run's per-prompt F1s with the baseline by prompt_id."""
    band, base = [], []
    for pid, f1 in zip(run["prompt_ids"], run["f1s"]):
        if pid in base_by_id:
            band.append(f1)
            base.append(base_by_id[pid])
    return band, base


# ---------------------------------------------------------------------------
# Accuracy sweep: qa_f1 vs decode_evict_budget (capacity), with paired CIs.
# ---------------------------------------------------------------------------
def run_sweep(args) -> None:
    outdir = args.output_dir or os.path.join("results", "capband_sweep", args.run_id)
    os.makedirs(outdir, exist_ok=True)
    nat = ["--natural-gen"]  # accuracy mode: respect EOS + per-task max_gen

    # Two sweep axes share one code path. "capacity": fix watermark, vary C.
    # "watermark": fix C, vary the low-watermark ratio (evict-to depth).
    points: list  # block counts (capacity axis) or watermark ratios
    if args.sweep_axis == "watermark":
        axis_name = "watermark"
        points = [float(x) for x in args.watermarks.split(",") if x]
    else:
        axis_name = "capacity"
        raw = [int(x) for x in args.capacities.split(",") if x]
        # --budget-unit tokens: interpret --capacities as KV token budgets and
        # convert to block capacity C = ceil(tokens / block_size). Lets a sweep
        # be specified in the memory unit papers report (tokens), model-agnostic.
        if args.budget_unit == "tokens":
            bs = args.block_size
            points = [max(1, (t + bs - 1) // bs) for t in raw]
            print(
                f"[capacity-band] budget-unit=tokens: {raw} tokens -> {points} "
                f"blocks (block_size={bs})"
            )
        else:
            points = raw

    # Baseline: no geo, natural generation -> reference F1 per prompt.
    base = _spawn(
        args,
        "baseline",
        HUGE_CAPACITY,
        os.path.join(outdir, f"{args.run_id}_base.json"),
        extra_flags=nat,
        label="base",
    )
    base_by_id = dict(zip(base["prompt_ids"], base["f1s"]))
    base_mean_f1 = base["qa_f1"]

    policies = [
        policy.strip()
        for policy in args.comparison_policies.split(",")
        if policy.strip()
    ]
    supported = {"v_redundancy", "paged_eviction", "recency"}
    if not policies or not set(policies) <= supported:
        raise ValueError(
            f"--comparison-policies must be a subset of {sorted(supported)}"
        )

    rows: list[dict] = []
    arms = [
        (policy, variant)
        for policy in policies
        for variant in _sweep_variants(args, policy)
    ]
    for policy, variant in arms:
        for i, pt in enumerate(points):
            if axis_name == "watermark":
                cap, wm, tag = args.capacity, float(pt), f"wm{pt}"
                budget_tokens = cap * args.block_size
            else:
                cap, wm, tag = int(pt), args.watermark, f"cap{pt}"
                budget_tokens = (
                    int(args.capacities.split(",")[i])
                    if args.budget_unit == "tokens"
                    else cap * args.block_size
                )
            run_tag = f"{policy}_{variant}_{tag}" if variant else f"{policy}_{tag}"
            variant_flags: list[str] = []
            if variant:
                floor, kappa = R2R_VARIANTS[variant]
                if floor:
                    variant_flags.append("--r2r-persistent-floor")
                if kappa is not None:
                    variant_flags += ["--r2r-dispersion-kappa", str(kappa)]
            run = _spawn(
                args,
                "band",
                cap,
                os.path.join(outdir, f"{args.run_id}_{run_tag}.json"),
                extra_flags=nat + variant_flags,
                label=run_tag,
                watermark=wm,
                eviction_policy=policy,
            )
            band_f1, base_f1 = _align_f1(run, base_by_id)
            mean_delta, lo, hi = _paired_delta_ci(
                band_f1, base_f1, seed=args.seed
            )
            rows.append(
                {
                    "model": os.path.basename(args.model.rstrip("/")),
                    "task": args.tasks,
                    "eviction_policy": policy,
                "r2r_variant": variant or None,
                "r2r_persistent_floor": run.get("r2r_persistent_floor"),
                "r2r_dispersion_kappa": run.get("r2r_dispersion_kappa"),
                "redundancy_mode": getattr(args, "redundancy_mode", None),
                "value_blend_beta": getattr(args, "value_blend_beta", None),
                "value_norm_protect_quantile": getattr(
                    args, "value_norm_protect_quantile", None
                ),
                "block_prototype_mode": getattr(args, "block_prototype_mode", "mean"),
                "score_sampled_layers": (
                    getattr(args, "score_sampled_layers", None) or "all"
                ),
                "query_alignment_weight": getattr(
                    args, "query_alignment_weight", None
                ),
                "query_tail_tokens": getattr(args, "query_tail_tokens", None),
                "candidate_expansion_factor": run.get(
                    "candidate_expansion_factor"
                ),
                "r2r_cover_depth": run.get("r2r_cover_depth"),
                "r2r_query_aggregation": run.get("r2r_query_aggregation"),
                "r2r_relevance_signal": run.get("r2r_relevance_signal"),
                "block_size": args.block_size,
                "capacity": cap,
                "budget_tokens": budget_tokens,
                "watermark": wm,
                "n": len(band_f1),
                "qa_f1": run["qa_f1"],
                "metric_name": run.get("metric_name", "qa_f1"),
                "qa_f1_delta": round(mean_delta, 4),
                "delta_ci_lo": round(lo, 4),
                "delta_ci_hi": round(hi, 4),
                "throughput_tok_s": run.get("throughput_tok_s", 0.0),
                "mean_tpot_ms": run.get("mean_tpot_ms", 0.0),
                "mean_ttft_ms": run.get("mean_ttft_ms", 0.0),
                "decode_evictions": run["decode_evictions"],
                "steady_blocks": run["steady_blocks"],
                "peak_blocks": run["peak_blocks"],
                "mean_out_len": run["mean_out_len"],
                "mean_answer_logprob": run.get("mean_answer_logprob"),
                "mean_answer_probability": run.get("mean_answer_probability"),
                "mean_verbalized_confidence": run.get(
                    "mean_verbalized_confidence"
                ),
                "verbalized_confidence_parse_rate": run.get(
                    "verbalized_confidence_parse_rate"
                ),
                    "confidence_pass_wall_s": run.get("confidence_pass_wall_s"),
                }
            )

    csv_path = os.path.join(outdir, "capband_accuracy.csv")
    fields = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    axis_col = "watermark" if axis_name == "watermark" else "capacity"
    fixed = (
        f"capacity={args.capacity}" if axis_name == "watermark"
        else f"watermark={args.watermark}"
    )
    metric = rows[0].get("metric_name", "qa_f1") if rows else "qa_f1"
    print("\n" + "=" * 78)
    print(f"CAPACITY-BAND ACCURACY SWEEP ({axis_name})  run_id={args.run_id}")
    print(f"  task={args.tasks}  n_prompts={base['num_prompts']}  "
          f"metric={metric}  baseline={base_mean_f1}")
    print(f"  {fixed}  policies={','.join(policies)}  "
          f"gpu_blocks={base['num_gpu_blocks']}")
    print("-" * 78)
    print(f"  {'policy':>16} {axis_col:>8} {metric:>8} {'Δ':>8} {'95% CI':>18} "
          f"{'tok/s':>8} {'tpot_ms':>8} {'evicts':>7} {'steady':>7}")
    for r in rows:
        ci = f"[{r['delta_ci_lo']:+.3f},{r['delta_ci_hi']:+.3f}]"
        print(f"  {r['eviction_policy']:>16} {r[axis_col]:>8} "
              f"{r['qa_f1']:>8.3f} {r['qa_f1_delta']:>+8.3f} "
              f"{ci:>18} {r['throughput_tok_s']:>8.1f} {r['mean_tpot_ms']:>8.2f} "
              f"{r['decode_evictions']:>7} {r['steady_blocks']:>7}")
    print("=" * 78)
    print(f"wrote {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--longbench-dir", default="data/longbench/data")
    p.add_argument("--tasks", default="multifieldqa_en")
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument("--max-prompt-len", type=int, default=4096)
    p.add_argument("--reserve", type=int, default=512)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-gen", type=int, default=512,
                   help="Forced generation length (ignore_eos) so decode regrows "
                        "to capacity.")
    p.add_argument("--capacity", type=int, default=64,
                   help="Per-request block cap C (decode_evict_budget).")
    p.add_argument("--watermark", type=float, default=0.75,
                   help="Low-watermark ratio; evict down to floor(watermark*C).")
    p.add_argument("--interval", type=int, default=1,
                   help="decode_evict_interval (step throttle).")
    p.add_argument("--warmup-pages", type=int, default=1)
    p.add_argument("--eviction-policy",
                   choices=("v_redundancy", "paged_eviction", "recency",
                            "random", "value_l2"),
                   default="v_redundancy",
                   help="Which blocks the band drops: v_redundancy (scored, the "
                        "thesis), recency (oldest-first == StreamingLLM at matched "
                        "memory), value_l2 (lowest value-L2-norm == Paged Eviction "
                        "at matched memory), or random. warmup_pages pins the sink "
                        "for recency.")
    p.add_argument(
        "--comparison-policies",
        default="v_redundancy",
        help="Comma-separated policies for --sweep. The baseline is shared.",
    )
    p.add_argument("--eviction-seed", type=int, default=0,
                   help="Seed for the 'random' eviction policy.")
    p.add_argument("--query-tail-protect-frac", type=float, default=None,
                   help="Tail-protect (query-aware recency floor): never V-evict "
                        "the last ceil(frac * blocks) blocks, where the LongBench "
                        "question lives. Off (pinned behavior) when unset/0.")
    p.add_argument("--positional-cosh-alpha", type=float, default=None,
                   help="Positional cosh (sech-bump) re-weight of v_redundancy "
                        "droppability: multiply each block's score by "
                        "1/cosh(alpha * x), x in [-1,1] over blocks, so eviction "
                        "is pulled toward the middle and away from the sink/tail. "
                        "Off (pinned behavior) when unset/0. v_redundancy only.")
    p.add_argument("--redundancy-mode", default=None,
                   choices=("pairwise", "greedy", "coverage"),
                   help="How v_redundancy ranks blocks: pairwise (score-once, the "
                        "pinned default) or greedy (iterative peel that protects "
                        "the surviving twin of a near-duplicate pair). Off (pinned "
                        "pairwise) when unset. v_redundancy only.")
    p.add_argument("--value-blend-beta", type=float, default=None,
                   help="Blend value-L2 norm into v_redundancy droppability: "
                        "zscore(redundancy) - beta*zscore(value_norm), so high-"
                        "value blocks are protected. Off when unset/0. "
                        "v_redundancy only.")
    p.add_argument("--value-norm-protect-quantile", type=float, default=None,
                   help="Phase-2 norm-constrained redundancy: protect blocks whose "
                        "value-L2 norm is at/above this per-request quantile, then "
                        "evict the most-redundant of the rest. Budget stays exact "
                        "(lowest-norm protections relax first). In [0, 1); q=0 ~ "
                        "value_l2 cut, q~1 ~ pure redundancy. Off when unset. "
                        "v_redundancy only.")
    p.add_argument(
        "--block-prototype-mode",
        choices=("mean", "quarters", "mean_top2_norm"),
        default="mean",
        help="Experimental within-block V representation for pairwise "
             "redundancy. 'mean' is pinned; 'quarters' keeps four sub-block "
             "means; 'mean_top2_norm' keeps the mean and two strongest tokens.",
    )
    p.add_argument(
        "--value-l2-block-reduction",
        choices=("sum", "max_token", "max_token_head"),
        default="sum",
        help="Within-block reduction for value_l2. 'sum' is the pinned "
        "PagedEviction block-score reproduction; max modes preserve a rare "
        "strong token or token/head cell while still evicting whole blocks.",
    )
    p.add_argument(
        "--score-sampled-layers",
        default=None,
        help="Raw scoring-layer spec. Required to be an explicit <=8-layer "
             "subset when actual query capture is enabled.",
    )
    p.add_argument(
        "--query-alignment-weight",
        type=float,
        default=None,
        help="Protect blocks with high actual causal QK attention mass. Off "
             "when unset/0; v_redundancy only.",
    )
    p.add_argument(
        "--query-relevance-protect-quantile",
        type=float,
        default=None,
        help="Protect blocks at or above this causal QK-attention quantile, "
        "then rank the remaining blocks by finalized V-redundancy. Off when "
        "unset; budget-based pairwise v_redundancy only.",
    )
    p.add_argument("--enable-query-tiebreak", action="store_true")
    p.add_argument("--query-tail-tokens", type=int, default=32)
    p.add_argument(
        "--incremental-decode-scoring",
        action="store_true",
        help="Phase 6: reuse copied survivor anchors/similarities across physical "
        "decode compactions. Exact pairwise mean V-redundancy only.",
    )
    p.add_argument(
        "--candidate-expansion-factor",
        type=float,
        default=None,
        help="R2R Stage-1 candidate multiplier; unset uses the legacy selector.",
    )
    p.add_argument("--r2r-cover-depth", type=int, default=2)
    p.add_argument("--r2r-persistent-floor", action="store_true")
    p.add_argument("--r2r-dispersion-kappa", type=float, default=None)
    p.add_argument(
        "--r2r-variants",
        default="",
        help="Comma-separated v1,floor,dispersion,v2 R2R sweep arms; "
        "empty preserves the original single-arm sweep.",
    )
    p.add_argument(
        "--r2r-query-aggregation", choices=("max", "mean"), default="max"
    )
    p.add_argument(
        "--r2r-relevance-signal",
        choices=("key_anchor", "attention_mass"),
        default="key_anchor",
    )
    p.add_argument(
        "--enable-tracing",
        action="store_true",
        help="Enable scorer phase timing and incremental row-reuse telemetry.",
    )
    p.add_argument(
        "--trace-log-path",
        default=None,
        help="Profiler path prefix; defaults to --dump-json in --single mode.",
    )
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument(
        "--confidence-logprobs-topk",
        type=int,
        default=0,
        help="Capture top-k decode logprobs for confidence diagnostics. "
        "Use 0 to disable (default), or 2..20 when enabled.",
    )
    p.add_argument(
        "--confidence-eval",
        action="store_true",
        help="Record answer logprob and independent verbalized confidence.",
    )
    p.add_argument("--confidence-logprobs", type=int, default=1)
    p.add_argument("--confidence-max-tokens", type=int, default=8)
    p.add_argument("--sample-ms", type=int, default=5)
    p.add_argument("--max-rep-threshold", type=int, default=50,
                   help="Fail if any output repeats a token this many times.")
    p.add_argument("--run-id", default="capband")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=0,
                   help="Bootstrap RNG seed for the sweep's paired CIs.")
    # Accuracy sweep mode (vs the default acceptance smoke).
    p.add_argument("--sweep", action="store_true",
                   help="Run the qa_f1-vs-{capacity,watermark} accuracy sweep "
                        "instead of the acceptance smoke.")
    p.add_argument("--sweep-axis", choices=("capacity", "watermark"),
                   default="capacity",
                   help="Which axis to sweep: capacity (vary C at fixed "
                        "watermark) or watermark (vary watermark at fixed C).")
    p.add_argument("--capacities", default="32,48,64,96,128",
                   help="Comma-separated decode_evict_budget values "
                        "(--sweep-axis capacity). Interpreted per --budget-unit.")
    p.add_argument("--budget-unit", choices=("blocks", "tokens"), default="blocks",
                   help="Unit for --capacities: 'blocks' (default, back-compat) "
                        "uses the values as block counts C directly; 'tokens' "
                        "treats them as KV token budgets and converts to blocks "
                        "C = ceil(tokens / block_size).")
    p.add_argument("--watermarks", default="0.5,0.625,0.75,0.875",
                   help="Comma-separated low-watermark ratios "
                        "(--sweep-axis watermark).")
    # Internal single-engine mode.
    p.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--mode", choices=("band", "baseline"), default="band")
    p.add_argument("--natural-gen", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--dump-json", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.single:
        run_single(args)
    elif args.sweep:
        run_sweep(args)
    else:
        run_smoke(args)


if __name__ == "__main__":
    main()
