#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GeoKV eviction ablation: admission vs decode, over a budget sweep (in-engine).

This driver ablates *where* the KV-eviction budget is spent, using the decoupled
fraction-of-prompt knobs (``prefill_evict_frac`` / ``decode_evict_frac``). Each
target is resolved per request from that request's prompt block count ``P``, so a
single fraction is comparable across models, tasks, and prompt lengths. All three
regimes keep the ``v_redundancy`` policy (the thesis):

* **prefill_only** — evict at admission to ``ceil(frac * P)`` blocks; no decode
  eviction. (``frac == 1.0`` is inert -> token-identical to the baseline.)
* **decode_only** — admit the full prompt; during decode cap the cache at
  ``C = ceil(frac * P)`` and evict down to ``ceil(watermark * C)``.
* **combined** — strict admission (``prefill_evict_frac = frac``, swept) with a
  fixed lenient decode (``decode_evict_frac = --combined-decode-frac``). This is
  the "strict admit / lenient decode" hypothesis arm.

For each (task, regime, frac) the driver spawns one clean engine subprocess
(``--single``), scores QA-F1 with natural generation (respect EOS), and compares
per-prompt F1 to that task's no-geo baseline with a paired bootstrap CI. It also
records the achieved memory (steady/peak batch occupancy, decode-eviction count)
so the accuracy tradeoff can be plotted against realized compression rather than
the nominal knob.

The dedicated ``R2R`` stage runs the algorithm-prescribed candidate-factor
grids under admission-band and decode-drip triggers, then adds de-duplicated
cover-depth, query-aggregation, and relevance-signal arms. R2R diagnostics from
the scorer trace are flattened into the result CSV.

Runs the engine IN-PROCESS (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) so the occupancy
sampler thread and ``collective_rpc`` counter read work. Run inside a PBS GPU job
(torch is broken on the login node): see ``benchmarks/geo_eviction_ablation_vllm.pbs``.
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

# Watermark-band regimes (swept over --fracs) + fixed-rate drip regimes (Stage
# A/B, using --blocks-list / --decode-blocks-per-step) + the pressure-gated drip
# (Stage C, using --blocks-list x --watermarks-list).
REGIMES = (
    "prefill_only",
    "decode_only",
    "combined",
    "decode_rate",
    "prefill_drip",
    "decode_pressure",
)


def _resolve_block_pool(llm) -> Any:
    try:
        core = llm.llm_engine.engine_core.engine_core
        return core.scheduler.kv_cache_manager.block_pool
    except AttributeError as e:
        raise RuntimeError("in-process vLLM block pool is unavailable") from e


def _resolve_scheduler(llm) -> Any:
    try:
        return llm.llm_engine.engine_core.engine_core.scheduler
    except AttributeError as e:
        raise RuntimeError("in-process vLLM scheduler is unavailable") from e


def _occupancy_sampler(
    block_pool: Any,
    scheduler: Any,
    stop: threading.Event,
    ts: list[float],
    used: list[int],
    running: list[int],
    waiting: list[int],
    sample_ms: int,
) -> None:
    num_blocks = block_pool.num_gpu_blocks
    interval = max(sample_ms, 1) / 1000.0
    start = time.perf_counter()
    while not stop.is_set():
        try:
            in_use = num_blocks - block_pool.get_num_free_blocks()
        except Exception:
            break
        ts.append(round(time.perf_counter() - start, 4))
        used.append(int(in_use))
        try:
            running.append(len(scheduler.running))
            waiting.append(len(scheduler.waiting))
        except Exception:
            running.append(0)
            waiting.append(0)
        time.sleep(interval)


def _read_preemptions(llm) -> int:
    try:
        metrics = llm.get_metrics()
    except Exception:
        return -1
    for metric in metrics:
        if getattr(metric, "name", "") == "vllm:num_preemptions":
            return int(getattr(metric, "value", 0))
    return 0


def _read_decode_evictions(llm) -> int:
    def _get(worker) -> int:
        policy = getattr(worker.model_runner, "geo_eviction_policy", None)
        return -1 if policy is None else int(policy._num_decode_evictions)

    try:
        core = llm.llm_engine.engine_core.engine_core
        return int(core.model_executor.collective_rpc(_get)[0])
    except Exception as e:
        print(f"[ablation] WARN could not read decode evictions: {e}")
        return -1


def _max_repetition_run(text: str) -> int:
    tokens = text.split()
    if not tokens:
        return 0
    best = run = 1
    for previous, current in zip(tokens, tokens[1:]):
        run = run + 1 if previous == current else 1
        best = max(best, run)
    return best


def _paired_delta_ci(
    scores: list[float],
    baseline: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    import random

    count = min(len(scores), len(baseline))
    if count == 0:
        return 0.0, 0.0, 0.0
    deltas = [scores[i] - baseline[i] for i in range(count)]
    rng = random.Random(seed)
    bootstraps = [
        statistics.fmean(deltas[rng.randrange(count)] for _ in range(count))
        for _ in range(n_boot)
    ]
    bootstraps.sort()
    low = bootstraps[int((alpha / 2) * n_boot)]
    high = bootstraps[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return statistics.fmean(deltas), low, high


def _align_f1(
    run: dict, baseline_by_id: dict[str, float]
) -> tuple[list[float], list[float]]:
    scores: list[float] = []
    baseline: list[float] = []
    for prompt_id, score in zip(run["prompt_ids"], run["f1s"]):
        if prompt_id in baseline_by_id:
            scores.append(score)
            baseline.append(baseline_by_id[prompt_id])
    return scores, baseline


def _geo_config(args) -> dict[str, Any] | None:
    """Build the geo_kv config block for one regime, or None for the baseline.

    Maps the ablation regime + swept knob onto the decoupled per-end knobs. Every
    regime uses physical reclaim; ``eviction_policy`` selects WHICH blocks the
    band drops (``v_redundancy`` is the thesis; ``recency``/``random``/``value_l2``/
    ``paged_eviction`` are matched-memory baselines). The two drip regimes use
    ``decode_evict_blocks_per_step`` (a fixed-rate decode drain) instead of the
    watermark band.
    """
    regime = args.regime
    if regime == "baseline":
        return None
    gk: dict[str, Any] = {
        "experiment_mode": "geo_uniform",
        "eviction_policy": args.eviction_policy,
        "score_sampled_layers": args.score_sampled_layers,
        "block_score_aggregation": "mean",
        "warmup_pages": args.warmup_pages,
        "decode_evict_watermark": args.watermark,
        "decode_evict_interval": args.interval,
        "physical_reclaim": True,
    }
    # Greedy (iterative) redundancy peel + value-norm blend. Omitted when unset so
    # a v_redundancy run stays byte-identical to the pinned pairwise scorer. Only
    # meaningful for eviction_policy=v_redundancy.
    if getattr(args, "redundancy_mode", None):
        gk["redundancy_mode"] = args.redundancy_mode
    if getattr(args, "value_blend_beta", None):
        gk["value_blend_beta"] = args.value_blend_beta
    if getattr(args, "value_norm_protect_quantile", None) is not None:
        gk["value_norm_protect_quantile"] = args.value_norm_protect_quantile
    if getattr(args, "candidate_expansion_factor", None) is not None:
        if not args.dump_json:
            raise ValueError("R2R single-run tracing requires --dump-json")
        gk.update(
            {
                "candidate_expansion_factor": args.candidate_expansion_factor,
                "r2r_cover_depth": args.r2r_cover_depth,
                "r2r_query_aggregation": args.r2r_query_aggregation,
                "r2r_relevance_signal": args.r2r_relevance_signal,
                "enable_tracing": True,
                "log_path": args.dump_json,
            }
        )
    if regime == "prefill_only":
        gk["prefill_evict_frac"] = args.frac
    elif regime == "decode_only":
        gk["decode_evict_frac"] = args.frac
    elif regime == "combined":
        gk["prefill_evict_frac"] = args.frac  # strict admission (swept)
        gk["decode_evict_frac"] = args.combined_decode_frac  # fixed lenient decode
    elif regime == "decode_rate":
        # Stage A: prefill fully open, decode drips N blocks/fire (N swept).
        gk["decode_evict_blocks_per_step"] = args.decode_blocks_per_step
    elif regime == "prefill_drip":
        # Stage B: strict admission (frac swept) + fixed decode drip at N*.
        gk["prefill_evict_frac"] = args.frac
        gk["decode_evict_blocks_per_step"] = args.decode_blocks_per_step
    elif regime == "decode_pressure":
        # Stage C: the drip (N swept) gated on global cache pressure (watermark
        # swept). Prefill fully open; eviction fires only when the pool is
        # >= pressure_watermark full.
        gk["decode_evict_blocks_per_step"] = args.decode_blocks_per_step
        gk["decode_evict_pressure_watermark"] = args.pressure_watermark
    else:
        raise ValueError(f"unknown regime: {regime!r}")
    return gk


# ---------------------------------------------------------------------------
# Single engine config (one subprocess per config; clean GPU memory)
# ---------------------------------------------------------------------------
def run_single(args) -> None:
    # geo hooks are V2-runner only; force it before importing vllm. In-process
    # engine so the sampler thread + collective_rpc can reach engine internals.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from benchmark_kv_eviction_accuracy import DATASET2METRIC, score_prediction
    from benchmark_kv_eviction_accuracy_vllm import build_prompts

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(tokenizer, args)
    if not prompts:
        raise RuntimeError("no prompts built; check --tasks/--num-prompts")

    geo_cfg = _geo_config(args)
    kwargs: dict[str, Any] = dict(
        model=args.model,
        enforce_eager=True,  # required by the geo_kv eviction guard
        attention_backend="FLEX_ATTENTION",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        async_scheduling=False,  # physical_reclaim requires the sync scheduler
        enable_prefix_caching=False,  # physical_reclaim is incompatible
        disable_log_stats=False,
    )
    if geo_cfg is not None:
        kwargs["additional_config"] = {"geo_kv": geo_cfg}
    # Shrink the KV pool so global cache pressure binds (Stage C on tasks that do
    # not naturally saturate). Applied to every arm INCLUDING the baseline so the
    # pressure-eviction vs native-preemption comparison is at a matched budget.
    if args.num_gpu_blocks_override:
        kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override
    print(
        f"[ablation] building engine: regime={args.regime} frac={args.frac} "
        f"decode_blocks_per_step={args.decode_blocks_per_step} "
        f"combined_decode_frac={args.combined_decode_frac} "
        f"watermark={args.watermark} pressure_watermark={args.pressure_watermark} "
        f"num_gpu_blocks_override={args.num_gpu_blocks_override} "
        f"geo={geo_cfg is not None}"
    )
    llm = LLM(**kwargs)
    block_pool = _resolve_block_pool(llm)
    scheduler = _resolve_scheduler(llm)
    num_gpu_blocks = int(block_pool.num_gpu_blocks)

    reqs = [TokensPrompt(prompt_token_ids=p["token_ids"]) for p in prompts]
    # Accuracy mode: respect EOS + per-task max_gen so qa_f1 reflects real answer
    # quality. Admission eviction fires at end-of-prefill; decode eviction fires
    # only if generation regrows the (compacted) cache back to its capacity.
    sps = [SamplingParams(temperature=0.0, max_tokens=p["max_gen"]) for p in prompts]

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
    preemptions = _read_preemptions(llm)
    texts = [o.outputs[0].text for o in outs]
    out_lens = [len(o.outputs[0].token_ids) for o in outs]
    token_ids = [list(o.outputs[0].token_ids) for o in outs]
    # Score with each task's canonical LongBench metric: token-F1 for QA tasks,
    # ROUGE-L for the summarization tasks (gov_report, multi_news). The column is
    # still named "qa_f1"/"f1s" for schema stability across stages; the actual
    # metric per task is recorded in "metric" below.
    f1s = [
        score_prediction(p["task"], t, p["answers"])
        for p, t in zip(prompts, texts)
    ]
    max_rep = max((_max_repetition_run(t) for t in texts), default=0)

    peak = max(used) if used else 0
    tail = used[len(used) // 2 :] if used else []
    steady = int(statistics.median(tail)) if tail else 0
    peak_running = max(running) if running else 0
    peak_waiting = max(waiting) if waiting else 0

    result = {
        "run_id": args.run_id,
        "model": args.model,
        "regime": args.regime,
        "task": args.tasks,
        "frac": args.frac,
        "decode_blocks_per_step": args.decode_blocks_per_step,
        "combined_decode_frac": args.combined_decode_frac,
        "watermark": args.watermark,
        "pressure_watermark": args.pressure_watermark,
        "num_prompts": len(prompts),
        "num_gpu_blocks": num_gpu_blocks,
        "num_gpu_blocks_override": args.num_gpu_blocks_override or 0,
        "preemptions": preemptions,
        "peak_running": peak_running,
        "peak_waiting": peak_waiting,
        "peak_blocks": peak,
        "steady_blocks": steady,
        "decode_evictions": decode_evictions,
        "mean_out_len": round(statistics.fmean(out_lens), 1) if out_lens else 0,
        "min_out_len": min(out_lens) if out_lens else 0,
        "qa_f1": round(statistics.fmean(f1s), 4) if f1s else 0.0,
        "metric": DATASET2METRIC.get(args.tasks, "qa_f1"),
        "eviction_policy": (
            "baseline" if geo_cfg is None else args.eviction_policy
        ),
        # Method knobs, recorded so a greedy/blend run is distinguishable on disk
        # from pinned pairwise v_redundancy. None when off or on the baseline.
        "redundancy_mode": (
            None if geo_cfg is None else getattr(args, "redundancy_mode", None)
        ),
        "value_blend_beta": (
            None if geo_cfg is None else getattr(args, "value_blend_beta", None)
        ),
        "value_norm_protect_quantile": (
            None
            if geo_cfg is None
            else getattr(args, "value_norm_protect_quantile", None)
        ),
        "candidate_expansion_factor": (
            None if geo_cfg is None else geo_cfg.get("candidate_expansion_factor")
        ),
        "r2r_cover_depth": None if geo_cfg is None else geo_cfg.get("r2r_cover_depth"),
        "r2r_query_aggregation": (
            None if geo_cfg is None else geo_cfg.get("r2r_query_aggregation")
        ),
        "r2r_relevance_signal": (
            None if geo_cfg is None else geo_cfg.get("r2r_relevance_signal")
        ),
        "max_repetition_run": max_rep,
        "wall_s": round(wall, 2),
        # Per-prompt F1s + ids let the orchestrator pair and bootstrap CIs.
        "f1s": [round(x, 4) for x in f1s],
        "prompt_ids": [p["prompt_id"] for p in prompts],
        "prompt_blocks": [p["num_blocks"] for p in prompts],
        "prompt_blocks_sum": sum(p["num_blocks"] for p in prompts),
        # Token ids drive the frac=1.0 prefill_only == baseline self-check.
        "token_ids": token_ids,
    }
    if args.dump_json:
        with open(args.dump_json, "w") as f:
            json.dump(result, f)
    summary = {k: v for k, v in result.items() if k != "token_ids"}
    print("[ablation] result:", json.dumps(summary, indent=2))
    if geo_cfg is not None and geo_cfg.get("enable_tracing"):
        try:
            llm.llm_engine.engine_core.shutdown()
            print("[ablation] engine shut down; R2R trace flushed")
        except Exception as e:
            print(f"[ablation] warning: engine shutdown failed: {e}")


# ---------------------------------------------------------------------------
# Orchestration: spawn each engine config as a subprocess.
# ---------------------------------------------------------------------------
def _spawn(
    args,
    regime: str,
    frac: float,
    dump: str,
    label: str,
    blocks_per_step: int = 0,
    pressure_watermark: float = 0.0,
    watermark: float | None = None,
    candidate_expansion_factor: float | None = None,
    r2r_cover_depth: int = 2,
    r2r_query_aggregation: str = "max",
    r2r_relevance_signal: str = "key_anchor",
) -> dict:
    """Spawn one engine subprocess for one arm and return its result dict.

    An arm is (regime, frac, blocks_per_step, pressure_watermark, watermark); the
    band regimes vary ``frac``, the drip regimes vary ``blocks_per_step``, the
    pressure regime varies both ``blocks_per_step`` and ``pressure_watermark``, and
    the low-watermark band (stage D) varies ``watermark`` (the evict-down-to
    target). ``watermark`` defaults to the run-wide ``args.watermark``.
    """
    if watermark is None:
        watermark = args.watermark
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single",
        "--regime",
        regime,
        "--frac",
        str(frac),
        "--decode-blocks-per-step",
        str(blocks_per_step),
        "--pressure-watermark",
        str(pressure_watermark),
        "--num-gpu-blocks-override",
        str(args.num_gpu_blocks_override or 0),
        "--combined-decode-frac",
        str(args.combined_decode_frac),
        "--watermark",
        str(watermark),
        "--interval",
        str(args.interval),
        "--warmup-pages",
        str(args.warmup_pages),
        "--score-sampled-layers",
        args.score_sampled_layers,
        "--eviction-policy",
        args.eviction_policy,
        *(
            ["--redundancy-mode", str(args.redundancy_mode)]
            if getattr(args, "redundancy_mode", None)
            else []
        ),
        *(
            ["--candidate-expansion-factor", str(candidate_expansion_factor)]
            if candidate_expansion_factor is not None
            else []
        ),
        "--r2r-cover-depth",
        str(r2r_cover_depth),
        "--r2r-query-aggregation",
        r2r_query_aggregation,
        "--r2r-relevance-signal",
        r2r_relevance_signal,
        *(
            ["--value-blend-beta", str(args.value_blend_beta)]
            if getattr(args, "value_blend_beta", None)
            else []
        ),
        *(
            [
                "--value-norm-protect-quantile",
                str(args.value_norm_protect_quantile),
            ]
            if getattr(args, "value_norm_protect_quantile", None) is not None
            else []
        ),
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
        "--sample-ms",
        str(args.sample_ms),
        "--run-id",
        f"{args.run_id}_{label}",
        "--dump-json",
        dump,
    ]
    print(
        f"[ablation] spawning: {label} "
        f"(regime={regime} frac={frac} blocks_per_step={blocks_per_step})"
    )
    subprocess.run(cmd, check=True)
    with open(dump) as f:
        result = json.load(f)
    profile_path = f"{dump}.scorer_profile.json"
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            result["scorer_profile"] = json.load(f)
    return result


def _build_arms(args) -> list[dict]:
    """Resolve the sweep arms for the selected stage.

    Each arm is ``{regime, frac, blocks, pressure_wm, tag}``. The baseline is
    spawned separately (it is every stage's shared reference).

    - ``legacy``: the original {regimes} x {fracs} band sweep.
    - ``A`` (decode-rate): prefill fully open, decode drips N in --blocks-list.
      Swept axis is N; frac is pinned to 1.0 (inert admission).
    - ``B`` (prefill ablation @ N*): fixed decode drip --nstar, sweep
      prefill_evict_frac over --fracs.
    - ``C`` (pressure-gated drip): 2-D sweep of --watermarks-list x --blocks-list.
      Prefill fully open; the drip fires only when the pool is >= watermark full.
    - ``D`` (low-watermark band): the decode capacity band at ``frac == 1.0`` (fill
      to the whole prompt, C == P), sweeping the LOW watermark over
      --low-watermarks-list. "Fill to 100%, then evict down to X%." Unlike C this
      is a per-request band (no global pool gate), so every task fires it.
    """
    stage = args.stage
    if stage == "R2R":
        if args.eviction_policy != "v_redundancy":
            raise ValueError("R2R stage requires --eviction-policy v_redundancy")
        if args.redundancy_mode not in (None, "pairwise"):
            raise ValueError("R2R stage requires pairwise redundancy")
        if args.value_blend_beta or args.value_norm_protect_quantile is not None:
            raise ValueError("R2R stage is incompatible with value refinements")
        band_ns = [float(x) for x in args.r2r_band_n.split(",") if x.strip()]
        drip_ns = [float(x) for x in args.r2r_drip_n.split(",") if x.strip()]
        depths = [int(x) for x in args.r2r_depths.split(",") if x.strip()]
        aggregations = [
            x.strip() for x in args.r2r_aggregations.split(",") if x.strip()
        ]
        signals = [x.strip() for x in args.r2r_signals.split(",") if x.strip()]
        factors = band_ns + drip_ns + [
            args.r2r_band_reference_n,
            args.r2r_drip_reference_n,
        ]
        if not factors or any(factor < 1 for factor in factors):
            raise ValueError("R2R candidate expansion factors must be >= 1")
        if any(depth < 0 for depth in depths):
            raise ValueError("R2R cover depths must be >= 0")
        if not set(aggregations) <= {"max", "mean"}:
            raise ValueError("R2R aggregations must be max or mean")
        if not set(signals) <= {"key_anchor", "attention_mass"}:
            raise ValueError("unknown R2R relevance signal")
        if not 0 < args.r2r_band_frac <= 1:
            raise ValueError("--r2r-band-frac must be in (0, 1]")
        if args.r2r_drip_blocks < 1:
            raise ValueError("--r2r-drip-blocks must be >= 1")
        arms: list[dict] = []
        seen: set[tuple] = set()

        def add_arm(
            trigger: str,
            factor: float,
            depth: int = 2,
            aggregation: str = "max",
            signal: str = "key_anchor",
        ) -> None:
            key = (trigger, factor, depth, aggregation, signal)
            if key in seen:
                return
            seen.add(key)
            drip = trigger == "drip"
            arms.append(
                {
                    "regime": "decode_rate" if drip else "prefill_only",
                    "frac": 1.0 if drip else args.r2r_band_frac,
                    "blocks": args.r2r_drip_blocks if drip else 0,
                    "pressure_wm": 0.0,
                    "candidate_factor": factor,
                    "cover_depth": depth,
                    "query_aggregation": aggregation,
                    "relevance_signal": signal,
                    "tag": (
                        f"r2r_{trigger}_n{factor:g}_d{depth}_"
                        f"{aggregation}_{signal}"
                    ),
                }
            )

        for factor in band_ns:
            add_arm("band", factor)
        for factor in drip_ns:
            add_arm("drip", factor)
        for trigger, factor in (
            ("band", args.r2r_band_reference_n),
            ("drip", args.r2r_drip_reference_n),
        ):
            for depth in depths:
                add_arm(trigger, factor, depth=depth)
            for aggregation in aggregations:
                add_arm(trigger, factor, aggregation=aggregation)
            for signal in signals:
                add_arm(trigger, factor, signal=signal)
        return arms
    if stage == "A":
        blocks = [int(x) for x in args.blocks_list.split(",") if x.strip()]
        return [
            {"regime": "decode_rate", "frac": 1.0, "blocks": n, "pressure_wm": 0.0,
             "tag": f"decode_rate_n{n}"}
            for n in blocks
        ]
    if stage == "B":
        nstar = int(args.nstar)
        if nstar < 1:
            raise ValueError("stage B requires --nstar >= 1 (the winning N)")
        fracs = [float(x) for x in args.fracs.split(",") if x.strip()]
        return [
            {"regime": "prefill_drip", "frac": f, "blocks": nstar, "pressure_wm": 0.0,
             "tag": f"prefill_drip_f{f}_n{nstar}"}
            for f in fracs
        ]
    if stage == "C":
        blocks = [int(x) for x in args.blocks_list.split(",") if x.strip()]
        wms = [float(x) for x in args.watermarks_list.split(",") if x.strip()]
        return [
            {"regime": "decode_pressure", "frac": 1.0, "blocks": n,
             "pressure_wm": w, "tag": f"decode_pressure_wm{w}_n{n}"}
            for w in wms
            for n in blocks
        ]
    if stage == "D":
        # Low-watermark band: decode_only at frac=1.0 (capacity C == P, fill to the
        # whole prompt) evicting down to each low watermark. Per-request band, so
        # no pool override is needed. Swept axis is the low watermark.
        lows = [float(x) for x in args.low_watermarks_list.split(",") if x.strip()]
        return [
            {"regime": "decode_only", "frac": 1.0, "blocks": 0, "pressure_wm": 0.0,
             "watermark": w, "tag": f"lowwm_{w}"}
            for w in lows
        ]
    # legacy band sweep
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    for r in regimes:
        if r not in REGIMES:
            raise ValueError(f"unknown regime {r!r}; choose from {list(REGIMES)}")
    fracs = [float(x) for x in args.fracs.split(",") if x.strip()]
    return [
        {"regime": regime, "frac": frac, "blocks": 0, "pressure_wm": 0.0,
         "tag": f"{regime}_f{frac}"}
        for regime in regimes
        for frac in fracs
    ]


def run_ablation(args) -> None:
    """Sweep the selected stage's arms for one (model, task) vs the baseline."""
    outdir = args.output_dir or os.path.join("results", "ablation", args.run_id)
    os.makedirs(outdir, exist_ok=True)

    arms = _build_arms(args)

    # Baseline: no geo, natural generation -> reference F1 + occupancy per prompt.
    base = _spawn(
        args, "baseline", 1.0, os.path.join(outdir, f"{args.run_id}_base.json"), "base"
    )
    base_by_id = dict(zip(base["prompt_ids"], base["f1s"]))
    base_steady = base["steady_blocks"]
    prompt_blocks_sum = base["prompt_blocks_sum"]

    rows: list[dict] = []
    selfcheck_notes: list[str] = []
    for arm in arms:
        regime, frac, blocks, pressure_wm, tag = (
            arm["regime"], arm["frac"], arm["blocks"], arm["pressure_wm"], arm["tag"]
        )
        # Stage D varies the low watermark per arm; every other stage uses the
        # single global --watermark.
        arm_watermark = arm.get("watermark", args.watermark)
        run = _spawn(
            args,
            regime,
            frac,
            os.path.join(outdir, f"{args.run_id}_{tag}.json"),
            tag,
            blocks_per_step=blocks,
            pressure_watermark=pressure_wm,
            watermark=arm_watermark,
            candidate_expansion_factor=arm.get("candidate_factor"),
            r2r_cover_depth=arm.get("cover_depth", 2),
            r2r_query_aggregation=arm.get("query_aggregation", "max"),
            r2r_relevance_signal=arm.get("relevance_signal", "key_anchor"),
        )
        band_f1, base_f1 = _align_f1(run, base_by_id)
        mean_delta, lo, hi = _paired_delta_ci(band_f1, base_f1, seed=args.seed)
        realized = (
            round(run["steady_blocks"] / prompt_blocks_sum, 4)
            if prompt_blocks_sum
            else 0.0
        )
        profile = run.get("scorer_profile", {})
        r2r_selection = profile.get("r2r_selection", {})
        coherence = profile.get("query_direction_coherence", {})
        logdet = profile.get("retained_value_logdet", {})
        rows.append(
            {
                "model": os.path.basename(args.model.rstrip("/")),
                "task": args.tasks,
                "eviction_policy": run.get("eviction_policy", args.eviction_policy),
                "redundancy_mode": run.get("redundancy_mode"),
                "value_blend_beta": run.get("value_blend_beta"),
                "value_norm_protect_quantile": run.get(
                    "value_norm_protect_quantile"
                ),
                "candidate_expansion_factor": run.get(
                    "candidate_expansion_factor"
                ),
                "r2r_cover_depth": run.get("r2r_cover_depth"),
                "r2r_query_aggregation": run.get("r2r_query_aggregation"),
                "r2r_relevance_signal": run.get("r2r_relevance_signal"),
                "r2r_candidate_coverage": r2r_selection.get(
                    "candidate_coverage"
                ),
                "r2r_selection_fires": r2r_selection.get("fires", 0),
                "r2r_cover_1": r2r_selection.get("reasons", {}).get(
                    "cover_1", 0
                ),
                "r2r_cover_2": r2r_selection.get("reasons", {}).get(
                    "cover_2", 0
                ),
                "r2r_backfill": r2r_selection.get("reasons", {}).get(
                    "backfill", 0
                ),
                "query_direction_coherence_mean": coherence.get("mean"),
                "retained_value_logdet_mean": logdet.get("mean"),
                "regime": regime,
                "frac": frac,
                "decode_blocks_per_step": blocks,
                "pressure_watermark": pressure_wm,
                "combined_decode_frac": args.combined_decode_frac,
                "watermark": arm_watermark,
                "num_gpu_blocks_override": args.num_gpu_blocks_override or 0,
                "n": len(band_f1),
                "qa_f1": run["qa_f1"],
                "metric": run.get("metric", "qa_f1"),
                "qa_f1_delta": round(mean_delta, 4),
                "delta_ci_lo": round(lo, 4),
                "delta_ci_hi": round(hi, 4),
                "realized_retention": realized,
                "decode_evictions": run["decode_evictions"],
                "preemptions": run.get("preemptions", -1),
                "peak_running": run.get("peak_running", 0),
                "peak_waiting": run.get("peak_waiting", 0),
                "steady_blocks": run["steady_blocks"],
                "peak_blocks": run["peak_blocks"],
                "baseline_steady_blocks": base_steady,
                "baseline_preemptions": base.get("preemptions", -1),
                "mean_out_len": run["mean_out_len"],
            }
        )

        # Self-check: prefill_only @ frac=1.0 retains the whole prompt, so it
        # must be token-identical to the no-geo baseline (the frac path is
        # inert at full budget). Catches wiring bugs cheaply. (Drip stages have
        # no inert arm -- N>=1 always drains -- so this only applies to legacy.)
        if regime == "prefill_only" and frac == 1.0:
            same = run["token_ids"] == base["token_ids"]
            selfcheck_notes.append(
                f"prefill_only@1.0 == baseline (token-identical): {same}"
            )
            if not same:
                n_div = sum(
                    1 for a, b in zip(run["token_ids"], base["token_ids"]) if a != b
                )
                selfcheck_notes.append(
                    f"  WARNING: diverged in {n_div}/{len(base['token_ids'])} "
                    "prompts -- the inert frac path is NOT a no-op"
                )

    csv_path = os.path.join(outdir, "ablation_accuracy.csv")
    fields = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    model_name = os.path.basename(args.model.rstrip("/"))
    print("\n" + "=" * 92)
    print(f"EVICTION ABLATION  run_id={args.run_id}")
    print(f"  model={model_name}  task={args.tasks}  n_prompts={base['num_prompts']}")
    print(
        f"  baseline qa_f1={base['qa_f1']}  baseline steady={base_steady} blk  "
        f"prompt_blocks_sum={prompt_blocks_sum}"
    )
    print(
        f"  combined_decode_frac={args.combined_decode_frac}  "
        f"watermark={args.watermark}  gpu_blocks={base['num_gpu_blocks']}"
    )
    print(
        f"  baseline preemptions={base.get('preemptions', -1)}  "
        f"num_gpu_blocks_override={args.num_gpu_blocks_override or 0}"
    )
    print("-" * 92)
    print(
        f"  {'regime':>13} {'N/step':>6} {'p_wm':>5} {'qa_f1':>7} {'Δf1':>8} "
        f"{'95% CI':>18} {'ret':>6} {'evicts':>7} {'preempt':>7} {'steady':>7}"
    )
    for r in rows:
        ci = f"[{r['delta_ci_lo']:+.3f},{r['delta_ci_hi']:+.3f}]"
        print(
            f"  {r['regime']:>13} {r['decode_blocks_per_step']:>6} "
            f"{r['pressure_watermark']:>5} "
            f"{r['qa_f1']:>7.3f} {r['qa_f1_delta']:>+8.3f} {ci:>18} "
            f"{r['realized_retention']:>6.3f} "
            f"{r['decode_evictions']:>7} {r.get('preemptions', -1):>7} "
            f"{r['steady_blocks']:>7}"
        )
    print("-" * 92)
    for note in selfcheck_notes:
        print(f"  self-check: {note}")
    print("=" * 92)
    print(f"wrote {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--longbench-dir", default="data/longbench/data")
    p.add_argument(
        "--tasks",
        default="multifieldqa_en",
        help="Single task name (one PBS job per model x task).",
    )
    p.add_argument("--num-prompts", type=int, default=150)
    p.add_argument("--max-prompt-len", type=int, default=4096)
    p.add_argument("--reserve", type=int, default=512)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument(
        "--max-gen",
        type=int,
        default=0,
        help="Override generation length; 0 => each task's natural "
        "max_gen (respect EOS).",
    )
    p.add_argument(
        "--watermark",
        type=float,
        default=0.75,
        help="Low-watermark ratio for decode: evict to ceil(wm*C).",
    )
    p.add_argument(
        "--interval", type=int, default=1, help="decode_evict_interval (step throttle)."
    )
    p.add_argument("--warmup-pages", type=int, default=1)
    p.add_argument(
        "--score-sampled-layers",
        default="3,10,17,24",
        help="Layers used for GeoKV scoring and query capture (at most eight).",
    )
    p.add_argument(
        "--eviction-policy",
        choices=(
            "v_redundancy",
            "recency",
            "random",
            "value_l2",
            "paged_eviction",
        ),
        default="v_redundancy",
        help="Which blocks the band drops: v_redundancy (scored, the thesis) or "
        "the matched-memory baselines recency (oldest-first == StreamingLLM), "
        "random (seeded), value_l2 (reference-code score), or paged_eviction "
        "(published V/K ratio). All drop the same COUNT, so "
        "the accuracy contrast is at matched retained memory.",
    )
    p.add_argument(
        "--redundancy-mode",
        choices=("pairwise", "greedy"),
        default=None,
        help="How v_redundancy ranks blocks: pairwise (score-once, the pinned "
        "default) or greedy (iterative peel that protects the surviving twin of a "
        "near-duplicate pair). Off (pinned pairwise) when unset. v_redundancy only.",
    )
    p.add_argument(
        "--value-blend-beta",
        type=float,
        default=None,
        help="Blend value-L2 norm into v_redundancy droppability: "
        "zscore(redundancy) - beta*zscore(value_norm), protecting high-value "
        "blocks. Off when unset/0. v_redundancy only.",
    )
    p.add_argument(
        "--value-norm-protect-quantile",
        type=float,
        default=None,
        help="Phase-2 norm-constrained redundancy: protect blocks whose value-L2 "
        "norm is at/above this per-request quantile, then evict the most-redundant "
        "of the rest. Budget stays exact (lowest-norm protections relax first). "
        "In [0, 1); q=0 ~ value_l2 cut, q~1 ~ pure redundancy. Off when unset. "
        "v_redundancy only.",
    )
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--sample-ms", type=int, default=5)
    p.add_argument("--run-id", default="ablation")
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--seed", type=int, default=0, help="Bootstrap RNG seed for the paired CIs."
    )
    # Ablation sweep controls.
    p.add_argument(
        "--stage",
        choices=("legacy", "A", "B", "C", "D", "R2R"),
        default="legacy",
        help="legacy = {regimes} x {fracs} band sweep; "
        "A = decode-rate drip sweep over --blocks-list (prefill open); "
        "B = prefill_evict_frac ablation over --fracs at a fixed drip --nstar; "
        "C = pressure-gated drip sweep over --watermarks-list x --blocks-list; "
        "D = low-watermark band sweep over --low-watermarks-list at frac=1.0 "
        "(fill to 100%% of the prompt, evict down to each low watermark); "
        "R2R = candidate, guard, aggregation, and relevance-signal sweep under "
        "both admission-band and decode-drip triggers.",
    )
    p.add_argument(
        "--r2r-band-n",
        default="1,2,3,4",
        help="R2R stage: admission-band candidate expansion factors.",
    )
    p.add_argument(
        "--r2r-drip-n",
        default="2,4,16,64",
        help="R2R stage: decode-drip candidate expansion factors.",
    )
    p.add_argument(
        "--r2r-depths",
        default="0,1,2",
        help="R2R stage: cover depths evaluated at each trigger's reference n.",
    )
    p.add_argument(
        "--r2r-aggregations",
        default="max,mean",
        help="R2R stage: query-window reductions at each trigger's reference n.",
    )
    p.add_argument(
        "--r2r-signals",
        default="key_anchor,attention_mass",
        help="R2R stage: Stage-2 signals at each trigger's reference n.",
    )
    p.add_argument("--r2r-band-reference-n", type=float, default=2.0)
    p.add_argument("--r2r-drip-reference-n", type=float, default=16.0)
    p.add_argument(
        "--r2r-band-frac",
        type=float,
        default=0.75,
        help="R2R stage: retained prompt fraction for admission-band arms.",
    )
    p.add_argument(
        "--r2r-drip-blocks",
        type=int,
        default=1,
        help="R2R stage: blocks evicted per decode drip fire.",
    )
    p.add_argument(
        "--regimes",
        default="prefill_only,decode_only,combined",
        help="Comma-separated regimes (legacy stage only).",
    )
    p.add_argument(
        "--fracs",
        default="1.0,0.75,0.5,0.375,0.25",
        help="Comma-separated fraction-of-prompt budgets (full->small). "
        "Used by legacy + stage B (the prefill_evict_frac axis).",
    )
    p.add_argument(
        "--blocks-list",
        default="1,2,4,8",
        help="Stage A/C: comma-separated decode drip rates N (blocks per fire).",
    )
    p.add_argument(
        "--watermarks-list",
        default="0.8,0.9,0.95",
        help="Stage C: comma-separated pressure FILL watermarks (fire the drip "
        "when the global pool is >= this fraction full).",
    )
    p.add_argument(
        "--low-watermarks-list",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
        help="Stage D: comma-separated LOW watermarks (evict-down-to targets as a "
        "fraction of the per-request capacity C == P). Fill to 100%%, drain to "
        "each.",
    )
    p.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        default=0,
        help="Cap the KV pool to this many blocks (0 => engine default). Used to "
        "make cache pressure bind on tasks that do not naturally saturate; "
        "applied to every arm including the baseline for a matched budget.",
    )
    p.add_argument(
        "--nstar",
        type=int,
        default=0,
        help="Stage B: the winning drip rate N* (blocks per fire) to fix.",
    )
    p.add_argument(
        "--combined-decode-frac",
        type=float,
        default=0.9,
        help="Fixed lenient decode fraction for the combined regime.",
    )
    # Internal single-engine mode.
    p.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--regime", choices=("baseline",) + REGIMES, default="baseline")
    p.add_argument("--frac", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument(
        "--decode-blocks-per-step",
        type=int,
        default=0,
        help=argparse.SUPPRESS,  # per-subprocess drip rate (0 == off)
    )
    p.add_argument(
        "--pressure-watermark",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,  # per-subprocess pressure FILL watermark (0 == off)
    )
    p.add_argument("--candidate-expansion-factor", type=float, default=None)
    p.add_argument("--r2r-cover-depth", type=int, default=2)
    p.add_argument(
        "--r2r-query-aggregation", choices=("max", "mean"), default="max"
    )
    p.add_argument(
        "--r2r-relevance-signal",
        choices=("key_anchor", "attention_mass"),
        default="key_anchor",
    )
    p.add_argument("--dump-json", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.single:
        run_single(args)
    else:
        run_ablation(args)


if __name__ == "__main__":
    main()
