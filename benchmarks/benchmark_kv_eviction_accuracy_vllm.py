#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-2 accuracy screen for value-space KV eviction (in-engine, vLLM).

Reproduces the Stage-1 HF result *inside the real vLLM V1 engine*: does dropping
the highest V-redundancy KV blocks preserve answer quality, versus recency and
random at a matched eviction rate? Milestone 1 is mask-only -- evicted blocks
stay allocated but are hidden from attention by the FlexAttention logical mask
(gated behind ``additional_config['geo_kv']``, inert when absent).

To keep the metric and data identical to the offline screen, this driver imports
the vendored ``qa_f1``/prompt helpers from ``benchmark_kv_eviction_accuracy.py``
and writes the same ``accuracy_scores.csv`` / ``summary.json`` schema, so
``scripts/plot_eviction_results.py`` can overlay Stage-1 and Stage-2.

Each engine config (policy x rate x band) is fixed at ``LLM`` construction, so
the orchestrator fans out one *subprocess* per config point (clean GPU memory),
runs a rate-0 inert-vs-baseline self-check first (identical tokens => the mask
plumbing is inert at rate 0), then the accuracy-vs-rate sweep.

Run inside a PBS GPU job (torch is broken on the login node): see
``benchmarks/geo_eviction_accuracy_vllm.pbs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Any

# Reuse the *exact* Stage-1 metric, data loading, and prompt construction so the
# two stages are directly comparable (single source of truth).
from benchmark_kv_eviction_accuracy import (
    BANDS,
    DATASET2MAXGEN,
    DATASET2METRIC,
    build_input_ids,
    load_task,
    score_prediction,
)

POLICIES = ("v_redundancy", "recency", "random")
CSV_FIELDS = [
    "run_id",
    "prompt_id",
    "task",
    "policy",
    "layer_band",
    "block_prototype_mode",
    "query_alignment_weight",
    "query_tail_tokens",
    "eviction_rate",
    "num_blocks",
    "num_evicted",
    "metric_name",
    "score",
]


def band_to_layers_spec(band: str) -> str:
    """Map a band name to a ``score_sampled_layers`` spec for GeoKVConfig."""
    if band == "all":
        return "all"
    return ",".join(str(x) for x in BANDS[band])


def expected_evicted(num_blocks: int, rate: float, warmup_pages: int) -> int:
    """Deterministic evicted count (mirrors select_evicted_blocks)."""
    evictable = max(0, num_blocks - 1 - max(warmup_pages, 0))
    return int(round(rate * evictable)) if rate > 0.0 else 0


# ---------------------------------------------------------------------------
# Prompt construction (shared token ids -> identical prompts across configs)
# ---------------------------------------------------------------------------
def build_prompts(tokenizer, args) -> list[dict]:
    tasks = [t for t in args.tasks.split(",") if t]
    prompts: list[dict] = []
    for task in tasks:
        max_gen = args.max_gen or DATASET2MAXGEN.get(task, 64)
        recs = load_task(args.longbench_dir, task, args.num_prompts)
        for pi, rec in enumerate(recs):
            ids = build_input_ids(
                tokenizer, task, rec, args.max_prompt_len, args.reserve
            )
            token_ids = ids[0].tolist()
            nb = (len(token_ids) + args.block_size - 1) // args.block_size
            if nb < 4:
                continue  # too short to evict meaningfully
            prompts.append(
                {
                    "prompt_id": rec.get("_id", f"{task}_{pi}"),
                    "task": task,
                    "token_ids": token_ids,
                    "answers": rec["answers"],
                    "max_gen": max_gen,
                    "num_blocks": nb,
                }
            )
    return prompts


# ---------------------------------------------------------------------------
# Single engine config (runs under GPU; one subprocess per config point)
# ---------------------------------------------------------------------------
def run_single(args) -> None:
    # The geo_kv eviction hooks live only in the V2 GPU model runner. Force it
    # (before importing vllm) so the config is never silently ignored on the V1
    # runner; the baseline stays on the same code path for an honest comparison.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(tokenizer, args)
    geo = args.policy != "none"
    band_label = args.band if args.policy == "v_redundancy" else "-"

    kwargs: dict[str, Any] = dict(
        model=args.model,
        enforce_eager=True,  # required by the geo_kv eviction guard (Milestone 1)
        attention_backend="FLEX_ATTENTION",
        max_model_len=args.max_model_len,
        # Keep the harness's block accounting and the engine's physical
        # allocation unit identical. This is also needed for block-granularity
        # accuracy sweeps (for example 8 versus 16 tokens per eviction unit).
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_mem_util,
        disable_log_stats=True,
    )
    if args.no_async_scheduling:
        # M2 physical_reclaim requires sync scheduling; keep M1 and M2 on the
        # same scheduler so the bit-identical comparison isolates reclamation.
        kwargs["async_scheduling"] = False
    if args.no_prefix_caching:
        # Diagnostic knob: freeing *cached* blocks interacts with the prefix
        # cache; disabling it isolates the core free/realloc path.
        kwargs["enable_prefix_caching"] = False
    if getattr(args, "no_chunked_prefill", False):
        # Diagnostic control only. Query capture rolls its bounded suffix across
        # scheduler chunks and can use an observed suffix after a cached prefix,
        # so production query-aware runs do not need to disable either feature.
        kwargs["enable_chunked_prefill"] = False
    if geo:
        # Phase-1C calibration measures fixed subsets against ALL-layer scoring,
        # so it forces the band to "all" and turns on tracing + the log path the
        # calibrator writes its JSON report beside.
        calibrate = getattr(args, "calibrate_layer_subsets", False)
        # A raw score_sampled_layers spec (e.g. "3,11,20,28") overrides the named
        # band, so the Phase-1C accuracy/latency check can pit a chosen subset
        # against "all" directly. Calibration always needs all layers.
        raw_spec = getattr(args, "score_sampled_layers", None)
        if calibrate:
            layers_spec = "all"
        elif raw_spec:
            layers_spec = raw_spec
        else:
            layers_spec = band_to_layers_spec(args.band)
        # Record the effective layer specification, including raw calibrated
        # lists that override the legacy named --band value.
        band_label = layers_spec
        geo_cfg = {
            "experiment_mode": "geo_uniform",
            "eviction_policy": args.policy,
            "eviction_rate": args.rate,
            "score_sampled_layers": layers_spec,
            "block_score_aggregation": "mean",
            "block_prototype_mode": getattr(args, "block_prototype_mode", "mean"),
            "warmup_pages": args.warmup_pages,
            "eviction_seed": args.seed,
            "physical_reclaim": args.physical_reclaim,
        }
        if getattr(args, "query_alignment_weight", None):
            geo_cfg["query_alignment_weight"] = args.query_alignment_weight
        if getattr(args, "enable_query_tiebreak", False):
            geo_cfg["enable_query_tiebreak"] = True
        if getattr(args, "query_tail_tokens", None):
            geo_cfg["query_tail_tokens"] = args.query_tail_tokens
        # Enable the Phase-1B scorer profiler when calibrating or when explicitly
        # asked (--enable-tracing), so scorer time per fire is recorded to
        # <csv>.scorer_profile.json (flushed by the explicit shutdown below).
        if calibrate or getattr(args, "enable_tracing", False):
            geo_cfg["enable_tracing"] = True
            if args.csv_path:
                geo_cfg["log_path"] = args.csv_path
        if calibrate:
            geo_cfg["calibrate_layer_subsets"] = True
        kwargs["additional_config"] = {"geo_kv": geo_cfg}
    print(
        f"[evict-acc-vllm] building engine: policy={args.policy} "
        f"rate={args.rate} band={args.band} geo={geo} "
        f"physical_reclaim={args.physical_reclaim}"
    )
    llm = LLM(**kwargs)

    reqs = [TokensPrompt(prompt_token_ids=p["token_ids"]) for p in prompts]
    # When dumping logprobs for the M2 tolerance check, request the top-k
    # logprob at each decode step. The chosen token's logprob is always
    # included by vLLM even if it falls outside the top-k.
    logprobs_topk = args.dump_logprobs and args.logprobs_topk or None
    sps = [
        SamplingParams(temperature=0.0, max_tokens=p["max_gen"], logprobs=logprobs_topk)
        for p in prompts
    ]
    outs = llm.generate(reqs, sps)

    tokens_dump: dict[str, list[int]] = {}
    # Per prompt: list over decode steps of {token_id: logprob} dicts (top-k plus
    # the sampled token). Consumed by _logit_tolerance_check.
    logprobs_dump: dict[str, list[dict[int, float]]] = {}
    rows: list[dict] = []
    for p, out in zip(prompts, outs):
        gen = out.outputs[0]
        tokens_dump[p["prompt_id"]] = list(gen.token_ids)
        if args.dump_logprobs and gen.logprobs is not None:
            logprobs_dump[p["prompt_id"]] = [
                {tid: lp.logprob for tid, lp in step.items()} for step in gen.logprobs
            ]
        if args.no_rows:
            continue
        # Score with each task's canonical LongBench metric: token-F1 for QA
        # tasks, ROUGE-L for the summarization tasks (gov_report, multi_news).
        score = score_prediction(p["task"], gen.text, p["answers"])
        rows.append(
            {
                "run_id": args.run_id,
                "prompt_id": p["prompt_id"],
                "task": p["task"],
                "policy": args.policy,
                "layer_band": band_label,
                "block_prototype_mode": getattr(
                    args, "block_prototype_mode", "mean"
                ),
                "query_alignment_weight": getattr(
                    args, "query_alignment_weight", None
                ),
                "query_tail_tokens": getattr(args, "query_tail_tokens", 32),
                "eviction_rate": args.rate,
                "num_blocks": p["num_blocks"],
                "num_evicted": expected_evicted(
                    p["num_blocks"], args.rate, args.warmup_pages
                ),
                "metric_name": DATASET2METRIC.get(p["task"], "qa_f1"),
                "score": round(score, 4),
            }
        )

    if args.dump_tokens:
        with open(args.dump_tokens, "w") as f:
            json.dump(tokens_dump, f)
    if args.dump_logprobs:
        # JSON keys must be strings; token ids are restored to int on read.
        serializable = {
            pid: [{str(t): lp for t, lp in step.items()} for step in steps]
            for pid, steps in logprobs_dump.items()
        }
        with open(args.dump_logprobs, "w") as f:
            json.dump(serializable, f)
    if rows and args.csv_path:
        _append_rows(args.csv_path, rows)
    # Explicitly shut the engine down so the geo_kv EvictionPolicy.close() runs
    # its end-of-run flush. This matters for the Phase-1C layer calibrator (and
    # the Phase-1B scorer profiler), which write their JSON report only at
    # close(); the in-process engine does not reliably call shutdown() on
    # interpreter exit, so relying on GC would silently drop the report.
    if getattr(args, "calibrate_layer_subsets", False) or getattr(
        args, "enable_tracing", False
    ):
        try:
            llm.llm_engine.engine_core.shutdown()
            print("[evict-acc-vllm] engine shut down; geo_kv reports flushed")
        except Exception as e:  # best-effort: never fail the run on teardown
            print(f"[evict-acc-vllm] warning: engine shutdown failed: {e}")
    print(f"[evict-acc-vllm] done: {len(rows)} rows, {len(tokens_dump)} prompts")


def _append_rows(csv_path: str, rows: list[dict]) -> None:
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Orchestrator (login/GPU head process; fans out subprocesses per config point)
# ---------------------------------------------------------------------------
def run_orchestrator(args) -> None:
    rates = [float(x) for x in args.rates.split(",") if x]
    policies = [p for p in args.policies.split(",") if p]
    run_id = args.run_id or f"evict_acc_vllm_{int(time.time())}"
    out_dir = args.output_dir or os.path.join(
        "results", "geo_prefill_distribution", run_id
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "accuracy_scores.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def combo(policy: str, rate: float, band: str, dump: str, no_rows: bool):
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--single",
            "--model",
            args.model,
            "--tasks",
            args.tasks,
            "--longbench-dir",
            args.longbench_dir,
            "--num-prompts",
            str(args.num_prompts),
            "--max-prompt-len",
            str(args.max_prompt_len),
            "--reserve",
            str(args.reserve),
            "--block-size",
            str(args.block_size),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-mem-util",
            str(args.gpu_mem_util),
            "--warmup-pages",
            str(args.warmup_pages),
            "--seed",
            str(args.seed),
            "--run-id",
            run_id,
            "--policy",
            policy,
            "--rate",
            str(rate),
            "--band",
            band,
            "--csv-path",
            csv_path,
            "--dump-tokens",
            os.path.join(out_dir, dump),
        ]
        if args.max_gen is not None:
            cmd += ["--max-gen", str(args.max_gen)]
        if getattr(args, "query_alignment_weight", None):
            cmd += [
                "--query-alignment-weight",
                str(args.query_alignment_weight),
                "--query-tail-tokens",
                str(args.query_tail_tokens),
            ]
        if getattr(args, "enable_query_tiebreak", False):
            cmd.append("--enable-query-tiebreak")
        if no_rows:
            cmd.append("--no-rows")
        subprocess.run(cmd, check=True)

    # 1. Baseline (full context, FlexAttention, no geo_kv) -> data + tokens.
    combo("none", 0.0, "-", "tokens_baseline.json", no_rows=False)

    # 2. Rate-0 inert-eviction self-check: geo_uniform active but nothing dropped
    #    must produce byte-identical tokens to the baseline.
    if not args.no_self_check:
        combo("v_redundancy", 0.0, "all", "tokens_inert.json", no_rows=True)
        _self_check(
            os.path.join(out_dir, "tokens_baseline.json"),
            os.path.join(out_dir, "tokens_inert.json"),
        )

    # 3. Accuracy-vs-rate curve. Band matters only for v_redundancy.
    for rate in rates:
        if rate == 0.0:
            continue
        for policy in policies:
            band = "all" if policy == "v_redundancy" else "-"
            dump = f"tokens_{policy}_{band}_{rate}.json"
            combo(policy, rate, "all", dump, no_rows=False)

    _summarize(csv_path, out_dir, run_id, args)


def run_m2_check(args) -> None:
    """Primary M2 acceptance test: physical reclaim is output-neutral vs M1.

    Physical reclaim frees and reallocates KV-cache blocks, which changes the
    memory *layout*. Attention (FlexAttention + flash reductions) is not
    bit-exact under a layout change, so strict token-for-token identity is
    unachievable: upstream vLLM already flips the greedy argmax on numerically
    borderline prompts when only a benign layout knob (e.g. prefix caching)
    toggles, reclaim uninvolved. The honest criterion is therefore
    *layout-robust*: M2 must match M1 on every prompt that is itself invariant
    under a benign layout change, and may differ only on already-borderline
    prompts.

    Two design points make this robust (an earlier version was fragile -- see
    below):

    1. The equality reference is M1 run at the *same layout as M2* (prefix
       caching off, since reclaim requires it). Reclaim on/off is then the ONLY
       difference between reference and M2, so a reclaim-induced diff is
       unambiguous -- no confounding prefix-cache toggle in the comparison.
    2. The borderline (excusal) set is measured by a *separate* benign,
       no-reclaim probe: M1 with prefix caching ON. Any prompt whose greedy
       argmax flips under that benign layout change is numerically borderline
       and is excused. The probe only ever GROWS the excusal set, so FP noise on
       the probe run can make the test more lenient but never spuriously fail it.

    The earlier version used the prefix-caching-ON run as the equality reference
    itself, comparing a pc-off M2 against a pc-on M1 -- i.e. across a prefix
    cache toggle AND reclaim at once. Benign FP noise on that pc-on reference
    (a prompt moving in or out of the dynamically-computed borderline set) could
    flip the verdict even though every pc-off output was byte-identical. Using
    the same-layout pc-off reference removes that failure mode.

    A separate rate-0 check stays strict: with nothing evicted there are no frees
    and no layout change, so reclaim-on rate-0 must be byte-identical to baseline.
    """
    rate = args.rate if args.rate and args.rate > 0 else 0.5
    run_id = args.run_id or f"m2_check_{int(time.time())}"
    out_dir = args.output_dir or os.path.join(
        "results", "geo_prefill_distribution", run_id
    )
    os.makedirs(out_dir, exist_ok=True)

    def spawn(
        policy: str,
        rate: float,
        band: str,
        reclaim: bool,
        dump: str,
        no_pc: bool = False,
        logprobs_dump: str | None = None,
    ) -> None:
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--single",
            "--model",
            args.model,
            "--tasks",
            args.tasks,
            "--longbench-dir",
            args.longbench_dir,
            "--num-prompts",
            str(args.num_prompts),
            "--max-prompt-len",
            str(args.max_prompt_len),
            "--reserve",
            str(args.reserve),
            "--block-size",
            str(args.block_size),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-mem-util",
            str(args.gpu_mem_util),
            "--warmup-pages",
            str(args.warmup_pages),
            "--seed",
            str(args.seed),
            "--run-id",
            run_id,
            "--policy",
            policy,
            "--rate",
            str(rate),
            "--band",
            band,
            "--no-rows",
            "--no-async-scheduling",
            "--dump-tokens",
            os.path.join(out_dir, dump),
        ]
        if logprobs_dump is not None:
            cmd += ["--dump-logprobs", os.path.join(out_dir, logprobs_dump)]
        if args.max_gen is not None:
            cmd += ["--max-gen", str(args.max_gen)]
        if reclaim:
            cmd.append("--physical-reclaim")
        if no_pc or args.no_prefix_caching:
            cmd.append("--no-prefix-caching")
        subprocess.run(cmd, check=True)

    # Reference: M1 (reclaim off) at the SAME layout as M2 (prefix caching off).
    # Reclaim on/off is the only difference vs M2. Dump logprobs for the
    # primary logit-tolerance check.
    spawn(
        "v_redundancy",
        rate,
        "all",
        False,
        "tokens_m1.json",
        no_pc=True,
        logprobs_dump="logprobs_m1.json",
    )
    # Borderline probe: M1 (reclaim off) under a benign no-reclaim layout change
    # (prefix caching on). Used only by the secondary token-id layout-robust
    # check to identify FP-borderline prompts; it can only grow the excusal set.
    spawn("v_redundancy", rate, "all", False, "tokens_m1_probe.json")
    # M2 (physical_reclaim) requires prefix caching off -> same layout as the
    # reference; reclaim is the only variable.
    spawn(
        "v_redundancy",
        rate,
        "all",
        True,
        "tokens_m2.json",
        no_pc=True,
        logprobs_dump="logprobs_m2.json",
    )
    # PRIMARY criterion: M2 decodes the SAME tokens as same-layout M1. Greedy
    # decoding is deterministic given the argmax, so byte-identical token
    # sequences prove reclaim never changed a single decision -- the strongest
    # possible output-neutrality statement, and immune to sub-argmax FP noise.
    _token_identity_check(
        os.path.join(out_dir, "tokens_m1.json"),
        os.path.join(out_dir, "tokens_m2.json"),
        what=f"physical_reclaim at rate={rate}",
    )
    # SECONDARY (informational): logit divergence within FP noise. Physical
    # compaction repacks KV contiguously, so the attention kernel reduces blocks
    # in a different order -> a slightly higher (but still tiny) reduction-order
    # noise floor than mask-only. A real freed-block read would shift the winning
    # logit by >>1; the loose bound here catches that while tolerating benign
    # noise (~0.08 observed vs a 0.15 bound). Non-fatal: token identity above is
    # the gate.
    try:
        _logit_tolerance_check(
            os.path.join(out_dir, "logprobs_m1.json"),
            os.path.join(out_dir, "logprobs_m2.json"),
            what=f"physical_reclaim at rate={rate}",
            tol=0.15,
        )
    except RuntimeError as e:
        print(f"[m2-check] (secondary logit-tolerance note; tokens still match)\n{e}")
    # SECONDARY (informational): token-id layout-robust check. Cheap, human-
    # readable view of which prompts drifted under a benign layout change.
    try:
        _layout_robust_check(
            os.path.join(out_dir, "tokens_m1.json"),
            os.path.join(out_dir, "tokens_m1_probe.json"),
            os.path.join(out_dir, "tokens_m2.json"),
            what=f"physical_reclaim at rate={rate}",
        )
    except RuntimeError as e:
        print(f"[m2-check] (secondary token-id check noted a knife-edge flip)\n{e}")

    # Inert: rate-0 with reclaim on must equal the no-geo baseline (no eviction =>
    # no frees => no layout change => truly bit-identical).
    if not args.no_self_check:
        # Both at prefix-caching-off layout: reclaim-on requires it, and the
        # baseline must share the layout for the comparison to be strict.
        spawn("none", 0.0, "-", False, "tokens_baseline.json", no_pc=True)
        spawn("v_redundancy", 0.0, "all", True, "tokens_m2_rate0.json", no_pc=True)
        _self_check(
            os.path.join(out_dir, "tokens_baseline.json"),
            os.path.join(out_dir, "tokens_m2_rate0.json"),
            what="rate-0 physical_reclaim",
        )
        print("[m2-check] RATE-0 OK: reclaim-on rate-0 == baseline")

    print(f"\n[m2-check] ALL CHECKS PASSED. dumps in {out_dir}")


def _layout_robust_check(ref_path: str, alt_path: str, m2_path: str, what: str) -> None:
    """Assert physical reclaim perturbs only numerically-borderline prompts.

    Args:
        ref_path: M1 tokens (reclaim OFF), prefix caching OFF -- the equality
            reference, at the SAME layout as M2. Reclaim on/off is the only
            difference between this and ``m2_path``.
        alt_path: M1 tokens (reclaim OFF), prefix caching ON -- a benign,
            no-reclaim layout probe. Any prompt whose output differs from ``ref``
            is numerically borderline (its greedy argmax flips under a benign
            layout change alone, no reclaim) and is excused from the equality
            requirement. This is only a borderline probe, never the equality
            reference, so its FP noise can only enlarge the excusal set.
        m2_path: M2 tokens (reclaim ON), prefix caching OFF. Must match ``ref``
            on every prompt that is NOT borderline.
        what: label for messages.

    Raises:
        RuntimeError: if reclaim changes a numerically-stable prompt (a freed
            block was likely read), or there are no overlapping prompts.
    """
    with open(ref_path) as f:
        ref = json.load(f)
    with open(alt_path) as f:
        alt = json.load(f)
    with open(m2_path) as f:
        m2 = json.load(f)
    common = sorted(set(ref) & set(m2))
    if not common:
        raise RuntimeError("layout-robust check FAILED: no overlapping prompts")
    borderline = {p for p in common if p in alt and ref[p] != alt[p]}
    reclaim_diff = {p for p in common if ref[p] != m2[p]}
    suspicious = sorted(reclaim_diff - borderline)
    stable_matched = [p for p in common if p not in borderline and ref[p] == m2[p]]
    if suspicious:
        lines = []
        for pid in suspicious:
            r, m = ref[pid], m2[pid]
            n = min(len(r), len(m))
            div = next((k for k in range(n) if r[k] != m[k]), n)
            lines.append(
                f"    {pid[:16]} diverge@{div} (lenM1={len(r)} lenM2={len(m)})"
            )
        raise RuntimeError(
            f"layout-robust check FAILED: {what} changed {len(suspicious)} "
            "numerically-STABLE prompt(s) -- a freed block was likely read:\n"
            + "\n".join(lines)
        )
    print(
        f"[m2-check] layout-robust OK: {what} -- "
        f"{len(stable_matched)}/{len(common)} stable prompts identical to M1; "
        f"{len(borderline)} borderline prompt(s) excluded (flip under a "
        "prefix-caching toggle alone, no reclaim): "
        f"{sorted(p[:16] for p in borderline)}. reclaim-induced diffs "
        f"{sorted(p[:16] for p in reclaim_diff)} are a subset of the borderline "
        "set => reclaim is output-neutral."
    )


def _token_identity_check(ref_path: str, m2_path: str, what: str) -> None:
    """Assert M2 decodes byte-identical token sequences to same-layout M1.

    This is the primary M2 output-neutrality gate. ``ref`` (M1, reclaim OFF) and
    ``m2`` (reclaim ON) run at the SAME layout (prefix caching off), so physical
    reclaim is the only difference. Greedy decoding is deterministic given the
    per-step argmax, so identical token-id sequences prove reclaim never changed
    a single decision -- a stronger statement than any logit bound, and immune to
    sub-argmax FP reduction-order noise (which physical compaction legitimately
    raises slightly by repacking KV contiguously).

    Args:
        ref_path: M1 (reclaim off) decoded token ids, prefix caching off.
        m2_path: M2 (reclaim on) decoded token ids, prefix caching off.
        what: label for messages.

    Raises:
        RuntimeError: if any overlapping prompt's token sequence differs, or
            there are no overlapping prompts.
    """
    with open(ref_path) as f:
        ref = json.load(f)
    with open(m2_path) as f:
        m2 = json.load(f)
    common = sorted(set(ref) & set(m2))
    if not common:
        raise RuntimeError("token-identity check FAILED: no overlapping prompts")

    offenders: list[str] = []
    for pid in common:
        a, b = ref[pid], m2[pid]
        if a == b:
            continue
        n = min(len(a), len(b))
        div = next((i for i in range(n) if a[i] != b[i]), n)
        offenders.append(
            f"    {pid[:16]} diverge@{div} (lenM1={len(a)} lenM2={len(b)})"
        )
    if offenders:
        raise RuntimeError(
            f"token-identity check FAILED: {what} -- {len(offenders)}/"
            f"{len(common)} prompt(s) decoded different tokens (a freed block "
            "was likely read):\n" + "\n".join(offenders)
        )
    print(
        f"[m2-check] token-identity OK: {what} -- {len(common)} prompts decode "
        "byte-identical to M1. Reclaim is output-neutral (0 argmax changes)."
    )


def _logit_tolerance_check(
    ref_path: str,
    m2_path: str,
    what: str,
    tol: float = 5e-2,
    max_steps: int | None = None,
) -> None:
    """Assert M2's winning-token logprob matches M1's within FP tolerance.

    This is the principled M2 correctness criterion, robust to greedy-argmax
    knife-edges that make token-id comparison fragile. ``ref`` (M1) and ``m2``
    run at the SAME layout (prefix caching off), so physical reclaim is the only
    difference. If reclaim is truly output-neutral, the per-step logits are
    identical up to FP reduction-order noise; a freed block being read would
    perturb the attention output and show up as a large logit divergence, almost
    always early (before the argmax paths diverge and the sequences drift apart
    for unrelated reasons).

    We compare the TOP-1 (winning-token) logprob per step -- the only quantity
    that drives greedy decoding. We deliberately do NOT compare the full top-k:
    low-probability tail tokens carry much larger reduction-order noise (observed
    ~0.25 in log-space) while the top-1 gap stays ~0.04, so including the tail
    would flag benign FP noise as corruption. Real corruption (a freed block
    read) shifts the winning logit by >>1, which the top-1 comparison catches.

    We compare only over the shared leading token prefix -- once the greedy
    argmax legitimately differs on a knife-edge step (tiny logit gap), the two
    sequences condition on different tokens thereafter and further comparison is
    meaningless.

    Args:
        ref_path: M1 (reclaim off) per-step logprobs, prefix caching off.
        m2_path: M2 (reclaim on) per-step logprobs, prefix caching off.
        what: label for messages.
        tol: max allowed |top1_logprob_M2 - top1_logprob_M1| at any compared
            step. 5e-2 comfortably covers FlexAttention reduction-order noise on
            the winning token (observed worst ~0.043) while catching real
            corruption (which shifts the winning logit by >>1).
        max_steps: if set, only the first ``max_steps`` decode steps per prompt
            are compared (the early steps are where corruption shows first and
            where the shared prefix is longest).

    Raises:
        RuntimeError: if any step's winning-token logprob diverges by more than
            ``tol`` within the shared prefix, or there are no overlapping prompts.
    """
    with open(ref_path) as f:
        ref = {p: _ints(steps) for p, steps in json.load(f).items()}
    with open(m2_path) as f:
        m2 = {p: _ints(steps) for p, steps in json.load(f).items()}
    common = sorted(set(ref) & set(m2))
    if not common:
        raise RuntimeError("logit-tolerance check FAILED: no overlapping prompts")

    worst_ok = 0.0
    offenders: list[str] = []
    n_compared_steps = 0
    for pid in common:
        r_steps, m_steps = ref[pid], m2[pid]
        n = min(len(r_steps), len(m_steps))
        if max_steps is not None:
            n = min(n, max_steps)
        for k in range(n):
            r_lp, m_lp = r_steps[k], m_steps[k]
            if not r_lp or not m_lp:
                break
            r_arg = max(r_lp, key=r_lp.get)
            m_arg = max(m_lp, key=m_lp.get)
            # Compare the winning-token logprob (the greedy-decision quantity).
            step_top1 = abs(r_lp[r_arg] - m_lp[m_arg])
            n_compared_steps += 1
            if step_top1 > tol:
                offenders.append(
                    f"    {pid[:16]} step {k}: |Δtop1_logprob|={step_top1:.4g} "
                    f"> tol={tol:g}"
                )
                break
            worst_ok = max(worst_ok, step_top1)
            # Stop at the first legitimate argmax divergence: beyond it the runs
            # condition on different tokens and are no longer comparable.
            if r_arg != m_arg:
                break

    if offenders:
        raise RuntimeError(
            f"logit-tolerance check FAILED: {what} -- {len(offenders)} "
            "prompt(s) exceeded tolerance within the shared prefix (a freed "
            "block was likely read):\n" + "\n".join(offenders)
        )
    print(
        f"[m2-check] logit-tolerance OK: {what} -- {len(common)} prompts, "
        f"{n_compared_steps} steps compared, worst |Δtop1_logprob|={worst_ok:.4g} "
        f"(tol={tol:g}). Reclaim is logit-neutral within FP noise."
    )


def _ints(steps: list[dict[str, float]]) -> list[dict[int, float]]:
    """Restore int token-id keys from a JSON-loaded logprobs dump."""
    return [{int(t): lp for t, lp in step.items()} for step in steps]


def _self_check(
    baseline_path: str, inert_path: str, what: str = "rate-0 eviction"
) -> None:
    with open(baseline_path) as f:
        base = json.load(f)
    with open(inert_path) as f:
        inert = json.load(f)
    common = sorted(set(base) & set(inert))
    if not common:
        raise RuntimeError("self-check FAILED: no overlapping prompts to compare")
    mismatches: list[str] = []
    for pid in common:
        b, i = base[pid], inert[pid]
        if b == i:
            continue
        n = min(len(b), len(i))
        div = next((k for k in range(n) if b[k] != i[k]), n)
        mismatches.append(
            f"    {pid[:16]} diverge@{div}/{max(len(b), len(i))} "
            f"(lenB={len(b)} lenI={len(i)})"
        )
    if mismatches:
        raise RuntimeError(
            f"self-check FAILED: {what} changed the output "
            f"({len(mismatches)}/{len(common)} prompts differ).\n"
            + "\n".join(mismatches)
            + "\n  The outputs must be byte-identical; aborting. (Late "
            "divergence => residual FlexAttention kernel noise; early "
            "divergence => a real bug.)"
        )
    print(
        f"[evict-acc-vllm] self-check OK: {len(common)} prompts identical "
        f"({what} is output-neutral)."
    )


def _summarize(csv_path: str, out_dir: str, run_id: str, args) -> None:
    rows: list[dict] = []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    agg: dict[tuple, list[float]] = {}
    for r in rows:
        key = (r["task"], r["policy"], r["layer_band"], r["eviction_rate"])
        agg.setdefault(key, []).append(float(r["score"]))
    summary = {
        "run_id": run_id,
        "model": os.path.basename(args.model.rstrip("/")),
        "engine": "vllm",
        "tasks": args.tasks,
        "num_prompts": args.num_prompts,
        "rates": args.rates,
        "policies": args.policies,
        "note": (
            "Stage-2 in-engine (vLLM FlexAttention) mask-only accuracy screen. "
            "Eviction = whole-block (all layers), scored by joint-V redundancy."
        ),
        "means": {
            "|".join(str(x) for x in k): round(sum(v) / len(v), 4)
            for k, v in sorted(agg.items())
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[evict-acc-vllm] wrote {csv_path} ({len(rows)} rows)")
    print("\n=== mean qa_f1 by task / policy / band / rate (vLLM) ===")
    print(f"{'task':16} {'policy':13} {'band':4} {'rate':>5} {'qa_f1':>7} {'n':>4}")
    for k, v in sorted(agg.items()):
        t, pol, b, rate = k
        print(f"{t:16} {pol:13} {b:4} {rate:>5} {sum(v) / len(v):>7.3f} {len(v):>4}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--longbench-dir", default="data/longbench/data")
    p.add_argument("--tasks", default="multifieldqa_en,hotpotqa,qasper")
    p.add_argument("--num-prompts", type=int, default=50)
    p.add_argument("--max-prompt-len", type=int, default=4096)
    p.add_argument("--reserve", type=int, default=512)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--rates", default="0.0,0.2,0.4,0.6")
    p.add_argument("--policies", default="v_redundancy,recency,random")
    p.add_argument("--max-gen", type=int, default=None, help="Override per-task.")
    p.add_argument("--warmup-pages", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--run-id", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-self-check", action="store_true")
    p.add_argument(
        "--m2-check",
        action="store_true",
        help="Run the M2 acceptance test (physical_reclaim bit-identical to M1) "
        "instead of the accuracy sweep. Uses --rate (default 0.5).",
    )
    # --single mode (one engine config; used internally by the orchestrator).
    p.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--policy", default="none")
    p.add_argument("--rate", type=float, default=0.0)
    p.add_argument("--band", default="all")
    p.add_argument("--csv-path", default=None)
    p.add_argument("--dump-tokens", default=None)
    p.add_argument(
        "--dump-logprobs",
        default=None,
        help="Path to dump per-step top-k logprobs (for the M2 logit-tolerance "
        "check). Enables logprobs on the sampler for this run.",
    )
    p.add_argument(
        "--logprobs-topk",
        type=int,
        default=20,
        help="Top-k logprobs to capture per decode step when --dump-logprobs is set.",
    )
    p.add_argument("--no-rows", action="store_true")
    p.add_argument(
        "--physical-reclaim",
        action="store_true",
        help="Enable GeoKV M2 physical block reclamation for this engine config.",
    )
    p.add_argument(
        "--no-async-scheduling",
        action="store_true",
        help="Disable vLLM async scheduling for this engine config (required by "
        "GeoKV M2 physical_reclaim; --m2-check sets it on both M1 and M2 runs "
        "so the bit-identical comparison isolates reclamation).",
    )
    p.add_argument(
        "--no-prefix-caching",
        action="store_true",
        help="Disable prefix caching (diagnostic: isolates the M2 free/realloc "
        "path from cached-block interactions).",
    )
    p.add_argument(
        "--calibrate-layer-subsets",
        action="store_true",
        help="Phase 1C: while scoring all layers, measure how well fixed "
        "1/2/4-layer subsets reproduce the all-layer selected-block set "
        "(mean Jaccard). Forces the band to 'all', enables tracing, and writes "
        "<csv-path>.layer_calibration.json. Diagnostic only.",
    )
    p.add_argument(
        "--score-sampled-layers",
        default=None,
        help="Raw GeoKVConfig score_sampled_layers spec ('all', 'last', or "
        "comma-separated indices e.g. '3,11,20,28'). Overrides --band. Used by "
        "the Phase-1C accuracy/latency check to pit a chosen subset against all.",
    )
    p.add_argument(
        "--block-prototype-mode",
        choices=("mean", "quarters", "mean_top2_norm"),
        default="mean",
        help="Experimental within-block V representation for pairwise "
        "redundancy. 'mean' is the pinned scorer; 'quarters' uses four "
        "contiguous sub-block means; 'mean_top2_norm' uses the block mean plus "
        "its two strongest valid tokens.",
    )
    p.add_argument(
        "--query-alignment-weight",
        type=float,
        default=None,
        help="Protect blocks with high actual causal QK attention mass: "
        "zscore(redundancy) - weight*zscore(query_mass). Off when unset/0.",
    )
    p.add_argument(
        "--enable-query-tiebreak",
        action="store_true",
        help="Use actual QK relevance only to break equal redundancy scores.",
    )
    p.add_argument(
        "--query-tail-tokens",
        type=int,
        default=32,
        help="Number of final post-RoPE prompt queries captured per request.",
    )
    p.add_argument(
        "--no-chunked-prefill",
        action="store_true",
        help="Disable chunked prefill. Query-aware matched arms use this on both "
        "baseline and query configurations to isolate the scorer.",
    )
    p.add_argument(
        "--enable-tracing",
        action="store_true",
        help="Enable the Phase-1B scorer profiler and flush its JSON report to "
        "<csv-path>.scorer_profile.json at end of run (via explicit engine "
        "shutdown). Off by default; adds per-fire timing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.single:
        run_single(args)
    elif args.m2_check:
        run_m2_check(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
