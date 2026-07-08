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
    build_input_ids,
    load_task,
    qa_f1_max,
)

POLICIES = ("v_redundancy", "recency", "random")
CSV_FIELDS = [
    "run_id",
    "prompt_id",
    "task",
    "policy",
    "layer_band",
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
        gpu_memory_utilization=args.gpu_mem_util,
        disable_log_stats=True,
    )
    if geo:
        kwargs["additional_config"] = {
            "geo_kv": {
                "experiment_mode": "geo_uniform",
                "eviction_policy": args.policy,
                "eviction_rate": args.rate,
                "score_sampled_layers": band_to_layers_spec(args.band),
                "block_score_aggregation": "mean",
                "warmup_pages": args.warmup_pages,
                "eviction_seed": args.seed,
            }
        }
    print(
        f"[evict-acc-vllm] building engine: policy={args.policy} "
        f"rate={args.rate} band={args.band} geo={geo}"
    )
    llm = LLM(**kwargs)

    reqs = [TokensPrompt(prompt_token_ids=p["token_ids"]) for p in prompts]
    sps = [SamplingParams(temperature=0.0, max_tokens=p["max_gen"]) for p in prompts]
    outs = llm.generate(reqs, sps)

    tokens_dump: dict[str, list[int]] = {}
    rows: list[dict] = []
    for p, out in zip(prompts, outs):
        gen = out.outputs[0]
        tokens_dump[p["prompt_id"]] = list(gen.token_ids)
        if args.no_rows:
            continue
        score = qa_f1_max(gen.text, p["answers"])
        rows.append(
            {
                "run_id": args.run_id,
                "prompt_id": p["prompt_id"],
                "task": p["task"],
                "policy": args.policy,
                "layer_band": band_label,
                "eviction_rate": args.rate,
                "num_blocks": p["num_blocks"],
                "num_evicted": expected_evicted(
                    p["num_blocks"], args.rate, args.warmup_pages
                ),
                "metric_name": "qa_f1",
                "score": round(score, 4),
            }
        )

    if args.dump_tokens:
        with open(args.dump_tokens, "w") as f:
            json.dump(tokens_dump, f)
    if rows and args.csv_path:
        _append_rows(args.csv_path, rows)
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


def _self_check(baseline_path: str, inert_path: str) -> None:
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
            "self-check FAILED: rate-0 eviction changed the output "
            f"({len(mismatches)}/{len(common)} prompts differ).\n"
            + "\n".join(mismatches)
            + "\n  The no-eviction path must be byte-identical to upstream; "
            "aborting. (Late divergence => residual FlexAttention kernel noise; "
            "early divergence => a real mask-logic bug.)"
        )
    print(
        f"[evict-acc-vllm] self-check OK: {len(common)} prompts identical "
        "(rate-0 eviction is inert)."
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
    # --single mode (one engine config; used internally by the orchestrator).
    p.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--policy", default="none")
    p.add_argument("--rate", type=float, default=0.0)
    p.add_argument("--band", default="all")
    p.add_argument("--csv-path", default=None)
    p.add_argument("--dump-tokens", default=None)
    p.add_argument("--no-rows", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.single:
        run_single(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
