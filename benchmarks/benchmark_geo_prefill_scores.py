#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline driver for geometric KV-cache eviction experiments (Option A).

This is the launcher + config packer for the geo_kv validation experiments. It
mirrors ``plan.md``'s experiment flags on the command line, packs them into
``additional_config={"geo_kv": {...}}``, and drives vLLM via the offline
``LLM`` API (score computation needs in-process KV-cache access, which an HTTP
client cannot reach).

Phase 1 scope: wire the flags end to end and run generation with no behavior
change. When ``--kv-eviction-experiment-mode`` is ``none`` (default), vLLM runs
exactly as upstream. When set to ``score_only``, the worker logs that the
experiment is enabled; the actual per-head score CSV is produced by the worker
in a later phase.

Example (single-node smoke test):

    .venv/bin/python benchmarks/benchmark_geo_prefill_scores.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --num-prompts 8 --input-len 4096 --output-len 1 \
        --enable-prefix-caching \
        --kv-eviction-experiment-mode score_only \
        --geo-score-sampled-layers all --geo-score-sampled-kv-heads all
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline driver for geometric KV eviction experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- model / engine ----------------------------------------------------
    eng = p.add_argument_group("engine")
    eng.add_argument("--model", required=True)
    eng.add_argument("--max-model-len", type=int, default=4096)
    eng.add_argument("--tensor-parallel-size", type=int, default=1)
    eng.add_argument("--dtype", default="auto")
    eng.add_argument("--seed", type=int, default=0)
    eng.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    eng.add_argument("--enable-prefix-caching", action="store_true")
    eng.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        default=None,
        help="Force the number of GPU KV blocks (sets the physical budget).",
    )
    eng.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs (recommended for score computation phases).",
    )

    # -- workload ----------------------------------------------------------
    work = p.add_argument_group("workload")
    work.add_argument("--num-prompts", type=int, default=8)
    work.add_argument(
        "--input-len",
        type=int,
        default=4096,
        help="Prompt length in tokens (fixed for every prompt).",
    )
    work.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Number of leading tokens shared across all prompts.",
    )
    work.add_argument("--output-len", type=int, default=1)
    work.add_argument(
        "--prompt-source",
        default="synthetic",
        choices=["synthetic", "longbench"],
        help="synthetic random tokens (pipeline test) or real LongBench text.",
    )
    work.add_argument(
        "--longbench-task",
        default="multifieldqa_en",
        help="LongBench task name (JSONL under --longbench-dir).",
    )
    work.add_argument(
        "--longbench-dir",
        default="data/longbench/data",
        help="Directory of extracted LongBench <task>.jsonl files.",
    )

    # -- geo_kv experiment flags (mirror plan.md) --------------------------
    geo = p.add_argument_group("geo_kv experiment")
    geo.add_argument(
        "--kv-eviction-experiment-mode",
        default="none",
        choices=[
            "none",
            "score_only",
            "geo_emergent",
            "geo_uniform",
            "geo_frozen_emergent",
            "attention_budget",
            "proxy_attention_budget",
        ],
    )
    geo.add_argument("--enable-kv-eviction-tracing", action="store_true")
    geo.add_argument("--kv-eviction-log-path", default=None)
    geo.add_argument("--geo-total-page-cap", type=int, default=None)
    geo.add_argument("--geo-admission-threshold", type=float, default=None)
    geo.add_argument("--geo-warmup-pages", type=int, default=None)
    geo.add_argument("--geo-score-sampled-layers", default="all")
    geo.add_argument("--geo-score-sampled-kv-heads", default="all")
    geo.add_argument(
        "--geo-score-token-level",
        action="store_true",
        help="Diagnostic: also score redundancy per token (no block pooling), "
        "writing token_level_scores.csv alongside per_head_scores.csv.",
    )
    geo.add_argument(
        "--geo-token-sample-cap",
        type=int,
        default=4096,
        help="Max tokens per request used for token-level scoring (bounds "
        "the O(N^2) similarity cost).",
    )
    geo.add_argument(
        "--geo-score-norm-variants",
        action="store_true",
        help="Diagnostic: also score redundancy under raw/center/whiten "
        "normalizations (block + token units) -> normalization_scores.csv.",
    )
    geo.add_argument(
        "--geo-norm-variants",
        default="raw,center_request,whiten_request",
        help="Comma-separated normalization variants to evaluate.",
    )
    geo.add_argument(
        "--geo-score-block-positional",
        action="store_true",
        help="Diagnostic: emit per-(request,block) joint-V redundancy (mean "
        "over layers) -> block_positional_scores.csv for the positional control.",
    )
    geo.add_argument(
        "--geo-head-redundancy-stat",
        default="percentile_90",
        choices=["max", "topk_mean", "percentile_90"],
    )
    geo.add_argument(
        "--geo-block-score-aggregation",
        default="percentile_90",
        choices=["mean", "max", "topk_mean", "percentile_90"],
    )
    geo.add_argument("--geo-topk-frac", type=float, default=0.1)
    geo.add_argument("--geo-freeze-after-prefill", action="store_true")
    geo.add_argument(
        "--geo-overflow-eviction-mode",
        default="repeated_single",
        choices=["repeated_single"],
    )

    # -- output ------------------------------------------------------------
    out = p.add_argument_group("output")
    out.add_argument(
        "--output-dir",
        default=None,
        help="Where experiment artifacts are written. Defaults to "
        "results/geo_prefill_distribution/<run_id>.",
    )
    out.add_argument("--run-id", default=None)
    out.add_argument(
        "--dataset",
        default="synthetic_fixed",
        help="Free-form dataset label recorded in per_head_scores.csv.",
    )

    return p.parse_args()


def build_geo_kv_dict(args: argparse.Namespace) -> dict:
    """Pack CLI flags into the additional_config['geo_kv'] block."""
    return {
        "experiment_mode": args.kv_eviction_experiment_mode,
        "enable_tracing": args.enable_kv_eviction_tracing,
        "log_path": args.kv_eviction_log_path,
        "total_page_cap": args.geo_total_page_cap,
        "admission_threshold": args.geo_admission_threshold,
        "warmup_pages": args.geo_warmup_pages,
        "score_sampled_layers": args.geo_score_sampled_layers,
        "score_sampled_kv_heads": args.geo_score_sampled_kv_heads,
        "score_token_level": args.geo_score_token_level,
        "token_sample_cap": args.geo_token_sample_cap,
        "score_norm_variants": args.geo_score_norm_variants,
        "norm_variants": args.geo_norm_variants,
        "score_block_positional": args.geo_score_block_positional,
        "head_redundancy_stat": args.geo_head_redundancy_stat,
        "block_score_aggregation": args.geo_block_score_aggregation,
        "topk_frac": args.geo_topk_frac,
        "freeze_after_prefill": args.geo_freeze_after_prefill,
        "overflow_eviction_mode": args.geo_overflow_eviction_mode,
        "output_dir": args.output_dir,
        "run_id": args.run_id,
        "dataset": args.dataset,
    }


def build_prompts(
    num_prompts: int,
    input_len: int,
    prefix_len: int,
    seed: int,
    vocab_lo: int = 10,
    vocab_hi: int = 30000,
) -> list[list[int]]:
    """Build fixed-length synthetic token-id prompts.

    A common ``prefix_len`` prefix is shared across all prompts (useful with
    prefix caching); the remainder is per-prompt random ids. Ids stay in a
    conservative range valid for common tokenizers. Synthetic prompts are fine
    for wiring/plumbing; realistic datasets can be added in a later phase.
    """
    prefix_len = max(0, min(prefix_len, input_len))
    base = random.Random(seed)
    prefix = [base.randint(vocab_lo, vocab_hi) for _ in range(prefix_len)]
    prompts: list[list[int]] = []
    for i in range(num_prompts):
        rng = random.Random(seed * 1_000_003 + i)
        suffix_len = input_len - prefix_len
        suffix = [rng.randint(vocab_lo, vocab_hi) for _ in range(suffix_len)]
        prompts.append(prefix + suffix)
    return prompts


def build_longbench_prompts(
    data_dir: str,
    task: str,
    num_prompts: int,
    input_len: int,
    tokenizer,
) -> list[list[int]]:
    """Load real long-context prompts from a local LongBench task JSONL.

    Each example's ``context`` (the long document) is tokenized with the model
    tokenizer and truncated to ``input_len`` tokens. Examples yielding fewer
    than two tokens are skipped.
    """
    import json

    path = os.path.join(data_dir, f"{task}.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"LongBench task file not found: {path}")
    prompts: list[list[int]] = []
    with open(path) as f:
        for line in f:
            if len(prompts) >= num_prompts:
                break
            ctx = json.loads(line).get("context", "")
            if not ctx:
                continue
            # Cap chars before tokenizing to avoid encoding huge documents.
            ids = tokenizer.encode(ctx[: input_len * 8])
            if len(ids) < 2:
                continue
            prompts.append(ids[:input_len])
    if not prompts:
        raise SystemExit(f"No usable prompts from {path}")
    return prompts


def main() -> None:
    args = parse_args()
    if args.prompt_source == "longbench" and args.dataset == "synthetic_fixed":
        args.dataset = f"longbench_{args.longbench_task}"

    # Prompts are capped at input_len tokens; the engine rejects a request when
    # prompt_len + output_len exceeds max_model_len. Give the sampled tokens room
    # so full-length prompts (== input_len) are not dropped.
    needed_len = args.input_len + args.output_len
    if args.max_model_len < needed_len:
        print(
            f"[geo_kv driver] max_model_len {args.max_model_len} < input_len + "
            f"output_len ({needed_len}); bumping max_model_len to {needed_len}."
        )
        args.max_model_len = needed_len

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        "results", "geo_prefill_distribution", run_id
    )
    # Resolve to an absolute path: the scorer runs in the EngineCore subprocess,
    # so a relative path could resolve against a different cwd.
    output_dir = os.path.abspath(output_dir)
    args.run_id = run_id
    args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    geo_kv = build_geo_kv_dict(args)

    print("=" * 70)
    print(f"[geo_kv driver] run_id={run_id}")
    print(f"[geo_kv driver] output_dir={output_dir}")
    print(f"[geo_kv driver] experiment_mode={geo_kv['experiment_mode']}")
    print(f"[geo_kv driver] geo_kv={json.dumps(geo_kv)}")
    print("=" * 70)

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        enforce_eager=args.enforce_eager,
        additional_config={"geo_kv": geo_kv},
    )

    if args.prompt_source == "longbench":
        prompts = build_longbench_prompts(
            args.longbench_dir,
            args.longbench_task,
            args.num_prompts,
            args.input_len,
            llm.get_tokenizer(),
        )
        lengths = sorted({len(p) for p in prompts})
        print(
            f"[geo_kv driver] loaded {len(prompts)} LongBench prompts "
            f"(task={args.longbench_task}, token lengths={lengths[:5]}...)"
        )
    else:
        prompts = build_prompts(
            num_prompts=args.num_prompts,
            input_len=args.input_len,
            prefix_len=args.prefix_len,
            seed=args.seed,
        )
    sampling = SamplingParams(
        max_tokens=args.output_len, temperature=0.0, ignore_eos=True
    )

    t0 = time.perf_counter()
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts], sampling)
    elapsed = time.perf_counter() - t0

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(
        f"[geo_kv driver] generated {len(outputs)} sequences, "
        f"{total_out_tokens} output tokens in {elapsed:.2f}s"
    )

    # Phase 1 confirmation artifact. Later phases write per_head_scores.csv and
    # summary.json into the same directory (from inside the worker).
    run_config = {
        "run_id": run_id,
        "model": args.model,
        "num_prompts": args.num_prompts,
        "input_len": args.input_len,
        "prefix_len": args.prefix_len,
        "output_len": args.output_len,
        "prompt_source": args.prompt_source,
        "longbench_task": args.longbench_task
        if args.prompt_source == "longbench"
        else None,
        "enable_prefix_caching": args.enable_prefix_caching,
        "num_gpu_blocks_override": args.num_gpu_blocks_override,
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "geo_kv": geo_kv,
        "elapsed_s": elapsed,
        "total_output_tokens": total_out_tokens,
    }
    run_config_path = os.path.join(output_dir, "run_config.json")
    with open(run_config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"[geo_kv driver] wrote {run_config_path}")


if __name__ == "__main__":
    main()
