# SPDX-License-Identifier: Apache-2.0
"""Synthetic CUDA timing probe for R2R v2 token-max relevance."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.v1.geo_kv.scoring import (
    block_key_anchor_relevance,
    block_token_max_relevance,
)


def time_cuda(fn, warmup: int, repeats: int) -> dict[str, float]:
    """Measure CUDA latency and peak temporary allocation for one scorer."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    durations: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end))
    peak = torch.cuda.max_memory_allocated() - baseline
    return {
        "median_ms": statistics.median(durations),
        "mean_ms": statistics.mean(durations),
        "std_ms": statistics.stdev(durations) if repeats > 1 else 0.0,
        "peak_extra_mib": max(peak, 0) / (1024 * 1024),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", default="16,64,256")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.query_heads % args.kv_heads or min(
        args.window, args.block_size, args.head_dim, args.repeats
    ) < 1:
        raise ValueError("invalid benchmark shape or repeat count")
    torch.manual_seed(0)
    cases: list[dict[str, object]] = []
    for blocks in (int(x) for x in args.blocks.split(",")):
        if blocks < 1 or blocks * args.block_size < args.window:
            raise ValueError("each cache must contain the query window")
        keys = torch.randn(
            blocks, args.block_size, args.kv_heads, args.head_dim,
            device="cuda", dtype=torch.bfloat16,
        )
        queries = torch.randn(
            args.window, args.query_heads, args.head_dim,
            device="cuda", dtype=torch.bfloat16,
        )
        lengths = torch.full(
            (blocks,), args.block_size, device="cuda", dtype=torch.long
        )
        indices = torch.arange(blocks, device="cuda")
        cheap = time_cuda(
            lambda: block_key_anchor_relevance(queries, keys, lengths),
            args.warmup, args.repeats,
        )
        full = time_cuda(
            lambda: block_token_max_relevance(
                queries, keys, lengths, indices,
                total_valid=blocks * args.block_size,
            ),
            args.warmup, args.repeats,
        )
        band_size = min(blocks, max(4, blocks // 4))
        band = indices[:band_size]
        boundary = time_cuda(
            lambda: block_token_max_relevance(
                queries, keys.index_select(0, band),
                lengths.index_select(0, band), band,
                total_valid=blocks * args.block_size,
            ),
            args.warmup, args.repeats,
        )
        cases.append({
            "blocks": blocks,
            "band_blocks": band_size,
            "cheap_key_anchor": cheap,
            "token_max_full": full,
            "token_max_band": boundary,
        })
    output = {
        "device": torch.cuda.get_device_name(),
        "window": args.window,
        "block_size": args.block_size,
        "query_heads": args.query_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "cases": cases,
    }
    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
