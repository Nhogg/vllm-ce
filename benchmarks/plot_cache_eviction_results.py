# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot cache-eviction benchmark JSON outputs."""

import argparse
import csv
import json
import os
from glob import glob
from pathlib import Path
from typing import Any


POLICY_ORDER = {
    "lru": 0,
    "random": 1,
    "rand": 1,
    "arc": 2,
    "paged": 3,
}


def load_result(path: Path) -> dict[str, Any]:
    with path.open() as f:
        result = json.load(f)
    result["_path"] = str(path)
    result["_name"] = path.stem
    return result


def cache_eviction_type(result: dict[str, Any]) -> str:
    if result.get("cache_eviction_type"):
        return result["cache_eviction_type"]
    active_policy = result.get("active_kv_eviction_policy") or "none"
    return "active_kv" if active_policy != "none" else "prefix"


def cache_eviction_policy(result: dict[str, Any]) -> str:
    if result.get("cache_eviction_policy"):
        return result["cache_eviction_policy"]
    active_policy = result.get("active_kv_eviction_policy") or "none"
    if active_policy != "none":
        return active_policy
    return result.get("eviction_policy", "unknown")


def cache_eviction_label(result: dict[str, Any]) -> str:
    if result.get("cache_eviction_label"):
        return result["cache_eviction_label"]

    policy = cache_eviction_policy(result)
    if cache_eviction_type(result) == "active_kv":
        budget = result.get("active_kv_eviction_cache_budget_tokens")
        if budget is not None:
            return f"{policy}-{budget}"
    return policy


def sort_key(result: dict[str, Any]) -> tuple[int, str, str]:
    policy = cache_eviction_policy(result)
    return (
        POLICY_ORDER.get(policy, 100),
        cache_eviction_label(result),
        result["_name"],
    )


def unique_labels(results: list[dict[str, Any]]) -> list[str]:
    labels = [cache_eviction_label(result) for result in results]
    if len(labels) == len(set(labels)):
        return labels

    seen: dict[str, int] = {}
    unique = []
    for label, result in zip(labels, results):
        seen[label] = seen.get(label, 0) + 1
        if seen[label] == 1:
            unique.append(label)
        else:
            unique.append(f"{label} [{result['_name']}]")
    return unique


def write_summary_csv(results: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "label",
        "cache_eviction_type",
        "cache_eviction_policy",
        "eviction_policy",
        "active_kv_eviction_policy",
        "active_kv_eviction_cache_budget_tokens",
        "throughput_tok_s",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
        "n",
        "total_input_tokens",
        "total_output_tokens",
        "wall_time_s",
        "model",
        "input_length_range",
        "prefix_len",
        "output_len",
        "num_prompts",
        "repeat_count",
        "warmup_rounds",
        "num_gpu_blocks_override",
        "max_model_len",
        "seed",
        "path",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            stats = result.get("latency_stats", {})
            writer.writerow(
                {
                    "name": result["_name"],
                    "label": cache_eviction_label(result),
                    "cache_eviction_type": cache_eviction_type(result),
                    "cache_eviction_policy": cache_eviction_policy(result),
                    "eviction_policy": result.get("eviction_policy"),
                    "active_kv_eviction_policy": result.get(
                        "active_kv_eviction_policy"
                    ),
                    "active_kv_eviction_cache_budget_tokens": result.get(
                        "active_kv_eviction_cache_budget_tokens"
                    ),
                    "throughput_tok_s": result.get("throughput_tok_s"),
                    "mean_ms": stats.get("mean_ms"),
                    "p50_ms": stats.get("p50_ms"),
                    "p95_ms": stats.get("p95_ms"),
                    "p99_ms": stats.get("p99_ms"),
                    "min_ms": stats.get("min_ms"),
                    "max_ms": stats.get("max_ms"),
                    "n": stats.get("n"),
                    "total_input_tokens": result.get("total_input_tokens"),
                    "total_output_tokens": result.get("total_output_tokens"),
                    "wall_time_s": result.get("wall_time_s"),
                    "model": result.get("model"),
                    "input_length_range": result.get("input_length_range"),
                    "prefix_len": result.get("prefix_len"),
                    "output_len": result.get("output_len"),
                    "num_prompts": result.get("num_prompts"),
                    "repeat_count": result.get("repeat_count"),
                    "warmup_rounds": result.get("warmup_rounds"),
                    "num_gpu_blocks_override": result.get("num_gpu_blocks_override"),
                    "max_model_len": result.get("max_model_len"),
                    "seed": result.get("seed"),
                    "path": result["_path"],
                }
            )


def plot_results(results: list[dict[str, Any]], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vllm-matplotlib")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it in your venv, "
            "or run with --no-plot."
        ) from exc

    labels = unique_labels(results)
    x = list(range(len(results)))
    latency_metrics = ["p50_ms", "p95_ms", "p99_ms", "mean_ms"]

    fig, axes = plt.subplots(3, 1, figsize=(max(8, len(results) * 1.8), 12))

    throughputs = [result.get("throughput_tok_s", 0.0) for result in results]
    axes[0].bar(x, throughputs)
    axes[0].set_ylabel("Throughput (tok/s)")
    axes[0].set_title("Cache Eviction Throughput")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.3)

    width = 0.18
    for offset, metric in enumerate(latency_metrics):
        values = [
            result.get("latency_stats", {}).get(metric, 0.0)
            for result in results
        ]
        positions = [i + (offset - 1.5) * width for i in x]
        axes[1].bar(positions, values, width=width, label=metric)
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("End-to-End Latency Summary")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    latency_samples = [
        result.get("per_request_latency_ms", [])
        for result in results
    ]
    axes[2].boxplot(latency_samples, tick_labels=labels, showfliers=False)
    axes[2].set_ylabel("Latency (ms)")
    axes[2].set_title("Per-Request Latency Distribution")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json")))
        else:
            matches = [Path(match) for match in sorted(glob(str(path)))]
            paths.extend(matches or [path])
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot JSON outputs from cache eviction benchmark runs."
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="Result JSON files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("cache_eviction_results.png"),
        help="Path for the generated plot.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("cache_eviction_results.csv"),
        help="Path for a flat CSV summary.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only write the CSV summary; do not require matplotlib.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.results)
    if not paths:
        raise SystemExit("No result JSON files found.")

    results = [load_result(path) for path in paths]
    results.sort(key=sort_key)

    write_summary_csv(results, args.summary_csv)
    print(f"Wrote {args.summary_csv}")
    if not args.no_plot:
        plot_results(results, args.output_png)
        print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
