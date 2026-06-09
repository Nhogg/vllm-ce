# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot cache-eviction benchmark JSON outputs."""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from glob import glob
from math import ceil
from pathlib import Path
from typing import Any, Callable, NamedTuple


POLICY_ORDER = {
    "lru": 0,
    "random": 1,
    "rand": 1,
    "arc": 2,
    "paged": 3,
}

class MetricSpec(NamedTuple):
    key: str
    title: str
    group: str
    getter: Callable[[dict[str, Any]], float | None]


def nested_value(result: dict[str, Any], *paths: str) -> float | None:
    for path in paths:
        value: Any = result
        for part in path.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
    return None


def avg_ms_per_output_token(result: dict[str, Any]) -> float | None:
    explicit = nested_value(
        result,
        "time_per_output_token_ms",
        "tpot_ms",
        "latency_stats.time_per_output_token_ms",
    )
    if explicit is not None:
        return explicit

    wall_time_s = nested_value(result, "wall_time_s")
    total_output_tokens = nested_value(result, "total_output_tokens")
    if wall_time_s is None or not total_output_tokens:
        return None
    return wall_time_s * 1000.0 / total_output_tokens


REQUESTED_METRICS = [
    MetricSpec(
        "time_to_first_token_ms",
        "Time to First Token (ms, lower is better)",
        "latency",
        lambda result: nested_value(
            result,
            "time_to_first_token_ms",
            "ttft_ms",
            "latency_stats.time_to_first_token_ms",
            "latency_stats.ttft_ms",
        ),
    ),
    MetricSpec(
        "inter_token_latency_ms",
        "Inter-Token Latency (ms, lower is better)",
        "latency",
        lambda result: nested_value(
            result,
            "inter_token_latency_ms",
            "itl_ms",
            "latency_stats.inter_token_latency_ms",
            "latency_stats.itl_ms",
        ),
    ),
    MetricSpec(
        "time_per_output_token_ms",
        "Avg E2E Time per Output Token (ms, lower is better)",
        "latency",
        avg_ms_per_output_token,
    ),
    MetricSpec(
        "end_to_end_latency_ms",
        "End-to-End Latency Mean (ms, lower is better)",
        "latency",
        lambda result: nested_value(
            result,
            "end_to_end_latency_ms",
            "latency_stats.mean_ms",
        ),
    ),
    MetricSpec(
        "tail_latency_p95_ms",
        "Tail Latency P95 (ms, lower is better)",
        "latency",
        lambda result: nested_value(
            result,
            "tail_latency_p95_ms",
            "latency_stats.p95_ms",
        ),
    ),
    MetricSpec(
        "tail_latency_p99_ms",
        "Tail Latency P99 (ms, lower is better)",
        "latency",
        lambda result: nested_value(
            result,
            "tail_latency_p99_ms",
            "tail_latency_ms",
            "latency_stats.p99_ms",
        ),
    ),
    MetricSpec(
        "tokens_per_second",
        "Tokens / sec (higher is better)",
        "throughput",
        lambda result: nested_value(result, "tokens_per_second", "throughput_tok_s"),
    ),
    MetricSpec(
        "prefix_cache_hit_rate",
        "Prefix Cache Hit Rate (higher is better)",
        "cache",
        lambda result: nested_value(result, "prefix_cache_hit_rate", "cache_hit_rate"),
    ),
    MetricSpec(
        "cache_miss_penalty_ms",
        "Cache Miss Penalty (ms, lower is better)",
        "cache",
        lambda result: nested_value(
            result,
            "cache_miss_penalty_ms",
            "cache_miss_penalty",
        ),
    ),
    MetricSpec(
        "recompute_cost_avoided",
        "Recompute Cost Avoided (higher is better)",
        "cache",
        lambda result: nested_value(
            result,
            "recompute_cost_avoided",
            "recompute_cost_avoided_tokens",
        ),
    ),
    MetricSpec(
        "eviction_regret",
        "Eviction Regret (lower is better)",
        "cache",
        lambda result: nested_value(result, "eviction_regret"),
    ),
    MetricSpec(
        "gpu_kv_cache_util",
        "GPU KV Cache Utilization (higher is better)",
        "cache",
        lambda result: nested_value(
            result,
            "gpu_kv_cache_util",
            "gpu_kv_cache_utilization",
        ),
    ),
]
METRIC_BY_KEY = {metric.key: metric for metric in REQUESTED_METRICS}


def load_result(path: Path) -> dict[str, Any]:
    with path.open() as f:
        result = json.load(f)
    result["_path"] = str(path)
    result["_name"] = path.stem
    return result


def metric_value(result: dict[str, Any], metric: MetricSpec) -> float | None:
    return metric.getter(result)


def available_metrics(results: list[dict[str, Any]]) -> list[MetricSpec]:
    return [
        metric
        for metric in REQUESTED_METRICS
        if any(metric_value(result, metric) is not None for result in results)
    ]


def missing_metrics(results: list[dict[str, Any]]) -> list[MetricSpec]:
    return [
        metric
        for metric in REQUESTED_METRICS
        if all(metric_value(result, metric) is None for result in results)
    ]


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


def group_results(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[cache_eviction_label(result)].append(result)
    return dict(grouped)


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


def write_summary_csv(
    results: list[dict[str, Any]],
    output_csv: Path,
    metrics: list[MetricSpec],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "label",
        "cache_eviction_type",
        "cache_eviction_policy",
        "eviction_policy",
        "active_kv_eviction_policy",
        "active_kv_eviction_cache_budget_tokens",
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
    fields.extend(metric.key for metric in metrics if metric.key not in fields)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {
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
            row.update(
                {
                    metric.key: metric_value(result, metric)
                    for metric in metrics
                }
            )
            writer.writerow(row)


def draw_metric_axes(
    plt: Any,
    results: list[dict[str, Any]],
    metrics: list[MetricSpec],
) -> Any:
    grouped = group_results(results)
    labels = sorted(grouped, key=lambda label: sort_key(grouped[label][0]))
    cols = min(3, max(1, len(metrics)))
    rows = ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.2 * rows))
    if hasattr(axes, "ravel"):
        axes_list = list(axes.ravel())
    else:
        axes_list = [axes]
    for axis, metric in zip(axes_list, metrics):
        means: list[float] = []
        stds: list[float] = []
        for label in labels:
            values = [
                value
                for result in grouped[label]
                if (value := metric_value(result, metric)) is not None
            ]
            if values:
                means.append(sum(values) / len(values))
                if len(values) > 1:
                    mean = means[-1]
                    variance = sum(
                        (value - mean) ** 2 for value in values
                    ) / len(values)
                    stds.append(variance**0.5)
                else:
                    stds.append(0.0)
            else:
                means.append(0.0)
                stds.append(0.0)

        x = list(range(len(labels)))
        axis.bar(x, means, yerr=stds, capsize=4)
        axis.set_title(metric.title)
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.3)

    if len(axes_list) > len(metrics):
        for axis in axes_list[len(metrics) :]:
            axis.axis("off")
    return fig


def plot_results(results: list[dict[str, Any]], output_png: Path) -> list[Path]:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vllm-matplotlib")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it in your venv, "
            "or run with --no-plot."
        ) from exc

    written_paths: list[Path] = []
    metrics = available_metrics(results)
    metric_groups = [
        ("latency", "Latency Metrics"),
        ("throughput", "Throughput Metrics"),
        ("cache", "Cache Metrics"),
    ]

    for group, title in metric_groups:
        group_metrics = [metric for metric in metrics if metric.group == group]
        if not group_metrics:
            continue
        suffix = "" if group == "latency" else f"_{group}"
        path = output_png.with_name(f"{output_png.stem}{suffix}{output_png.suffix}")
        fig = draw_metric_axes(plt, results, group_metrics)
        fig.suptitle(f"Cache Eviction Comparison: {title}", y=1.02)
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written_paths.append(path)

    grouped = group_results(results)
    labels = sorted(grouped, key=lambda label: sort_key(grouped[label][0]))

    boxplot_path = output_png.with_name(
        f"{output_png.stem}_latency_boxplot{output_png.suffix}"
    )
    latency_samples = [
        [
            value
            for result in grouped[label]
            for value in result.get("per_request_latency_ms", [])
        ]
        for label in labels
    ]
    if any(samples for samples in latency_samples):
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5))
        ax.boxplot(latency_samples, tick_labels=labels, showfliers=False)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Per-Request Latency Distribution (lower is better)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(boxplot_path, dpi=160)
        plt.close(fig)
        written_paths.append(boxplot_path)
    return written_paths


def expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.json")))
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

    results = []
    for path in paths:
        try:
            results.append(load_result(path))
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON file: {path}", file=sys.stderr)
    results.sort(key=sort_key)

    if not results:
        raise SystemExit("No valid result JSON files found.")

    models = {result.get("model") for result in results}
    shapes = {
        (
            result.get("input_length_range"),
            result.get("prefix_len"),
            result.get("output_len"),
            result.get("max_model_len"),
        )
        for result in results
    }
    if len(models) > 1 or len(shapes) > 1:
        print(
            "Warning: input directory contains multiple benchmark shapes or models; "
            "plots compare them as-is.",
            file=sys.stderr,
        )

    write_summary_csv(results, args.summary_csv, REQUESTED_METRICS)
    print(f"Wrote {args.summary_csv}")

    missing = missing_metrics(results)
    if missing:
        missing_names = ", ".join(metric.key for metric in missing)
        print(
            "Warning: requested metrics missing from these JSONs: "
            f"{missing_names}",
            file=sys.stderr,
        )

    if not args.no_plot:
        for path in plot_results(results, args.output_png):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
