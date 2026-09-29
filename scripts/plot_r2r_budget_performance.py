#!/usr/bin/env python3
"""Plot performance metrics from the all-method explicit-budget sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

MODEL_ORDER = (
    ("Llama-3.2-1B-Instruct", "Llama 3.2 1B"),
    ("Llama-3.2-3B-Instruct", "Llama 3.2 3B"),
    ("Llama-3.1-8B-Instruct", "Llama 3.1 8B"),
    ("DeepSeek-R1-Distill-Qwen-32B", "DeepSeek R1 Qwen 32B"),
    ("Qwen3.8-Flash-Next", "Qwen 3.8 Flash Next"),
)
TASKS = (
    ("qasper", "Qasper"),
    ("gov_report", "GovReport"),
    ("multi_news", "MultiNews"),
    ("hotpotqa", "HotpotQA"),
    ("multifieldqa_en", "MultiFieldQA"),
)
METHODS = (
    ("v_redundancy", "R2R", "#159947", "D"),
    ("paged_eviction", "PagedEviction", "#d81b60", "^"),
    ("recency", "StreamingLLM", "#e67e22", "s"),
)
METRICS = (
    ("throughput_tok_s", "Output throughput (tok/s)", "throughput"),
    ("mean_ttft_ms", "Mean TTFT (ms)", "ttft"),
    ("mean_tpot_ms", "Mean TPOT (ms)", "tpot"),
    ("steady_blocks", "Steady GPU KV blocks", "steady_kv_blocks"),
    ("decode_evictions", "Decode eviction events", "decode_evictions"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/r2r_budget_all_methods_accuracy"),
        help="Root produced by the all-method explicit-budget sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to ROOT/charts/performance.",
    )
    return parser.parse_args()


def load_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = "r2r_budget_all_methods_*/capband_accuracy.csv"
    for path in sorted(root.glob(pattern)):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no completed budget CSVs found under {root}")
    return rows


def load_baselines(root: Path) -> dict[tuple[str, str], dict[str, object]]:
    baselines: dict[tuple[str, str], dict[str, object]] = {}
    for path in sorted(root.glob("r2r_budget_all_methods_*/*_base.json")):
        with path.open() as handle:
            data = json.load(handle)
        baselines[(str(data["model"]), str(data["task"]))] = data
    return baselines


def plot_metric(
    rows: list[dict[str, str]],
    baselines: dict[tuple[str, str], dict[str, object]],
    field: str,
    ylabel: str,
    output: Path,
) -> None:
    observed_models = {row["model"] for row in rows}
    models = [item for item in MODEL_ORDER if item[0] in observed_models]
    models.extend((name, name) for name in sorted(observed_models - {m for m, _ in models}))
    capacities = sorted({int(row["capacity"]) for row in rows})
    fig, axes = plt.subplots(
        len(models),
        5,
        figsize=(20, 3.5 * len(models)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    for i, (model, model_label) in enumerate(models):
        for j, (task, task_label) in enumerate(TASKS):
            ax = axes[i, j]
            panel = [
                row
                for row in rows
                if row["model"] == model and row["task"] == task
            ]
            for policy, label, color, marker in METHODS:
                method_rows = sorted(
                    (row for row in panel if row["eviction_policy"] == policy),
                    key=lambda row: int(row["capacity"]),
                )
                if not method_rows:
                    continue
                ax.plot(
                    [int(row["capacity"]) for row in method_rows],
                    [float(row[field]) for row in method_rows],
                    color=color,
                    label=label,
                    marker=marker,
                    markersize=4.5,
                    linewidth=1.8,
                )
            baseline = baselines.get((model, task))
            if baseline is not None and field != "decode_evictions":
                ax.axhline(
                    float(baseline[field]),
                    color="#303030",
                    linestyle="--",
                    linewidth=1.4,
                    label="No eviction",
                )
            if i == 0:
                ax.set_title(task_label, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{model_label}\n{ylabel}")
            if i == len(models) - 1:
                ax.set_xlabel("KV capacity C (blocks)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(capacities, [str(value) for value in capacities])
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(f"{ylabel} across methods and per-request KV capacities")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.root)
    baselines = load_baselines(args.root)
    output_dir = args.output_dir or args.root / "charts" / "performance"
    for field, ylabel, stem in METRICS:
        output = output_dir / f"{stem}_by_kv_budget.png"
        plot_metric(rows, baselines, field, ylabel, output)
        print(f"wrote {output}")
    print(f"plotted {len(rows)} method/budget measurements")


if __name__ == "__main__":
    main()
