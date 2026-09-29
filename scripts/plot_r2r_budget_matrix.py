#!/usr/bin/env python3
"""Plot all eviction methods over explicit block capacities."""

from __future__ import annotations

import argparse
import csv
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
    ("v_redundancy", "R2R", "#2563eb"),
    ("paged_eviction", "PagedEviction", "#f59e0b"),
    ("recency", "StreamingLLM", "#dc2626"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/r2r_budget_all_methods_accuracy"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_rows(root: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(root.glob("r2r_budget_all_methods_*/capband_accuracy.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no completed budget CSVs found under {root}")
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.root)
    observed_models = {row["model"] for row in rows}
    models = [item for item in MODEL_ORDER if item[0] in observed_models]
    models.extend((name, name) for name in sorted(observed_models - {m for m, _ in models}))
    capacities = sorted({int(row["capacity"]) for row in rows})
    output = args.output or args.root / "charts" / "accuracy_by_kv_budget.png"
    fig, axes = plt.subplots(
        len(models), 5, figsize=(20, 3.7 * len(models)), sharex=True,
        constrained_layout=True, squeeze=False,
    )
    for i, (model, model_label) in enumerate(models):
        for j, (task, task_label) in enumerate(TASKS):
            ax = axes[i, j]
            panel = [
                r
                for r in rows
                if r["model"] == model and r["task"] == task
            ]
            for policy, label, color in METHODS:
                method_rows = sorted(
                    (
                        r
                        for r in panel
                        if r["eviction_policy"] == policy
                    ),
                    key=lambda row: int(row["capacity"]),
                )
                if not method_rows:
                    continue
                x = [int(r["capacity"]) for r in method_rows]
                y = [float(r["qa_f1"]) for r in method_rows]
                baseline = [
                    y0 - float(r["qa_f1_delta"])
                    for y0, r in zip(y, method_rows)
                ]
                low = [
                    baseline[k] + float(r["delta_ci_lo"])
                    for k, r in enumerate(method_rows)
                ]
                high = [
                    baseline[k] + float(r["delta_ci_hi"])
                    for k, r in enumerate(method_rows)
                ]
                ax.errorbar(
                    x,
                    y,
                    yerr=(
                        [max(0.0, y0 - lo) for y0, lo in zip(y, low)],
                        [max(0.0, hi - y0) for y0, hi in zip(y, high)],
                    ),
                    marker="o",
                    color=color,
                    capsize=2,
                    label=label,
                )
            if panel:
                first = panel[0]
                baseline_score = (
                    float(first["qa_f1"]) - float(first["qa_f1_delta"])
                )
                ax.axhline(
                    baseline_score,
                    color="#111827",
                    linestyle="--",
                    linewidth=1,
                    label="No eviction",
                )
            if i == 0:
                ax.set_title(task_label)
            if j == 0:
                ax.set_ylabel(f"{model_label}\nLongBench accuracy")
            if i == len(models) - 1:
                ax.set_xlabel("KV capacity C (blocks)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(
                capacities, [str(value) for value in capacities]
            )
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Accuracy across methods and per-request KV capacities")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"plotted {len(rows)} budget points to {output}")


if __name__ == "__main__":
    main()
