#!/usr/bin/env python3
"""Aggregate and plot the 3-model x 5-task R2R evaluation matrix."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODELS = (
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Llama-3.1-8B-Instruct",
)
TASKS = ("qasper", "gov_report", "multi_news", "hotpotqa", "multifieldqa_en")
MODEL_LABELS = {model: label for model, label in zip(MODELS, ("1B", "3B", "8B"))}
TASK_LABELS = {
    "qasper": "Qasper",
    "gov_report": "GovReport",
    "multi_news": "MultiNews",
    "hotpotqa": "HotpotQA",
    "multifieldqa_en": "MultiFieldQA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/r2r_full_matrix"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def load_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("r2r_full_*/ablation_accuracy.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["source_csv"] = str(path)
                rows.append(row)
    if not rows:
        raise SystemExit(f"no completed matrix CSVs found under {root}")
    return rows


def trigger(row: dict[str, str]) -> str:
    return "drip" if row.get("regime") == "decode_rate" else "band"


def method(row: dict[str, str]) -> str:
    policy = row.get("eviction_policy")
    if policy == "paged_eviction":
        return "paged"
    if policy == "recency":
        return "streamingllm"
    return "r2r"


def selected_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["task"], trigger(row), method(row))].append(row)
    selected = []
    for candidates in grouped.values():
        # "Best R2R" is descriptive/post-hoc, not a held-out model selection.
        selected.append(max(candidates, key=lambda row: number(row, "qa_f1")))
    return selected


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def accuracy_heatmaps(rows: list[dict[str, str]], output: Path) -> None:
    lookup = {
        (r["model"], r["task"], trigger(r), method(r)): r
        for r in rows
    }
    fig, axes = plt.subplots(2, 3, figsize=(17, 7), constrained_layout=True)
    for ax, trig, method_key in zip(
        axes.flat,
        ("band", "band", "band", "drip", "drip", "drip"),
        (
            "r2r",
            "paged",
            "streamingllm",
            "r2r",
            "paged",
            "streamingllm",
        ),
    ):
        values = np.full((len(MODELS), len(TASKS)), np.nan)
        for i, model in enumerate(MODELS):
            for j, task in enumerate(TASKS):
                row = lookup.get((model, task, trig, method_key))
                if row:
                    values[i, j] = number(row, "qa_f1_delta")
        image = ax.imshow(values, cmap="RdYlGn", vmin=-0.08, vmax=0.08, aspect="auto")
        for i in range(len(MODELS)):
            for j in range(len(TASKS)):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:+.3f}", ha="center", va="center")
        ax.set_xticks(range(len(TASKS)), [TASK_LABELS[t] for t in TASKS], rotation=30)
        ax.set_yticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
        title = {
            "r2r": "Best observed R2R",
            "paged": "PagedEviction",
            "streamingllm": "StreamingLLM",
        }[method_key]
        ax.set_title(f"{title} · {trig}")
    fig.colorbar(image, ax=axes, label="LongBench score delta vs no eviction")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def efficiency_bars(rows: list[dict[str, str]], output: Path) -> None:
    categories = (
        ("band", "r2r"),
        ("band", "paged"),
        ("band", "streamingllm"),
        ("drip", "r2r"),
        ("drip", "paged"),
        ("drip", "streamingllm"),
    )
    labels = (
        "R2R band",
        "Paged band",
        "Streaming band",
        "R2R drip",
        "Paged drip",
        "Streaming drip",
    )
    metrics = (
        ("throughput_tok_s", "baseline_throughput_tok_s", "Throughput / baseline"),
        ("ttft_ms_mean", "baseline_ttft_ms_mean", "TTFT / baseline"),
        ("tpot_ms_mean", "baseline_tpot_ms_mean", "TPOT / baseline"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, (metric, baseline, title) in zip(axes, metrics):
        values = []
        for trig, method_key in categories:
            matches = [
                r for r in rows
                if trigger(r) == trig
                and method(r) == method_key
                and number(r, baseline) > 0
            ]
            ratios = [number(r, metric) / number(r, baseline) for r in matches]
            values.append(float(np.mean(ratios)) if ratios else np.nan)
        ax.bar(
            range(6),
            values,
            color=(
                "#3b82f6",
                "#f59e0b",
                "#dc2626",
                "#2563eb",
                "#d97706",
                "#b91c1c",
            ),
        )
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_xticks(range(6), labels, rotation=35, ha="right")
        ax.set_title(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def frontier(rows: list[dict[str, str]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    styles = {"band": "o", "drip": "s"}
    for row in rows:
        baseline = number(row, "baseline_throughput_tok_s")
        if baseline <= 0:
            continue
        arm_method = method(row)
        ax.scatter(
            number(row, "throughput_tok_s") / baseline,
            number(row, "qa_f1_delta"),
            marker=styles[trigger(row)],
            color={
                "r2r": "#2563eb",
                "paged": "#f59e0b",
                "streamingllm": "#dc2626",
            }[arm_method],
            alpha=0.55,
            s=35,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(1, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Output-token throughput / no-eviction baseline")
    ax.set_ylabel("LongBench score delta")
    ax.set_title("Accuracy–throughput operating points (all matrix arms)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir or args.root / "charts"
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.root)
    selected = selected_rows(rows)
    write_csv(output / "all_arms.csv", rows)
    write_csv(output / "best_observed_and_comparisons.csv", selected)
    accuracy_heatmaps(selected, output / "accuracy_delta_heatmaps.png")
    efficiency_bars(selected, output / "efficiency_ratios.png")
    frontier(rows, output / "accuracy_throughput_frontier.png")
    cells = {(row["model"], row["task"]) for row in rows}
    print(f"wrote charts for {len(cells)}/15 completed model-task cells to {output}")


if __name__ == "__main__":
    main()
