#!/usr/bin/env python3
"""Validate and plot the matched R2R v1/v2 block-budget quality sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

MODELS = {
    "Llama-3.2-1B-Instruct": "Llama 3.2 1B",
    "Llama-3.2-3B-Instruct": "Llama 3.2 3B",
    "Llama-3.1-8B-Instruct": "Llama 3.1 8B",
}
MODEL_TAGS = {
    "Llama-3.2-1B-Instruct": "1b",
    "Llama-3.2-3B-Instruct": "3b",
    "Llama-3.1-8B-Instruct": "8b",
}
TASKS = {
    "qasper": "Qasper",
    "gov_report": "GovReport",
    "multi_news": "MultiNews",
    "hotpotqa": "HotpotQA",
    "multifieldqa_en": "MultiFieldQA",
}
CAPACITIES = (16, 32, 64, 128, 256)
VARIANTS = ("v1", "v2")
METRICS = {
    "qa_f1": ("accuracy", "Task score"),
    "throughput_tok_s": ("throughput", "Output throughput (tok/s)"),
    "mean_ttft_ms": ("ttft", "Mean TTFT (ms)"),
    "mean_tpot_ms": ("tpot", "Mean TPOT (ms)"),
    "steady_blocks": ("steady_blocks", "Steady GPU KV blocks"),
}
COLORS = {"v1": "#667085", "floor": "#2f78bd", "dispersion": "#e7902b", "v2": "#009e73"}


def _finite(value: str | float | int, context: str) -> float:
    """Parse a finite numeric result or reject an incomplete measurement."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {context}: {value!r}")
    return parsed


def load_matrix(
    root: Path,
    models: tuple[str, ...] = tuple(MODELS),
    tasks: tuple[str, ...] = tuple(TASKS),
    capacities: tuple[int, ...] = CAPACITIES,
    variants: tuple[str, ...] = VARIANTS,
) -> tuple[dict[tuple[str, str, str, int], dict[str, str]], dict]:
    """Read a complete sweep, rejecting missing or duplicate model/task arms."""
    rows: dict[tuple[str, str, str, int], dict[str, str]] = {}
    baselines: dict[tuple[str, str], dict] = {}
    expected_cells = {(model, task) for model in models for task in tasks}
    cell_dirs = sorted(root.glob("r2r_v2_*"))
    for cell_dir in cell_dirs:
        if not cell_dir.is_dir():
            continue
        csv_path = cell_dir / "capband_accuracy.csv"
        base_paths = list(cell_dir.glob("*_base.json"))
        if not csv_path.is_file() or len(base_paths) != 1:
            raise ValueError(f"incomplete model/task cell: {cell_dir}")
        with base_paths[0].open() as handle:
            base = json.load(handle)
        cell = (str(base["model"]), str(base["task"]))
        if cell not in expected_cells or cell in baselines:
            raise ValueError(f"unexpected or duplicate model/task cell: {cell}")
        baselines[cell] = base
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row["model"],
                    row["task"],
                    row["r2r_variant"],
                    int(row["capacity"]),
                )
                if key[:2] != cell or key in rows:
                    raise ValueError(f"mismatched or duplicate arm: {key}")
                if row["eviction_policy"] != "v_redundancy":
                    raise ValueError(f"unexpected eviction policy: {key}")
                expected_floor = row["r2r_variant"] in ("floor", "v2")
                if row["r2r_persistent_floor"] != str(expected_floor):
                    raise ValueError(f"floor setting inconsistent with arm: {key}")
                expected_kappa = row["r2r_variant"] in ("dispersion", "v2")
                kappa = row["r2r_dispersion_kappa"]
                if expected_kappa != (kappa not in ("", "None")):
                    raise ValueError(f"dispersion setting inconsistent: {key}")
                for metric in METRICS:
                    _finite(row[metric], f"{key}/{metric}")
                rows[key] = row
    if set(baselines) != expected_cells:
        missing = sorted(expected_cells - set(baselines))
        raise ValueError(f"missing {len(missing)} model/task cells: {missing[:5]}")
    expected = {
        (model, task, variant, capacity)
        for model, task in expected_cells
        for variant in variants
        for capacity in capacities
    }
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise ValueError(f"incomplete grid: missing {missing[:5]}; extra {extra[:5]}")
    return rows, baselines


def paired_delta(
    v1: dict, v2: dict, n_boot: int = 2000, seed: int = 0
) -> tuple[int, float, float, float]:
    """Return prompt-paired v2-minus-v1 score and percentile bootstrap CI."""
    ids1, ids2 = list(v1["prompt_ids"]), list(v2["prompt_ids"])
    if len(ids1) != len(set(ids1)) or len(ids2) != len(set(ids2)):
        raise ValueError("duplicate prompt IDs in variant JSON")
    if len(ids1) != len(v1["f1s"]) or len(ids2) != len(v2["f1s"]):
        raise ValueError("prompt ID/score length mismatch")
    if set(ids1) != set(ids2) or not ids1:
        raise ValueError("v1/v2 prompt sets differ or are empty")
    score1 = dict(zip(ids1, v1["f1s"]))
    score2 = dict(zip(ids2, v2["f1s"]))
    differences = [
        _finite(score2[pid], "v2 score") - _finite(score1[pid], "v1 score")
        for pid in ids1
    ]
    center = statistics.fmean(differences)
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(differences[rng.randrange(len(ids1))] for _ in ids1)
        for _ in range(n_boot)
    )
    return (
        len(ids1),
        center,
        samples[int(0.025 * n_boot)],
        samples[min(n_boot - 1, int(0.975 * n_boot))],
    )


def paired_rows(root: Path, rows: dict, capacities: tuple[int, ...]) -> list[dict]:
    """Build v2-v1 accuracy and efficiency comparisons from raw run JSONs."""
    comparisons = []
    for model, task in sorted({key[:2] for key in rows}):
        cell_dir = root / f"r2r_v2_{MODEL_TAGS[model]}_{task}"
        stem = cell_dir.name
        for capacity in capacities:
            arm1 = rows[(model, task, "v1", capacity)]
            arm2 = rows[(model, task, "v2", capacity)]
            matched_fields = (
                "metric_name",
                "block_size",
                "watermark",
                "score_sampled_layers",
                "candidate_expansion_factor",
                "r2r_cover_depth",
                "r2r_query_aggregation",
                "r2r_relevance_signal",
            )
            for field in matched_fields:
                if arm1.get(field) != arm2.get(field):
                    raise ValueError(f"unmatched {field}: {model}/{task}/{capacity}")
            run_paths = [
                cell_dir / f"{stem}_v_redundancy_{variant}_cap{capacity}.json"
                for variant in VARIANTS
            ]
            if any(not path.is_file() for path in run_paths):
                raise ValueError(f"missing raw paired runs: {run_paths}")
            with run_paths[0].open() as handle:
                run1 = json.load(handle)
            with run_paths[1].open() as handle:
                run2 = json.load(handle)
            if (run1["model"], run1["task"]) != (model, task):
                raise ValueError(f"v1 raw run mismatch: {run_paths[0]}")
            if (run2["model"], run2["task"]) != (model, task):
                raise ValueError(f"v2 raw run mismatch: {run_paths[1]}")
            n, delta, low, high = paired_delta(run1, run2)
            if n != int(arm1["n"]) or n != int(arm2["n"]):
                raise ValueError(f"prompt count mismatch: {model}/{task}/{capacity}")
            for arm, run in ((arm1, run1), (arm2, run2)):
                measured = statistics.fmean(
                    _finite(score, "raw prompt score") for score in run["f1s"]
                )
                if abs(measured - float(arm["qa_f1"])) > 0.0001:
                    raise ValueError(
                        f"CSV/raw score mismatch: {model}/{task}/{capacity}"
                    )
            comparisons.append(
                {
                    "model": model,
                    "task": task,
                    "capacity": capacity,
                    "metric_name": arm1["metric_name"],
                    "n": n,
                    "v1_score": arm1["qa_f1"],
                    "v2_score": arm2["qa_f1"],
                    "score_delta_v2_minus_v1": round(delta, 6),
                    "score_delta_ci_lo": round(low, 6),
                    "score_delta_ci_hi": round(high, 6),
                    **{
                        f"{metric}_delta_v2_minus_v1": round(
                            float(arm2[metric]) - float(arm1[metric]), 6
                        )
                        for metric in METRICS
                        if metric != "qa_f1"
                    },
                }
            )
    return comparisons


def plot_metric(rows: dict, baselines: dict, metric: str, output: Path) -> None:
    """Plot one model-by-task grid with v1 and v2 as separate curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [model for model in MODELS if model in {key[0] for key in rows}]
    tasks = [task for task in TASKS if task in {key[1] for key in rows}]
    capacities = sorted({key[3] for key in rows})
    variants = [name for name in COLORS if name in {key[2] for key in rows}]
    fig, axes = plt.subplots(
        len(models),
        len(tasks),
        figsize=(19, 3.3 * len(models)),
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )
    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            ax = axes[i, j]
            for variant in variants:
                curve = [rows[(model, task, variant, cap)] for cap in capacities]
                ax.plot(
                    capacities,
                    [float(row[metric]) for row in curve],
                    marker="o",
                    linewidth=1.6,
                    markersize=4,
                    color=COLORS[variant],
                    label=variant.upper(),
                )
            base = baselines[(model, task)]
            if metric in base:
                ax.axhline(
                    float(base[metric]),
                    color="#333333",
                    ls="--",
                    lw=1,
                    label="No eviction",
                )
            if i == 0:
                ax.set_title(TASKS[task])
            if j == 0:
                ax.set_ylabel(f"{MODELS[model]}\n{METRICS[metric][1]}")
            if i == len(models) - 1:
                ax.set_xlabel("KV capacity ceiling (blocks)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(capacities, [str(cap) for cap in capacities])
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle(f"R2R variants: {METRICS[metric][1]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_paired_delta(comparisons: list[dict], output: Path) -> None:
    """Plot prompt-paired v2-minus-v1 accuracy with bootstrap intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [model for model in MODELS if model in {r["model"] for r in comparisons}]
    tasks = [task for task in TASKS if task in {r["task"] for r in comparisons}]
    fig, axes = plt.subplots(
        len(models),
        len(tasks),
        figsize=(19, 3.3 * len(models)),
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )
    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            ax = axes[i, j]
            panel = sorted(
                (
                    row
                    for row in comparisons
                    if row["model"] == model and row["task"] == task
                ),
                key=lambda row: row["capacity"],
            )
            capacities = [row["capacity"] for row in panel]
            centers = [row["score_delta_v2_minus_v1"] for row in panel]
            lower = [
                max(0, center - row["score_delta_ci_lo"])
                for center, row in zip(centers, panel)
            ]
            upper = [
                max(0, row["score_delta_ci_hi"] - center)
                for center, row in zip(centers, panel)
            ]
            ax.errorbar(
                capacities,
                centers,
                yerr=[lower, upper],
                color=COLORS["v2"],
                marker="o",
                markersize=4,
                linewidth=1.5,
                capsize=2,
            )
            ax.axhline(0, color="#333333", ls="--", lw=1)
            if i == 0:
                ax.set_title(TASKS[task])
            if j == 0:
                ax.set_ylabel(f"{MODELS[model]}\nV2 − V1 task score")
            if i == len(models) - 1:
                ax.set_xlabel("KV capacity ceiling (blocks)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(capacities, [str(cap) for cap in capacities])
            ax.grid(alpha=0.2)
    fig.suptitle("Prompt-paired R2R v2 effect (95% bootstrap intervals)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    """Validate the complete matrix, then emit paired comparisons and grids."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("results/r2r_v2_quality_matrix")
    )
    args = parser.parse_args()
    rows, baselines = load_matrix(args.root)
    comparisons = paired_rows(args.root, rows, CAPACITIES)
    charts = args.root / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    with (charts / "paired_v2_vs_v1.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    for metric, (stem, _) in METRICS.items():
        plot_metric(rows, baselines, metric, charts / f"{stem}_by_kv_budget.png")
    plot_paired_delta(comparisons, charts / "paired_score_delta_by_kv_budget.png")
    wins = sum(row["score_delta_v2_minus_v1"] > 0 for row in comparisons)
    losses = sum(row["score_delta_v2_minus_v1"] < 0 for row in comparisons)
    print(
        f"Validated {len(rows)} arms, {len(comparisons)} matched comparisons; "
        f"v2 score wins/losses/ties: {wins}/{losses}/"
        f"{len(comparisons) - wins - losses}. Wrote {charts}"
    )


if __name__ == "__main__":
    main()
