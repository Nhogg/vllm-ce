#!/usr/bin/env python3
"""Validate and plot the matched four-arm R2R v2 floor/dispersion ablation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import analyze_r2r_v2_quality as quality

VARIANTS = ("v1", "floor", "dispersion", "v2")
EFFECTS = ("floor", "dispersion", "v2")


def effect_rows(
    root: Path,
    rows: dict,
    baselines: dict,
    capacities: tuple[int, ...],
) -> list[dict]:
    """Validate raw arms and return prompt-paired effects against v1.

    Args:
        root: Matrix output directory.
        rows: Validated sweep CSV rows keyed by model, task, arm, and budget.
        baselines: Raw no-eviction baselines keyed by model and task.
        capacities: Expected block budgets.

    Returns:
        Three arm-minus-v1 rows and one factorial-interaction row per cell
        and capacity. The interaction is v2 - floor - dispersion + v1.
    """
    # This also validates baseline provenance and the original v1/v2 pair.
    quality.paired_rows(root, rows, baselines, capacities)
    out = []
    for model, task in sorted(baselines):
        stem = f"r2r_v2_{quality.MODEL_TAGS[model]}_{task}"
        cell_dir = root / stem
        baseline = baselines[(model, task)]
        for capacity in capacities:
            runs = {}
            for variant in VARIANTS:
                arm = rows[(model, task, variant, capacity)]
                path = cell_dir / f"{stem}_v_redundancy_{variant}_cap{capacity}.json"
                if not path.is_file():
                    raise ValueError(f"missing raw ablation arm: {path}")
                with path.open() as handle:
                    run = json.load(handle)
                quality._validate_raw_run(run, arm, baseline, variant, capacity)
                measured = statistics.fmean(
                    quality._finite(score, "raw prompt score") for score in run["f1s"]
                )
                if abs(measured - float(arm["qa_f1"])) > 0.0001:
                    raise ValueError(
                        f"CSV/raw score mismatch: {model}/{task}/{variant}/{capacity}"
                    )
                runs[variant] = run
            ids = runs["v1"]["prompt_ids"]
            for variant in EFFECTS:
                n, delta, low, high = quality.paired_delta(runs["v1"], runs[variant])
                arm = rows[(model, task, variant, capacity)]
                control = rows[(model, task, "v1", capacity)]
                out.append(
                    {
                        "model": model,
                        "task": task,
                        "capacity": capacity,
                        "metric_name": arm["metric_name"],
                        "effect": f"{variant}_minus_v1",
                        "n": n,
                        "score_delta": round(delta, 6),
                        "score_delta_ci_lo": round(low, 6),
                        "score_delta_ci_hi": round(high, 6),
                        **{
                            f"{metric}_delta": round(
                                float(arm[metric]) - float(control[metric]), 6
                            )
                            for metric in quality.METRICS
                            if metric != "qa_f1"
                        },
                    }
                )
            aligned = {
                variant: dict(zip(run["prompt_ids"], run["f1s"]))
                for variant, run in runs.items()
            }
            interaction = [
                quality._finite(aligned["v2"][pid], "v2 score")
                - quality._finite(aligned["floor"][pid], "floor score")
                - quality._finite(aligned["dispersion"][pid], "dispersion score")
                + quality._finite(aligned["v1"][pid], "v1 score")
                for pid in ids
            ]
            n, delta, low, high = quality.paired_delta(
                {"prompt_ids": ids, "f1s": [0.0] * len(ids)},
                {"prompt_ids": ids, "f1s": interaction},
            )
            out.append(
                {
                    "model": model,
                    "task": task,
                    "capacity": capacity,
                    "metric_name": quality.EXPECTED_METRICS[task],
                    "effect": "interaction",
                    "n": n,
                    "score_delta": round(delta, 6),
                    "score_delta_ci_lo": round(low, 6),
                    "score_delta_ci_hi": round(high, 6),
                    **{
                        f"{metric}_delta": round(
                            float(rows[(model, task, "v2", capacity)][metric])
                            - float(rows[(model, task, "floor", capacity)][metric])
                            - float(rows[(model, task, "dispersion", capacity)][metric])
                            + float(rows[(model, task, "v1", capacity)][metric]),
                            6,
                        )
                        for metric in quality.METRICS
                        if metric != "qa_f1"
                    },
                }
            )
    return out


def plot_effects(rows: list[dict], output: Path) -> None:
    """Plot prompt-paired accuracy effects with bootstrap intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [
        model for model in quality.MODELS if model in {row["model"] for row in rows}
    ]
    tasks = [task for task in quality.TASKS if task in {row["task"] for row in rows}]
    capacities = sorted({row["capacity"] for row in rows})
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
            panel = {
                (row["effect"], row["capacity"]): row
                for row in rows
                if row["model"] == model and row["task"] == task
            }
            for variant in EFFECTS:
                curve = [panel[(f"{variant}_minus_v1", cap)] for cap in capacities]
                centers = [row["score_delta"] for row in curve]
                lower = [
                    max(0, center - row["score_delta_ci_lo"])
                    for center, row in zip(centers, curve)
                ]
                upper = [
                    max(0, row["score_delta_ci_hi"] - center)
                    for center, row in zip(centers, curve)
                ]
                ax.errorbar(
                    capacities,
                    centers,
                    yerr=[lower, upper],
                    color=quality.COLORS[variant],
                    marker="o",
                    markersize=3,
                    linewidth=1.3,
                    capsize=2,
                    label=variant,
                )
            ax.axhline(0, color="#333333", ls="--", lw=1)
            if i == 0:
                ax.set_title(quality.TASKS[task])
            if j == 0:
                ax.set_ylabel(f"{quality.MODELS[model]}\nΔ task score vs v1")
            if i == len(models) - 1:
                ax.set_xlabel("KV capacity ceiling (blocks)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(capacities, [str(cap) for cap in capacities])
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("Prompt-paired R2R ablation effects (95% bootstrap intervals)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    """Validate the four-arm matrix, then write paired effects and plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("results/r2r_v2_ablation_matrix")
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    rows, baselines = quality.load_matrix(args.root, variants=VARIANTS)
    effects = effect_rows(args.root, rows, baselines, quality.CAPACITIES)
    if args.check_only:
        print(
            f"PASS: {len(baselines)} cells, {len(rows)} arms, "
            f"{len(effects)} paired effects including interactions"
        )
        return
    charts = args.root / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    with (charts / "paired_ablation_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)
    for metric, (stem, _) in quality.METRICS.items():
        quality.plot_metric(
            rows, baselines, metric, charts / f"{stem}_by_kv_budget.png"
        )
    plot_effects(effects, charts / "paired_ablation_effects.png")
    print(f"Validated {len(rows)} arms; wrote {len(effects)} effects to {charts}")


if __name__ == "__main__":
    main()
