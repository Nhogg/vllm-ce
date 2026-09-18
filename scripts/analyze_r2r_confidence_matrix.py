#!/usr/bin/env python3
"""Aggregate implicit and verbalized confidence for the full R2R matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

QA_TASKS = {"qasper", "hotpotqa", "multifieldqa_en"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("results/r2r_confidence_matrix")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--qa-correct-threshold", type=float, default=0.5)
    parser.add_argument("--ece-bins", type=int, default=10)
    return parser.parse_args()


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def brier(confidence: list[float], correct: list[float]) -> float | None:
    if not confidence:
        return None
    return statistics.fmean((p - y) ** 2 for p, y in zip(confidence, correct))


def ece(confidence: list[float], correct: list[float], bins: int) -> float | None:
    if not confidence:
        return None
    total = len(confidence)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            i
            for i, value in enumerate(confidence)
            if low <= value < high or (index == bins - 1 and value == 1.0)
        ]
        if members:
            mean_confidence = statistics.fmean(confidence[i] for i in members)
            accuracy = statistics.fmean(correct[i] for i in members)
            error += len(members) / total * abs(mean_confidence - accuracy)
    return error


def arm_method(run: dict[str, Any]) -> str:
    policy = run.get("eviction_policy")
    if policy == "baseline":
        return "baseline"
    if policy == "paged_eviction":
        return "paged"
    if policy == "recency":
        return "streamingllm"
    return "r2r"


def trigger(run: dict[str, Any]) -> str:
    return "drip" if run.get("regime") == "decode_rate" else "band"


def reference_arm(run: dict[str, Any]) -> bool:
    method = arm_method(run)
    if method != "r2r":
        return method in {"paged", "streamingllm"}
    expected_n = 16.0 if trigger(run) == "drip" else 2.0
    return (
        float(run.get("candidate_expansion_factor") or 0) == expected_n
        and run.get("r2r_cover_depth") == 2
        and run.get("r2r_query_aggregation") == "max"
        and run.get("r2r_relevance_signal") == "key_anchor"
    )


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(root.glob("r2r_confidence_*/*.json")):
        if path.name.endswith(".scorer_profile.json"):
            continue
        run = json.loads(path.read_text())
        if run.get("confidence_eval"):
            run["source_json"] = str(path)
            runs.append(run)
    if not runs:
        raise SystemExit(f"no confidence result JSON files found under {root}")
    return runs


def summarize_run(
    run: dict[str, Any], threshold: float, bins: int
) -> dict[str, Any]:
    scores = [float(value) for value in run["f1s"]]
    implicit = [
        float(item["geometric_mean_probability"])
        for item in run["answer_logprob_metrics"]
    ]
    verbal_pairs = [
        (float(confidence), score)
        for confidence, score in zip(run["verbalized_confidence"], scores)
        if confidence is not None
    ]
    verbal = [pair[0] for pair in verbal_pairs]
    verbal_scores = [pair[1] for pair in verbal_pairs]
    task = run["task"]
    row: dict[str, Any] = {
        "model": Path(run["model"]).name,
        "task": task,
        "method": arm_method(run),
        "trigger": trigger(run),
        "candidate_expansion_factor": run.get("candidate_expansion_factor"),
        "r2r_cover_depth": run.get("r2r_cover_depth"),
        "r2r_query_aggregation": run.get("r2r_query_aggregation"),
        "r2r_relevance_signal": run.get("r2r_relevance_signal"),
        "num_prompts": len(scores),
        "mean_task_score": statistics.fmean(scores),
        "mean_implicit_confidence": statistics.fmean(implicit),
        "mean_answer_logprob": statistics.fmean(
            float(item["mean_logprob"])
            for item in run["answer_logprob_metrics"]
        ),
        "mean_verbalized_confidence": (
            statistics.fmean(verbal) if verbal else None
        ),
        "verbalized_parse_rate": len(verbal) / len(scores),
        "implicit_score_correlation": correlation(implicit, scores),
        "verbalized_score_correlation": correlation(verbal, verbal_scores),
        "source_json": run["source_json"],
    }
    if task in QA_TASKS:
        correct = [float(score >= threshold) for score in scores]
        verbal_correct = [
            float(score >= threshold) for score in verbal_scores
        ]
        row.update(
            {
                "qa_accuracy_at_threshold": statistics.fmean(correct),
                "implicit_brier": brier(implicit, correct),
                "implicit_ece": ece(implicit, correct, bins),
                "verbalized_brier": brier(verbal, verbal_correct),
                "verbalized_ece": ece(verbal, verbal_correct, bins),
            }
        )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def comparison_plot(rows: list[dict[str, Any]], output: Path) -> None:
    reference = [row for row in rows if row.get("_reference", False)]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in reference:
        grouped[(row["trigger"], row["method"])].append(row)
    categories = [
        (trigger_name, method)
        for trigger_name in ("band", "drip")
        for method in ("r2r", "paged", "streamingllm")
    ]
    labels = [f"{method}\n{trigger_name}" for trigger_name, method in categories]
    metrics = (
        ("mean_answer_logprob", "Mean answer log probability"),
        ("mean_verbalized_confidence", "Mean verbalized confidence"),
        ("implicit_score_correlation", "Implicit confidence–score correlation"),
        ("verbalized_score_correlation", "Verbal confidence–score correlation"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, (field, title) in zip(axes.flat, metrics):
        values = []
        for category in categories:
            observed = [
                float(row[field])
                for row in grouped.get(category, [])
                if row.get(field) is not None and math.isfinite(float(row[field]))
            ]
            values.append(statistics.fmean(observed) if observed else np.nan)
        colors = ("#2563eb", "#f59e0b", "#dc2626") * 2
        ax.bar(range(len(categories)), values, color=colors)
        ax.set_xticks(range(len(categories)), labels)
        ax.set_title(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0 <= args.qa_correct_threshold <= 1:
        raise SystemExit("--qa-correct-threshold must be in [0, 1]")
    if args.ece_bins < 2:
        raise SystemExit("--ece-bins must be at least 2")
    output = args.output_dir or args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.root)
    rows = []
    for run in runs:
        row = summarize_run(run, args.qa_correct_threshold, args.ece_bins)
        row["_reference"] = reference_arm(run)
        rows.append(row)
    baselines = {
        (row["model"], row["task"]): row
        for row in rows
        if row["method"] == "baseline"
    }
    delta_fields = (
        "mean_answer_logprob",
        "mean_implicit_confidence",
        "mean_verbalized_confidence",
        "implicit_brier",
        "verbalized_brier",
    )
    for row in rows:
        baseline = baselines.get((row["model"], row["task"]))
        if baseline is None or row["method"] == "baseline":
            continue
        for field in delta_fields:
            value = row.get(field)
            base_value = baseline.get(field)
            row[f"{field}_delta_vs_baseline"] = (
                float(value) - float(base_value)
                if value is not None and base_value is not None
                else None
            )
    write_csv(output / "confidence_all_arms.csv", rows)
    comparison_plot(rows, output / "confidence_comparison.png")
    print(f"analyzed {len(runs)} runs into {output}")


if __name__ == "__main__":
    main()
