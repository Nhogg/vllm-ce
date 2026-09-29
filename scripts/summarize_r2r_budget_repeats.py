#!/usr/bin/env python3
"""Validate five budget-sweep repeats and report per-arm mean and sample SD."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

MODELS = (
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Llama-3.1-8B-Instruct",
)
TASKS = ("qasper", "gov_report", "multi_news", "hotpotqa", "multifieldqa_en")
POLICIES = ("none", "v_redundancy", "paged_eviction", "recency")
METRICS = (
    "qa_f1",
    "qa_f1_delta",
    "throughput_tok_s",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "steady_blocks",
    "peak_blocks",
    "decode_evictions",
    "mean_out_len",
    "mean_answer_logprob",
    "mean_answer_probability",
    "mean_verbalized_confidence",
    "verbalized_confidence_parse_rate",
    "confidence_pass_wall_s",
)
REQUIRED_METRICS = (
    "qa_f1",
    "throughput_tok_s",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "steady_blocks",
    "peak_blocks",
    "mean_out_len",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("accuracy", "confidence"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--capacities", default="16,32,64,128,256")
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="Comma-separated expected model directory names.",
    )
    return parser.parse_args()


def numeric(value: object, context: str) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value at {context}: {value!r}")
    return result


def load_repeat(
    repeat_dir: Path,
    capacities: tuple[int, ...],
    mode: str,
    models: tuple[str, ...] = MODELS,
) -> dict[tuple[str, str, str, int], dict[str, object]]:
    if not repeat_dir.is_dir():
        raise ValueError(f"missing repeat directory: {repeat_dir}")
    rows: dict[tuple[str, str, str, int], dict[str, object]] = {}
    cell_dirs = sorted(repeat_dir.glob("r2r_budget_all_methods_*"))
    if len(cell_dirs) != len(models) * len(TASKS):
        raise ValueError(
            f"{repeat_dir}: expected {len(models) * len(TASKS)} cells, "
            f"found {len(cell_dirs)}"
        )
    for cell_dir in cell_dirs:
        csv_path = cell_dir / "capband_accuracy.csv"
        if not csv_path.is_file():
            raise ValueError(f"missing completed cell CSV: {csv_path}")
        with csv_path.open(newline="") as handle:
            cell_rows = list(csv.DictReader(handle))
        baseline_paths = list(cell_dir.glob("*_base.json"))
        if len(baseline_paths) != 1:
            raise ValueError(f"{cell_dir}: expected one baseline JSON")
        with baseline_paths[0].open() as handle:
            baseline = json.load(handle)
        model = str(baseline["model"])
        task = str(baseline["task"])
        if model not in models or task not in TASKS:
            raise ValueError(f"unexpected model/task in {baseline_paths[0]}")
        base_key = (model, task, "none", 0)
        if base_key in rows:
            raise ValueError(f"duplicate baseline: {base_key}")
        if mode == "confidence" and not baseline.get("confidence_eval"):
            raise ValueError(f"missing baseline confidence data: {baseline_paths[0]}")
        rows[base_key] = {
            **baseline,
            "model": model,
            "task": task,
            "eviction_policy": "none",
            "capacity": 0,
            "n": baseline["num_prompts"],
            "decode_evictions": None,
        }
        if len(cell_rows) != (len(POLICIES) - 1) * len(capacities):
            raise ValueError(
                f"{csv_path}: expected {(len(POLICIES) - 1) * len(capacities)} "
                f"arms, found {len(cell_rows)}"
            )
        for row in cell_rows:
            key = (
                row["model"],
                row["task"],
                row["eviction_policy"],
                int(row["capacity"]),
            )
            if key in rows:
                raise ValueError(f"duplicate arm in {csv_path}: {key}")
            if key[0:2] != (model, task):
                raise ValueError(f"model/task mismatch in {csv_path}: {key}")
            if mode == "confidence" and not row.get("mean_answer_logprob"):
                raise ValueError(f"missing confidence data in {csv_path}: {key}")
            rows[key] = row
    expected = {
        (model, task, policy, cap)
        for model in models
        for task in TASKS
        for policy in POLICIES
        for cap in ((0,) if policy == "none" else capacities)
    }
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        unexpected = sorted(set(rows) - expected)
        raise ValueError(
            f"{repeat_dir}: missing {missing[:5]}; unexpected {unexpected[:5]}"
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    repeats: list[dict[tuple[str, str, str, int], dict[str, object]]],
    mode: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    first = repeats[0]
    for key in sorted(first):
        model, task, policy, capacity = key
        arms = [repeat[key] for repeat in repeats]
        metadata = (
            "n",
            "block_size",
            "watermark",
            "metric_name",
            "score_sampled_layers",
            "candidate_expansion_factor",
            "r2r_cover_depth",
            "r2r_query_aggregation",
            "r2r_relevance_signal",
        )
        for field in metadata:
            values = [str(arm.get(field, "")) for arm in arms]
            if len(set(values)) != 1:
                raise ValueError(f"{key}: {field} differs across repeats: {values}")
        if policy == "none" and any(
            arm["prompt_ids"] != arms[0]["prompt_ids"] for arm in arms[1:]
        ):
            raise ValueError(f"{key}: baseline prompt IDs differ across repeats")
        summary: dict[str, object] = {
            "mode": mode,
            "model": model,
            "task": task,
            "eviction_policy": policy,
            "capacity": capacity,
            "n_prompts": arms[0]["n"],
            "metric_name": arms[0].get("metric_name", ""),
            "repeats": len(repeats),
        }
        for metric in METRICS:
            values = [
                numeric(arm.get(metric), f"{key} repeat {i} {metric}")
                for i, arm in enumerate(arms, start=1)
            ]
            if metric in REQUIRED_METRICS and any(value is None for value in values):
                raise ValueError(f"{key}: missing required metric {metric}")
            if mode == "confidence" and metric in (
                "mean_answer_logprob",
                "mean_answer_probability",
                "confidence_pass_wall_s",
            ) and any(value is None for value in values):
                raise ValueError(f"{key}: missing confidence metric {metric}")
            observed = [value for value in values if value is not None]
            summary[f"{metric}_n"] = len(observed)
            summary[f"{metric}_mean"] = (
                statistics.fmean(observed) if observed else ""
            )
            summary[f"{metric}_std"] = (
                statistics.stdev(observed) if len(observed) >= 2 else ""
            )
        summary_rows.append(summary)
        for repeat_index, arm in enumerate(arms, start=1):
            raw_rows.append(
                {
                    "repeat": repeat_index,
                    "mode": mode,
                    "model": model,
                    "task": task,
                    "eviction_policy": policy,
                    "capacity": capacity,
                    **{metric: arm.get(metric, "") for metric in METRICS},
                }
            )
    return raw_rows, summary_rows


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2 for sample SD")
    try:
        capacities = tuple(int(value) for value in args.capacities.split(","))
        if capacities != tuple(sorted(set(capacities))) or min(capacities) < 2:
            raise ValueError("capacities must be sorted, unique, and >= 2")
        models = tuple(value.strip() for value in args.models.split(","))
        if (
            not models
            or any(not value for value in models)
            or len(set(models)) != len(models)
        ):
            raise ValueError("--models must contain unique, nonempty model names")
        repeats = [
            load_repeat(
                args.root / f"repeat_{index:02d}", capacities, args.mode, models
            )
            for index in range(1, args.repeats + 1)
        ]
        raw_rows, summary_rows = summarize(repeats, args.mode)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"cannot summarize incomplete/inconsistent sweep: {error}")
    output = args.output_dir or args.root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "all_repeats.csv", raw_rows)
    write_csv(output / "mean_std.csv", summary_rows)
    print(f"validated {len(repeats)} repeats and {len(summary_rows)} arms")
    print(f"wrote {output / 'mean_std.csv'}")


if __name__ == "__main__":
    main()
