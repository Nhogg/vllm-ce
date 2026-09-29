#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy grid for GeoKV, matched policies, and external PagedEviction.

Reads the per-cell JSON dumps written by ``benchmark_kv_capacity_band_vllm.py``
under ``results/baselines/baselines_<tag>_<policy>_<task>/*.json`` (one file per
capacity point plus a ``_base.json`` full-cache reference). Each cell stores the
scalar metric in ``qa_f1`` with ``metric_name`` telling whether it is QA F1 or
ROUGE-L.

Renders a model x task grid (rows = 1B/3B/8B, columns = the five LongBench
tasks). Each panel plots absolute accuracy against KV capacity C for recency,
in-tree block value-L2, and GeoKV over a dashed full-cache reference. With
``--include-external-paged``, the end-to-end fork curve and its own dotted
full-cache reference are also required and rendered. A single legend row sits
under the figure title; every panel is titled "<Task> <Model>".

Usage:
    python scripts/plot_baseline_capacity_grid.py \
        --glob 'results/baselines/baselines_*/*.json' \
        --glob 'results/accuracy_sweep_current/geokv_current_*/*.json' \
        --out writeups/presentation/figs/baselines_capacity_grid.png
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Use a serif face for a cleaner, paper-style look. Times New Roman is not
# installed on this host, so fall back through metric-compatible serifs down to
# DejaVu Serif (always bundled with matplotlib).
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif",
]
plt.rcParams["mathtext.fontset"] = "dejavuserif"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _geokv_style import (  # noqa: E402
    BLUE,
    GREEN,
    INK,
    MUTED,
    VERMILLION,
    style_ax,
)

# Column (task) and row (model) order for the grid.
TASK_ORDER = ["qasper", "gov_report", "multi_news", "hotpotqa", "multifieldqa_en"]
TASK_TITLE = {
    "qasper": "Qasper",
    "gov_report": "GovReport",
    "multi_news": "MultiNews",
    "hotpotqa": "HotpotQA",
    "multifieldqa_en": "MultiFieldQA",
}
MODEL_ORDER = ["1B", "3B", "8B", "Qwen2.5-32B-Instruct-AWQ"]

# Our method (v_redundancy) drawn against the two matched-memory baselines (all
# lines) plus the full-cache reference (a dashed horizontal rule). Ours is last
# in POLICY_ORDER so it draws on top of the baselines.
POLICY_ORDER = ["value_l2", "recency", "v_redundancy"]
CURRENT_POLICY_ORDER = ["paged_eviction", "recency", "v_redundancy"]
QUERY_POLICY = "query_v_redundancy"
EXTERNAL_POLICY = "external_paged_global"
PAGED_PURPLE = "#CC79A7"
POLICY_STYLE = {
    "value_l2": {
        "label": "Block value-L2 (Paged-style)",
        "color": BLUE,
        "marker": "o",
    },
    "recency": {"label": "StreamingLLM", "color": VERMILLION, "marker": "s"},
    "paged_eviction": {
        "label": "PagedEviction",
        "color": PAGED_PURPLE,
        "marker": "^",
    },
    "v_redundancy": {
        "label": "GeoKV cosh1.5",
        "color": GREEN,
        "marker": "D",
    },
    QUERY_POLICY: {
        "label": "GeoKV + query",
        "color": "#009E73",
        "marker": "P",
    },
    EXTERNAL_POLICY: {
        "label": "PagedEviction (during-prefill)",
        "color": PAGED_PURPLE,
        "marker": "^",
    },
}
FULL_LABEL = "Full cache (no compression)"
EXTERNAL_FULL_LABEL = "Paged full cache"

# Pretty y-axis label per stored metric name.
METRIC_LABEL = {"qa_f1": "QA F1", "rouge_l": "ROUGE-L"}
EXPECTED_CAPACITIES = [16, 32, 64, 128, 256]


def _model_short(model: str) -> str:
    """'Llama-3.2-1B-Instruct' -> '1B' (falls back to the raw string)."""
    for tag in MODEL_ORDER:
        if tag in model or tag.lower() in model.lower():
            return tag
    return model


def _collect(
    json_paths: list[str], prefer_query_aware_full: bool = False
) -> tuple[dict, dict, dict, dict, dict]:
    """Parse cells into curves, full-cache references, and per-task metric names.

    Returns:
        curves: ``[model_short][task][policy]`` -> list of ``(C, value)`` sorted
            by capacity, for the line policies only.
        full: ``[model_short][task]`` -> ``value`` full-cache reference.
        metric: ``[task]`` -> stored ``metric_name`` (e.g. ``qa_f1``,
            ``rouge_l``); the scored metric is task-determined.
        model_name: ``[model_short]`` -> full model name from the JSON (e.g.
            ``Llama-3.2-1B-Instruct``), for panel titles.
    """
    curves: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    full: dict = defaultdict(dict)
    external_full: dict = defaultdict(dict)
    query_full: set[tuple[str, str]] = set()
    metric: dict = {}
    model_name: dict = {}
    for p in json_paths:
        if "smoke" in p:
            continue
        with open(p) as f:
            d = json.load(f)
        full_model = d.get("model", "?")
        model = _model_short(full_model)
        task = d.get("task", "?")
        policy = d.get("eviction_policy", "?")
        if (
            policy == "v_redundancy"
            and d.get("query_relevance_protect_quantile") is not None
        ):
            policy = QUERY_POLICY
        val = d.get("qa_f1")
        if val is None:
            continue
        model_name.setdefault(model, full_model)
        metric.setdefault(task, d.get("metric_name", "qa_f1"))
        if policy == "none" or p.endswith("_base.json"):
            full_table = (
                external_full
                if d.get("runtime") == "external_vllm_paged_eviction"
                else full
            )
            previous = full_table[model].get(task)
            if previous is not None and previous != float(val):
                is_query_base = "queryaware_" in d.get("run_id", "")
                key = (model, task)
                if prefer_query_aware_full and is_query_base:
                    full_table[model][task] = float(val)
                    query_full.add(key)
                    continue
                if prefer_query_aware_full and key in query_full:
                    continue
                if prefer_query_aware_full and full_table is full:
                    # Multiple historical baseline epochs may disagree. The
                    # later query-aware base will become authoritative; ignore
                    # these older conflicts only under the explicit flag.
                    continue
                raise ValueError(
                    "conflicting full-cache controls for "
                    f"{model}/{task}: {previous} vs {float(val)} ({p})"
                )
            full_table[model][task] = float(val)
            if "queryaware_" in d.get("run_id", ""):
                query_full.add((model, task))
        else:
            cap = d.get("capacity")
            if cap is not None:
                duplicate = next(
                    (
                        old_val
                        for old_cap, old_val in curves[model][task][policy]
                        if old_cap == int(cap)
                    ),
                    None,
                )
                if duplicate is not None:
                    raise ValueError(
                        "duplicate curve point for "
                        f"{model}/{task}/{policy}/C={int(cap)}; "
                        "use explicit non-overlapping --glob inputs"
                    )
                curves[model][task][policy].append((int(cap), float(val)))
    for model in curves:
        for task in curves[model]:
            for policy in curves[model][task]:
                curves[model][task][policy].sort(key=lambda t: t[0])
    return curves, full, external_full, metric, model_name


def _validate_complete(
    curves: dict,
    full: dict,
    external_full: dict,
    include_external_paged: bool,
    include_query_aware: bool,
    policy_order: list[str],
    expected_capacities: list[int],
    expected_models: list[str],
) -> None:
    errors: list[str] = []
    for model in expected_models:
        for task in TASK_ORDER:
            if task not in full.get(model, {}):
                errors.append(f"missing full control: {model}/{task}")
            required_policies = list(policy_order)
            if include_query_aware:
                required_policies.append(QUERY_POLICY)
            for policy in required_policies:
                caps = [
                    cap
                    for cap, _ in curves.get(model, {}).get(task, {}).get(policy, [])
                ]
                if caps != expected_capacities:
                    errors.append(
                        f"incomplete curve: {model}/{task}/{policy}: {caps}"
                    )
            if include_external_paged:
                if task not in external_full.get(model, {}):
                    errors.append(f"missing Paged fork full control: {model}/{task}")
                caps = [
                    cap
                    for cap, _ in curves.get(model, {})
                    .get(task, {})
                    .get(EXTERNAL_POLICY, [])
                ]
                if caps != EXPECTED_CAPACITIES:
                    errors.append(
                        f"incomplete curve: {model}/{task}/{EXTERNAL_POLICY}: {caps}"
                    )
    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n  ... {len(errors) - 20} more"
        raise ValueError(f"capacity grid is incomplete:\n{preview}{suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--glob",
        action="append",
        dest="globs",
        default=None,
        help="glob of per-cell JSON dumps; repeat to combine explicit baseline "
        "and GeoKV result roots. Defaults to results/baselines/*/*.json. The "
        "smoke dir is skipped in _collect.",
    )
    ap.add_argument(
        "--out",
        default="writeups/presentation/figs/baselines_capacity_grid.png",
    )
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Render a diagnostic subset instead of requiring the full 3x5 grid.",
    )
    ap.add_argument(
        "--include-external-paged",
        action="store_true",
        help="Require and render the end-to-end vLLM-PagedEviction fork arm.",
    )
    ap.add_argument(
        "--include-query-aware",
        action="store_true",
        help="Require and render the hard query-protection GeoKV arm.",
    )
    ap.add_argument(
        "--prefer-query-aware-full",
        action="store_true",
        help="Use the same-snapshot query campaign full-cache controls when "
        "older baseline roots contain conflicting full-cache outputs.",
    )
    ap.add_argument(
        "--current-all-methods",
        action="store_true",
        help="Render the current R2R/PagedEviction/StreamingLLM comparison at "
        "the capacities selected by --current-capacities.",
    )
    ap.add_argument(
        "--current-capacities",
        default="16,32,54,128,256",
        help="Expected block capacities for --current-all-methods.",
    )
    ap.add_argument(
        "--expected-models",
        default="1B,3B,8B",
        help="Comma-separated model names required for completeness checks.",
    )
    args = ap.parse_args()

    patterns = args.globs or ["results/baselines/*/*.json"]
    json_paths = sorted({p for pattern in patterns for p in globmod.glob(pattern)})
    if not json_paths:
        raise SystemExit(f"no baseline JSON found for {patterns!r}")
    curves, full, external_full, metric, model_name = _collect(
        json_paths, args.prefer_query_aware_full
    )
    policy_order = (
        list(CURRENT_POLICY_ORDER)
        if args.current_all_methods
        else list(POLICY_ORDER)
    )
    expected_capacities = (
        [int(value) for value in args.current_capacities.split(",")]
        if args.current_all_methods
        else EXPECTED_CAPACITIES
    )
    if args.current_all_methods and (
        not expected_capacities
        or expected_capacities != sorted(set(expected_capacities))
    ):
        ap.error("--current-capacities must be sorted, unique block counts")
    expected_models = [value.strip() for value in args.expected_models.split(",")]
    if not expected_models or any(not value for value in expected_models):
        ap.error("--expected-models must contain nonempty model names")
    if args.current_all_methods:
        POLICY_STYLE["v_redundancy"]["label"] = "R2R (ours)"

    if not args.allow_incomplete:
        _validate_complete(
            curves,
            full,
            external_full,
            args.include_external_paged,
            args.include_query_aware,
            policy_order,
            expected_capacities,
            expected_models,
        )

    if args.include_query_aware:
        policy_order.append(QUERY_POLICY)
    if args.include_external_paged:
        policy_order.append(EXTERNAL_POLICY)

    models = [m for m in MODEL_ORDER if m in curves] or sorted(curves)
    tasks = TASK_ORDER  # fixed column order
    nrows, ncols = len(models), len(tasks)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.5 * ncols, 2.8 * nrows),
        squeeze=False, sharex=True,
    )

    for ri, model in enumerate(models):
        for ci, task in enumerate(tasks):
            ax = axes[ri][ci]
            style_ax(ax, grid_axis="both")

            # Full-cache reference: dashed horizontal rule.
            ref = full.get(model, {}).get(task)
            if ref is not None:
                ax.axhline(ref, color=MUTED, ls="--", lw=1.3, zorder=1)
            external_ref = external_full.get(model, {}).get(task)
            if args.include_external_paged and external_ref is not None:
                ax.axhline(
                    external_ref,
                    color=PAGED_PURPLE,
                    ls=":",
                    lw=1.3,
                    zorder=1,
                )

            for policy in policy_order:
                pts = curves.get(model, {}).get(task, {}).get(policy, [])
                if not pts:
                    continue
                xs = [c for c, _ in pts]
                ys = [v for _, v in pts]
                st = POLICY_STYLE[policy]
                ax.plot(
                    xs, ys, marker=st["marker"], color=st["color"],
                    lw=2.0, ms=6, zorder=3,
                )

            ax.set_title(
                f"{TASK_TITLE.get(task, task)} — {model_name.get(model, model)}",
                fontsize=11, fontweight="bold", color=INK,
            )
            ax.set_xscale("log", base=2)
            all_caps = sorted(
                {c for pol in policy_order
                 for c, _ in curves.get(model, {}).get(task, {}).get(pol, [])}
            )
            if all_caps:
                ax.set_xticks(all_caps)
                ax.set_xticklabels([str(c) for c in all_caps])
            ax.margins(x=0.06)
            # Metric is task-determined (QA F1 vs ROUGE-L), so label every
            # panel's y-axis with its own metric rather than a single left-edge
            # label -- otherwise the ROUGE-L columns would read as QA F1.
            mname = metric.get(task, "qa_f1")
            ax.set_ylabel(METRIC_LABEL.get(mname, mname), fontsize=10, color=INK)
            if ri == nrows - 1:
                ax.set_xlabel("KV capacity C (blocks)", fontsize=10, color=INK)

    # Compact legend only; intentionally no figure-level summary/title text.
    handles = [
        Line2D([0], [0], color=POLICY_STYLE[p]["color"],
               marker=POLICY_STYLE[p]["marker"], lw=2.0, ms=6,
               label=POLICY_STYLE[p]["label"])
        for p in policy_order
    ]
    handles.append(
        Line2D([0], [0], color=MUTED, ls="--", lw=1.3, label=FULL_LABEL)
    )
    if args.include_external_paged:
        handles.append(
            Line2D(
                [0],
                [0],
                color=PAGED_PURPLE,
                ls=":",
                lw=1.3,
                label=EXTERNAL_FULL_LABEL,
            )
        )
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles), frameon=False, fontsize=10.5,
        handletextpad=0.6, columnspacing=1.8,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
