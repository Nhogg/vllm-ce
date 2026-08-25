# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the LongBench task tables used by the eviction-accuracy campaign.

Pure-table checks (no model, no GPU): every scored task must have both a prompt
template and a generation budget, and each template must consume exactly the
``{context}`` and ``{input}`` placeholders. A second, data-dependent check
formats one real record per task and is skipped when the LongBench jsonl files
are not present (e.g. generic CI).
"""

import json
import os
import sys

import pytest

_BENCH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "benchmarks"
)
sys.path.insert(0, os.path.abspath(_BENCH))

from benchmark_kv_eviction_accuracy import (  # noqa: E402
    DATASET2MAXGEN,
    DATASET2METRIC,
    DATASET2PROMPT,
    SUMMARIZATION_TASKS,
    rouge_l_max,
    score_prediction,
)

# The tasks the campaign sweeps (the original 3 + the 4-task expansion).
CAMPAIGN_TASKS = [
    "multifieldqa_en",
    "hotpotqa",
    "qasper",
    "2wikimqa",
    "musique",
    "narrativeqa",
    "passage_retrieval_en",
]

# Summarization tasks (context-only, ROUGE-L scored) added for the baseline
# campaign. Their templates key only on {context} (empty {input}).
SUMM_TASKS = list(SUMMARIZATION_TASKS)
ALL_TASKS = CAMPAIGN_TASKS + SUMM_TASKS

_DATA_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data", "longbench", "data"
    )
)


@pytest.mark.parametrize("task", CAMPAIGN_TASKS)
def test_task_has_template_and_maxgen(task):
    assert task in DATASET2PROMPT, f"{task} missing prompt template"
    assert task in DATASET2MAXGEN, f"{task} missing max_gen"
    assert DATASET2MAXGEN[task] > 0


@pytest.mark.parametrize("task", CAMPAIGN_TASKS)
def test_template_placeholders(task):
    tmpl = DATASET2PROMPT[task]
    # Both placeholders present; formatting with them leaves nothing unfilled.
    filled = tmpl.format(context="CTX", input="Q")
    assert "{context}" not in filled and "{input}" not in filled
    assert "CTX" in filled and "Q" in filled


@pytest.mark.parametrize("task", CAMPAIGN_TASKS)
def test_real_record_formats(task):
    path = os.path.join(_DATA_DIR, f"{task}.jsonl")
    if not os.path.exists(path):
        pytest.skip(f"LongBench data not present: {path}")
    with open(path) as f:
        rec = json.loads(f.readline())
    for key in ("context", "input", "answers"):
        assert key in rec, f"{task} record missing {key!r}"
    out = DATASET2PROMPT[task].format(context=rec["context"], input=rec["input"])
    assert len(out) > 0
    assert rec["answers"], f"{task} has empty answers"


# --- Summarization tasks (ROUGE-L, context-only) ---------------------------


@pytest.mark.parametrize("task", SUMM_TASKS)
def test_summ_task_has_template_and_maxgen(task):
    assert task in DATASET2PROMPT, f"{task} missing prompt template"
    assert task in DATASET2MAXGEN, f"{task} missing max_gen"
    # Summaries need room; guard against a QA-sized budget slipping in.
    assert DATASET2MAXGEN[task] >= 256


@pytest.mark.parametrize("task", SUMM_TASKS)
def test_summ_task_is_rouge_scored(task):
    assert DATASET2METRIC[task] == "rouge_l"


@pytest.mark.parametrize("task", SUMM_TASKS)
def test_summ_template_needs_only_context(task):
    # Context-only tasks: {context} fills the whole template even with an empty
    # {input} (records carry input=""), and nothing is left unfilled.
    tmpl = DATASET2PROMPT[task]
    filled = tmpl.format(context="CTX", input="")
    assert "{context}" not in filled and "{input}" not in filled
    assert "CTX" in filled


@pytest.mark.parametrize("task", SUMM_TASKS)
def test_summ_real_record_formats(task):
    path = os.path.join(_DATA_DIR, f"{task}.jsonl")
    if not os.path.exists(path):
        pytest.skip(f"LongBench data not present: {path}")
    with open(path) as f:
        rec = json.loads(f.readline())
    for key in ("context", "answers"):
        assert key in rec, f"{task} record missing {key!r}"
    out = DATASET2PROMPT[task].format(
        context=rec["context"], input=rec.get("input", "")
    )
    assert len(out) > 0
    assert rec["answers"], f"{task} has empty answers"


# --- Metric dispatch --------------------------------------------------------


def test_every_campaign_and_summ_task_has_a_metric():
    for task in ALL_TASKS:
        assert task in DATASET2METRIC, f"{task} missing from DATASET2METRIC"
        assert DATASET2METRIC[task] in ("qa_f1", "rouge_l")


def test_score_prediction_dispatches_by_task():
    # QA task -> token-F1: exact match scores 1.0.
    assert score_prediction("hotpotqa", "Paris", ["Paris"]) == 1.0
    # Summarization task -> ROUGE-L: identical text scores 1.0.
    ref = "the agency issued a detailed multiyear procurement report"
    assert score_prediction("gov_report", ref, [ref]) == 1.0


def test_rouge_l_max_handles_empty_and_multi_answer():
    # Empty prediction -> 0.0 (no crash).
    assert rouge_l_max("", ["some gold summary"]) == 0.0
    # Max over the answer list: a perfect match to one gold answer wins.
    got = rouge_l_max("the cat sat", ["totally different", "the cat sat"])
    assert got == 1.0


def test_unknown_task_defaults_to_qa_f1():
    # Tasks not in the metric table fall back to qa_f1 (back-compat).
    assert score_prediction("some_new_task", "Paris", ["Paris"]) == 1.0
