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
    DATASET2PROMPT,
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
