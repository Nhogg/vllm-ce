# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the opt-in GeoKV scorer profiler (plan Phase 1B).

Covers the engine-independent behavior: disabled is a true no-op (shared null
context, no samples), enabled records per-phase timings and per-fire counts, the
summary carries median/p95/total, and a JSON report round-trips to disk.
"""

import json

import pytest

from vllm.v1.geo_kv.scorer_profiler import PHASES, ScorerProfiler


def test_disabled_profiler_is_a_noop():
    prof = ScorerProfiler(enabled=False)
    # The disabled path must reuse one shared context object (no per-call alloc).
    a = prof.section("gather")
    b = prof.section("similarity")
    assert a is b
    with prof.section("gather"):
        pass
    prof.record_fire(10, 4)
    # Nothing recorded, and dump() is a silent no-op.
    summ = prof.summary()
    assert summ["fires"] == 0
    assert summ["phases"] == {}
    prof.dump()  # must not raise


def test_enabled_profiler_records_phases_and_fires():
    prof = ScorerProfiler(enabled=True)  # no device -> wall-clock timing
    for _ in range(5):
        with prof.section("gather"):
            pass
        with prof.section("similarity"):
            pass
        prof.record_fire(num_blocks=12, num_layers=3)
    summ = prof.summary()
    assert summ["fires"] == 5
    assert set(summ["phases"]) == {"gather", "similarity"}
    for name in ("gather", "similarity"):
        ph = summ["phases"][name]
        assert ph["count"] == 5
        assert ph["total_s"] >= 0.0
        # median/p95/mean are present and finite.
        assert ph["median_ms"] >= 0.0
        assert ph["p95_ms"] >= ph["median_ms"] or ph["count"] < 2
    assert summ["blocks_per_fire"]["median"] == 12
    assert summ["layers_per_fire"]["median"] == 3


def test_phase_order_is_stable_and_documented():
    # The report iterates PHASES in pipeline order; every canonical name is unique.
    assert len(set(PHASES)) == len(PHASES)
    assert PHASES[0] == "gather"
    assert PHASES.index("query_attention") < PHASES.index("similarity")
    assert "selection" in PHASES and "transfer" in PHASES


def test_summary_only_includes_phases_that_fired():
    prof = ScorerProfiler(enabled=True)
    with prof.section("aggregate"):
        pass
    prof.record_fire(1, 1)
    summ = prof.summary()
    assert list(summ["phases"]) == ["aggregate"]


def test_dump_writes_json_report(tmp_path):
    log_path = str(tmp_path / "run.log")
    prof = ScorerProfiler(enabled=True, log_path=log_path)
    with prof.section("selection"):
        pass
    prof.record_fire(8, 2)
    prof.dump()
    report_path = tmp_path / "run.log.scorer_profile.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert data["fires"] == 1
    assert "selection" in data["phases"]


def test_dump_without_fires_writes_nothing(tmp_path):
    log_path = str(tmp_path / "run.log")
    prof = ScorerProfiler(enabled=True, log_path=log_path)
    prof.dump()  # no fires recorded
    assert not (tmp_path / "run.log.scorer_profile.json").exists()


def test_query_window_summary():
    prof = ScorerProfiler(enabled=True)
    prof.record_fire(8, 2)
    for size in (7, 32, 16):
        prof.record_query_window(size)
    summary = prof.summary()["queries_per_fire"]
    assert summary == {"median": 16.0, "p95": 32.0, "min": 7.0, "max": 32.0}


def test_query_direction_coherence_summary():
    prof = ScorerProfiler(enabled=True)
    prof.record_fire(8, 2)
    for value in (0.25, 1.0, 0.5):
        prof.record_query_direction_coherence(value)
    summary = prof.summary()["query_direction_coherence"]
    assert summary == {
        "count": 3,
        "mean": pytest.approx(7 / 12),
        "median": 0.5,
        "p95": 1.0,
        "min": 0.25,
        "max": 1.0,
    }


def test_retained_value_logdet_summary():
    prof = ScorerProfiler(enabled=True)
    prof.record_fire(8, 2)
    prof.record_retained_value_logdet([1.0, 3.0])
    prof.record_retained_value_logdet([2.0])

    assert prof.summary()["retained_value_logdet"] == {
        "count": 3,
        "mean": 2.0,
        "median": 2.0,
        "p95": 3.0,
        "min": 1.0,
        "max": 3.0,
    }


def test_r2r_selection_summary():
    prof = ScorerProfiler(enabled=True)
    prof.record_fire(8, 2)
    prof.record_r2r_selection(4, 8, {"cover_1": 2, "backfill": 1})
    prof.record_r2r_selection(3, 6, {"cover_1": 1, "cover_2": 2})

    assert prof.summary()["r2r_selection"] == {
        "fires": 2,
        "candidate_blocks": 7,
        "eligible_blocks": 14,
        "candidate_coverage": 0.5,
        "reasons": {"cover_1": 3, "backfill": 1, "cover_2": 2},
    }


def test_incremental_update_summary():
    prof = ScorerProfiler(enabled=True)
    prof.record_incremental_update(total_blocks=8, recomputed_blocks=8)
    prof.record_incremental_update(total_blocks=8, recomputed_blocks=2)
    prof.record_fire(8, 2)

    summary = prof.summary()["incremental_updates"]
    assert summary == {
        "count": 2,
        "total_block_rows": 16,
        "recomputed_block_rows": 10,
        "reused_fraction": 0.375,
        "recomputed_median": 2.0,
        "recomputed_p95": 8.0,
    }
    fire = prof.summary()["scoring_fires"][0]
    assert fire["num_blocks"] == 8
    assert fire["num_layers"] == 2
    assert fire["incremental_total_rows"] == 16
    assert fire["incremental_recomputed_rows"] == 10


def test_scoring_fire_details_do_not_leak_prior_fire_samples():
    prof = ScorerProfiler(enabled=True)
    with prof.section("gather"):
        pass
    prof.record_fire(64, 1)
    with prof.section("similarity"):
        pass
    prof.record_fire(64, 1)

    first, second = prof.summary()["scoring_fires"]
    assert set(first["phases_s"]) == {"gather"}
    assert set(second["phases_s"]) == {"similarity"}
