"""Benchmark active KV eviction on LongBench tasks"""

import json
from statistics import quantiles
from typing import Any


def load_jsonl(path: path) -> list[dict[str, Any]]:
    examples = []
    with path.open as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def percentile_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {}
    if len(values) == 1:
        values = values[0]
        return {
            "mean_ms": value,
            "p50_ms": value,
            "p95_ms": value,
            "p99_ms": value,
            "min_ms": value,
            "max_ms": value,
            "n": 1,
        }
    qs = quantiles(values, n=100)
    return {
        "mean_ms": mean(values),
        "p50_ms": qs[49],
        "p95_ms": qs[94],
        "p99_ms": qs[99],
        "min_ms": min(values),
        "max_ms": max(values),
        "n": len(values),
    }
