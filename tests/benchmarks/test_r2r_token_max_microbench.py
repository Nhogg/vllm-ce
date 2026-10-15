"""CPU checks for the token-max CUDA probe's bounded band matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from benchmark_r2r_token_max_cuda import band_sizes  # noqa: E402


def test_band_sizes_deduplicate_and_exclude_full_pool():
    assert band_sizes("4,8,8,16,32,64", 16) == [4, 8]
    assert band_sizes("4,8,16,32,64", 256) == [4, 8, 16, 32, 64]


@pytest.mark.parametrize("spec", ["", "0,4", "-1,4", "four,8"])
def test_band_sizes_reject_invalid_spec(spec):
    with pytest.raises(ValueError, match="positive integers"):
        band_sizes(spec, 16)
