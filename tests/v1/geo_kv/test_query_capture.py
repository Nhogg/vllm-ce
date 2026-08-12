# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.v1.geo_kv.query_capture import QueryCapturer


def _capturer(tail_tokens: int = 2) -> QueryCapturer:
    return QueryCapturer(
        [(3, 2, 2, 0.5), (7, 2, 2, 0.5)],
        max_num_reqs=16,
        tail_tokens=tail_tokens,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_query_capture_gathers_each_requests_tail_without_cross_talk():
    cap = _capturer()
    cap.begin_step(
        np.array([9, 4], dtype=np.int32),
        np.array([0, 3, 7], dtype=np.int32),
        np.array([3, 4], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
        np.array([3, 4], dtype=np.int32),
        np.array([True, True]),
    )
    q = torch.arange(7 * 2 * 2, dtype=torch.float32).reshape(7, 2, 2)
    cap.capture(3, q)
    assert torch.equal(cap.get(9, 3), q[1:3])
    assert torch.equal(cap.get(4, 3), q[5:7])
    assert cap.get(9, 7) is None
    assert cap.scale(3) == 0.5


def test_query_capture_rolls_query_tail_across_prefill_chunks():
    cap = _capturer(tail_tokens=4)
    cap.begin_step(
        np.array([2], dtype=np.int32),
        np.array([0, 3], dtype=np.int32),
        np.array([3], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([5], dtype=np.int32),
        np.array([True]),
    )
    first = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    cap.capture(3, first)
    cap.begin_step(
        np.array([2], dtype=np.int32),
        np.array([0, 2], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([3], dtype=np.int32),
        np.array([5], dtype=np.int32),
        np.array([True]),
    )
    second = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) + 100
    cap.capture(3, second)
    assert torch.equal(cap.get(2, 3), torch.cat((first[-2:], second)))


def test_query_capture_uses_observed_suffix_after_cached_prefix():
    cap = _capturer(tail_tokens=4)
    cap.begin_step(
        np.array([2], dtype=np.int32),
        np.array([0, 2], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([8], dtype=np.int32),
        np.array([10], dtype=np.int32),
        np.array([True]),
    )
    suffix = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    cap.capture(3, suffix)
    assert torch.equal(cap.get(2, 3), suffix)


def test_query_capture_allows_short_unchunked_prompt_and_dummy_noop():
    cap = _capturer(tail_tokens=4)
    args = (
        np.array([2], dtype=np.int32),
        np.array([0, 2], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([True]),
    )
    cap.begin_step(*args)
    q = torch.ones(2, 2, 2)
    cap.capture(3, q)
    assert cap.get(2, 3).shape[0] == 2
    cap.begin_step(*args, dummy=True)
    cap.capture(3, q)
    assert cap.get(2, 3) is None


def test_query_capture_survives_request_slot_reassignment():
    cap = _capturer(tail_tokens=4)
    cap.begin_step(
        np.array([2], dtype=np.int32),
        np.array([0, 3], dtype=np.int32),
        np.array([3], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([5], dtype=np.int32),
        np.array([True]),
    )
    first = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    for layer in (3, 7):
        cap.capture(layer, first)
    cap.stash_request("request-a", 2)
    assert cap.restore_request("request-a", 5)

    cap.begin_step(
        np.array([5], dtype=np.int32),
        np.array([0, 2], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([3], dtype=np.int32),
        np.array([5], dtype=np.int32),
        np.array([True]),
    )
    second = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) + 100
    cap.capture(3, second)
    assert torch.equal(cap.get(5, 3), torch.cat((first[-2:], second)))
