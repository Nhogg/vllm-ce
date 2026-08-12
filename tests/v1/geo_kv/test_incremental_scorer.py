# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for generation-safe incremental V-redundancy scoring."""

import torch

from vllm.v1.geo_kv.incremental_scorer import IncrementalRedundancyState
from vllm.v1.geo_kv.scoring import block_joint_redundancy


def _update(
    state: IncrementalRedundancyState,
    anchors: torch.Tensor,
    valid_lens: tuple[int, ...],
    generation: int,
) -> tuple[torch.Tensor, list[int]]:
    changed = state.changed_positions(valid_lens, generation)
    changed_anchors = anchors[changed]
    scores = state.update(
        valid_lens, generation, changed, changed_anchors
    )
    return scores, changed


def test_first_update_matches_full_and_builds_all_rows():
    torch.manual_seed(0)
    anchors = torch.randn(8, 2, 4)
    state = IncrementalRedundancyState.empty(generation=0)

    scores, changed = _update(state, anchors, (4,) * 7 + (2,), generation=0)

    torch.testing.assert_close(scores, block_joint_redundancy(anchors), rtol=0, atol=0)
    assert changed == list(range(8))


def test_growth_recomputes_only_changed_tail_and_matches_full():
    torch.manual_seed(1)
    first = torch.randn(5, 2, 4)
    state = IncrementalRedundancyState.empty(generation=0)
    _update(state, first, (4, 4, 4, 4, 2), generation=0)

    grown = torch.cat((first.clone(), torch.randn(1, 2, 4)))
    grown[4] = torch.randn(2, 4)  # formerly partial tail is now complete
    scores, changed = _update(state, grown, (4, 4, 4, 4, 4, 1), generation=0)

    assert changed == [4, 5]
    torch.testing.assert_close(scores, block_joint_redundancy(grown), rtol=0, atol=0)


def test_compaction_prunes_survivors_then_reuses_them_exactly():
    torch.manual_seed(2)
    anchors = torch.randn(8, 1, 6)
    state = IncrementalRedundancyState.empty(generation=0)
    _update(state, anchors, (4,) * 7 + (3,), generation=0)
    evicted = torch.tensor([False, False, True, False, True, False, False, False])

    state.retain(evicted, generation=1)
    survivors = anchors[~evicted]
    grown = torch.cat((survivors, torch.randn(1, 1, 6)))
    scores, changed = _update(
        state, grown, (4, 4, 4, 4, 4, 4, 1), generation=1
    )

    # The retained partial anchor becomes complete and the appended block is new.
    assert changed == [5, 6]
    torch.testing.assert_close(scores, block_joint_redundancy(grown), rtol=0, atol=0)


def test_generation_mismatch_forces_safe_full_rebuild():
    torch.manual_seed(3)
    anchors = torch.randn(4, 1, 4)
    state = IncrementalRedundancyState.empty(generation=0)
    _update(state, anchors, (4, 4, 4, 2), generation=0)

    replacement = torch.randn_like(anchors)
    scores, changed = _update(state, replacement, (4, 4, 4, 2), generation=1)

    assert changed == [0, 1, 2, 3]
    torch.testing.assert_close(
        scores, block_joint_redundancy(replacement), rtol=0, atol=0
    )


def test_bad_compaction_mask_invalidates_instead_of_reusing():
    anchors = torch.randn(4, 1, 3)
    state = IncrementalRedundancyState.empty(generation=0)
    _update(state, anchors, (4, 4, 4, 2), generation=0)

    state.retain(torch.tensor([False, True]), generation=1)

    assert state.num_blocks == 0
    assert state.generation == 1
