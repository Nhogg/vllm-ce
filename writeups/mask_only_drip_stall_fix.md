# Mask-only decode drip stall fix

## Scope

This change fixes Phase R0's mask-only fixed-rate decode eviction stall. It is
limited to the decode drip's mask-only accounting and a focused lifecycle
regression test. Physical reclamation and watermark-band behavior are unchanged.

## Problem

`decode_evict_blocks_per_step=N` promises to hide `N` new interior blocks on
each eviction fire. In mask-only mode the block table is never compacted, so
previously hidden blocks remain in `evicted_store`.

The old path always selected a total of `N` blocks. It assigned previously
hidden blocks an infinite droppability score so they could not become visible
again. After the first fire those old blocks occupied the entire top-`N`
selection. OR-merging that selection into `evicted_store` added no new blocks,
so subsequent fires stalled.

## Implementation

In `vllm/v1/geo_kv/eviction_policy.py`, the mask-only drip now includes the
number of previously hidden blocks in its cumulative eviction target:

```text
budget = count - blocks_per_step - previously_hidden
```

The existing selector still reselects the old mask first, but its larger total
eviction target now has room for `N` additional blocks. Existing sink, anchor,
tail-protection, and selector clamping behavior remain authoritative when fewer
than `N` blocks are available.

The adjustment is made only when `blocks_per_step` is active and only after the
physical-reclaim branch has returned. The physical drip and both capacity-band
paths therefore retain their prior budget calculations.

## Regression coverage

`test_mask_only_drip_hides_new_blocks_on_each_fire` in
`tests/v1/geo_kv/test_capacity_band_lifecycle.py` creates an eight-block request
with a two-block mask-only drip and fires it twice. It verifies that:

- the first fire hides exactly two blocks;
- the second fire preserves every previously hidden block;
- the second fire increases the hidden count to four; and
- the decode-eviction event counter advances on both fires.

## Verification

Passed:

```bash
.venv/bin/python -m py_compile \
  vllm/v1/geo_kv/eviction_policy.py \
  tests/v1/geo_kv/test_capacity_band_lifecycle.py
git diff --check
```

The focused pytest command was attempted:

```bash
.venv/bin/python -m pytest \
  tests/v1/geo_kv/test_capacity_band_lifecycle.py -k 'drip' -v
```

It failed while importing `torch` from `tests/conftest.py`, before test
collection, because the existing environment's `libtorch_cuda.so` could not
resolve `ncclCommWindowDeregister`. This is an environment/library mismatch,
not a test failure.

The repository's `pre-commit` and `ruff` executables are not installed in the
current shell or `.venv`, so lint hooks could not be run without modifying the
environment.

## Suggested commit

Stage only the implementation, regression test, and this writeup. The worktree
contains unrelated untracked benchmark and script files that should not be
included.

Suggested subject:

```text
fix(geo-kv): advance mask-only decode drip
```

Per the repository contribution policy, add the appropriate AI-assistance and
human sign-off trailers when committing. The human submitter should review all
three staged files and rerun the focused test after repairing the PyTorch/NCCL
environment.
