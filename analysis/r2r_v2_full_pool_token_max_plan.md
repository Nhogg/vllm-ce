# Next R2R v2 ablation: full-pool token-max relevance

Status: design-only. Do not alter the queued floor+dispersion quality sweep
(`195668`) or interpret the 10-prompt rank audit as an accuracy result.

## Why this is the next test

The 2026-10-12 audit found that cheap key-anchor relevance and full token-max
selected different blocks on all 10 sampled prefill fires. A boundary-only
rerank missed full-pool selections even with a 32-rank margin, and the sampled
bounded bands were slower than full scoring on the A100 probe. Thus the
boundary shortcut failed its gate. The remaining question is whether the
*full* token-max decision improves LongBench task scores enough to justify
its real end-to-end cost.

## Minimal implementation, once the current quality run is stable

1. Add an opt-in `r2r_relevance_signal="token_max"` choice. Leave
   `key_anchor` and `attention_mass` semantics unchanged. Require R2R mode,
   physical reclaim, and captured queries as the existing relevance paths do.
2. Reuse `block_token_max_relevance` from `scoring.py` and the audit's
   32-block chunking to bound temporary memory. Compute valid-token and
   causal QK maxima for each sampled layer, then aggregate layers exactly as
   the current key-anchor path does. The stage-1 residual shortlist, trusted
   versus flagged quotas, protected prompt floor, cover guard, and backfill
   must be byte-identical between arms; only the stage-2 relevance order may
   change. Do not combine numeric key-anchor and token-max scores.
3. Disable the expensive rank audit during the applied token-max run to avoid
   scoring QK twice. Keep per-fire selector timing and peak memory metrics so
   the measured overhead is visible. Confirm `r2r_relevance_signal` and κ=2
   appear in each raw JSON/CSV output.
4. Add CPU tests for causal masking, partial blocks, GQA head mapping,
   sampled-layer aggregation, exact eviction budget, and a constructed case
   where one relevant token reverses the key-anchor selection. Check the
   opt-in default leaves existing v1 and floor+dispersion runs unchanged.

## Experiment design

First use the same 3B model, five LongBench tasks, exact prompt IDs and seeds,
and 16/32/64/128/256 block capacity ceilings as the current sweep. Compare
`v2 + key_anchor` against `v2 + full token_max` with identical κ=2, one sink
block, pinned prompt tail, sampled layers `3,10,17,24`, candidate factor 4,
cover depth 2, and 0.75 low-watermark. Record the task's native score (QA F1
or ROUGE-L), paired per-prompt differences and confidence intervals, TTFT,
TPOT, throughput, steady/peak KV blocks, evictions, and scorer time/peak
temporary memory. Label the outputs distinctly; do not merge them into the
current `v1,v2` CSV or overwrite the queued run's directory.

If the five-task pilot shows a credible quality benefit without unacceptable
performance cost, extend to all three Llama models and repeat runs for
run-to-run variability. Do not choose a margin or other parameter using the
same task scores that will be reported as the final evaluation.

## Decision rule

Do not infer that token-max is better merely because the eviction mask
changed. Advance it only if paired task scores improve on a meaningful range
of budgets/tasks and the measured end-to-end overhead is acceptable. If
accuracy is mixed, inspect per-task and per-budget regressions before any
broader rollout. If quality is neutral or worse, keep the cheaper key-anchor
path. The incremental directional pair-cost cache remains a separate later
performance pass.

Source results: `results/r2r_v2/token_max_microbench.json` and
`results/r2r_v2/rank_audit/summary.json`. Meeting-facing summary:
`writeups/meeting_2026-10-12_r2r_v2_result.md`.
