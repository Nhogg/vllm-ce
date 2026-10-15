# R2R v2: implementation decisions

Status: floor and dispersion phase implemented as opt-in flags. Standalone
token-max timing and opt-in full-vs-boundary rank diagnostics have completed
their first small GPU probe; token-max is still not wired into the applied
eviction decision. Incremental pair-cost caching remains deferred. This resolves the open choices in the
revised v2 proposal against the current `fix/geo-kv` selector. The existing
v1 experiments remain the control. In particular, these choices do not claim
that a protective change cannot reduce accuracy: a fixed budget means every
newly protected block displaces some other block.

## 1. Persistent floor: protect the prompt, plus the live decode anchor

Use the existing `warmup_pages` sink count (one in the current sweeps). Add a
request-local, immutable set of **logical prompt block identities** covering
the last `query_tail_tokens` prompt tokens. For prompt length `P` and block
size `s`, the pinned set is the blocks intersecting
`[max(0, P - query_tail_tokens), P)`. At `s=16` and a 32-token window this is
two blocks when aligned and can be three when unaligned. It is not a
budget-dependent fraction or a measured question length.

Derive the logical indices from known prompt length at request setup. During
prefill, protect each pinned block as soon as it is materialized, including
when admission/eviction occurs before prefill completes. Keep the set after
prefill; do not redefine it from the current last block. Translate these
logical identities to current physical block positions after every compaction,
or store stable IDs in the block table. A request with no prefill eviction
still initializes this state. Clear it on request teardown/reuse.

Keep the selector's existing **current final block** protection as a separate
decode/write anchor. The complete exclusion set is thus sink blocks, pinned
prompt-tail blocks, and the current final block. Deduplicate overlaps. If a
requested budget is smaller than this floor, fail configuration/admission
explicitly rather than silently evicting protected blocks or exceeding the
budget. Retain protected blocks as possible *covers* for other blocks; they
are excluded only as eviction candidates.

Why: the moving final block alone ceases to protect the prompt's question once
decode appends blocks. The pinned set follows prompt content through physical
compaction, while the live final block preserves the existing decode invariant.

## 2. Dispersion: a conservative, self-relative guard

For each eligible block, sampled layer, and KV head, compute the value anchor
`v_bar = mean_t V_t` and distances `a_t = ||V_t - v_bar||_2` over **valid**
tokens. Define

```
rho = max_t(a_t) / max(median_t(a_t), eps_scale)
eps_scale = float32_machine_epsilon * max_t(||V_t||_2)
```

If both numerator and denominator are zero, define `rho = 1`. Use a
predeclared default `kappa = 2`: flag a block when the median across sampled
layers of the maximum head-wise `rho` is greater than 2. The median-layer
quorum avoids treating one anomalous layer as a certain needle, while the
head-wise maximum respects that eviction affects every KV head. Keep this
aggregation configurable for an ablation; do not tune it on LongBench scores.

Use median *distance*, not MAD of distances, as the denominator. MAD measures
how tightly the distances cluster around each other, not their typical size;
in high dimension it can approach zero for ordinary blocks and flag almost
everything. The proposed ratio is deliberately simple, but `kappa=2` is a
heuristic, not a statistical guarantee or a universal calibration. Inspect
flag rates by task, model, budget, layer, and head before interpreting it as
needle detection. A block with only one valid token is unflagged: it has no
within-block evidence of an outlier.

The guard is a **priority**, not an absolute prohibition. Let `T` be trusted
eligible blocks and `G` flagged eligible blocks; `b` is the exact number that
must be evicted.

- If `|T| >= b`, form the Stage-1 shortlist only from `T`, taking the lowest
  residual-cost `min(n*b, |T|)` blocks. Never add flagged blocks merely to
  fill the `n*b` expansion target.
- If `|T| < b`, every trusted block must be evicted. Choose the remaining
  `b - |T|` from flagged blocks using the normal R2R ranking and cover guard.
  The flagged shortlist may be expanded to `min(n*(b-|T|), |G|)`, but final
  selection must respect the two separate quotas. Log the forced-flagged
  count. If `|T| + |G| < b`, fail the capacity operation explicitly.

For Stage 1, keep the production **directional, worst-head projection
residual** and its sampled-layer aggregation, rather than replacing it with
plain cosine. Keep stable tie-breaking by logical block order. Compute
anchors for protected blocks too, since they may cover eligible blocks.

## 3. Relevance: preserve v1 now; test token-max separately

The cheap v1 relevance is the maximum query/key-anchor dot product over the
captured query tail and query heads, not `mean(W) dot K_anchor`. Preserve that
behavior in the first v2 quality run. The query buffer is fixed-size and
rolling; its reset after eviction is unchanged.

Do **not** enable boundary-only token-max in the same run as the floor and
dispersion changes. It needs a separate GPU timing and ranking audit first:

1. Implement a microbenchmark that computes valid-token, causal, GQA-aware
   `max_{q,t}(q dot K_t) * qk_scale` from retained Q/K tensors at several
   budgets and eviction sizes. Attention logits are not already retained by
   the forward pass, so this is real work and memory traffic.
2. Compare full-candidate token-max ranks with cheap ranks offline on sampled
   eviction events. Measure how often a block **outside** a proposed boundary
   would cross the final keep/evict decision. An inside-band flip counter alone
   cannot detect this miss.
3. Only if the audit shows an acceptable miss rate and cost, test a bounded
   boundary implementation as its own quality ablation. Choose the band size
   from the observed *last examined cheap rank* of the cover guard, not its
   cover-1/cover-2/backfill counts. Use a cold-start fixed cap and report band
   saturation during large prefill evictions.

If a boundary variant is tested, rerank **within the band only** and retain
the cheap order outside it. Do not sort a mixture of cheap and token-max
numeric scores: max over tokens is systematically on another scale from a
mean-pooled key score. A band around the raw `b`-th candidate is not a
correctness guarantee: a hidden needle can be ranked arbitrarily far away by
the cheap score. The out-of-band audit decides whether this approximation is
worth running; if not, benchmark another bounded design or full-candidate
token-max rather than claiming the boundary catches all needles.

## 4. Incremental pair costs: defer until scoring is stable

Do not adapt the current `IncrementalRedundancyState` cosine cache as a
drop-in. R2R's pair cost is directional, per-head residual with a worst-head
reduction. A future cache must key entries by stable block identity, layer,
and content generation; update a partially filled block; delete evicted
blocks; consider each new block as a possible better cover for **every**
survivor; and preserve deterministic tie-breaking. Physical compaction alone
must not invalidate unchanged logical pairs.

Require tests comparing full and incremental pair-cost matrices, covers, and
final eviction sets across append, partial-block growth, eviction,
compaction, and request reset. Include the case where a new block supersedes
an existing cover without any eviction. Numerical tolerance is appropriate
for floating-point costs; final selected IDs should match exactly under
stable tie-breaking. Treat this as a separate performance pass, not part of
the first quality run. `O(deltaB * B)` is a target for new-pair computation,
not an end-to-end complexity guarantee for cover updates or bookkeeping.

## Validation and rollout

1. Add CPU/selector tests for misaligned prompt tails, prefill eviction before
   prompt end, decode growth, physical compaction, exact budget, protected
   covers, and no-eligible-block behavior. Add synthetic dispersion tests for
   homogeneous blocks, one-token outliers, zero/near-zero spread, partial
   blocks, and multi-head/layer aggregation. Test strict trusted-first quotas.
2. Run matched **floor-only**, **dispersion-only**, and **combined** ablations
   against the unchanged v1 control before the full matrix. The combined
   result alone cannot attribute a gain or regression. Record exact paired
   prompts, seeds, accuracy, TTFT, TPOT, throughput, and GPU memory at budgets
   16/32/64/128/256; report variability across repeated runs.
3. Record floor size and overlaps, trusted/flagged eligible counts, forced
   flagged evictions, cover-1/cover-2/backfill decisions, last examined rank,
   and selector time. These are diagnostics, not additional tuning targets.
4. Run the token-max microbenchmark/rank audit independently. Only then
   decide whether its next quality ablation is boundary-only or full-pool.
5. Redesign incremental pair maintenance after the scoring policy is fixed,
   with parity tests and a separate performance comparison.

This order deliberately rejects the revised proposal's claim that floor plus
dispersion can only help or no-op. Both alter which other blocks survive at a
fixed budget; either can hurt accuracy. It also keeps the 75%-to-100% prefill
watermark question outside R2R scoring. Change that policy only in a separate
experiment with an explicit control.

## First token-max gate result (2026-10-12)

The one-GPU synthetic probe used an A100 40 GB, a 32-query window, 16-token
blocks, and the 3B audit model's 24 query heads / 8 KV heads / 128 head
dimension. At 16/64/256 blocks, median cheap-key-anchor time was
0.293/0.292/0.295 ms per scorer call; full-pool token-max was
0.333/0.331/0.335 ms. Quarter-pool bounded bands were
0.361/0.360/0.371 ms, including candidate gathers. Bands saved temporary
memory but were not faster at these sizes. These isolated timings are not
end-to-end TTFT/TPOT measurements.

The first live rank audit used 10 `multifieldqa_en` prompts on the 3B model,
with κ=2 and 10 prefill eviction fires. Full-pool token-max changed the cheap
selector on all 10 fires (15.1 selected-block swaps on average). Boundary
bands with margins 4/8/16/32 matched the full selection on **0/10** fires;
every fire had at least one newly full-selected block outside each band. The
widest margin still missed 8.0 selected blocks on average. The profiler also
recorded candidate coverage 1608/1982 (81.1%) and only 6 flagged eligible
blocks, with no forced flagged evictions, in this sample.

**Decision:** do not enable the proposed boundary-only token-max relevance.
Its approximation missed on every sampled fire and did not win synthetic
latency. This is one model/task and 10 prompts, not a general accuracy result:
we do not yet know whether full-pool token-max improves LongBench scores.
After the floor+dispersion quality sweep finishes, the next candidate is a
separate full-pool token-max quality/performance ablation against that v2
control, not an unqualified claim that token-max is better. Keep the
directional pair-cost cache redesign deferred until selection logic settles.
The concrete follow-up design is in `analysis/r2r_v2_full_pool_token_max_plan.md`.

Artifacts: `results/r2r_v2/token_max_microbench.json` and
`results/r2r_v2/rank_audit/summary.json`. The rank summary is generated from
`audit.json.scorer_profile.json` by
`benchmarks/analyze_r2r_rank_audit.py`.

## Full v1 vs combined-v2 quality result (2026-10-14)

PBS job `196082` exited zero. The validator accepted all 15 model/task cells,
150 arms, and 75 prompt-paired comparisons at 16/32/64/128/256 blocks. Each
task used 200 prompts except `multifieldqa_en` (150). The 0.75 watermark,
κ=2, key-anchor relevance, and sampled layers matched the submitted protocol.
The source checksum snapshot still matched after completion. Figures and the
paired CSV are in `results/r2r_v2_quality_matrix/charts/`.

Across the 75 comparisons, combined v2 beat v1 in 40, lost in 25, and tied in
10. This count is not a pooled task score: QA F1 and ROUGE-L have different
meanings, and the five budgets within each model/task are not independent.
Prompt-bootstrap 95% intervals excluded zero in 10 positive and eight
negative comparisons; 57 crossed zero. They are not adjusted for the 75
comparisons and do not measure run-to-run variability.

The pattern is heterogeneous. HotpotQA improved in 10/15 comparisons (five
positive intervals excluding zero), while Multi-News lost in 9/15 (six
negative intervals excluding zero). The largest observed loss was 8B
MultiFieldQA at 16 blocks: QA F1 delta −0.0849, paired 95% interval
[−0.1301, −0.0422]. At 256 blocks, ten comparisons tied exactly, consistent
with little or no eviction pressure in those cells. Do not claim a general v2
win or infer which component caused an individual gain/loss from this combined
comparison.

Reported v2 TTFT was higher in 59/75 comparisons, with a mean cellwise delta
of +147 ms; throughput and TPOT were mixed. These are single-run wall-clock
measurements, not repeated-run estimates, so treat the magnitude cautiously.
The next quality and overhead experiment should isolate **floor-only** and
**dispersion-only** alongside v1 and combined v2 on matched prompts. A
four-arm analyzer should report each arm minus v1 and the factorial
interaction `v2 - floor - dispersion + v1`, including paired bootstrap
intervals. Do not switch to boundary-only token-max based on this result;
the earlier rank audit still argues against that shortcut.

## Current-code anchors

- `vllm/v1/geo_kv/eviction_policy.py`: `select_evicted_r2r` and capacity paths.
- `vllm/v1/geo_kv/scoring.py`: `block_pair_residual_cost` and
  `block_key_anchor_relevance`.
- `vllm/v1/geo_kv/config.py`: `warmup_pages`, `query_tail_tokens`, and existing
  optional fractional tail protection.
- `vllm/v1/geo_kv/incremental_scorer.py`: current cosine-only incremental
  state, not an R2R pair-cost cache.
- `analysis/r2r_algorithm.md`: original R2R description; this document
  records proposed v2 decisions without changing that baseline.
