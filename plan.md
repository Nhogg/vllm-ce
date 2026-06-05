# KV Cache Eviction Policy Ablation Study

## Goal

Systematically compare GPU-side KV cache eviction policies in vLLM to identify
which policies best balance cache hit rate, latency, and GPU utilization across
representative workloads. This feeds into implementing and validating a new
eviction policy (paged eviction) and establishing a baseline for future work.

---

## Eviction Policies to Study

| Policy | Status | Notes |
|--------|--------|-------|
| **LRU** | Existing (baseline) | `FreeKVCacheBlockQueue` in `vllm/v1/core/kv_cache_utils.py` |
| **Random** | To implement | Randomly select victim block from free pool |
| **Paged** | To implement | From paper: [cite paper — clarify which one] |
| **ARC** | Existing (CPU offload only) | `vllm/v1/kv_offload/cpu/policies/arc.py` — evaluate porting to GPU path |
| **FIFO** | Possibly trivial | Already implicit in some paths; evaluate cost of adding explicitly |

> **Action needed:** Confirm which "paged eviction" paper is the reference
> (e.g., vAttention, MemGPT, or another). The implementation scope changes
> significantly depending on whether "paged" means block-granularity CLOCK,
> frequency-weighted, or something else.

---

## Metrics

### Primary (keep regardless)

| Metric | Description | Where to collect |
|--------|-------------|-----------------|
| **TTFT** | Time to first token (ms) | `vllm/v1/metrics/loggers.py`, existing histogram |
| **ITL / TPOT** | Inter-token latency / time per output token (ms) | Same |
| **E2E latency** | Request wall-clock time (ms), p50/p95/p99 | Same |
| **Tokens/sec** | Output throughput | Derived from ITL × batch |
| **Prefix cache hit rate** | Fraction of prompt tokens served from cache | `CachingMetrics` in `vllm/v1/metrics/stats.py` |

### Secondary (whittle after initial runs)

| Metric | Description | Where to collect |
|--------|-------------|-----------------|
| **Cache miss penalty** | TTFT uplift on a cache miss vs. hit | Compute as TTFT_miss − TTFT_hit; tag requests at scheduler |
| **Recompute cost avoided** | FLOPs/tokens that would have been recomputed without cache | Estimate: hit_blocks × block_size × model_flops_per_token |
| **Eviction regret** | Fraction of evicted blocks later re-requested within a window | Track block hashes post-eviction; count re-requests in `BlockPool` |
| **GPU KV cache utilization** | % of KV cache capacity occupied | `gpu_cache_usage_perc` already in `SchedulerStats` |
| **Tail latency (p99/p999)** | Worst-case latency under cache pressure | Extend existing histograms |

---

## Architecture: Making Eviction Pluggable on the GPU Path

Currently the GPU cache has hardcoded LRU. The CPU offload path already has a
clean pluggable policy abstraction (`vllm/v1/kv_offload/cpu/policies/base.py`).
We will mirror that pattern for the GPU path.

### Files to create / modify

```
vllm/v1/core/eviction/
    __init__.py
    base.py          # Abstract EvictionPolicy: select_victim(free_blocks) → block_id
    lru.py           # Wrap existing FreeKVCacheBlockQueue behavior
    random.py        # Random victim selection
    paged.py         # Paged eviction (TBD on paper spec)

vllm/v1/core/block_pool.py          # Wire policy into _maybe_evict_cached_block()
vllm/config/cache.py                # Add eviction_policy: str config flag
```

The key invariant: **the block pool still owns ref-counting and hash maps**;
the eviction policy only answers "which free block should I evict next?"
It must not touch allocator state directly.

---

## Implementation Plan

### Phase 0: Instrumentation (do first — needed for all metrics)

1. Add `eviction_policy` config flag to `CacheConfig` (default: `"lru"`).
2. Add per-eviction event tagging to `KVCacheEvictionEvent`: which policy made
   the decision, block hash, idle time, lifetime.
3. Add eviction regret tracker: ring-buffer of recently evicted block hashes,
   checked on each `allocate_slots()` call to detect "we just evicted that".
4. Expose cache miss penalty: tag each `allocate_slots()` call with whether the
   prefix lookup was a hit or miss; record TTFT separately for each group.

### Phase 1: Implement Random Policy

- `RandomEvictionPolicy`: at eviction time, pick a uniformly random block from
  the free-but-cached pool (blocks with `ref_cnt == 0` and a cached hash).
- Expected to perform worse than LRU — useful as a lower bound.
- Implementation effort: ~50 LOC.

### Phase 2: Implement Paged Eviction

- Depends on paper clarification.
- Likely involves block access frequency or recency-frequency hybrid (CLOCK-Pro,
  LIRS, etc.).
- Implementation effort: TBD.

### Phase 3: Port ARC to GPU Path (optional / stretch)

- ARC already exists for CPU offload; porting to GPU means adapting ghost list
  logic to block-pool free list semantics.
- Evaluate complexity vs. expected gain first.

### Phase 4: Benchmark Harness

Extend `benchmarks/benchmark_prefix_caching.py` with:
- `--eviction-policy {lru,random,paged,arc}` flag
- Workload sweep: synthetic (high-reuse, low-reuse, mixed), ShareGPT, long-doc
- Cache pressure sweep: fill cache to 50%, 80%, 100% before measuring
- Output CSV with all primary and secondary metrics per request

---

## Workloads for Ablation

| Workload | Rationale |
|----------|-----------|
| **High-reuse synthetic** | Repeated identical prefixes → maximal cache benefit |
| **Low-reuse synthetic** | Random prefixes → stress eviction policy decision quality |
| **Mixed (80/20)** | 80% repeated prefixes, 20% novel → realistic chat/API traffic |
| **ShareGPT** | Real conversation dataset, multi-turn |
| **Long-document QA** | Long shared context, many queries → tests block-granularity policies |
| **Cache-saturated** | Fill cache to 100% then stream requests → pure eviction pressure |

---

## Experimental Protocol

1. Fix model (e.g., Llama-3-8B), GPU (single A100/H100), vLLM concurrency.
2. Fix random seed for workload generation.
3. Run each (policy × workload × cache-fill) triple independently with a
   warm-up period discarded.
4. Collect 10k requests per run minimum.
5. Report p50/p95/p99 for latency metrics; mean ± std for throughput/hit-rate.
6. Statistical significance: Mann-Whitney U test for latency comparisons.

---

## Open Questions / Decisions Needed

- [ ] Which paper defines "paged eviction"? Need citation/link.
- [ ] Should we study GPU-only, or also GPU+CPU-offload tier interaction?
- [ ] What model(s) to fix for benchmarks? (Larger model = more cache pressure)
- [ ] Do we need multi-GPU / disaggregated prefill scenarios?
- [ ] Is eviction regret the right secondary metric, or should we use "wasted
      eviction rate" (evicted blocks never re-requested = wasted eviction work)?

---

## Milestones

| # | Deliverable | Depends on |
|---|-------------|------------|
| M0 | Instrumentation + config flag | — |
| M1 | Random policy + harness working end-to-end | M0 |
| M2 | Baseline LRU vs. Random ablation results | M1 |
| M3 | Paged eviction implemented | Paper clarified, M0 |
| M4 | Full ablation results (all policies × all workloads) | M2, M3 |
| M5 | Analysis: metric correlation, policy recommendation | M4 |
