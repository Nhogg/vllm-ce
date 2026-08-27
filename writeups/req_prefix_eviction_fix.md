# User request ID prefix eviction fix

## Scope

This change fixes Phase R0's synthetic-request ID collision in both active
GeoKV eviction and score-only prefill analysis.

## Problem

GeoKV treated every request ID beginning with `req_` as an internal dummy
request. `req_` is also a plausible user-generated prefix, so legitimate
requests with IDs such as `req_customer_0` were silently excluded from prefill
scoring, prefill eviction, and decode eviction.

vLLM's `InputBatch.make_dummy` does generate IDs beginning with `req_`, but the
GPU model runner already identifies those executions with `dummy_run=True` and
does not call the GeoKV scorer or eviction policy for them. Inferring dummy
status from the request ID is therefore unnecessary.

Kernel warmup batches use the separately reserved `_warmup_` prefix and can
reach request-discovery helpers outside the normal model-runner guard, so that
specific exclusion remains.

## Implementation

The broad tuple `("_warmup_", "req_")` was replaced with the single reserved
kernel-warmup prefix in:

- `vllm/v1/geo_kv/eviction_policy.py`; and
- `vllm/v1/geo_kv/prefill_scorer.py`.

Both prefill request-discovery paths and decode eviction now skip only request
IDs beginning with `_warmup_`. Normal requests beginning with `req_` follow the
same scoring and eviction paths as every other user request.

## Regression coverage

The capacity-band lifecycle tests now verify that:

- `_warmup_0_` is still excluded from prefill eviction;
- `req_customer_0` completes normal physical prefill eviction;
- score-only request discovery includes `req_customer_0` but excludes a warmup
  request in the same batch; and
- decode eviction ignores a prefilling request and a warmup request while
  evicting `req_customer_2` in the same batch.

## Verification

Passed:

```bash
.venv/bin/python -m py_compile \
  vllm/v1/geo_kv/eviction_policy.py \
  vllm/v1/geo_kv/prefill_scorer.py \
  tests/v1/geo_kv/test_capacity_band_lifecycle.py
git diff --check
```

The focused pytest command was attempted:

```bash
.venv/bin/python -m pytest \
  tests/v1/geo_kv/test_capacity_band_lifecycle.py \
  -k 'request_id or warmup or synthetic' -v
```

It failed while importing `torch` from `tests/conftest.py`, before test
collection, because the existing environment's `libtorch_cuda.so` could not
resolve `ncclCommWindowDeregister`. This is the same environment/library issue
encountered while verifying the mask-only drip fix, not a test assertion
failure.

The human submitter should review all changed lines and rerun the focused tests
after repairing the PyTorch/NCCL environment.
