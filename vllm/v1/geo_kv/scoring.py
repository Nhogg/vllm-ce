# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Geometric KV redundancy scoring (Option A, score_only).

Pure tensor math with no vLLM runner dependencies, so it can be unit-tested in
isolation. See ``plan.md`` Phase 3 for the definitions:

For each retained block ``b`` at (layer ``l``, KV head ``h``):
    anchor[b] = L2_normalize(mean over the block's valid tokens of K[b])
    redundancy[b] = max over b2 != b of cosine(anchor[b], anchor[b2])

Then per (layer, head) we aggregate redundancy across blocks into
mean / max / p90 / std, plus the number of blocks scored.
"""

from __future__ import annotations

import torch


def compute_prefill_head_scores(
    k_blocks: torch.Tensor, valid_lens: torch.Tensor
) -> dict[str, object] | None:
    """Compute per-(KV head) redundancy statistics over a request's blocks.

    Args:
        k_blocks: K vectors for the request's retained blocks at one layer,
            shape ``(B, block_size, H, D)`` (B blocks, H KV heads, D head dim).
        valid_lens: ``(B,)`` int tensor giving the number of valid tokens in
            each block (the final block may be partially filled).

    Returns:
        A dict with numpy arrays of shape ``(H,)`` under keys
        ``mean``/``max``/``p90``/``std`` and an int ``num_blocks``; or ``None``
        if fewer than 2 blocks (redundancy is undefined with a single block).
    """
    B, S, _, _ = k_blocks.shape
    if B < 2:
        return None

    device = k_blocks.device
    k = k_blocks.to(torch.float32)

    # Mean over valid tokens per block -> per-(block, head) anchor.
    pos = torch.arange(S, device=device).view(1, S)
    mask = (pos < valid_lens.view(B, 1)).to(torch.float32)  # (B, S)
    denom = valid_lens.clamp(min=1).to(torch.float32).view(B, 1, 1)
    summed = (k * mask.view(B, S, 1, 1)).sum(dim=1)  # (B, H, D)
    anchors = summed / denom
    anchors = torch.nn.functional.normalize(anchors, dim=-1)  # unit vectors

    # Per-head pairwise cosine (anchors are unit norm => dot == cosine).
    a = anchors.permute(1, 0, 2).contiguous()  # (H, B, D)
    sim = torch.bmm(a, a.transpose(1, 2))  # (H, B, B)
    eye = torch.eye(B, device=device, dtype=torch.bool).view(1, B, B)
    sim = sim.masked_fill(eye, float("-inf"))
    redundancy = sim.max(dim=2).values  # (H, B): max cosine to any other block

    def to_np(t: torch.Tensor):
        return t.detach().to(torch.float32).cpu().numpy()

    return {
        "mean": to_np(redundancy.mean(dim=1)),
        "max": to_np(redundancy.max(dim=1).values),
        "p90": to_np(torch.quantile(redundancy, 0.9, dim=1)),
        "std": to_np(redundancy.std(dim=1)),  # unbiased; B >= 2 guaranteed
        "num_blocks": B,
    }


def compute_prefill_token_scores(
    k_tokens: torch.Tensor, sample_cap: int | None = None
) -> dict[str, object] | None:
    """Compute per-(KV head) redundancy at the TOKEN level (no block pooling).

    Diagnostic for whether block mean-pooling manufactures the near-uniform
    redundancy seen in ``compute_prefill_head_scores``. Here each token's own K
    vector is the unit of comparison:

        redundancy[t] = max over t2 != t of cosine(K[t], K[t2])

    aggregated per head into mean / max / p90 / std over tokens. This is the
    granularity KeyDiff operates at; it is a read-only measurement and is not
    actionable under Option A (vLLM evicts whole blocks, not tokens).

    Args:
        k_tokens: valid-token K vectors for a request at one layer, shape
            ``(N, H, D)`` (N tokens, H KV heads, D head dim).
        sample_cap: if set and ``N`` exceeds it, evenly subsample this many
            tokens to bound the O(N^2) similarity cost.

    Returns:
        A dict with numpy arrays of shape ``(H,)`` under keys
        ``mean``/``max``/``p90``/``std`` and an int ``num_tokens``; or ``None``
        if fewer than 2 tokens.
    """
    N = k_tokens.shape[0]
    if N < 2:
        return None

    device = k_tokens.device
    k = k_tokens.to(torch.float32)
    if sample_cap is not None and sample_cap < N:
        # Evenly spaced indices preserve coverage across the document.
        idx = torch.linspace(0, N - 1, sample_cap, device=device).round().long()
        idx = torch.unique(idx)
        k = k.index_select(0, idx)
        N = k.shape[0]

    k = torch.nn.functional.normalize(k, dim=-1)  # unit vectors per token/head
    a = k.permute(1, 0, 2).contiguous()  # (H, N, D)
    sim = torch.bmm(a, a.transpose(1, 2))  # (H, N, N) cosine (unit norm)
    eye = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
    sim = sim.masked_fill(eye, float("-inf"))
    redundancy = sim.max(dim=2).values  # (H, N): max cosine to any other token

    def to_np(t: torch.Tensor):
        return t.detach().to(torch.float32).cpu().numpy()

    return {
        "mean": to_np(redundancy.mean(dim=1)),
        "max": to_np(redundancy.max(dim=1).values),
        "p90": to_np(torch.quantile(redundancy, 0.9, dim=1)),
        "std": to_np(redundancy.std(dim=1)),  # unbiased; N >= 2 guaranteed
        "num_tokens": N,
    }


def _head_redundancy(x: torch.Tensor) -> torch.Tensor:
    """Per-head max-cosine redundancy for token vectors ``x`` of shape (N,H,D).

    Returns ``(H, N)``: for each head, each token's max cosine to any OTHER
    token in that head (self excluded).
    """
    N = x.shape[0]
    xn = torch.nn.functional.normalize(x, dim=-1)
    a = xn.permute(1, 0, 2).contiguous()  # (H, N, D)
    sim = torch.bmm(a, a.transpose(1, 2))  # (H, N, N)
    eye = torch.eye(N, device=x.device, dtype=torch.bool).view(1, N, N)
    sim = sim.masked_fill(eye, float("-inf"))
    return sim.max(dim=2).values  # (H, N)


def compute_token_diagnostics(
    k_tokens: torch.Tensor,
    v_tokens: torch.Tensor,
    sample_cap: int | None = None,
) -> dict[str, object] | None:
    """Richer token-level diagnostics for the "look at all tensors" experiment.

    Beyond the per-head K cosine redundancy of ``compute_prefill_token_scores``,
    this also measures (a) per-head redundancy on the V (value) tensors, (b)
    per-head norm distributions of the RAW K/V vectors (magnitude is discarded
    by cosine but is a known importance proxy), and (c) joint redundancy/norm in
    the concatenated all-heads space (the per-token footprint that eviction
    actually acts on). Still read-only; no cache mutation.

    Args:
        k_tokens: K vectors for a request at one layer, shape ``(N, H, D)``.
        v_tokens: V vectors for the same tokens, shape ``(N, H, D)``.
        sample_cap: if set and ``N`` exceeds it, evenly subsample this many
            tokens to bound the O(N^2) similarity cost.

    Returns:
        A dict with per-head ``(H,)`` numpy arrays nested under ``k_red``/
        ``v_red`` (mean/max/p90/std) and ``k_norm``/``v_norm`` (mean/std/p90/
        cov), per-layer float scalars (``joint_red_*``, ``joint_norm_*``), and
        an int ``num_tokens``; or ``None`` if fewer than 2 tokens.
    """
    N = k_tokens.shape[0]
    if N < 2:
        return None

    device = k_tokens.device
    k = k_tokens.to(torch.float32)
    v = v_tokens.to(torch.float32)
    if sample_cap is not None and sample_cap < N:
        idx = torch.linspace(0, N - 1, sample_cap, device=device).round().long()
        idx = torch.unique(idx)
        k = k.index_select(0, idx)
        v = v.index_select(0, idx)
        N = k.shape[0]

    H, D = k.shape[1], k.shape[2]

    def to_np(t: torch.Tensor):
        return t.detach().to(torch.float32).cpu().numpy()

    def norm_stats(nrm: torch.Tensor) -> dict[str, object]:
        # nrm: (N, H) per-token per-head magnitudes.
        m = nrm.mean(dim=0)
        s = nrm.std(dim=0)
        return {
            "mean": to_np(m),
            "std": to_np(s),
            "p90": to_np(torch.quantile(nrm, 0.9, dim=0)),
            "cov": to_np(s / m.clamp_min(1e-9)),
        }

    def red_stats(r: torch.Tensor) -> dict[str, object]:
        # r: (H, N) per-head per-token redundancy.
        return {
            "mean": to_np(r.mean(dim=1)),
            "max": to_np(r.max(dim=1).values),
            "p90": to_np(torch.quantile(r, 0.9, dim=1)),
            "std": to_np(r.std(dim=1)),
        }

    # Per-head norms on RAW vectors (magnitude, pre-normalization).
    k_norm = norm_stats(k.norm(dim=-1))
    v_norm = norm_stats(v.norm(dim=-1))

    # Per-head cosine redundancy for K and V.
    k_red = red_stats(_head_redundancy(k))
    v_red = red_stats(_head_redundancy(v))

    def scalar(t: torch.Tensor) -> float:
        return float(t.detach().to(torch.float32).cpu())

    # Joint all-heads space: concat heads -> per-token (H*D) footprint. The
    # concatenated cosine is a norm-weighted blend of the per-head cosines, so
    # this is the redundancy of the object eviction actually drops. Computed for
    # both K and V: whether the V content signal survives head aggregation (vs
    # the value-norm fingerprint sneaking back via norm weighting) is the whole
    # question for eviction, which can only act on the joined footprint.
    eyej = torch.eye(N, device=device, dtype=torch.bool)

    def joint_stats(vecs: torch.Tensor) -> dict[str, float]:
        f = vecs.reshape(N, H * D)
        nrm = f.norm(dim=1)  # (N,)
        fn = torch.nn.functional.normalize(f, dim=-1)
        sim = (fn @ fn.t()).masked_fill(eyej, float("-inf"))  # (N, N)
        red = sim.max(dim=1).values  # (N,)
        nm = nrm.mean()
        return {
            "red_mean": scalar(red.mean()),
            "red_std": scalar(red.std()),
            "red_p90": scalar(torch.quantile(red, 0.9)),
            "norm_mean": scalar(nm),
            "norm_std": scalar(nrm.std()),
            "norm_cov": scalar(nrm.std() / nm.clamp_min(1e-9)),
        }

    jk = joint_stats(k)
    jv = joint_stats(v)
    return {
        "num_tokens": N,
        "k_red": k_red,
        "v_red": v_red,
        "k_norm": k_norm,
        "v_norm": v_norm,
        "joint_red_mean": jk["red_mean"],
        "joint_red_std": jk["red_std"],
        "joint_red_p90": jk["red_p90"],
        "joint_norm_mean": jk["norm_mean"],
        "joint_norm_std": jk["norm_std"],
        "joint_norm_cov": jk["norm_cov"],
        "joint_v_red_mean": jv["red_mean"],
        "joint_v_red_std": jv["red_std"],
        "joint_v_red_p90": jv["red_p90"],
        "joint_v_norm_mean": jv["norm_mean"],
        "joint_v_norm_std": jv["norm_std"],
        "joint_v_norm_cov": jv["norm_cov"],
    }


NORM_VARIANTS = ("raw", "center_request", "whiten_request")


def block_anchors(k_blocks: torch.Tensor, valid_lens: torch.Tensor) -> torch.Tensor:
    """Mean-pool K over each block's valid tokens -> ``(B, H, D)`` anchors.

    Unnormalized (magnitude preserved) so downstream normalization variants can
    center/whiten before the final L2 normalize.
    """
    B, S, _, _ = k_blocks.shape
    device = k_blocks.device
    k = k_blocks.to(torch.float32)
    pos = torch.arange(S, device=device).view(1, S)
    mask = (pos < valid_lens.view(B, 1)).to(torch.float32)  # (B, S)
    denom = valid_lens.clamp(min=1).to(torch.float32).view(B, 1, 1)
    return (k * mask.view(B, S, 1, 1)).sum(dim=1) / denom  # (B, H, D)


def block_prototypes(
    v_blocks: torch.Tensor,
    valid_lens: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Represent each block by one or more valid-token V prototypes.

    Args:
        v_blocks: Value vectors shaped ``(B, S, H, D)``.
        valid_lens: Number of valid tokens in each block, shaped ``(B,)``.
        mode: ``mean``, ``quarters``, or ``mean_top2_norm``.

    Returns:
        A pair ``(prototypes, valid)``. ``prototypes`` has shape
        ``(B, P, H, D)`` and ``valid`` has shape ``(B, P)``. Invalid prototypes
        occur only for empty quarters in a partial final block and are excluded
        from the conservative redundancy reduction.
    """
    B, S, H, D = v_blocks.shape
    device = v_blocks.device
    v = v_blocks.to(torch.float32)
    pos = torch.arange(S, device=device).view(1, S)
    token_valid = pos < valid_lens.view(B, 1)
    denom = valid_lens.clamp(min=1).to(torch.float32).view(B, 1, 1)
    mean = (v * token_valid.view(B, S, 1, 1)).sum(dim=1) / denom

    if mode == "mean":
        return mean[:, None], (valid_lens > 0)[:, None]

    if mode == "quarters":
        # tensor_split covers non-divisible block sizes without dropping tokens.
        groups = torch.tensor_split(torch.arange(S, device=device), 4)
        protos: list[torch.Tensor] = []
        proto_valid: list[torch.Tensor] = []
        for idx in groups:
            group_mask = token_valid[:, idx]
            count = group_mask.sum(dim=1)
            total = (
                v[:, idx] * group_mask.view(B, idx.numel(), 1, 1)
            ).sum(dim=1)
            protos.append(
                total / count.clamp(min=1).to(torch.float32).view(B, 1, 1)
            )
            proto_valid.append(count > 0)
        return torch.stack(protos, dim=1), torch.stack(proto_valid, dim=1)

    if mode == "mean_top2_norm":
        # Score tokens jointly across KV heads, then retain the two strongest
        # valid token vectors in addition to the mean. For a one-token partial
        # block, the second gathered token is marked invalid and ignored.
        token_norm = torch.linalg.vector_norm(v, dim=-1).mean(dim=2)
        token_norm = token_norm.masked_fill(~token_valid, float("-inf"))
        topk = min(2, S)
        top_idx = torch.topk(token_norm, k=topk, dim=1).indices
        gather_idx = top_idx.view(B, topk, 1, 1).expand(B, topk, H, D)
        strongest = v.gather(1, gather_idx)
        ranks = torch.arange(topk, device=device).view(1, topk)
        strongest_valid = ranks < valid_lens.clamp(max=topk).view(B, 1)
        return (
            torch.cat((mean[:, None], strongest), dim=1),
            torch.cat(((valid_lens > 0)[:, None], strongest_valid), dim=1),
        )

    raise ValueError(f"unknown block prototype mode: {mode!r}")


def block_multi_prototype_redundancy(
    prototypes: torch.Tensor,
    prototype_valid: torch.Tensor,
) -> torch.Tensor:
    """Conservative whole-block redundancy from multiple V prototypes.

    Each valid prototype finds its best cosine match in every *other* block.
    The block score is the minimum of those coverage values, so a block is
    highly droppable only when all of its retained within-block features are
    represented elsewhere.

    Args:
        prototypes: ``(B, P, H, D)`` block prototypes.
        prototype_valid: ``(B, P)`` validity mask.

    Returns:
        ``(B,)`` droppability; higher means every prototype is more redundant.
    """
    B, P, _, _ = prototypes.shape
    if B < 2:
        return torch.zeros(B, device=prototypes.device, dtype=torch.float32)
    f = torch.nn.functional.normalize(
        prototypes.reshape(B, P, -1).to(torch.float32), dim=-1
    )
    flat = f.reshape(B * P, -1)
    sim = (flat @ flat.t()).reshape(B, P, B, P)
    same_block = torch.eye(B, device=prototypes.device, dtype=torch.bool)
    sim = sim.masked_fill(same_block[:, None, :, None], float("-inf"))
    sim = sim.masked_fill(~prototype_valid[None, None, :, :], float("-inf"))
    coverage = sim.amax(dim=(2, 3))
    # Invalid source prototypes must not lower the conservative block minimum.
    coverage = coverage.masked_fill(~prototype_valid, float("inf"))
    score = coverage.amin(dim=1)
    return torch.where(torch.isfinite(score), score, torch.zeros_like(score))


def block_joint_redundancy(anchors: torch.Tensor) -> torch.Tensor:
    """Per-block joint (all-heads) max-cosine redundancy.

    Concatenates each block's per-head anchor into one vector (the block's whole
    footprint, i.e. the eviction unit) and returns, per block, its max cosine to
    any OTHER block. Used for the per-block V-redundancy positional control.

    Args:
        anchors: ``(B, H, D)`` block anchors (e.g. from :func:`block_anchors`).

    Returns:
        ``(B,)`` tensor of per-block max cosine to any other block.
    """
    B = anchors.shape[0]
    f = torch.nn.functional.normalize(anchors.reshape(B, -1), dim=-1)
    eye = torch.eye(B, device=anchors.device, dtype=torch.bool)
    sim = (f @ f.t()).masked_fill(eye, float("-inf"))
    return sim.max(dim=1).values


def block_joint_similarity(anchors: torch.Tensor) -> torch.Tensor:
    """Whole-block cosine matrix used by global budget-aware coverage.

    Unlike :func:`block_joint_redundancy`, the diagonal remains the block's
    self-similarity. A kept-set facility-location objective needs that diagonal
    so selecting a block fully covers its own information.
    """
    block_count = anchors.shape[0]
    normalized = torch.nn.functional.normalize(
        anchors.reshape(block_count, -1), dim=-1
    )
    return normalized @ normalized.t()


def block_joint_redundancy_greedy(
    anchors: torch.Tensor,
    protect: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy (iterative) per-block redundancy score.

    :func:`block_joint_redundancy` scores every block against *all* other blocks
    at once, so both halves of a near-duplicate pair score maximally redundant
    and a top-k budget cut drops *both* -- annihilating the information that was
    duplicated (the whole point of keeping one copy). This greedy variant peels
    one block at a time: it drops the currently-most-redundant block, then
    recomputes each survivor's max cosine against the *surviving* set only, so
    once one twin is dropped its partner is no longer redundant and is protected.

    Rather than return a single peel step, it runs the peel to completion over
    all evictable blocks and encodes the **eviction order as a descending score**
    (first-peeled == highest), so the existing ``argsort(descending)`` budget
    selector consumes it unchanged and reaches any budget by taking a prefix.
    The peel order is prefix-consistent: the block chosen at step ``i`` depends
    only on the survivors after steps ``1..i-1``, independent of the final count,
    so one full peel serves every budget.

    Args:
        anchors: ``(B, H, D)`` block anchors (e.g. from :func:`block_anchors`).
        protect: optional ``(B,)`` bool mask of blocks that must never be peeled
            (sink / anchor / tail). Protected blocks stay in the comparison set
            (survivors a duplicate can hide behind) but are never chosen, and
            receive the lowest scores. ``None`` == nothing protected here (the
            downstream selector still enforces sink/anchor keep-rules).

    Returns:
        ``(B,)`` per-block droppability; higher == peeled earlier == more
        droppable. Protected blocks receive the smallest scores (kept last).
    """
    B = anchors.shape[0]
    device = anchors.device
    if B < 2:
        return torch.zeros(B, device=device, dtype=torch.float32)
    f = torch.nn.functional.normalize(anchors.reshape(B, -1).to(torch.float32), dim=-1)
    sim = f @ f.t()  # (B, B) cosine
    sim.fill_diagonal_(float("-inf"))
    if protect is None:
        protect_mask = torch.zeros(B, dtype=torch.bool, device=device)
    else:
        protect_mask = protect.to(device=device, dtype=torch.bool)

    # alive[j] == block j is still a survivor a duplicate can hide behind.
    alive = torch.ones(B, dtype=torch.bool, device=device)
    # Rank of eviction: earlier-peeled blocks get a larger score. Protected /
    # never-peeled blocks keep -inf and are shifted to the bottom afterward.
    scores = torch.full((B,), float("-inf"), device=device, dtype=torch.float32)
    num_evictable = int((~protect_mask).sum())
    sim_work = sim.clone()
    for step in range(num_evictable):
        # Each survivor's max cosine against the currently-alive set.
        red = sim_work.max(dim=1).values  # (B,)
        red = red.masked_fill(~alive, float("-inf"))
        red = red.masked_fill(protect_mask, float("-inf"))
        idx = int(red.argmax())
        if not torch.isfinite(red[idx]):
            break  # nothing left to peel
        # Descending score by peel order: first peeled scores highest.
        scores[idx] = float(num_evictable - step)
        alive[idx] = False
        # Drop it from the comparison set so partners stop counting it.
        sim_work[:, idx] = float("-inf")
    return scores


def block_value_l2(
    v_blocks: torch.Tensor,
    valid_lens: torch.Tensor,
    reduction: str = "sum",
) -> torch.Tensor:
    """Per-block value-L2-norm score (the Paged-Eviction ``value_l2`` signal).

    With ``reduction="sum"``, replicates ``KVCachePruner.get_block_score`` from
    the Paged-Eviction fork
    (``vllm/attention/kvcache_prunner.py``): score a block by the L2 norm of its
    value vectors, averaged over KV heads and summed over the block's tokens ---
    ``norm(V, p=2, dim=-1).mean(heads).sum(tokens)``. The fork ignores padding in
    the final partial block; here we mask to a block's valid tokens, which is
    strictly more correct and only differs on the last block.

    A *high* norm means the block carries strong value signal, so Paged Eviction
    drops the *lowest*-norm blocks. Callers negate this to obtain a droppability
    score (higher == more droppable), matching the v_redundancy convention that
    the budget selector's ``argsort(descending)`` consumes.

    Args:
        v_blocks: V vectors for a request's blocks at one layer, shape
            ``(B, block_size, H, D)``.
        valid_lens: ``(B,)`` int tensor of valid token counts per block.

        reduction: ``sum`` (pinned block-score reproduction), ``max_token``
            (protect a block with any strong token after averaging heads), or
            ``max_token_head`` (protect a block with any strong token/head
            cell). The max variants are whole-block approximations to the
            fork's active token/head-granular pruning path.

    Returns:
        ``(B,)`` per-block value-L2-norm (magnitude; NOT yet negated).
    """
    B, S, _, _ = v_blocks.shape
    device = v_blocks.device
    v = v_blocks.to(torch.float32)
    pos = torch.arange(S, device=device).view(1, S)
    mask = (pos < valid_lens.view(B, 1)).to(torch.float32)  # (B, S)
    per_tok_head = torch.norm(v, p=2, dim=-1)  # (B, S, H)
    per_tok = per_tok_head.mean(dim=2)  # (B, S): mean over heads
    if reduction == "sum":
        return (per_tok * mask).sum(dim=1)
    if reduction == "max_token":
        return per_tok.masked_fill(~mask.bool(), float("-inf")).max(dim=1).values
    if reduction == "max_token_head":
        valid = mask.bool().unsqueeze(-1)
        return per_tok_head.masked_fill(~valid, float("-inf")).amax(dim=(1, 2))
    raise ValueError(f"unknown value-L2 block reduction: {reduction!r}")


def block_query_attention_mass(
    queries: torch.Tensor,
    k_blocks: torch.Tensor,
    valid_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Exact causal QK attention mass assigned to each whole KV block.

    ``queries`` contains the final observed queries, bounded by the configured
    tail size. The function performs grouped-query attention against the
    request's visible K cache, applies the valid-token and causal masks, and
    averages attention mass over query tokens and query heads. It intentionally
    omits V: this is a relevance signal used to protect blocks that the current
    question/decode query reads, not a replacement attention output.

    Args:
        queries: Post-RoPE queries shaped ``(W, Hq, D)``.
        k_blocks: Post-RoPE keys shaped ``(B, S, Hkv, D)``.
        valid_lens: Valid token count per block, shaped ``(B,)``.
        scale: Attention logit scale. Defaults to ``D**-0.5``.

    Returns:
        ``(B,)`` non-negative mass summing to approximately one.
    """
    if queries.ndim != 3 or k_blocks.ndim != 4:
        raise ValueError("queries must be (W,Hq,D) and k_blocks (B,S,Hkv,D)")
    W, Hq, D = queries.shape
    B, S, Hkv, key_dim = k_blocks.shape
    if W < 1 or B < 1 or key_dim != D or Hq % Hkv != 0:
        raise ValueError(
            "query/key shapes require W,B >= 1, equal head dims, and Hq % Hkv == 0"
        )

    q = queries.to(torch.float32).reshape(W, Hkv, Hq // Hkv, D)
    k = k_blocks.to(torch.float32)
    logits = torch.einsum("whgd,bshd->whgbs", q, k)
    logits.mul_(float(scale) if scale is not None else D**-0.5)

    device = k_blocks.device
    token_pos = torch.arange(S, device=device).view(1, S)
    token_valid = token_pos < valid_lens.view(B, 1)
    # The captured queries are the last W valid tokens of this request in the
    # current storage order. The runner reconstructs this tail across scheduler
    # chunks and request-slot reassignment; decode naturally supplies W=1.
    # Keep this scalar on device. ``int(valid_lens.sum())`` would introduce one
    # GPU-to-host synchronization per sampled layer in the production scorer.
    total_valid = valid_lens.sum()
    query_pos = total_valid - W + torch.arange(W, device=device)
    key_pos = (
        torch.arange(B, device=device).view(B, 1) * S + token_pos
    )
    causal_valid = token_valid.unsqueeze(0) & (
        key_pos.unsqueeze(0) <= query_pos.view(W, 1, 1)
    )
    logits = logits.masked_fill(
        ~causal_valid.view(W, 1, 1, B, S), float("-inf")
    )
    probs = torch.softmax(logits.flatten(-2), dim=-1).view(W, Hkv, Hq // Hkv, B, S)
    mass = probs.sum(dim=(0, 1, 2, 4)) / float(W * Hq)
    return mass.to(torch.float32)


def block_key_anchor_relevance(
    queries: torch.Tensor,
    k_blocks: torch.Tensor,
    valid_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Score whole blocks by query dot mean-pooled post-RoPE keys.

    Args:
        queries: Post-RoPE queries shaped ``(W, Hq, D)``.
        k_blocks: Post-RoPE keys shaped ``(B, S, Hkv, D)``.
        valid_lens: Valid token count per block, shaped ``(B,)``.
        scale: Query-key scale. Defaults to ``D**-0.5``.

    Returns:
        ``(B,)`` peak query-to-key-anchor relevance over query tokens and heads.
        The max is the conservative whole-block reduction required by vLLM's
        shared physical block after computing per-head dot products.

    Raises:
        ValueError: If query and key shapes are incompatible with GQA.
    """
    if queries.ndim != 3 or k_blocks.ndim != 4:
        raise ValueError("queries must be (W,Hq,D) and k_blocks (B,S,Hkv,D)")
    W, Hq, D = queries.shape
    B, _, Hkv, key_dim = k_blocks.shape
    if W < 1 or B < 1 or key_dim != D or Hq % Hkv != 0:
        raise ValueError(
            "query/key shapes require W,B >= 1, equal head dims, and Hq % Hkv == 0"
        )
    anchors = block_anchors(k_blocks, valid_lens)
    q = queries.to(torch.float32).reshape(W, Hkv, Hq // Hkv, D)
    relevance = torch.einsum("whgd,bhd->whgb", q, anchors).amax(dim=(0, 1, 2))
    relevance.mul_(float(scale) if scale is not None else D**-0.5)
    return relevance.to(torch.float32)


def refine_with_query_relevance(
    scores: torch.Tensor,
    relevance: torch.Tensor | None,
    weight: float | None,
    tiebreak_only: bool = False,
) -> torch.Tensor:
    """Lower droppability for query-relevant blocks.

    A positive ``weight`` combines per-request z-scored signals, making the
    coefficient independent of request length and native score scale. With no
    weight, ``tiebreak_only`` applies a bounded perturbation that can reorder
    equal redundancy scores but cannot cross the smallest nonzero score gap.
    Disabled inputs return the original tensor object unchanged.
    """
    if relevance is None or (not weight and not tiebreak_only):
        return scores
    if relevance.shape != scores.shape:
        raise ValueError("query relevance and block scores must have equal shape")
    if weight:
        return zscore(scores) - float(weight) * zscore(relevance)

    rel = relevance.to(device=scores.device, dtype=torch.float32)
    span = rel.max() - rel.min()
    if not torch.isfinite(span) or span <= 0:
        return scores
    rel01 = (rel - rel.min()) / span
    x = scores.to(torch.float32)
    uniq = torch.unique(x).sort().values
    if uniq.numel() > 1:
        gaps = uniq[1:] - uniq[:-1]
        min_gap = gaps[gaps > 0].min()
        eps = min_gap / 4.0
    else:
        eps = torch.tensor(
            torch.finfo(torch.float32).eps * max(1.0, float(x.abs().max())),
            device=x.device,
        )
    return x - eps * rel01


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-request z-score standardization of a ``(B,)`` score vector.

    Centers to mean 0 and scales to unit std across the request's blocks, so two
    heterogeneous signals (e.g. cosine redundancy and value-L2 norm) can be
    blended on a common, scale-free footing regardless of their native
    magnitudes. A near-constant vector (std < ``eps``) returns all-zeros.

    Args:
        x: ``(B,)`` per-block scores.
        eps: floor on the std to avoid amplifying a degenerate (constant) signal.

    Returns:
        ``(B,)`` standardized scores (mean ~0, std ~1), or zeros if degenerate.
    """
    x = x.to(torch.float32)
    if x.numel() < 2:
        return torch.zeros_like(x)
    sd = x.std()
    if not torch.isfinite(sd) or sd < eps:
        return torch.zeros_like(x)
    return (x - x.mean()) / sd


def positional_cosh_weights(
    num_blocks: int,
    alpha: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Per-block sech (``1/cosh``) positional weights over ``num_blocks``.

    Maps each block's position to ``x in [-1, 1]`` (first block -> -1, last ->
    +1, middle -> 0) and returns ``w[b] = 1/cosh(alpha * x_b)``. ``sech`` peaks
    at 1.0 in the middle (``x=0``) and decays symmetrically toward
    ``1/cosh(alpha)`` at the ends, so multiplying a droppability score by ``w``
    inflates the middle blocks (more droppable) and shields the ends (sink /
    tail). Larger ``alpha`` == steeper end protection; ``alpha == 0`` returns
    all-ones (inert).

    A single block maps to ``x = 0`` (weight 1.0); callers apply this only when
    there are >= 2 blocks to evict, so the degenerate case never re-weights.

    Args:
        num_blocks: Number of positions to weight (the request's block count).
        alpha: Non-negative steepness of the sech bump. 0 == all-ones.
        device: Device for the returned tensor.
        dtype: Float dtype for the returned tensor.

    Returns:
        ``(num_blocks,)`` tensor of weights in ``(0, 1]``.
    """
    if num_blocks <= 0:
        return torch.ones(0, device=device, dtype=dtype)
    if alpha <= 0.0 or num_blocks == 1:
        return torch.ones(num_blocks, device=device, dtype=dtype)
    # Positions -> [-1, 1] with both endpoints included (linspace, not arange).
    x = torch.linspace(-1.0, 1.0, num_blocks, device=device, dtype=torch.float32)
    w = 1.0 / torch.cosh(alpha * x)
    return w.to(dtype)


def _apply_norm(units: torch.Tensor, mode: str, eps: float = 1e-6) -> torch.Tensor:
    """Return cosine-ready unit vectors for a normalization ``mode``.

    Centering/whitening statistics are computed per (head, dim) over the ``M``
    units of THIS request. After the transform the vectors are L2-normalized;
    units whose residual norm is < ``eps`` are zeroed so a near-zero residual
    does not produce a noisy cosine (it then matches nothing and looks
    non-redundant).

    Args:
        units: ``(M, H, D)`` raw vectors (tokens or block anchors).
        mode: one of :data:`NORM_VARIANTS`.
        eps: floor for residual norm / whitening std.

    Returns:
        ``(M, H, D)`` L2-normalized (or zeroed) vectors ready for cosine.
    """
    if mode == "raw":
        z = units
    elif mode == "center_request":
        z = units - units.mean(dim=0, keepdim=True)
    elif mode == "whiten_request":
        mu = units.mean(dim=0, keepdim=True)
        sd = units.std(dim=0, keepdim=True).clamp_min(eps)
        z = (units - mu) / sd
    else:
        raise ValueError(f"unknown normalization mode: {mode!r}")
    resid = z.norm(dim=-1, keepdim=True)
    zn = torch.nn.functional.normalize(z, dim=-1)
    return torch.where(resid < eps, torch.zeros_like(zn), zn)


def compute_redundancy_variants(
    units: torch.Tensor,
    modes: list[str],
    sample_cap: int | None = None,
) -> dict[str, object] | None:
    """Per-head max-cosine redundancy under several normalizations.

    Tests whether the redundancy fingerprint is just a per-head "mean cone"
    (removed by ``center_request``) or lives in the covariance/anisotropy
    (targeted by ``whiten_request``). Same read-only cosine as elsewhere, only
    the pre-cosine normalization differs.

    Args:
        units: ``(M, H, D)`` raw vectors (per-token K, or block anchors from
            :func:`block_anchors`).
        modes: normalization modes to evaluate (subset of :data:`NORM_VARIANTS`).
        sample_cap: if set and ``M`` exceeds it, evenly subsample this many
            units to bound the O(M^2) cost.

    Returns:
        ``{mode: {mean/max/p90/std: (H,) numpy}}`` plus an int ``num_units``; or
        ``None`` if fewer than 2 units.
    """
    M = units.shape[0]
    if M < 2:
        return None
    x = units.to(torch.float32)
    device = x.device
    if sample_cap is not None and sample_cap < M:
        idx = torch.linspace(0, M - 1, sample_cap, device=device).round().long()
        idx = torch.unique(idx)
        x = x.index_select(0, idx)
        M = x.shape[0]

    eye = torch.eye(M, device=device, dtype=torch.bool).view(1, M, M)

    def to_np(t: torch.Tensor):
        return t.detach().to(torch.float32).cpu().numpy()

    out: dict[str, object] = {"num_units": M}
    for mode in modes:
        z = _apply_norm(x, mode)
        a = z.permute(1, 0, 2).contiguous()  # (H, M, D)
        sim = torch.bmm(a, a.transpose(1, 2))  # (H, M, M)
        sim = sim.masked_fill(eye, float("-inf"))
        r = sim.max(dim=2).values  # (H, M)
        out[mode] = {
            "mean": to_np(r.mean(dim=1)),
            "max": to_np(r.max(dim=1).values),
            "p90": to_np(torch.quantile(r, 0.9, dim=1)),
            "std": to_np(r.std(dim=1)),
        }
    return out


def block_valid_lens(
    prompt_len: int, num_blocks: int, block_size: int, device: torch.device
) -> torch.Tensor:
    """Valid token count per block for a prompt of ``prompt_len`` tokens.

    Full blocks hold ``block_size`` tokens; the final block may be partial.
    """
    idx = torch.arange(num_blocks, device=device)
    return (prompt_len - idx * block_size).clamp(min=0, max=block_size)
