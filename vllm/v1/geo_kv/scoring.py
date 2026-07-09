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
