#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-1 accuracy screen for value-space KV-cache eviction (HF reference).

This is a *measurement* harness, not a vLLM code path. It answers one question:
does dropping the highest V-redundancy KV *pages* preserve output quality,
relative to recency and random eviction at a matched eviction rate?

Why HuggingFace and not vLLM: vLLM's default FlashAttention kernel cannot mask an
arbitrary interior set of KV blocks without a kernel change (it only truncates the
tail via ``seq_lens``); the sole kernel-free in-engine path is the non-default
FlexAttention backend. To get a faithful go/no-go number cheaply we instead use
transformers, where dropping arbitrary KV positions is a 2D ``attention_mask``
plus explicit absolute ``position_ids`` (RoPE for survivors is preserved because
we never compact the cache). No vLLM engine code is touched; default behavior is
unchanged by construction.

Method, per LongBench prompt:
  1. Prefill once (full, unmasked) -> a ``DynamicCache`` + next-token logits.
  2. Score per-(layer, block) joint-V redundancy with the *same* metric used in
     the read-only experiments (``vllm/v1/geo_kv/scoring.py``), aggregated over a
     layer band into one droppability score per block (Option A: one whole-block
     decision applied across all layers).
  3. For each (policy, layer band, eviction rate), pick the evicted block set and
     decode the answer with a manual greedy loop that masks those blocks'
     positions. Score with LongBench ``qa_f1`` (vendored, pure-Python).

Run inside a PBS GPU job (torch is broken on the login node): see
``benchmarks/geo_eviction_accuracy.pbs``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same redundancy metric as the read-only experiments (single source of truth).
from vllm.v1.geo_kv.scoring import (
    block_anchors,
    block_joint_redundancy,
    block_valid_lens,
)

# Canonical LongBench prompt templates + generation budgets (THUDM/LongBench).
# Only the qa_f1 tasks are included; add rouge tasks + a rouge scorer later.
DATASET2PROMPT = {
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, "
        "answer the following question based on the above text, only give me "
        "the answer and do not output any other words.\n\nQuestion: {input}\n"
        "Answer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other words."
        "\n\nQuestion: {input}\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible."
        ' If the question cannot be answered based on the information in the '
        'article, write "unanswerable". If the question is a yes/no question, '
        'answer "yes", "no", or "unanswerable". Do not provide any explanation.'
        "\n\nArticle: {context}\n\n Answer the question based on the above "
        "article as concisely as you can, using a single phrase or sentence if "
        'possible. If the question cannot be answered based on the information '
        'in the article, write "unanswerable". If the question is a yes/no '
        'question, answer "yes", "no", or "unanswerable". Do not provide any '
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    # Multi-hop QA over passages -- same template as hotpotqa (LongBench canon).
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other words."
        "\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other words."
        "\n\nQuestion: {input}\nAnswer:"
    ),
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question as concisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\nStory: "
        "{context}\n\nNow, answer the question based on the story as concisely "
        "as you can, using a single phrase if possible. Do not provide any "
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    # Synthetic exact-retrieval needle probe: answer is "Paragraph N".
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please "
        "determine which paragraph the abstract is from.\n\n{context}\n\nThe "
        "following is an abstract.\n\n{input}\n\nPlease enter the number of the "
        "paragraph that the abstract is from. The answer format must be like "
        '"Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: '
    ),
    # Context-only summarization (no question): {input} is unused. Scored by
    # ROUGE-L, not qa_f1. Canonical THUDM/LongBench templates.
    "gov_report": (
        "You are given a report by a government agency. Write a one-page "
        "summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page "
        "summary of the report.\n\nSummary:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all "
        "news.\n\nNews:\n{context}\n\nNow, write a one-page summary of all the "
        "news.\n\nSummary:"
    ),
}
DATASET2MAXGEN = {
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "qasper": 128,
    "2wikimqa": 32,
    "musique": 32,
    "narrativeqa": 128,
    "passage_retrieval_en": 32,
    "gov_report": 512,
    "multi_news": 512,
}

# Which scorer each task uses. QA tasks -> token-F1; summarization -> ROUGE-L.
# Tasks absent here default to qa_f1 (keeps older callers working).
DATASET2METRIC = {
    "multifieldqa_en": "qa_f1",
    "hotpotqa": "qa_f1",
    "qasper": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "narrativeqa": "qa_f1",
    "passage_retrieval_en": "qa_f1",
    "gov_report": "rouge_l",
    "multi_news": "rouge_l",
}

# Summarization tasks have an empty {input}; their templates key only on
# {context}. Listed separately so table tests don't require an {input} slot.
SUMMARIZATION_TASKS = ("gov_report", "multi_news")

# Layer bands to score V-redundancy over. "mid" is the discriminative band found
# in the read-only work; "l0" is an ablation expected to underperform (high mean
# redundancy but low page-to-page spread); "all" is the flat-average contrast.
BANDS = {"mid": list(range(6, 18)), "l0": [0], "all": list(range(0, 32))}
POLICIES = ("v_redundancy", "recency", "random")


# ---------------------------------------------------------------------------
# LongBench qa_f1 metric (vendored, pure-Python; avoids lm_eval's eager
# rouge/jieba/fuzzywuzzy imports).
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD-style)."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def qa_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between a prediction and one gold answer."""
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


def qa_f1_max(prediction: str, answers: list[str]) -> float:
    """Max token-F1 over the gold answer list (LongBench convention)."""
    return max((qa_f1(prediction, a) for a in answers), default=0.0)


# ROUGE-L scorer is constructed lazily and cached: the stemmer import is heavy
# and QA-only runs must not pay for it.
_ROUGE_SCORER = None


def _get_rouge_scorer():
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer

        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _ROUGE_SCORER


def rouge_l_max(prediction: str, answers: list[str]) -> float:
    """Max ROUGE-L F-measure over the gold answer list (LongBench summ. tasks).

    Mirrors THUDM/LongBench: summarization tasks (gov_report, multi_news) score
    generated summaries by ROUGE-L rather than token-F1. Empty predictions score
    0.0 (the scorer handles this without raising).
    """
    if not prediction.strip():
        return 0.0
    scorer = _get_rouge_scorer()
    best = 0.0
    for a in answers:
        if not a:
            continue
        best = max(best, scorer.score(a, prediction)["rougeL"].fmeasure)
    return best


def score_prediction(task: str, prediction: str, answers: list[str]) -> float:
    """Dispatch to the task's canonical metric (qa_f1 or rouge_l).

    Args:
        task: LongBench task name.
        prediction: Model-generated text.
        answers: Gold answer list.

    Returns:
        The task's metric in [0, 1]; qa_f1 for tasks not in DATASET2METRIC.
    """
    if DATASET2METRIC.get(task, "qa_f1") == "rouge_l":
        return rouge_l_max(prediction, answers)
    return qa_f1_max(prediction, answers)


# ---------------------------------------------------------------------------
# Data + prompt building
# ---------------------------------------------------------------------------
def load_task(data_dir: str, task: str, num_prompts: int) -> list[dict]:
    path = os.path.join(data_dir, f"{task}.jsonl")
    recs: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
            if len(recs) >= num_prompts:
                break
    return recs


def build_input_ids(
    tokenizer,
    task: str,
    rec: dict,
    max_prompt_len: int,
    reserve: int,
) -> torch.Tensor:
    """Middle-truncate the context, format the task template, chat-template it.

    LongBench truncates the middle of the context (answer-relevant text sits at
    both ends). We truncate context tokens to a budget, then apply the model's
    chat template so the Instruct model is prompted correctly.
    """
    context = rec["context"]
    budget = max(256, max_prompt_len - reserve)
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    if len(ctx_ids) > budget:
        half = budget // 2
        ctx_ids = ctx_ids[:half] + ctx_ids[-(budget - half):]
        context = tokenizer.decode(ctx_ids)
    prompt_str = DATASET2PROMPT[task].format(context=context, input=rec["input"])
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_str}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return enc["input_ids"]  # (1, prompt_len)


# ---------------------------------------------------------------------------
# V-redundancy block scoring (reuses scoring.py on HF's DynamicCache V tensors)
# ---------------------------------------------------------------------------
@torch.no_grad()
def per_layer_block_redundancy(
    cache, prompt_len: int, block_size: int
) -> torch.Tensor:
    """Return (num_layers, num_blocks) per-block joint-V redundancy.

    Reads each layer's V ``[1, H, S, D]`` from the DynamicCache, reshapes to the
    ``(num_blocks, block_size, H, D)`` layout ``block_anchors`` expects, and
    computes per-block max-cosine redundancy in the concatenated all-heads space.
    """
    layers = cache.layers
    device = layers[0].values.device
    num_blocks = (prompt_len + block_size - 1) // block_size
    valid = block_valid_lens(prompt_len, num_blocks, block_size, device)
    pad = num_blocks * block_size - prompt_len
    out = torch.empty(len(layers), num_blocks, dtype=torch.float32, device=device)
    for i, layer in enumerate(layers):
        v = layer.values[0].to(torch.float32)  # (H, S, D)
        v = v[:, :prompt_len, :].permute(1, 0, 2).contiguous()  # (S, H, D)
        if pad:
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))  # pad tokens
        h, d = v.shape[1], v.shape[2]
        vb = v.reshape(num_blocks, block_size, h, d)
        out[i] = block_joint_redundancy(block_anchors(vb, valid))
    return out


def band_scores(per_layer: torch.Tensor, band: str) -> torch.Tensor:
    """Aggregate per-(layer,block) redundancy over a band -> (num_blocks,)."""
    idx = [layer for layer in BANDS[band] if layer < per_layer.shape[0]]
    return per_layer[idx].mean(dim=0)


# ---------------------------------------------------------------------------
# Eviction policy -> set of evicted block indices
# ---------------------------------------------------------------------------
def select_evicted(
    policy: str,
    scores: torch.Tensor | None,
    num_blocks: int,
    rate: float,
    seed: int,
) -> set[int]:
    """Choose which blocks to evict. The final block (decode anchor) is kept.

    All policies evict the same COUNT for a matched-rate comparison.
    """
    evictable = list(range(num_blocks - 1))  # keep the last (most recent) block
    k = int(round(rate * len(evictable)))
    if k <= 0:
        return set()
    if policy == "recency":  # drop the OLDEST k (StreamingLLM keeps recent)
        return set(evictable[:k])
    if policy == "random":
        rng = random.Random(seed)
        return set(rng.sample(evictable, k))
    if policy == "v_redundancy":
        assert scores is not None
        order = sorted(evictable, key=lambda b: float(scores[b]), reverse=True)
        return set(order[:k])
    raise ValueError(f"unknown policy: {policy!r}")


def prompt_mask(
    prompt_len: int, evicted: set[int], block_size: int, device
) -> torch.Tensor:
    """1D mask over prompt positions: 1 = keep, 0 = evicted block's positions."""
    mask = torch.ones(prompt_len, dtype=torch.long, device=device)
    for b in evicted:
        lo = b * block_size
        hi = min((b + 1) * block_size, prompt_len)
        mask[lo:hi] = 0
    return mask


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
@torch.no_grad()
def masked_decode(
    model,
    cache,
    prompt_len: int,
    first_logits: torch.Tensor,
    pmask: torch.Tensor,
    max_gen: int,
    eos_ids: set[int],
) -> list[int]:
    """Greedy decode with evicted prompt positions masked out.

    Explicit absolute ``position_ids`` keep RoPE aligned with the cached (never
    compacted) K. ``cache`` is mutated (grows); caller resets it between runs.
    """
    device = pmask.device
    logits = first_logits  # (1, vocab): logits after the prompt
    generated: list[int] = []
    for step in range(max_gen):
        tok = int(torch.argmax(logits[0]))
        generated.append(tok)
        if tok in eos_ids:
            break
        cur = torch.cat(
            [pmask, torch.ones(step + 1, dtype=torch.long, device=device)]
        ).unsqueeze(0)  # (1, prompt_len + step + 1)
        pos = torch.tensor([[prompt_len + step]], device=device)
        out = model(
            input_ids=torch.tensor([[tok]], device=device),
            past_key_values=cache,
            attention_mask=cur,
            position_ids=pos,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]
    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--longbench-dir", default="data/longbench/data")
    p.add_argument("--tasks", default="multifieldqa_en,hotpotqa,qasper")
    p.add_argument("--num-prompts", type=int, default=50)
    p.add_argument("--max-prompt-len", type=int, default=4096)
    p.add_argument("--reserve", type=int, default=512)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--rates", default="0.0,0.2,0.4,0.6")
    p.add_argument("--bands", default="mid,l0,all")
    p.add_argument("--policies", default="v_redundancy,recency,random")
    p.add_argument("--max-gen", type=int, default=None, help="Override per-task.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-self-check", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [t for t in args.tasks.split(",") if t]
    rates = [float(x) for x in args.rates.split(",") if x]
    bands = [b for b in args.bands.split(",") if b]
    policies = [p for p in args.policies.split(",") if p]
    run_id = args.run_id or f"evict_acc_{int(time.time())}"
    out_dir = args.output_dir or os.path.join(
        "results", "geo_prefill_distribution", run_id
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"[evict-acc] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    device = model.device
    eos_ids = {tokenizer.eos_token_id}
    for extra in ("<|eot_id|>", "<|end_of_text|>"):
        tid = tokenizer.convert_tokens_to_ids(extra)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.add(tid)
    eos_ids.discard(None)

    rows: list[dict] = []
    self_checked = args.no_self_check

    for task in tasks:
        max_gen = args.max_gen or DATASET2MAXGEN.get(task, 64)
        recs = load_task(args.longbench_dir, task, args.num_prompts)
        print(f"[evict-acc] task={task} prompts={len(recs)} max_gen={max_gen}")
        for pi, rec in enumerate(recs):
            ids = build_input_ids(
                tokenizer, task, rec, args.max_prompt_len, args.reserve
            ).to(device)
            prompt_len = ids.shape[1]
            nb = (prompt_len + args.block_size - 1) // args.block_size
            if nb < 4:
                continue  # too short to evict meaningfully

            with torch.no_grad():
                pref = model(input_ids=ids, use_cache=True)
            cache = pref.past_key_values
            first_logits = pref.logits[:, -1, :].clone()
            per_layer = per_layer_block_redundancy(
                cache, prompt_len, args.block_size
            )
            scores_by_band = {b: band_scores(per_layer, b) for b in bands}

            # One-time correctness self-check: masked decode with an all-keep
            # mask must match HF greedy generate for the first tokens.
            if not self_checked:
                _self_check(
                    model, tokenizer, ids, cache, first_logits, prompt_len,
                    device, eos_ids,
                )
                self_checked = True

            # Enumerate runs. Band only matters for v_redundancy; recency/random
            # are band-independent, so score them once (band label "-").
            combos: list[tuple[str, str, float]] = []
            for rate in rates:
                if rate == 0.0:
                    combos.append(("none", "-", 0.0))
                    continue
                for policy in policies:
                    if policy == "v_redundancy":
                        for b in bands:
                            combos.append((policy, b, rate))
                    else:
                        combos.append((policy, "-", rate))

            for policy, band, rate in combos:
                scores = scores_by_band.get(band) if policy == "v_redundancy" else None
                evicted = select_evicted(
                    policy, scores, nb, rate, seed=args.seed + pi
                )
                pmask = prompt_mask(prompt_len, evicted, args.block_size, device)
                gen = masked_decode(
                    model, cache, prompt_len, first_logits, pmask, max_gen, eos_ids
                )
                _reset(cache, prompt_len)
                text = tokenizer.decode(gen, skip_special_tokens=True)
                score = qa_f1_max(text, rec["answers"])
                rows.append({
                    "run_id": run_id,
                    "prompt_id": rec.get("_id", f"{task}_{pi}"),
                    "task": task,
                    "policy": policy,
                    "layer_band": band,
                    "eviction_rate": rate,
                    "num_blocks": nb,
                    "num_evicted": len(evicted),
                    "metric_name": "qa_f1",
                    "score": round(score, 4),
                })

    _write_outputs(out_dir, run_id, args, rows)


def _reset(cache, prompt_len) -> None:
    """Reset cache to prompt-only between decode runs (drop appended tokens)."""
    if hasattr(cache, "crop"):
        cache.crop(prompt_len)
    else:  # fallback: slice appended decode tokens off each layer's K/V.
        for layer in cache.layers:
            layer.keys = layer.keys[:, :, :prompt_len, :].contiguous()
            layer.values = layer.values[:, :, :prompt_len, :].contiguous()


@torch.no_grad()
def _self_check(
    model, tokenizer, ids, cache, first_logits, prompt_len, device, eos_ids
) -> None:
    """Masked decode with a full-keep mask must match HF greedy generate."""
    keep = torch.ones(prompt_len, dtype=torch.long, device=device)
    manual = masked_decode(
        model, cache, prompt_len, first_logits, keep, 8, eos_ids
    )
    _reset(cache, prompt_len)
    ref = model.generate(
        ids, max_new_tokens=8, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )[0, prompt_len:].tolist()
    n = min(len(manual), len(ref))
    if manual[:n] != ref[:n]:
        raise RuntimeError(
            "self-check FAILED: manual masked decode != HF generate\n"
            f"  manual={manual[:n]}\n  ref   ={ref[:n]}\n"
            "Mask/position bookkeeping is wrong; aborting."
        )
    print(f"[evict-acc] self-check OK (first {n} tokens match)")


def _write_outputs(out_dir, run_id, args, rows) -> None:
    import csv

    csv_path = os.path.join(out_dir, "accuracy_scores.csv")
    fields = [
        "run_id", "prompt_id", "task", "policy", "layer_band",
        "eviction_rate", "num_blocks", "num_evicted", "metric_name", "score",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Aggregate: mean score per (task, policy, band, rate).
    agg: dict[tuple, list[float]] = {}
    for r in rows:
        key = (r["task"], r["policy"], r["layer_band"], r["eviction_rate"])
        agg.setdefault(key, []).append(r["score"])
    summary = {
        "run_id": run_id,
        "model": os.path.basename(args.model.rstrip("/")),
        "tasks": args.tasks,
        "num_prompts": args.num_prompts,
        "rates": args.rates,
        "bands": args.bands,
        "policies": args.policies,
        "note": (
            "Stage-1 HF reference accuracy screen; not a vLLM code path. "
            "Eviction = whole-block (all layers), scored by joint-V redundancy."
        ),
        "means": {
            "|".join(str(x) for x in k): round(sum(v) / len(v), 4)
            for k, v in sorted(agg.items())
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[evict-acc] wrote {csv_path} ({len(rows)} rows)")
    print("\n=== mean qa_f1 by task / policy / band / rate ===")
    print(f"{'task':16} {'policy':13} {'band':4} {'rate':>5} {'qa_f1':>7} {'n':>4}")
    for k, v in sorted(agg.items()):
        t, pol, b, rate = k
        print(
            f"{t:16} {pol:13} {b:4} {rate:>5} "
            f"{sum(v) / len(v):>7.3f} {len(v):>4}"
        )


if __name__ == "__main__":
    main()
