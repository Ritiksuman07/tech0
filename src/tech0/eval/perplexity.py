"""WikiText-103 perplexity: the general-coherence health check.

A perplexity spike here (relative to the base pretrain) signals over-specialisation
and loss of basic linguistic competence.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _wikitext_texts(split: str, limit: int | None, smoke: bool):
    if smoke:
        return ["the quick brown fox jumps over the lazy dog . " * 40] * (limit or 20)
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    texts = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        text = ex["text"].strip()
        if text:
            texts.append(text)
    return texts


@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    device: str = "cpu",
    split: str = "test",
    seq_len: int = 2048,
    limit: int | None = 1000,
    smoke: bool = False,
) -> float:
    model.eval()
    texts = _wikitext_texts(split, limit, smoke)
    corpus = "\n\n".join(texts)
    tokens = tokenizer.encode(corpus)
    if len(tokens) < 2:
        return float("nan")

    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(tokens) - 1, seq_len):
        chunk = tokens[start : start + seq_len + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        logits = model(x)["logits"].float()
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tokens += y.numel()
    return math.exp(total_nll / max(1, total_tokens))
