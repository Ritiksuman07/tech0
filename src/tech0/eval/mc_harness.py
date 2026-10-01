"""Multiple-choice evaluation via length-normalized log-likelihood.

Covers: HellaSwag, ARC-Easy, PIQA, Winogrande.

Each task is normalised to examples of the form:
    {"context": str, "choices": [str, ...], "gold": int}
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn.functional as F


@torch.no_grad()
def continuation_logprob(model, tokenizer, context: str, continuation: str, device: str):
    """Mean log-probability of `continuation` given `context` (token-normalized)."""
    ctx_ids = tokenizer.encode(context) if context else [tokenizer.bos_token_id]
    cont_ids = tokenizer.encode(continuation)
    if not cont_ids:
        return float("-inf"), 1
    ids = torch.tensor([ctx_ids + cont_ids], dtype=torch.long, device=device)
    logits = model(ids)["logits"][0].float()
    logprobs = F.log_softmax(logits, dim=-1)
    start = len(ctx_ids)
    total = 0.0
    for i, token in enumerate(cont_ids):
        total += logprobs[start + i - 1, token].item()
    return total / len(cont_ids), len(cont_ids)


def evaluate_mc(model, tokenizer, examples: list[dict], device: str = "cpu", limit: int | None = None) -> float:
    model.eval()
    correct = 0
    total = 0
    for ex in examples[: limit or len(examples)]:
        scores = []
        for choice in ex["choices"]:
            score, _ = continuation_logprob(model, tokenizer, ex["context"], choice, device)
            scores.append(score)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct += int(pred == ex["gold"])
        total += 1
    return correct / max(1, total)


# ---- task loaders ------------------------------------------------------

def _iter_hellaswag(limit: int | None, split: str) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("hellaswag", split=split, streaming=True)
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        yield {"context": ex["ctx"], "choices": [" " + e for e in ex["endings"]], "gold": int(ex["label"])}


def _iter_arc_easy(limit: int | None, split: str) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("ai2_arc", "ARC-Easy", split=split, streaming=True)
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        labels = ex["choices"]["label"]
        gold = labels.index(ex["answerKey"]) if ex["answerKey"] in labels else 0
        yield {"context": ex["question"], "choices": ex["choices"]["text"], "gold": gold}


def _iter_piqa(limit: int | None, split: str) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("piqa", split=split, streaming=True)
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        yield {"context": ex["goal"], "choices": [ex["sol1"], ex["sol2"]], "gold": int(ex["label"])}


def _iter_winogrande(limit: int | None, split: str) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("winogrande", "winogrande_xl", split=split, streaming=True)
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        sentence = ex["sentence"]
        choices = [sentence.replace("_", ex["option1"]), sentence.replace("_", ex["option2"])]
        yield {"context": "", "choices": choices, "gold": int(ex["answer"]) - 1}


_LOADERS = {
    "hellaswag": _iter_hellaswag,
    "arc_easy": _iter_arc_easy,
    "piqa": _iter_piqa,
    "winogrande": _iter_winogrande,
}


def synthetic_mc(n: int = 20) -> list[dict]:
    """Offline examples for smoke tests: correct choice is the one seen in context."""
    examples = []
    for i in range(n):
        token = f"answer{i}"
        examples.append({"context": f"the correct value is {token}", "choices": [token, f"wrong{i}"], "gold": 0})
    return examples


def load_mc_task(name: str, split: str = "validation", limit: int | None = None, smoke: bool = False) -> list[dict]:
    if smoke:
        return synthetic_mc(limit or 20)
    if name not in _LOADERS:
        raise KeyError(f"Unknown MC task '{name}'. Known: {list(_LOADERS)}")
    return list(_LOADERS[name](limit, split))
