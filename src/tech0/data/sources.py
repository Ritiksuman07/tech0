"""Curated data sources + a synthetic fallback for smoke tests.

Phase 1 foundation mix (from the design doc):
  - FineWeb-Edu         (textbook-grade web text)
  - Cosmopedia v2       (synthetic textbook-style)
  - Wikipedia           (encyclopedic)
  - code corpus         (for the code specialisation benchmarks)
"""

from __future__ import annotations

import random
from typing import Iterator

# Registry: name -> load_dataset kwargs. `text_key` may also try fallbacks.
SOURCES: dict[str, dict] = {
    "fineweb_edu": {
        "name": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "text_key": "text",
    },
    "cosmopedia": {
        "name": "HuggingFaceTB/cosmopedia",
        "config": "web_samples_v2",
        "split": "train",
        "text_key": "text",
    },
    "wikipedia": {
        "name": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "text_key": "text",
    },
    "code": {
        "name": "codeparrot/codeparrot-clean",
        "config": None,
        "split": "train",
        "text_key": "code",
    },
}

# Source -> curriculum role. `target` is domain-specific, `general` is broad competence.
SOURCE_ROLE = {
    "fineweb_edu": "general",
    "cosmopedia": "general",
    "wikipedia": "general",
    "code": "target",
}

_TEXT_FALLBACK_KEYS = ("text", "content", "code", "raw_content")


def hf_stream(
    source: str,
    split: str | None = None,
    max_docs: int | None = None,
) -> Iterator[str]:
    """Stream documents from a HuggingFace dataset (requires `datasets`)."""
    from datasets import load_dataset

    spec = SOURCES[source]
    ds = load_dataset(spec["name"], spec["config"], split=split or spec["split"], streaming=True)
    text_key = spec["text_key"]
    for i, example in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        text = example.get(text_key)
        if not text:
            for key in _TEXT_FALLBACK_KEYS:
                if example.get(key):
                    text = example[key]
                    break
        if text and isinstance(text, str) and text.strip():
            yield text


def mixed_stream(
    sources: list[str],
    max_docs_per_source: int | None = None,
    split: str | None = None,
) -> Iterator[str]:
    """Interleave multiple HF sources (round-robin at the doc level).

    A source that fails to load (gated, renamed, offline) is skipped with a
    warning instead of aborting the whole run.
    """
    def safe(source: str) -> Iterator[str]:
        try:
            yield from hf_stream(source, split=split, max_docs=max_docs_per_source)
        except Exception as exc:  # noqa: BLE001
            print(f"[sources] skipping '{source}': {exc}")

    generators = [safe(s) for s in sources]
    active = list(generators)
    while active:
        nxt = []
        for gen in active:
            try:
                yield next(gen)
                nxt.append(gen)
            except StopIteration:
                continue
        active = nxt


_WORDS = (
    "the quick brown fox jumps over lazy dog model language tokens attention neural "
    "network training data gradient descent transformer embedding vector context window "
    "reasoning code function returns value list dictionary loop index matrix tensor "
    "probability distribution entropy cross validation benchmark accuracy loss optimizer "
    "science history geography mathematics physics biology chemistry algorithm complexity "
    "the a of and to in is it that for with as on by this are be from or an at"
).split()


def synthetic_corpus(n_docs: int = 2000, seed: int = 0, min_words: int = 20, max_words: int = 120) -> Iterator[str]:
    """Deterministic pseudo-text used for smoke tests and offline development."""
    rng = random.Random(seed)
    for _ in range(n_docs):
        n = rng.randint(min_words, max_words)
        yield " ".join(rng.choice(_WORDS) for _ in range(n))


def role_of(source: str) -> str:
    return SOURCE_ROLE.get(source, "general")
