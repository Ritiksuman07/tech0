"""Tokenise text streams and pack them into flat uint16 binary shards."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


def tokenize_to_bin(
    texts: Iterable[str],
    encode_fn,
    out_path: str | Path,
    dtype: np.dtype = np.uint16,
    chunk: int = 2_000_000,
    max_tokens: int | None = None,
    add_bos: bool = False,
) -> int:
    """Stream `texts`, tokenize with `encode_fn(text) -> list[int]`, append EOS
    after each document, and write a flat binary token file. Returns token count."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = np.empty(chunk, dtype=dtype)
    filled = 0
    total = 0

    def flush(n: int) -> None:
        if n:
            fh.write(buffer[:n].tobytes())

    with out_path.open("wb") as fh:
        for text in texts:
            ids = encode_fn(text)
            ids = [*ids, 2] if not add_bos else [1, *ids, 2]  # EOS / BOS ids
            for token in ids:
                buffer[filled] = token
                filled += 1
                total += 1
                if filled == chunk:
                    flush(filled)
                    filled = 0
            if max_tokens is not None and total >= max_tokens:
                break
        flush(filled)
    return total


def split_tokens(token_file: str | Path, val_ratio: float = 0.005, dtype: np.dtype = np.uint16):
    """Split a flat token file into train/val files by copying byte ranges."""
    src = Path(token_file)
    data = np.memmap(src, dtype=dtype, mode="r")
    n_val = max(1, int(len(data) * val_ratio))
    n_train = len(data) - n_val
    train_path = src.with_name("train.bin")
    val_path = src.with_name("val.bin")

    for path, chunk in ((train_path, data[:n_train]), (val_path, data[n_train:])):
        with path.open("wb") as fh:
            step = 5_000_000
            for start in range(0, len(chunk), step):
                fh.write(np.asarray(chunk[start : start + step]).tobytes())
    return str(train_path), str(val_path), int(n_train), int(n_val)
