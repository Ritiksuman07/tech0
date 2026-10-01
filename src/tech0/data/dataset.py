"""Datasets for pretraining (flat token bins) and SFT (masked chat)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class BinDataset:
    """Random fixed-length blocks from a flat uint16 token file (memmap)."""

    def __init__(self, path: str | Path, seq_len: int, dtype: np.dtype = np.uint16):
        self.path = Path(path)
        self.data = np.memmap(self.path, dtype=dtype, mode="r")
        self.seq_len = seq_len
        if len(self.data) < seq_len + 1:
            raise ValueError(f"{self.path} has {len(self.data)} tokens, need > {seq_len}")

    def __len__(self) -> int:
        return max(0, (len(self.data) - 1) // self.seq_len)

    def get_batch(self, batch_size: int, generator: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.seq_len
        idx = generator.integers(0, len(self), size=batch_size)
        x = np.stack([self.data[i * s : i * s + s] for i in idx]).astype(np.int64)
        y = np.stack([self.data[i * s + 1 : i * s + 1 + s] for i in idx]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.seq_len
        x = torch.from_numpy(self.data[i * s : i * s + s].astype(np.int64))
        y = torch.from_numpy(self.data[i * s + 1 : i * s + 1 + s].astype(np.int64))
        return x, y


def safe_bin_dataset(path: str | Path, seq_len: int, dtype: np.dtype = np.uint16) -> "BinDataset | None":
    """Return a BinDataset, or None if the file is missing / too small."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return BinDataset(path, seq_len, dtype)
    except ValueError:
        return None


class PackedTextDataset(Dataset):
    """Tokenize a list of texts, concatenate, and serve fixed-length blocks."""

    def __init__(self, texts: list[str], tokenizer, seq_len: int, eos_id: int):
        tokens: list[int] = []
        for text in texts:
            tokens.extend(tokenizer.encode(text))
            tokens.append(eos_id)
        self.tokens = np.asarray(tokens, dtype=np.int64)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.seq_len
        x = torch.from_numpy(self.tokens[i * s : i * s + s])
        y = torch.from_numpy(self.tokens[i * s + 1 : i * s + 1 + s])
        return x, y


def load_instruction_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _to_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in record:
        return record["messages"]
    prompt = record.get("instruction", "")
    if record.get("input"):
        prompt = f"{prompt}\n{record['input']}"
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": record.get("output", record.get("response", ""))},
    ]


def build_chat_example(
    tokenizer,
    messages: list[dict[str, str]],
    seq_len: int,
) -> tuple[list[int], list[int]]:
    """ChatML-ish encoding. Prompt tokens are masked out of the loss (label=-100).

    Layout: <|user|> prompt </s> <|assistant|> response </s>
    """
    eos = tokenizer.eos_token_id
    user_id = 3
    assistant_id = 4
    input_ids: list[int] = []
    labels: list[int] = []

    for message in messages:
        content = tokenizer.encode(message["content"])
        if message["role"] == "user":
            segment = [user_id, *content, eos]
            input_ids.extend(segment)
            labels.extend([IGNORE_INDEX] * len(segment))
        else:
            header = [assistant_id]
            input_ids.extend(header)
            labels.extend([IGNORE_INDEX] * len(header))
            input_ids.extend(content)
            labels.extend(content)
            input_ids.append(eos)
            labels.append(eos)

    input_ids = input_ids[:seq_len]
    labels = labels[:seq_len]
    return input_ids, labels


class InstructionDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, seq_len: int):
        self.records = load_instruction_records(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        messages = _to_messages(self.records[i])
        input_ids, labels = build_chat_example(self.tokenizer, messages, self.seq_len)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_instructions(batch, pad_token_id: int = 0):
    max_len = max(len(x) for x, _ in batch)
    x_padded, y_padded, attn = [], [], []
    for x, y in batch:
        pad = max_len - len(x)
        x_padded.append(torch.cat([x, torch.full((pad,), pad_token_id, dtype=torch.long)]))
        y_padded.append(torch.cat([y, torch.full((pad,), IGNORE_INDEX, dtype=torch.long)]))
        attn.append(torch.cat([torch.ones(len(x), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return torch.stack(x_padded), torch.stack(y_padded), torch.stack(attn)
