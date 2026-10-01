"""Train and use the Tech0 32k byte-level BPE tokenizer.

Special-token ids are fixed and MUST match configs/model_50m.yaml:
    0 <pad>, 1 <s> (bos), 2 </s> (eos), 3 <|user|>, 4 <|assistant|>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<|user|>", "<|assistant|>"]
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
USER_ID, ASSISTANT_ID = 3, 4


def train_tokenizer(
    iterator: Iterable[str],
    vocab_size: int = 32000,
    out_dir: str | Path = "data/tokenizer",
) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    out_path = out_dir / "tokenizer.json"
    tokenizer.save(str(out_path))
    (out_dir / "special_tokens.json").write_text(json.dumps(SPECIAL_TOKENS), encoding="utf-8")
    return str(out_path)


class Tech0Tokenizer:
    def __init__(self, path: str | Path):
        path = Path(path)
        if path.is_dir():
            path = path / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(path))
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [BOS_ID, *ids]
        if add_eos:
            ids = [*ids, EOS_ID]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def pad_token_id(self) -> int:
        return PAD_ID

    @property
    def eos_token_id(self) -> int:
        return EOS_ID

    @property
    def bos_token_id(self) -> int:
        return BOS_ID


def _load_texts(paths: list[str]) -> Iterator[str]:
    for p in paths:
        p = Path(p)
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Train the Tech0 32k BPE tokenizer")
    parser.add_argument("--input", nargs="+", required=True, help="Text files to train on")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--out-dir", default="data/tokenizer")
    args = parser.parse_args()
    out = train_tokenizer(_load_texts(args.input), args.vocab_size, args.out_dir)
    print(f"Saved tokenizer to {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
