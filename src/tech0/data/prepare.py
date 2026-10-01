"""Prepare tokenizer + packed token bins.

Real run:
    python -m tech0.data.prepare --config configs/pretrain.yaml \
        --sources fineweb_edu cosmopedia wikipedia code --max-docs 200000
Smoke run (offline, synthetic):
    python -m tech0.data.prepare --config configs/pretrain.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

from tech0.data.packing import split_tokens, tokenize_to_bin
from tech0.data.sources import mixed_stream, synthetic_corpus
from tech0.tokenizer.build_bpe import Tech0Tokenizer, train_tokenizer
from tech0.utils.config import apply_dotted_overrides, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Tech0 data")
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--sources", nargs="+", default=["fineweb_edu", "cosmopedia", "wikipedia", "code"])
    parser.add_argument("--max-docs", type=int, default=None, help="Per-source document cap")
    parser.add_argument("--max-tokens", type=int, default=None, help="Total token cap")
    parser.add_argument("--tokenizer-vocab", type=int, default=None)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_dotted_overrides(cfg, args.override)
    data_cfg = cfg.get("data", {})
    out_dir = Path(args.out_dir or Path(data_cfg.get("train_bin", "data/train.bin")).parent)
    tok_dir = Path(args.tokenizer_dir or data_cfg.get("tokenizer_dir", "data/tokenizer"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        tok_vocab = args.tokenizer_vocab or 1024
        max_tokens = args.max_tokens or 1_000_000
        tok_sample = list(synthetic_corpus(n_docs=1500))

        def stream():
            return synthetic_corpus(n_docs=8000)
    else:
        tok_vocab = args.tokenizer_vocab or 32000
        max_tokens = args.max_tokens
        print(f"[data] sampling tokenizer corpus from {args.sources}")
        tok_sample = list(islice(mixed_stream(args.sources, max_docs_per_source=args.max_docs), 20000))

        def stream():
            return mixed_stream(args.sources, max_docs_per_source=args.max_docs)

    tok_path = tok_dir / "tokenizer.json"
    if args.retrain or not tok_path.exists():
        print(f"[data] training tokenizer (vocab={tok_vocab})")
        train_tokenizer(iter(tok_sample), vocab_size=tok_vocab, out_dir=tok_dir)
    tokenizer = Tech0Tokenizer(tok_dir)

    raw_path = out_dir / "all.bin"
    print("[data] tokenizing corpus ->", raw_path)
    total = tokenize_to_bin(stream(), tokenizer.encode, raw_path, max_tokens=max_tokens)
    print(f"[data] wrote {total} tokens")

    train_path, val_path, n_train, n_val = split_tokens(raw_path, val_ratio=0.005)
    raw_path.unlink(missing_ok=True)

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "train_bin": train_path,
        "val_bin": val_path,
        "tokenizer_dir": str(tok_dir),
        "smoke": bool(args.smoke),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[data] train={n_train} val={n_val} vocab={tokenizer.vocab_size}")
    print(f"[data] meta written to {out_dir / 'meta.json'}")


if __name__ == "__main__":  # pragma: no cover
    main()
