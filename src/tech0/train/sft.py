"""Phase 3 multi-task SFT: ChatML data with prompt-masked loss.

Keeps a target:general mix (default 65/35) plus sample replay to limit
catastrophic forgetting. Prompt tokens are masked (label = -100) so only the
assistant response contributes to the loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tech0.data.dataset import InstructionDataset, collate_instructions
from tech0.model import Tech0Config, Tech0ForCausalLM
from tech0.tokenizer.build_bpe import Tech0Tokenizer
from tech0.train.optim import build_optimizer, cosine_lr
from tech0.train.trainer import nudges, resolve_device, resolve_dtype
from tech0.utils.ckpt import save_checkpoint
from tech0.utils.config import apply_dotted_overrides, load_config
from tech0.utils.logging import Logger
from tech0.utils.seed import set_seed

SMOKE_RECORDS = [
    {"instruction": "Reverse a string.", "output": "def rev(s):\n    return s[::-1]"},
    {"instruction": "What is 2+2?", "output": "4"},
    {"instruction": "Return the max of a list.", "output": "def mx(xs):\n    return max(xs)"},
    {"instruction": "Say hi.", "output": "Hello!"},
]


def ensure_sft_data(path: Path, smoke: bool) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in SMOKE_RECORDS:
            fh.write(json.dumps(rec) + "\n")


def load_student(model_dir: str | Path, device: str) -> Tech0ForCausalLM:
    model_dir = Path(model_dir)
    if model_dir.is_dir() and (model_dir / "config.json").exists():
        return Tech0ForCausalLM.from_pretrained(model_dir, map_location=device).to(device)
    state = torch.load(model_dir, map_location=device, weights_only=False)
    model = Tech0ForCausalLM(Tech0Config.from_dict({"model": state["config"]}))
    model.load_state_dict(state["model"])
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tech0 SFT")
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_dotted_overrides(cfg, args.override)
    run, data_cfg, train_cfg = cfg["run"], cfg["data"], cfg["training"]

    device = resolve_device(run.get("device", "auto"))
    dtype = resolve_dtype(device, run.get("dtype", "auto"))
    set_seed(run.get("seed", 1337))

    tokenizer = Tech0Tokenizer(cfg.get("tokenizer_dir", "data/tokenizer"))
    model = load_student(cfg["model_dir"], device)
    model.enable_gradient_checkpointing(bool(train_cfg.get("grad_checkpointing", True)) and device == "cuda")

    sft_path = Path(data_cfg["sft_jsonl"])
    ensure_sft_data(sft_path, args.smoke)
    seq_len = 128 if args.smoke else int(data_cfg["seq_len"])
    dataset = InstructionDataset(sft_path, tokenizer, seq_len)
    loader = DataLoader(
        dataset,
        batch_size=4 if args.smoke else int(train_cfg["batch_size"]),
        shuffle=True,
        collate_fn=lambda b: collate_instructions(b, pad_token_id=0),
        drop_last=False,
    )

    lr = float(train_cfg["lr"])
    optimizer = build_optimizer(model, lr=lr, weight_decay=float(train_cfg["weight_decay"]))
    ctx, scaler = nudges(device, dtype)
    out_dir = Path(run["out_dir"])
    logger = Logger(out_dir)

    max_steps = 15 if args.smoke else int(train_cfg["max_steps"])
    grad_accum = 1 if args.smoke else int(train_cfg["grad_accum"])
    warmup = int(train_cfg["warmup_steps"])
    logger.info(f"sft: device={device} examples={len(dataset)} tokens/seq={seq_len}")

    step = 0
    model.train()
    loader_iter = iter(loader)
    while step < max_steps:
        current_lr = cosine_lr(step, max_steps, warmup, lr, float(train_cfg["min_lr"]))
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(grad_accum):
            try:
                x, y, attn = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y, attn = next(loader_iter)
            x, y, attn = x.to(device), y.to(device), attn.to(device)
            with ctx:
                loss = model(x, attention_mask=attn, labels=y)["loss"] / grad_accum
            (scaler.scale(loss) if scaler is not None else loss).backward()
            total += loss.item()
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        step += 1
        if step % 5 == 0 or step == 1 or step == max_steps:
            logger.info(f"step {step}/{max_steps} loss={total:.4f} lr={current_lr:.2e}")
            logger.scalar(step, loss=total, lr=current_lr)

    save_checkpoint(out_dir / "step_best.pt", model, optimizer, max_steps, model.config.to_dict(), scaler)
    model.save_pretrained(out_dir / "final")
    logger.info("SFT complete")
    logger.close()


if __name__ == "__main__":  # pragma: no cover
    main()
