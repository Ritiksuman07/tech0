"""Phase 1/2 pretraining: next-token prediction with a cosine LR schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tech0.data.dataset import BinDataset
from tech0.model import Tech0Config, Tech0ForCausalLM
from tech0.train.optim import build_optimizer, cosine_lr
from tech0.train.trainer import nudges, resolve_device, resolve_dtype
from tech0.utils.ckpt import latest_checkpoint, load_checkpoint, save_checkpoint
from tech0.utils.config import apply_dotted_overrides, load_config
from tech0.utils.logging import Logger
from tech0.utils.seed import set_seed

SMOKE_MODEL = {
    "num_hidden_layers": 2,
    "hidden_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "intermediate_size": 352,
    "max_position_embeddings": 256,
}


@torch.no_grad()
def evaluate(model, dataset, batch_size, eval_iters, rng, device, ctx) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = dataset.get_batch(batch_size, rng)
        x, y = x.to(device), y.to(device)
        with ctx:
            loss = model(x, labels=y)["loss"]
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tech0 pretraining")
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_dotted_overrides(cfg, args.override)
    run_cfg, data_cfg, train_cfg = cfg["run"], cfg["data"], cfg["training"]

    device = resolve_device(run_cfg.get("device", "auto"))
    dtype = resolve_dtype(device, run_cfg.get("dtype", "auto"))
    set_seed(run_cfg.get("seed", 1337))

    train_bin = Path(data_cfg["train_bin"])
    meta_path = train_bin.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    model_cfg = load_config(args.model_config or cfg.get("model_config", "configs/model_50m.yaml"))
    model_dict = model_cfg.get("model", model_cfg)
    if meta.get("vocab_size"):
        model_dict["vocab_size"] = int(meta["vocab_size"])
    if args.smoke:
        model_dict.update(SMOKE_MODEL)

    model = Tech0ForCausalLM(Tech0Config.from_dict({"model": model_dict}))
    model.to(device)

    seq_len = int(data_cfg["seq_len"])
    batch_size = int(train_cfg["batch_size"])
    grad_accum = int(train_cfg["grad_accum"])
    max_steps = int(train_cfg["max_steps"])
    eval_iters = int(train_cfg["eval_iters"])

    if args.smoke:
        seq_len = min(seq_len, 128)
        batch_size = min(batch_size, 8)
        grad_accum = 1
        max_steps = min(max_steps, 20)
        eval_iters = min(eval_iters, 5)

    model.enable_gradient_checkpointing(bool(train_cfg.get("grad_checkpointing", True)) and device == "cuda")

    train_ds = BinDataset(train_bin, seq_len)
    val_bin = Path(data_cfg.get("val_bin", train_bin.parent / "val.bin"))
    val_ds = BinDataset(val_bin, seq_len) if val_bin.exists() else None

    optimizer = build_optimizer(
        model,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.95))),
    )
    ctx, scaler = nudges(device, dtype)

    out_dir = Path(args.out_dir or run_cfg["out_dir"])
    logger = Logger(out_dir)
    logger.info(f"device={device} dtype={dtype} params={model.num_parameters():,} tokens/step="
                f"{batch_size * grad_accum * seq_len:,}")

    rng = np.random.default_rng(run_cfg.get("seed", 1337))
    step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = latest_checkpoint(out_dir)
        if ckpt:
            info = load_checkpoint(ckpt, model, optimizer, scaler, map_location=device)
            step = info["step"]
            logger.info(f"resumed from {ckpt} at step {step}")

    lr, min_lr = float(train_cfg["lr"]), float(train_cfg["min_lr"])
    warmup = int(train_cfg["warmup_steps"])
    log_interval = int(train_cfg["log_interval"])
    eval_interval = int(train_cfg["eval_interval"])
    ckpt_interval = int(train_cfg["ckpt_interval"])
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    model.train()
    while step < max_steps:
        current_lr = cosine_lr(step, max_steps, warmup, lr, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(grad_accum):
            x, y = train_ds.get_batch(batch_size, rng)
            x, y = x.to(device), y.to(device)
            with ctx:
                loss = model(x, labels=y)["loss"] / grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        step += 1
        if step % log_interval == 0 or step == 1:
            logger.info(f"step {step}/{max_steps} loss={total_loss:.4f} lr={current_lr:.2e}")
            logger.scalar(step, loss=total_loss, lr=current_lr)

        if val_ds is not None and (step % eval_interval == 0 or step == max_steps):
            val_loss = evaluate(model, val_ds, batch_size, eval_iters, rng, device, ctx)
            logger.info(f"step {step} val_loss={val_loss:.4f} ppl={np.exp(val_loss):.2f}")
            logger.scalar(step, val_loss=val_loss, val_ppl=float(np.exp(val_loss)))
            if train_cfg.get("save_best", True) and val_loss < best_val:
                best_val = val_loss
                save_checkpoint(out_dir / "step_best.pt", model, optimizer, step, model_dict, scaler)

        if step % ckpt_interval == 0 or step == max_steps:
            save_checkpoint(out_dir / f"step_{step:07d}.pt", model, optimizer, step, model_dict, scaler)

    model.save_pretrained(out_dir / "final")
    logger.info("training complete")
    logger.close()


if __name__ == "__main__":  # pragma: no cover
    main()
