"""Knowledge distillation from a single teacher (Llama-3.2-3B).

Default `mode: sequence`:
    1. Teacher generates dense rationale/response text for seed prompts.
    2. Student is annealed on the synthetic token stream (CE).

Optional `mode: logit`:
    (1-a)*CE + a*tau^2*KL(student_topk || teacher_topk). Requires a shared
    vocabulary between student and teacher; guarded below because our 32k BPE
    and Llama's 128k vocab are not aligned.

Usage:
    python -m tech0.train.distill --config configs/distill.yaml            # generate + train
    python -m tech0.train.distill --config configs/distill.yaml --generate-only
    python -m tech0.train.distill --config configs/distill.yaml --smoke    # offline, no teacher
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tech0.data.dataset import BinDataset, safe_bin_dataset
from tech0.data.packing import split_tokens, tokenize_to_bin
from tech0.model import Tech0Config, Tech0ForCausalLM
from tech0.tokenizer.build_bpe import Tech0Tokenizer
from tech0.train.optim import build_optimizer, cosine_lr
from tech0.train.trainer import nudges, resolve_device, resolve_dtype, vram_gb
from tech0.utils.ckpt import save_checkpoint
from tech0.utils.config import apply_dotted_overrides, load_config
from tech0.utils.logging import Logger
from tech0.utils.seed import set_seed

DEFAULT_PROMPTS = [
    "Write a short function that returns the sum of a list of integers.",
    "Explain step by step how to compute the factorial of a number.",
    "Given a sentence, extract all named entities and their types.",
    "Reason carefully and solve: if a train travels 60 km in 45 minutes, what is its speed in km/h?",
    "Convert the following description into a JSON schema.",
    "Debug this Python snippet and explain the fix.",
    "Summarise the key idea of gradient descent for a beginner.",
    "Write a unit test for a function that reverses a string.",
]


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
    top_k: int = 100,
    ignore_index: int = -100,
) -> torch.Tensor:
    """(1 - alpha) * CE + alpha * tau^2 * KL(student || teacher) over top-k."""
    shift_student = student_logits[:, :-1, :]
    shift_teacher = teacher_logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    ce = F.cross_entropy(
        shift_student.reshape(-1, shift_student.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
    )

    k = min(top_k, shift_teacher.size(-1))
    topk_vals, topk_idx = torch.topk(shift_teacher, k, dim=-1)
    teacher_probs = F.softmax(topk_vals / temperature, dim=-1)
    student_logprobs = F.log_softmax(shift_student.gather(-1, topk_idx) / temperature, dim=-1)
    kl = F.kl_div(student_logprobs, teacher_probs, reduction="none").sum(-1)
    mask = (shift_labels != ignore_index).float()
    kl = (kl * mask).sum() / mask.sum().clamp(min=1.0)
    return (1.0 - alpha) * ce + alpha * (temperature ** 2) * kl


def _write_prompts(prompts: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for prompt in prompts:
            fh.write(json.dumps({"prompt": prompt}) + "\n")


def _load_prompts(path: Path) -> list[str]:
    if not path.exists():
        return list(DEFAULT_PROMPTS)
    prompts = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompts.append(rec.get("prompt") or rec.get("instruction") or rec.get("text", ""))
    return [p for p in prompts if p]


def generate_synthetic(cfg: dict, smoke: bool = False) -> Path:
    d = cfg["distillation"]
    data = cfg["data"]
    prompts_path = Path(data["prompts"])
    out_path = Path(data["out"])

    prompts = _load_prompts(prompts_path)
    if not prompts_path.exists():
        _write_prompts(prompts, prompts_path)

    if smoke:
        with out_path.open("w", encoding="utf-8") as fh:
            for prompt in prompts:
                fh.write(json.dumps({"prompt": prompt, "text": f"Question: {prompt}\nAnswer: " + prompt}) + "\n")
        print(f"[distill] wrote {len(prompts)} offline synthetic samples -> {out_path}")
        return out_path

    from tech0.teacher.hf_teacher import HFTeacher, TeacherConfig

    teacher = HFTeacher(TeacherConfig(
        model_name=cfg["teacher"]["model_name"],
        load_in_4bit=bool(cfg["teacher"].get("load_in_4bit", True)),
        min_vram_gb=float(cfg["teacher"].get("min_vram_gb", 8.0)),
        device=cfg["teacher"].get("device", "auto"),
    ))
    mode = d.get("mode", "sequence")
    if mode == "logit":
        raise RuntimeError(
            "logit KD requires a teacher whose vocabulary matches the student's "
            f"32k BPE (teacher vocab={teacher.vocab_size}). Use mode=sequence or supply a "
            "vocab-aligned teacher + projection."
        )

    num_samples = int(data.get("num_samples", 1000))
    generated = teacher.generate(
        prompts,
        max_new_tokens=int(data.get("max_new_tokens", 512)),
        temperature=float(data.get("temperature", 0.7)),
        batch_size=int(data.get("batch_size", 8)),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for i, text in enumerate(generated[:num_samples] if num_samples else generated):
            fh.write(json.dumps({"prompt": prompts[i % len(prompts)], "text": text}) + "\n")
    print(f"[distill] generated {len(generated)} samples -> {out_path}")
    return out_path


def train_student(cfg: dict, smoke: bool = False) -> None:
    s = cfg["student"]
    run = cfg["run"]
    data = cfg["data"]
    device = resolve_device(run.get("device", "auto"))
    dtype = resolve_dtype(device, run.get("dtype", "auto"))
    set_seed(run.get("seed", 1337))

    tokenizer = Tech0Tokenizer(cfg.get("tokenizer_dir", "data/tokenizer"))

    synthetic = Path(data["out"])
    texts = [json.loads(line)["text"] for line in synthetic.read_text(encoding="utf-8").splitlines() if line.strip()]
    if smoke:
        texts = texts * 300
    raw = synthetic.with_suffix(".bin")
    tokenize_to_bin(texts, tokenizer.encode, raw)
    train_bin, val_bin, n_train, n_val = split_tokens(raw, val_ratio=0.005)
    raw.unlink(missing_ok=True)

    model_cfg = load_config(s.get("model_config", "configs/model_50m.yaml"))
    model_dict = model_cfg.get("model", model_cfg)
    model_dict["vocab_size"] = tokenizer.vocab_size
    if smoke:
        model_dict.update({"num_hidden_layers": 2, "hidden_size": 128, "num_attention_heads": 4,
                           "num_key_value_heads": 2, "head_dim": 32, "intermediate_size": 352,
                           "max_position_embeddings": 256})
    model = Tech0ForCausalLM(Tech0Config.from_dict({"model": model_dict})).to(device)
    model.enable_gradient_checkpointing(device == "cuda")

    seq_len = int(s.get("seq_len", 1024))
    if smoke:
        seq_len = min(seq_len, 128)
    train_ds = BinDataset(train_bin, seq_len)
    val_ds = safe_bin_dataset(val_bin, seq_len)

    lr = float(s.get("lr", 2e-4))
    optimizer = build_optimizer(model, lr=lr, weight_decay=float(s.get("weight_decay", 0.1)))
    ctx, scaler = nudges(device, dtype)
    out_dir = Path(run["out_dir"])
    logger = Logger(out_dir)
    rng = np.random.default_rng(run.get("seed", 1337))

    max_steps = 20 if smoke else int(s.get("max_steps", 5000))
    batch_size = 8 if smoke else int(s.get("batch_size", 16))
    grad_accum = 1 if smoke else int(s.get("grad_accum", 8))
    logger.info(f"distill anneal: device={device} tokens/step={batch_size * grad_accum * seq_len:,}")

    model.train()
    for step in range(1, max_steps + 1):
        current_lr = cosine_lr(step - 1, max_steps, max(1, max_steps // 20), lr, lr * 0.05)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(grad_accum):
            x, y = train_ds.get_batch(batch_size, rng)
            x, y = x.to(device), y.to(device)
            with ctx:
                loss = model(x, labels=y)["loss"] / grad_accum
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
        if step % 5 == 0 or step == 1 or step == max_steps:
            logger.info(f"step {step}/{max_steps} loss={total:.4f}")
            logger.scalar(step, loss=total, lr=current_lr)

    save_checkpoint(out_dir / "step_best.pt", model, optimizer, max_steps, model_dict, scaler)
    model.save_pretrained(out_dir / "final")
    logger.info("distillation annealing complete")
    logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tech0 distillation")
    parser.add_argument("--config", default="configs/distill.yaml")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    if not args.smoke and vram_gb() and vram_gb() < 8:
        print(f"[distill] WARNING: only {vram_gb():.1f}GB VRAM; use --smoke locally and run the "
              "teacher on Colab/Kaggle.")

    cfg = load_config(args.config)
    apply_dotted_overrides(cfg, args.override)

    if not args.train_only:
        generate_synthetic(cfg, smoke=args.smoke)
    if not args.generate_only:
        train_student(cfg, smoke=args.smoke)


if __name__ == "__main__":  # pragma: no cover
    main()
