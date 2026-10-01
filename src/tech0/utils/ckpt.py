"""Checkpoint save/load with resume support (single-file torch checkpoints)."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import torch


def _atomic_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    config: dict | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    extra: dict | None = None,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "config": config or {},
        "extra": extra or {},
    }
    _atomic_save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    path = Path(path)
    state = torch.load(path, map_location=map_location, weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"State dict mismatch. missing={missing} unexpected={unexpected}")
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return {"step": state.get("step", 0), "config": state.get("config", {}), "extra": state.get("extra", {})}


def latest_checkpoint(ckpt_dir: str | Path) -> str | None:
    files = sorted(glob.glob(str(Path(ckpt_dir) / "step_*.pt")))
    return files[-1] if files else None
