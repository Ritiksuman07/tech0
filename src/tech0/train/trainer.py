"""Device/dtype resolution and autocast context shared by all trainers."""

from __future__ import annotations

import contextlib

import torch


def resolve_device(device_cfg: str = "auto") -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def resolve_dtype(device: str, dtype_cfg: str = "auto") -> torch.dtype:
    if dtype_cfg != "auto":
        return getattr(torch, dtype_cfg)
    if device == "cuda":
        # T4/P100 -> fp16 (no bf16); Ampere+ -> bf16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def nudges(device: str, dtype: torch.dtype):
    """Return (autocast_ctx, scaler). Scaler is only needed for cuda fp16."""
    if device == "cuda" and dtype in (torch.float16, torch.bfloat16):
        ctx = torch.autocast(device_type="cuda", dtype=dtype)
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    else:
        ctx = contextlib.nullcontext()
        scaler = None
    return ctx, scaler


def vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9
