"""Minimal rank-aware logger with optional Weights & Biases / TensorBoard-free CSV logging."""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path


class Logger:
    def __init__(self, run_dir: str | Path, enabled: bool = True, rank: int = 0):
        self.run_dir = Path(run_dir)
        self.enabled = enabled and rank == 0
        self.rank = rank
        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_file = None
        self._writer = None
        self._start = time.time()
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_csv(self) -> None:
        if not self.enabled or self._writer is not None:
            return
        self._csv_file = self._csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)

    def info(self, msg: str) -> None:
        if self.enabled:
            elapsed = time.time() - self._start
            print(f"[tech0 {elapsed:8.1f}s] {msg}", flush=True)

    def scalar(self, step: int, **metrics: float) -> None:
        if not self.enabled:
            return
        self._ensure_csv()
        row = {"step": step, "wall": round(time.time() - self._start, 3), **metrics}
        if self._writer is not None:
            if self._csv_file.tell() == 0:
                self._writer.writerow(list(row.keys()))
            self._writer.writerow(list(row.values()))
            self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._writer = None


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def suppress_non_master() -> None:  # pragma: no cover
    if not is_main_process():
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
