"""Lightweight config loader: YAML -> nested dict with attribute access and override support."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that supports attribute access, recursive merge and dotted overrides."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - attribute error path
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def deep_merge(self, other: dict) -> "Config":
        for key, value in other.items():
            if key in self and isinstance(self[key], dict) and isinstance(value, dict):
                Config(self[key]).deep_merge(value)
            else:
                self[key] = value
        return self

    def copy(self) -> "Config":  # type: ignore[override]
        return Config(copy.deepcopy(dict(self)))


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    cfg = Config()
    if path is not None:
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg.deep_merge(data)
    if overrides:
        cfg.deep_merge(overrides)
    return cfg


def apply_dotted_overrides(cfg: Config, pairs: list[str]) -> Config:
    """Apply ['a.b=1', 'c=hello'] style overrides in place."""
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Invalid override '{pair}', expected key=value")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        node: dict = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg
