"""Config loading. YAML on disk, dict in memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_PATH = Path(__file__).parent / "default_config.yaml"


def load(path: Path | None = None) -> dict[str, Any]:
    base = yaml.safe_load(DEFAULT_PATH.read_text())
    if path is not None and path != DEFAULT_PATH:
        override = yaml.safe_load(path.read_text()) or {}
        base = _merge(base, override)
    return base


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
