"""Per-film feature normalization + weighted scoring."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..manifest import Shot


def normalize_per_film(shots: list[Shot], feature_names: Iterable[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    n = len(shots)
    for name in feature_names:
        vals = np.array([s.features.get(name, 0.0) for s in shots], dtype=np.float64)
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if hi - lo < 1e-9:
            out[name] = np.zeros(n)
        else:
            out[name] = (vals - lo) / (hi - lo)
    return out


def score_shots(shots: list[Shot], weights: dict[str, float]) -> np.ndarray:
    feats = normalize_per_film(shots, weights.keys())
    n = len(shots)
    total = np.zeros(n)
    for name, w in weights.items():
        total = total + float(w) * feats[name]
    return total
