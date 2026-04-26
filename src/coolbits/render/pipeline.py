"""Orchestrate render: load manifest, score, select, optionally encode."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import manifest as mf
from ..util import parse_tc
from . import score as score_mod
from . import select as select_mod
from . import encode as encode_mod

log = logging.getLogger(__name__)


@dataclass
class RenderPlan:
    manifest_path: Path
    selected_indices: list[int]
    scores: np.ndarray
    intervals: list[tuple[float, float]]
    total_duration_s: float


def plan(
    manifest: mf.Manifest,
    config: dict[str, Any],
) -> RenderPlan:
    r_cfg = config["render"]
    weights = r_cfg["weights"]
    sel_cfg = r_cfg["selection"]

    # Drop weights for features not present in the manifest. This is the
    # "partial manifest still renderable" promise from PLAN.md.
    available = set()
    for shot in manifest.shots:
        available.update(shot.features.keys())
    effective = {k: v for k, v in weights.items() if k in available}
    if len(effective) < len(weights):
        missing = sorted(set(weights) - set(effective))
        log.warning("Skipping unavailable features: %s", missing)

    if not effective:
        raise RuntimeError("No usable features in manifest — cannot score")

    scores = score_mod.score_shots(manifest.shots, effective)

    selected = select_mod.select(
        manifest.shots,
        scores,
        mode=sel_cfg["mode"],
        threshold=float(sel_cfg["threshold"]),
        target_duration_s=float(sel_cfg["target_duration_s"]),
        min_shot_duration_s=float(sel_cfg["min_shot_duration_s"]),
        max_shot_duration_s=float(sel_cfg["max_shot_duration_s"]),
    )

    intervals = select_mod.pad_and_merge(
        manifest.shots,
        selected,
        edge_padding_s=float(r_cfg.get("edge_padding_s", 0.0)),
        bridge_gap_s=float(r_cfg.get("bridge_gap_s", 0.0)),
        source_duration_s=manifest.source_duration,
    )
    total = sum(e - s for s, e in intervals)
    return RenderPlan(
        manifest_path=Path(""),
        selected_indices=selected,
        scores=scores,
        intervals=intervals,
        total_duration_s=total,
    )


def encode(plan: RenderPlan, source: Path, output: Path, config: dict[str, Any]) -> None:
    r_cfg = config["render"]
    enc = r_cfg["encode"]
    encode_mod.render_intervals(
        source,
        plan.intervals,
        output,
        mode=enc["mode"],
        crf=int(enc["crf"]),
        preset=enc["preset"],
        video_codec=enc["video_codec"],
        audio_codec=enc["audio_codec"],
        audio_bitrate=enc["audio_bitrate"],
        audio_fade_ms=int(r_cfg.get("audio_fade_ms", 150)),
    )


def format_preview(
    manifest: mf.Manifest,
    plan: RenderPlan,
    *,
    max_lines: int | None = None,
) -> str:
    lines: list[str] = []
    lines.append(
        f"# Preview — {Path(manifest.source_path).name}\n"
        f"# Shots: {len(manifest.shots)}, "
        f"selected: {len(plan.selected_indices)}, "
        f"merged intervals: {len(plan.intervals)}, "
        f"output duration: {plan.total_duration_s:.1f}s\n"
    )

    rows = []
    feats_in_use = set()
    for i in plan.selected_indices:
        feats_in_use.update(manifest.shots[i].features.keys())
    feat_order = sorted(feats_in_use)
    header = (
        f"{'idx':>5}  {'start':>12}  {'end':>12}  {'dur':>6}  {'score':>7}  "
        + "  ".join(f"{k[:14]:>14}" for k in feat_order)
        + "  | dialogue / annotations"
    )
    rows.append(header)
    rows.append("-" * len(header))
    for i in plan.selected_indices:
        sh = manifest.shots[i]
        score = plan.scores[i]
        feat_cells = "  ".join(f"{sh.features.get(k, 0.0):14.3f}" for k in feat_order)
        d = (sh.dialogue[0] if sh.dialogue else "").strip()
        a = (sh.annotations[0] if sh.annotations else "").strip()
        tail = ""
        if d:
            tail += d[:60]
        if a:
            tail += f"  [{a[:40]}]"
        rows.append(
            f"{sh.index:>5}  {sh.start_tc:>12}  {sh.end_tc:>12}  "
            f"{sh.duration_s:>6.2f}  {score:>7.3f}  {feat_cells}  | {tail}"
        )

    if max_lines is not None and len(rows) > max_lines + 2:
        rows = rows[: max_lines + 2] + [f"... ({len(plan.selected_indices) - max_lines} more)"]

    return "\n".join(lines + rows)
