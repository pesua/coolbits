"""Shot selection: threshold or target-duration."""

from __future__ import annotations

import numpy as np

from ..manifest import Shot


def select(
    shots: list[Shot],
    scores: np.ndarray,
    *,
    mode: str,
    threshold: float,
    target_duration_s: float,
    min_shot_duration_s: float,
    max_shot_duration_s: float,
) -> list[int]:
    """Returns indices of selected shots, in chronological order."""
    eligible = [
        i
        for i, s in enumerate(shots)
        if min_shot_duration_s <= s.duration_s <= max_shot_duration_s
    ]

    if mode == "threshold":
        keep = [i for i in eligible if scores[i] >= threshold]
    elif mode == "target_duration":
        ranked = sorted(eligible, key=lambda i: scores[i], reverse=True)
        kept: list[int] = []
        total = 0.0
        for i in ranked:
            if total >= target_duration_s:
                break
            kept.append(i)
            total += shots[i].duration_s
        keep = kept
    else:
        raise ValueError(f"unknown selection mode: {mode}")
    return sorted(keep)


def pad_and_merge(
    shots: list[Shot],
    selected: list[int],
    *,
    edge_padding_s: float,
    bridge_gap_s: float,
    source_duration_s: float | None,
) -> list[tuple[float, float]]:
    """Returns merged [start, end) intervals in seconds, sorted.

    Two cool shots separated by a gap <= bridge_gap_s are joined and the
    intervening filler is kept — this is what produces the small amount of
    connective filler in the final cut."""
    from ..util import parse_tc

    if not selected:
        return []
    intervals = []
    for i in selected:
        sh = shots[i]
        s = max(0.0, parse_tc(sh.start_tc) - edge_padding_s)
        e = parse_tc(sh.end_tc) + edge_padding_s
        if source_duration_s is not None:
            e = min(e, source_duration_s)
        intervals.append((s, e))
    intervals.sort()
    merged: list[tuple[float, float]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s - cur_e <= bridge_gap_s:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged
