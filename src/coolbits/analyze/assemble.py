"""Fold subtitle streams + caption classifications onto shots by overlap."""

from __future__ import annotations

import logging
from typing import Iterable

from ..manifest import Shot
from ..util import overlap, parse_tc

log = logging.getLogger(__name__)


def attach_subtitle_features(
    shots: list[Shot],
    cues: Iterable,  # list[Cue] from analyze.subtitles
    annotation_labels: dict[str, str],
) -> None:
    cues_list = list(cues)
    for shot in shots:
        s = parse_tc(shot.start_tc)
        e = parse_tc(shot.end_tc)
        dur = max(1e-6, e - s)
        speech_overlap = 0.0
        words = 0
        ann_action = 0
        ann_ambient = 0
        ann_dialogue_adj = 0
        ann_music = 0
        diag_lines: list[str] = []
        ann_lines: list[str] = []
        for cue in cues_list:
            if cue.end_s <= s or cue.start_s >= e:
                continue
            ov = overlap(s, e, cue.start_s, cue.end_s)
            if ov <= 0:
                continue
            if cue.dialogue:
                speech_overlap += ov
                words += len(cue.dialogue.split())
                diag_lines.append(cue.dialogue)
            for ann in cue.annotations:
                ann_lines.append(ann)
                lab = annotation_labels.get(ann, "ambient")
                if lab == "action":
                    ann_action += 1
                elif lab == "ambient":
                    ann_ambient += 1
                elif lab == "dialogue-adjacent":
                    ann_dialogue_adj += 1
                elif lab == "music":
                    ann_music += 1

        shot.features["speech_density"] = float(min(1.0, speech_overlap / dur))
        shot.features["words_per_second"] = float(words / dur)
        shot.features["action_annotations"] = float(ann_action)
        shot.features["ambient_annotations"] = float(ann_ambient)
        shot.features["dialogue_adjacent_annotations"] = float(ann_dialogue_adj)
        shot.features["music_annotations"] = float(ann_music)
        shot.dialogue = diag_lines
        shot.annotations = ann_lines
