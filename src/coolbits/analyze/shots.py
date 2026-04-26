"""Shot detection via PySceneDetect.

The shot list is the spine of the manifest — every other feature
attaches per-shot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scenedetect import ContentDetector, SceneManager, open_video

from ..manifest import Manifest, Shot
from ..util import fmt_tc


def detect_shots(
    video_path: Path,
    *,
    threshold: float = 27.0,
    min_scene_len: int = 15,
) -> tuple[list[Shot], dict[str, Any]]:
    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    sm.detect_scenes(video=video, show_progress=False)
    scenes = sm.get_scene_list()
    fps = video.frame_rate

    shots: list[Shot] = []
    if not scenes:
        # Fall back to one shot covering the whole file.
        duration = float(video.duration.get_seconds())
        shots.append(
            Shot(
                index=0,
                start_tc=fmt_tc(0.0),
                end_tc=fmt_tc(duration),
                start_frame=0,
                end_frame=int(duration * fps),
                duration_s=duration,
            )
        )
    else:
        for i, (start, end) in enumerate(scenes):
            s = float(start.get_seconds())
            e = float(end.get_seconds())
            shots.append(
                Shot(
                    index=i,
                    start_tc=fmt_tc(s),
                    end_tc=fmt_tc(e),
                    start_frame=int(start.get_frames()),
                    end_frame=int(end.get_frames()),
                    duration_s=e - s,
                )
            )

    meta = {"fps": float(fps), "duration_s": float(video.duration.get_seconds())}
    return shots, meta


def attach_to_manifest(manifest: Manifest, shots: list[Shot]) -> None:
    manifest.shots = shots
    if "shots" not in manifest.features_present:
        manifest.features_present.append("shots")
