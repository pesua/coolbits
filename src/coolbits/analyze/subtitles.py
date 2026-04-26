"""Subtitle extraction & parsing.

Probes streams with ffprobe, extracts the chosen track via ffmpeg to SRT,
parses with `srt`, splits each cue into dialogue text and bracketed
non-speech annotations. Folds onto shots in `assemble`."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import srt

log = logging.getLogger(__name__)

# Bracketed cue markers: [WHIRRING], (DOOR SLAMS), {music plays}.
# Some SDH releases mix uppercase tags, italicized HTML, and speaker prefixes.
_BRACKET = re.compile(r"[\[\(\{][^\]\)\}]+[\]\)\}]")
_TAG = re.compile(r"<[^>]+>")
_SPEAKER = re.compile(r"^\s*[A-Z][A-Z0-9 \-']{1,20}:\s*")


@dataclass
class Cue:
    start_s: float
    end_s: float
    dialogue: str
    annotations: list[str]


def probe_subtitle_tracks(video_path: Path) -> list[dict[str, Any]]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "s",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(out.stdout or "{}")
    return data.get("streams", [])


def pick_track(
    streams: list[dict[str, Any]],
    *,
    prefer_languages: list[str],
    prefer_sdh: bool,
) -> dict[str, Any] | None:
    if not streams:
        return None

    def score(s: dict[str, Any]) -> tuple[int, int, int]:
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        lang = (tags.get("language") or "").lower()
        title = (tags.get("title") or "").lower()
        disp = s.get("disposition") or {}
        lang_rank = 0
        for i, want in enumerate(prefer_languages):
            if lang == want.lower():
                lang_rank = len(prefer_languages) - i
                break
        sdh_rank = 0
        if prefer_sdh and (
            disp.get("hearing_impaired")
            or "sdh" in title
            or "hearing" in title
            or "cc" in title
        ):
            sdh_rank = 1
        codec_rank = 1 if (s.get("codec_name") or "").lower() in {"subrip", "srt", "ass", "ssa", "mov_text"} else 0
        return (lang_rank, sdh_rank, codec_rank)

    ranked = sorted(streams, key=score, reverse=True)
    best = ranked[0]
    if score(best) == (0, 0, 0):
        return None
    return best


def extract_to_srt(video_path: Path, stream_index: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-map",
            f"0:{stream_index}",
            "-c:s",
            "srt",
            str(dest),
        ],
        check=True,
    )


def parse_srt(path: Path) -> list[Cue]:
    text = path.read_text(encoding="utf-8", errors="replace")
    cues: list[Cue] = []
    for sub in srt.parse(text):
        raw = _TAG.sub(" ", sub.content).replace("\n", " ").strip()
        anns = [m.group(0).strip("[]{}() ").strip() for m in _BRACKET.finditer(raw)]
        clean = _BRACKET.sub(" ", raw)
        clean = _SPEAKER.sub("", clean).strip()
        clean = re.sub(r"\s+", " ", clean)
        cues.append(
            Cue(
                start_s=sub.start.total_seconds(),
                end_s=sub.end.total_seconds(),
                dialogue=clean,
                annotations=[a for a in anns if a],
            )
        )
    return cues
