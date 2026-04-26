"""ffmpeg-based concat with re-encode and audio fades."""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def render_intervals(
    source: Path,
    intervals: list[tuple[float, float]],
    output: Path,
    *,
    mode: str,
    crf: int,
    preset: str,
    video_codec: str,
    audio_codec: str,
    audio_bitrate: str,
    audio_fade_ms: int,
) -> None:
    if not intervals:
        raise ValueError("no intervals to render")

    if mode == "stream_copy":
        _render_stream_copy(source, intervals, output)
        return

    n = len(intervals)
    fade_s = max(0.0, audio_fade_ms / 1000.0)

    parts: list[str] = []
    for i, (s, e) in enumerate(intervals):
        parts.append(
            f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}];"
        )
        afilter = (
            f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS"
        )
        if fade_s > 0 and (e - s) > 2 * fade_s:
            afilter += (
                f",afade=t=in:st=0:d={fade_s:.3f}"
                f",afade=t=out:st={(e - s) - fade_s:.3f}:d={fade_s:.3f}"
            )
        parts.append(afilter + f"[a{i}];")

    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
    parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]")
    filter_complex = "".join(parts)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "info",
        "-i",
        str(source),
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-map",
        "[outa]",
        "-c:v",
        video_codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        audio_codec,
        "-b:a",
        audio_bitrate,
        "-movflags",
        "+faststart",
        str(output),
    ]
    log.info("ffmpeg: %s", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def _render_stream_copy(source: Path, intervals: list[tuple[float, float]], output: Path) -> None:
    """Cuts on keyframes only — fast but imprecise. Useful for previews."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        listfile = td_path / "concat.txt"
        parts = []
        for i, (s, e) in enumerate(intervals):
            seg = td_path / f"seg_{i:04d}.mkv"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{s:.3f}",
                    "-to",
                    f"{e:.3f}",
                    "-i",
                    str(source),
                    "-c",
                    "copy",
                    str(seg),
                ],
                check=True,
            )
            parts.append(f"file {shlex.quote(str(seg))}")
        listfile.write_text("\n".join(parts))
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listfile),
                "-c",
                "copy",
                str(output),
            ],
            check=True,
        )
