"""Optional analyze-time proxy.

Some sources (4K HEVC remuxes) make every analyze stage slow because
software HEVC decode is the bottleneck. A one-shot transcode to a small
H.264 proxy turns three slow decode passes (shot detection, motion, CLIP)
into one slow encode plus three fast decodes — a net win on heavy sources.

The render stage still pulls from the original; the proxy is analyze-only."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _probe_height(video_path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=height",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(out.stdout or "{}").get("streams", [])
    return int(streams[0]["height"]) if streams else 0


def _should_build(cfg: dict[str, Any], src_height: int) -> bool:
    mode = (cfg.get("mode") or "auto").lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    # auto
    threshold = int(cfg.get("auto_threshold_height", 1440))
    return src_height >= threshold


def get_or_create(
    video_path: Path,
    cache_dir: Path,
    cfg: dict[str, Any],
) -> Path:
    """Return the path the analyze stages should read from.

    If proxy mode applies, builds (or reuses) cache_dir/proxy.mkv and
    returns it; otherwise returns video_path unchanged."""
    src_height = _probe_height(video_path)
    if not _should_build(cfg, src_height):
        return video_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = cache_dir / "proxy.mkv"
    height = int(cfg.get("height", 720))
    crf = int(cfg.get("crf", 23))
    preset = cfg.get("preset", "veryfast")

    # Reuse if proxy already covers this source.
    sentinel = cache_dir / "proxy_meta.txt"
    expected = f"{video_path.stat().st_size}|{height}|{crf}|{preset}"
    if proxy_path.exists() and sentinel.exists() and sentinel.read_text().strip() == expected:
        log.info("Proxy already built: %s", proxy_path)
        return proxy_path

    log.info(
        "Building analyze proxy at %dp CRF %d (source is %dp) — this is "
        "a one-shot transcode but pays off across shot detection, motion, "
        "and CLIP stages.",
        height,
        crf,
        src_height,
    )

    # -map 0:v:0 video, drop audio, copy all subtitle streams so the
    # subs stage can find SDH on the proxy directly. Even-numbered scale
    # value avoids libx264 chroma-alignment errors on odd inputs.
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-stats",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:s?",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-vf",
        f"scale=-2:{height}",
        "-c:s",
        "copy",
        "-an",
        str(proxy_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # Some MKV sub formats (e.g. PGS) refuse to be copied into
        # MP4-flavored containers. We're writing MKV, so this is
        # rare — but if it happens, retry without subs and let the
        # subtitle stage probe the original instead.
        log.warning("Proxy build with subs failed; retrying without subs")
        cmd_no_subs = [c for c in cmd if c not in ("0:s?",)]
        # Drop the second -map and -c:s copy too.
        cleaned: list[str] = []
        skip = 0
        for i, tok in enumerate(cmd):
            if skip:
                skip -= 1
                continue
            if tok == "-map" and i + 1 < len(cmd) and cmd[i + 1] == "0:s?":
                skip = 1
                continue
            if tok == "-c:s" and i + 1 < len(cmd) and cmd[i + 1] == "copy":
                skip = 1
                continue
            cleaned.append(tok)
        subprocess.run(cleaned, check=True)
        # Mark so subtitles stage knows to read original.
        (cache_dir / "proxy_no_subs").write_text("1")

    sentinel.write_text(expected)
    if proxy_path.exists():
        log.info(
            "Proxy ready: %.1f MiB (%s)",
            proxy_path.stat().st_size / 1024 / 1024,
            proxy_path,
        )
    return proxy_path


def has_subs(cache_dir: Path) -> bool:
    return not (cache_dir / "proxy_no_subs").exists()
