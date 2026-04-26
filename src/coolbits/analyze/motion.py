"""Motion energy.

Decode at 360p, compute mean abs inter-frame delta, bucket per shot.
Cheap; runs at multiple times realtime on a single core."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

from ..manifest import Shot

log = logging.getLogger(__name__)


def compute(
    video_path: Path,
    shots: list[Shot],
    *,
    decode_height: int = 360,
    sample_every_n: int = 2,
) -> dict[int, float]:
    """Returns {shot_index: motion_energy}.

    Streams downsampled grayscale frames from ffmpeg via stdout, accumulates
    per-second sums of abs inter-frame deltas, then maps onto shots.
    """
    # Probe original resolution to derive scaled width.
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    import json as _json

    info = _json.loads(probe.stdout)["streams"][0]
    src_w, src_h = int(info["width"]), int(info["height"])
    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    new_h = decode_height
    new_w = int(src_w * (new_h / src_h)) // 2 * 2

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={new_w}:{new_h},select=not(mod(n\\,{sample_every_n}))",
        "-vsync",
        "vfr",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "-",
    ]
    frame_size = new_w * new_h
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_size * 4)

    bucket = max(1, int(round(fps / sample_every_n)))  # ~1s buckets
    deltas_per_bucket: list[float] = []
    prev: np.ndarray | None = None
    accum = 0.0
    n_in_bucket = 0
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            cur = np.frombuffer(buf, dtype=np.uint8).reshape(new_h, new_w)
            if prev is not None:
                d = float(np.mean(np.abs(cur.astype(np.int16) - prev.astype(np.int16))))
                accum += d
                n_in_bucket += 1
                if n_in_bucket >= bucket:
                    deltas_per_bucket.append(accum / n_in_bucket)
                    accum, n_in_bucket = 0.0, 0
            prev = cur
        if n_in_bucket > 0:
            deltas_per_bucket.append(accum / n_in_bucket)
    finally:
        proc.stdout.close()
        proc.wait()

    out: dict[int, float] = {}
    for shot in shots:
        s_idx = int(shot.duration_s and 0 + max(0, int(shot.start_frame / fps)))
        e_idx = int(max(s_idx + 1, int(shot.end_frame / fps)))
        seg = deltas_per_bucket[s_idx:e_idx]
        out[shot.index] = float(np.mean(seg)) if seg else 0.0
    return out
