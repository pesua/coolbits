"""CLIP embedding + prompt scoring.

Sample N frames per shot, embed with OpenCLIP, persist embeddings to the
cache as a (num_shots × frames_per_shot × dim) float32 array. Compute
per-prompt cosine similarities and return them as per-shot scalars.

Frame extraction is one linear ffmpeg pass that emits subsampled frames
to stdout. Each desired sample timestamp is mapped to its nearest frame
in that stream, so we never re-decode the same GOP twice — critical on
large 4K HEVC sources where per-shot `-ss` seeks were the bottleneck."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

from ..manifest import Shot

log = logging.getLogger(__name__)


def _sample_timestamps(shot: Shot, frames_per_shot: int) -> list[float]:
    s = max(0.0, shot.duration_s)
    start = _to_seconds(shot.start_tc)
    end = _to_seconds(shot.end_tc)
    if frames_per_shot == 1 or s <= 0:
        return [start + s / 2]
    step = (end - start) / (frames_per_shot + 1)
    return [start + step * (i + 1) for i in range(frames_per_shot)]


def _to_seconds(tc: str) -> float:
    h, m, rest = tc.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def embed_and_score(
    video_path: Path,
    shots: list[Shot],
    *,
    cache_dir: Path,
    model_name: str,
    pretrained: str,
    frames_per_shot: int,
    sample_fps: float,
    prompts_pos: dict[str, str],
    prompts_neg: dict[str, str],
) -> dict[str, dict[int, float]]:
    """Returns {feature_name: {shot_index: score}} for all prompts.

    Cache lives at cache_dir/clip_embeddings.npy. We re-embed only when
    the (model, pretrained, frames_per_shot, sample_fps, num_shots) tuple
    changes; prompt scoring runs every call from cached embeddings."""
    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "clip_embeddings.npy"
    meta_path = cache_dir / "clip_meta.txt"

    expect_shape = (len(shots), frames_per_shot)
    expect_meta = (
        f"{model_name}|{pretrained}|{frames_per_shot}|{sample_fps}|{len(shots)}"
    )

    embeddings: np.ndarray | None = None
    if emb_path.exists() and meta_path.exists() and meta_path.read_text().strip() == expect_meta:
        cached = np.load(emb_path)
        if cached.shape[:2] == expect_shape:
            embeddings = cached
            log.info("Loaded CLIP embeddings from cache: %s", emb_path)

    log.info("Loading CLIP model %s/%s on %s", model_name, pretrained, device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    if embeddings is None:
        embeddings = _embed_batched(
            video_path,
            shots,
            model=model,
            preprocess=preprocess,
            device=device,
            frames_per_shot=frames_per_shot,
            sample_fps=sample_fps,
        )
        np.save(emb_path, embeddings)
        meta_path.write_text(expect_meta)
        log.info("Saved CLIP embeddings to %s", emb_path)

    # Score prompts from (cached or fresh) embeddings.
    all_prompts: dict[str, str] = {**prompts_pos, **prompts_neg}
    text_tokens = tokenizer(list(all_prompts.values())).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats_np = text_feats.cpu().numpy()  # (P, dim)

    shot_emb = embeddings.mean(axis=1)
    shot_emb = shot_emb / np.maximum(
        np.linalg.norm(shot_emb, axis=1, keepdims=True), 1e-8
    )
    scores = shot_emb @ text_feats_np.T  # (N, P)

    out: dict[str, dict[int, float]] = {name: {} for name in all_prompts}
    names = list(all_prompts.keys())
    for shot in shots:
        for j, name in enumerate(names):
            out[name][shot.index] = float(scores[shot.index, j])
    return out


def _embed_batched(
    video_path: Path,
    shots: list[Shot],
    *,
    model,
    preprocess,
    device: str,
    frames_per_shot: int,
    sample_fps: float,
) -> np.ndarray:
    """Single linear ffmpeg pass + map sampled frames onto shots."""
    import torch
    from PIL import Image
    from tqdm import tqdm

    SIZE = 224
    dim = model.visual.output_dim
    embeddings = np.zeros((len(shots), frames_per_shot, dim), dtype=np.float32)

    # Map each desired sample (shot, slot) to its nearest frame index in
    # the subsampled stream. Multiple slots can share a frame index (very
    # short shots) — that's fine; we'll embed the frame once and reuse.
    wanted: dict[int, list[tuple[int, int]]] = {}
    for shot in shots:
        for slot, t in enumerate(_sample_timestamps(shot, frames_per_shot)):
            idx = max(0, int(round(t * sample_fps)))
            wanted.setdefault(idx, []).append((shot.index, slot))

    if not wanted:
        return embeddings

    max_wanted = max(wanted)
    log.info(
        "CLIP frame extraction: %d unique frames at %.2f fps "
        "(covering %d shots, last frame idx %d ~ t=%.1fs)",
        len(wanted),
        sample_fps,
        len(shots),
        max_wanted,
        max_wanted / sample_fps,
    )

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        (
            f"fps={sample_fps},"
            f"scale={SIZE}:{SIZE}:force_original_aspect_ratio=increase,"
            f"crop={SIZE}:{SIZE}"
        ),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    frame_size = SIZE * SIZE * 3
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=frame_size * 64,
    )

    BATCH = 32
    batch_imgs: list = []
    batch_meta: list[tuple[int, int]] = []

    def flush() -> None:
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device=device)
        with torch.no_grad():
            feats = model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_np = feats.cpu().numpy()
        for (si, fi), v in zip(batch_meta, feats_np):
            embeddings[si, fi] = v
        batch_imgs.clear()
        batch_meta.clear()

    pbar = tqdm(total=len(wanted), desc="CLIP embedding frames")
    frame_idx = 0
    try:
        while frame_idx <= max_wanted:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            slots = wanted.get(frame_idx)
            if slots is not None:
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(SIZE, SIZE, 3).copy()
                img = preprocess(Image.fromarray(arr))
                for shot_idx, slot in slots:
                    batch_imgs.append(img)
                    batch_meta.append((shot_idx, slot))
                if len(batch_imgs) >= BATCH:
                    flush()
                pbar.update(1)
            frame_idx += 1
        flush()
    finally:
        pbar.close()
        try:
            proc.stdout.close()
        except BrokenPipeError:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # Backfill: shots whose wanted frames fell past stream end (rare —
    # rounding above a clip's last frame) keep their zero rows; their
    # CLIP scores end up at 0 and don't dominate the weighted sum.
    return embeddings
