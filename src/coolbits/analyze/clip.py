"""CLIP embedding + prompt scoring.

Sample N frames per shot, embed with OpenCLIP, persist embeddings to the
cache as a (num_shots × frames_per_shot × dim) float16 array. Compute
per-prompt cosine similarities and return them as per-shot scalars."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

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


def _read_frame(video_path: Path, t: float, size: int = 224) -> np.ndarray | None:
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0 or len(out.stdout) < size * size * 3:
        return None
    return np.frombuffer(out.stdout[: size * size * 3], dtype=np.uint8).reshape(size, size, 3).copy()


def embed_and_score(
    video_path: Path,
    shots: list[Shot],
    *,
    cache_dir: Path,
    model_name: str,
    pretrained: str,
    frames_per_shot: int,
    prompts_pos: dict[str, str],
    prompts_neg: dict[str, str],
) -> dict[str, dict[int, float]]:
    """Returns {feature_name: {shot_index: score}} for all prompts.

    Embeddings are cached under cache_dir/clip_embeddings.npy. If the cache
    matches (same shot count, same frames_per_shot, same model), we skip
    re-embedding and score from the cache.
    """
    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "clip_embeddings.npy"
    meta_path = cache_dir / "clip_meta.txt"

    expect_shape = (len(shots), frames_per_shot)
    expect_meta = f"{model_name}|{pretrained}|{frames_per_shot}|{len(shots)}"

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
        from PIL import Image
        from tqdm import tqdm

        dim = model.visual.output_dim
        embeddings = np.zeros((len(shots), frames_per_shot, dim), dtype=np.float32)
        batch_imgs: list[torch.Tensor] = []
        batch_idx: list[tuple[int, int]] = []
        BATCH = 32

        def flush() -> None:
            if not batch_imgs:
                return
            x = torch.stack(batch_imgs).to(device=device, dtype=dtype)
            with torch.no_grad():
                feats = model.encode_image(x)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.cpu().numpy()
            for (si, fi), v in zip(batch_idx, feats_np):
                embeddings[si, fi] = v
            batch_imgs.clear()
            batch_idx.clear()

        for shot in tqdm(shots, desc="CLIP embedding shots"):
            timestamps = _sample_timestamps(shot, frames_per_shot)
            for fi, t in enumerate(timestamps):
                arr = _read_frame(video_path, t)
                if arr is None:
                    continue
                img = Image.fromarray(arr)
                batch_imgs.append(preprocess(img))
                batch_idx.append((shot.index, fi))
                if len(batch_imgs) >= BATCH:
                    flush()
        flush()
        np.save(emb_path, embeddings)
        meta_path.write_text(expect_meta)
        log.info("Saved CLIP embeddings to %s", emb_path)

    # Score prompts.
    all_prompts: dict[str, str] = {**prompts_pos, **prompts_neg}
    text_tokens = tokenizer(list(all_prompts.values())).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats_np = text_feats.cpu().numpy()  # (P, dim)

    # mean-pool the per-shot frame embeddings, then cosine.
    shot_emb = embeddings.mean(axis=1)  # (N, dim)
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
