"""Drive the analysis stages end-to-end."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .. import manifest as mf
from . import shots as shots_stage
from . import subtitles as subs_stage
from . import captions as captions_stage
from . import motion as motion_stage
from . import clip as clip_stage
from . import assemble as assemble_stage

log = logging.getLogger(__name__)


def run(
    video_path: Path,
    *,
    workspace: Path,
    config: dict[str, Any],
    force: bool = False,
    skip_clip: bool = False,
    skip_motion: bool = False,
    skip_captions: bool = False,
) -> mf.Manifest:
    video_path = video_path.resolve()
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    log.info("Hashing source: %s", video_path.name)
    src_hash = mf.partial_hash(video_path)
    cache_d = mf.cache_dir(workspace, src_hash)
    cache_d.mkdir(parents=True, exist_ok=True)
    man_path = mf.manifest_path(workspace, src_hash)

    if man_path.exists() and not force:
        existing = mf.load(man_path)
        if existing.schema_version != mf.SCHEMA_VERSION:
            log.info("Manifest schema mismatch — re-analyzing from scratch")
            existing = None
    else:
        existing = None

    if existing is None:
        manifest = mf.Manifest(
            schema_version=mf.SCHEMA_VERSION,
            source_hash=src_hash,
            source_path=str(video_path),
            source_duration=None,
            analyzed_at=mf.now_iso(),
            tool_versions={},
            subtitle_track=None,
            features_present=[],
            shots=[],
        )
    else:
        manifest = existing

    # Stage 1 — shot detection.
    if "shots" in manifest.features_present and manifest.shots and not force:
        log.info("Stage: shot detection — cached (%d shots)", len(manifest.shots))
    else:
        log.info("Stage: shot detection")
        sd = config["analyze"]["shot_detection"]
        shots, sd_meta = shots_stage.detect_shots(
            video_path,
            threshold=sd["threshold"],
            min_scene_len=sd["min_scene_len"],
        )
        shots_stage.attach_to_manifest(manifest, shots)
        manifest.source_duration = sd_meta["duration_s"]
        manifest.tool_versions["scenedetect_fps"] = str(sd_meta["fps"])
        log.info("Detected %d shots over %.1fs", len(shots), sd_meta["duration_s"])
        mf.save(manifest, man_path)

    # Stage 2 — subtitle extraction. Cheap, always re-fold.
    log.info("Stage: subtitles")
    sub_cfg = config["analyze"]["subtitles"]
    streams = subs_stage.probe_subtitle_tracks(video_path)
    track = subs_stage.pick_track(
        streams,
        prefer_languages=sub_cfg["prefer_languages"],
        prefer_sdh=sub_cfg["prefer_sdh"],
    )
    cues: list = []
    if track is not None:
        srt_path = cache_d / "subs.srt"
        try:
            subs_stage.extract_to_srt(video_path, track["index"], srt_path)
            cues = subs_stage.parse_srt(srt_path)
            tags = track.get("tags") or {}
            manifest.subtitle_track = {
                "index": track["index"],
                "language": tags.get("language"),
                "title": tags.get("title"),
                "codec": track.get("codec_name"),
                "cue_count": len(cues),
            }
            log.info(
                "Extracted %d cues from subtitle stream %d (%s)",
                len(cues),
                track["index"],
                tags.get("language") or "unknown",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Subtitle extraction failed: %s", e)
            manifest.subtitle_track = None
    else:
        log.warning("No usable subtitle track found")

    # Stage 3 — caption classification.
    annotation_labels: dict[str, str] = {}
    if cues and not skip_captions:
        log.info("Stage: caption classification")
        unique_anns = {a for cue in cues for a in cue.annotations}
        if unique_anns:
            cap_cfg = config["analyze"]["caption_classification"]
            annotation_labels = captions_stage.classify_annotations(
                unique_anns,
                cache_path=cache_d / "captions.json",
                model=cap_cfg["model"],
                batch_size=cap_cfg["batch_size"],
                categories=tuple(cap_cfg["categories"]),
            )
            log.info("Classified %d unique annotations", len(unique_anns))

    assemble_stage.attach_subtitle_features(manifest.shots, cues, annotation_labels)
    if cues and "subtitles" not in manifest.features_present:
        manifest.features_present.append("subtitles")
    mf.save(manifest, man_path)

    # Stage 4 — motion energy.
    if not skip_motion:
        if "motion_energy" in manifest.features_present and not force:
            log.info("Stage: motion energy — cached")
        else:
            log.info("Stage: motion energy")
            m_cfg = config["analyze"]["motion"]
            motion_scores = motion_stage.compute(
                video_path,
                manifest.shots,
                decode_height=m_cfg["decode_height"],
                sample_every_n=m_cfg["sample_every_n_frames"],
            )
            for shot in manifest.shots:
                shot.features["motion_energy"] = motion_scores.get(shot.index, 0.0)
            if "motion_energy" not in manifest.features_present:
                manifest.features_present.append("motion_energy")
        mf.save(manifest, man_path)

    # Stage 5 — CLIP. Cache lives in cache_d/clip_embeddings.npy; the
    # embed_and_score routine already short-circuits embedding when the cache
    # is valid. We re-score prompts every run since that's cheap and lets users
    # add prompts without --force.
    if not skip_clip:
        log.info("Stage: CLIP embedding + prompt scoring")
        c_cfg = config["analyze"]["clip"]
        scores_per_prompt = clip_stage.embed_and_score(
            video_path,
            manifest.shots,
            cache_dir=cache_d,
            model_name=c_cfg["model"],
            pretrained=c_cfg["pretrained"],
            frames_per_shot=c_cfg["frames_per_shot"],
            prompts_pos=c_cfg["prompts"]["positive"],
            prompts_neg=c_cfg["prompts"]["negative"],
        )
        for prompt_name, per_shot in scores_per_prompt.items():
            for shot in manifest.shots:
                shot.features[prompt_name] = per_shot.get(shot.index, 0.0)
        if "clip" not in manifest.features_present:
            manifest.features_present.append("clip")
        mf.save(manifest, man_path)

    log.info("Analysis complete: %s", man_path)
    return manifest
