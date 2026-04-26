"""LLM caption classification.

Takes the unique set of bracketed annotations across all cues and asks
Claude to label each one. Cached on disk under cache/{hash}/captions.json
keyed by normalized annotation text so the same labels never get billed
twice."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ("action", "ambient", "dialogue-adjacent", "music")


def _normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def classify_annotations(
    annotations: set[str],
    *,
    cache_path: Path,
    model: str,
    batch_size: int = 80,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
) -> dict[str, str]:
    cache: dict[str, str] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            log.warning("Caption cache at %s is corrupt; starting fresh", cache_path)

    by_norm: dict[str, list[str]] = {}
    for ann in annotations:
        norm = _normalize(ann)
        if not norm:
            continue
        by_norm.setdefault(norm, []).append(ann)

    todo = [n for n in by_norm if n not in cache]
    if not todo:
        return {ann: cache[norm] for norm, anns in by_norm.items() for ann in anns}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning(
            "ANTHROPIC_API_KEY not set; skipping caption classification "
            "(treating all annotations as 'ambient')."
        )
        for n in todo:
            cache[n] = "ambient"
    else:
        try:
            from anthropic import Anthropic
        except ImportError:
            log.warning("anthropic SDK missing; skipping caption classification")
            for n in todo:
                cache[n] = "ambient"
        else:
            client = Anthropic(api_key=api_key)
            for i in range(0, len(todo), batch_size):
                batch = todo[i : i + batch_size]
                labels = _classify_batch(client, model, batch, categories)
                for n, lab in zip(batch, labels):
                    cache[n] = lab
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    return {ann: cache[norm] for norm, anns in by_norm.items() for ann in anns}


def _classify_batch(
    client: Any,
    model: str,
    items: list[str],
    categories: tuple[str, ...],
) -> list[str]:
    cats = ", ".join(categories)
    numbered = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(items))
    prompt = (
        "You label SDH subtitle non-speech annotations from films. "
        "For each numbered item, choose exactly one category from: "
        f"{cats}.\n\n"
        "Definitions:\n"
        "- action: violent/explosive/loud kinetic events (gunfire, explosions, "
        "engines roaring, debris, screams of effort).\n"
        "- ambient: passive environmental sound (wind, distant city, water).\n"
        "- dialogue-adjacent: speech-coloring cues (sighs, gasps, laughter, "
        "speaker labels).\n"
        "- music: score, song, or musical-cue annotations.\n\n"
        "Reply with one line per item, formatted exactly as `N: category`. "
        "Do not add commentary.\n\n"
        f"Items:\n{numbered}"
    )

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return _parse_labels(text, len(items), categories)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2**attempt)
    log.warning("LLM classification failed after retries: %s", last_err)
    return ["ambient"] * len(items)


def _parse_labels(text: str, n: int, categories: tuple[str, ...]) -> list[str]:
    labels = ["ambient"] * n
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*[:\.\)]\s*([\w\-]+)", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        cat = m.group(2).lower()
        if 0 <= idx < n and cat in categories:
            labels[idx] = cat
    return labels
