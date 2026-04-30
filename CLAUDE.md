# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Two-phase pipeline that turns a sci-fi film into an action-only supercut.

- **Analyze** (slow, once per film): shot detection → subtitle extraction → LLM caption labeling → motion energy → CLIP embedding/scoring. Output is a per-film manifest plus a feature cache.
- **Render** (fast, repeatable): score every shot, keep ones above a coolness threshold, bridge nearby cool shots with short filler, ffmpeg-concat with audio fades.

Selection is a **quality threshold, not a length target** — total output length is whatever the film yields.

## Commands

Requires `ffmpeg`/`ffprobe` on `PATH` and [`uv`](https://docs.astral.sh/uv/). Python ≥3.11.

```sh
uv sync                                         # install
uv run coolbits analyze "Film.mkv"              # phase 1, expensive
uv run coolbits preview "Film.mkv"              # text listing of selected shots
uv run coolbits render "Film.mkv" --out cut.mp4 # full render
```

Common flags:
- `-v` (verbose logging), `--config my.yaml` (deep-merged over `default_config.yaml`), `--workspace ./other` (defaults to `./.coolbits`).
- `analyze --force` re-runs everything; `--skip-clip / --skip-motion / --skip-captions` skip stages.

There is **no test suite** and **no linter configured** — don't invent commands. If you need to sanity-check a change, the cheapest end-to-end probe is `uv run coolbits preview` against an already-analyzed film.

## Architecture

### Manifest is the contract between phases

`src/coolbits/manifest.py` defines `Manifest` and `Shot`. Every signal — motion, dialogue density, action-annotation count, CLIP-prompt similarities — is a per-shot scalar dropped into `shot.features[<name>]`. The manifest also tracks `features_present` so render knows what's available.

Adding a new signal = (1) write a per-shot float into the manifest during analyze, (2) add a weight key under `render.weights` in config. No other code changes.

The render side is tolerant of missing features: weights for absent features are dropped with a warning (`render/pipeline.py::plan`), so a half-analyzed manifest still renders.

### Per-film normalization

`render/score.py::normalize_per_film` 0–1 scales each feature across the film before applying weights. Different films with different motion baselines stay comparable. **Don't normalize during analyze** — keep raw values in the manifest so re-runs and config tweaks remain meaningful.

### Caching layout (`./.coolbits/`)

- `manifests/<source_hash>.json` — the manifest. `source_hash` is a 32-char SHA256 of (filesize ‖ first/middle/last 1 MiB) — see `manifest.partial_hash`. Stable for big MKVs and cheap to compute.
- `cache/<source_hash>/` — raw intermediates: `subs.srt`, `captions.json` (LLM label cache, keyed by normalized annotation text), `clip_embeddings.npy` + `clip_meta.txt`, optional `proxy.mkv` + `proxy_meta.txt`.

### Stage idempotence

`analyze/pipeline.py` checks `features_present` and the schema version before deciding to re-run a stage. CLIP **always re-scores prompts** (cheap) but only re-embeds if `(model, pretrained, frames_per_shot, sample_fps, num_shots)` changed — so adding a new prompt to config and re-running analyze is fast. `--force` re-runs everything; schema-version mismatch wipes the manifest and re-runs from scratch.

### Analyze proxy

`analyze/proxy.py` builds a one-shot 720p H.264 proxy when the source is ≥1440p (configurable). All analyze stages then read from the proxy, but **render always reads from the original**. Subtitles are usually copy-able into the proxy MKV; if `proxy_no_subs` sentinel exists in the cache dir, the subtitle stage falls back to probing the original.

### CLIP frame extraction

`analyze/clip.py` uses **one linear ffmpeg pass** that emits subsampled frames (default 2 fps) over stdout, then maps each shot's desired sample timestamps to nearest frame indices in that stream. This was specifically designed to avoid per-shot `-ss` seeks, which thrashed GOP decoding on 4K HEVC. Don't switch back to per-shot seeking.

### Render bridging

`render/select.py::pad_and_merge` joins two selected shots whose gap is `≤ bridge_gap_s` and **keeps the intervening unselected shots** as-is. This is intentional — those gap-fillers are the ~10% connective tissue that makes the cut watchable.

### LLM annotation classification

`analyze/captions.py` extracts unique bracketed SDH annotations (e.g. `[explosion]`, `[soft music]`), classifies each into `action / ambient / dialogue-adjacent / music` via Claude, and caches by **normalized** annotation text (lowercase, alphanumeric-only). The same label is never billed twice across re-runs. Reads `ANTHROPIC_API_KEY` from env or `.env`; without it, classification falls back to "ambient" silently and the rest of the pipeline still works.

## Config

`src/coolbits/default_config.yaml` is the source of truth. User configs are deep-merged over it (`config.py::_merge`). The knobs that actually move output:

- `analyze.clip.prompts.{positive,negative}` — feature names here become the keys in `shot.features` and must match `render.weights`.
- `render.weights` — sign matters; negatives push dialogue/closeup shots out.
- `render.selection.threshold` — the coolness bar. Mode is `threshold` by default (length-agnostic); `target_duration` mode exists for length-capped cuts.
- `render.bridge_gap_s` — controls how much filler bridges adjacent cool shots.

## Conventions worth keeping

- The manifest is **JSON, not pickle** — it's reviewed by humans and diff-able.
- Saves are atomic (`tempfile` + `os.replace` in `manifest.save`). Don't write partial JSON.
- Workspace defaults to `./.coolbits` (gitignored). Don't commit it; don't commit `videos/` or video files.
- The CLI entry is `coolbits = "coolbits.cli:main"` from `pyproject.toml` — `cli.py` only orchestrates; real work lives in `analyze/pipeline.py` and `render/pipeline.py`.
