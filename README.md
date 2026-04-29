# coolbits

Take a sci-fi film, throw out the talky filler, keep the cool stuff.

Two-phase pipeline. **Analyze** runs once per film and is slow: shot detection, subtitle extraction, LLM caption labeling, motion energy, CLIP embedding. Output is a per-film manifest plus a feature cache. **Render** is fast and runs as many times as you want: it scores every shot from the manifest, keeps the ones above a coolness bar, glues nearby cool shots together with a little filler between them, and ffmpeg-concats them with audio fades.

The selection is a coolness threshold, not a target length. We want all the cool stuff, however long that turns out to be — plus a small amount of bridging filler so cool runs don't feel staccato.

## Install & run

Requires `ffmpeg`/`ffprobe` on `PATH` and [`uv`](https://docs.astral.sh/uv/).

```sh
uv sync

# one-time, expensive (≈8 min on CPU for a 2-hour film)
uv run coolbits analyze "Atlas (2024).mkv"

# fast — text listing of selected shots, no encoding
uv run coolbits preview "Atlas (2024).mkv"

# full render
uv run coolbits render "Atlas (2024).mkv" --out cut.mp4
```

LLM caption classification reads `ANTHROPIC_API_KEY` from the environment or a `.env` at the project root. Without it, classification falls back to "ambient" for every annotation; everything else still runs.

Tune via YAML: copy `src/coolbits/default_config.yaml`, edit, pass `--config my.yaml`. The interesting knobs are `analyze.clip.prompts`, `render.weights`, `render.selection.threshold`, and `render.bridge_gap_s`.

## How it works

- **Manifest = shots + features.** Shots are the unit. Every signal (motion, dialogue density, CLIP-action score, action-classified annotations) is a per-shot scalar. Adding a new signal means dropping a per-shot number into the manifest and a weight into the config — nothing else has to change.
- **Per-film normalization.** Features are 0–1 scaled across this film, so different films with different motion baselines stay comparable.
- **Caching.** Manifest under `.coolbits/manifests/<source_hash>.json`. Raw intermediates (extracted SRT, CLIP embeddings, caption-label cache) under `.coolbits/cache/<source_hash>/`. Adding a new CLIP prompt re-scores from cached embeddings — no re-decoding.
- **Stages are individually idempotent.** Re-running `analyze` after editing the config picks up new signals without redoing finished stages. `--force` re-runs everything.

## What's in MVP

Shot detection (PySceneDetect), subtitle extraction + dialogue/SDH-annotation split, LLM annotation classification (Claude), frame-diff motion energy, CLIP ViT-B/32 prompt scoring, weighted-sum selection with bridging, ffmpeg re-encode with audio fades.

MVP assumes English SDH subs are embedded in the source MKV. A film without SDH still renders — just with weaker positive-signal coverage.

## Backlog

**Performance (large 4K HEVC sources)**
- Optional analyze-time proxy. One-shot transcode of huge 4K HEVC sources to a 720p/1080p H.264 proxy used by all analyze stages (shot detect, motion, CLIP), while render still pulls from the original. Expected savings ≈ two full HEVC decode passes per film. Behind a config flag so small sources skip it.

**Fallbacks for films without SDH subtitles**
- Silero VAD for speech density.
- YAMNet for action / music / speech classification from audio.
- Demucs stem separation if VAD/YAMNet aren't enough.

**Quality**
- TransNetV2 for shot detection on films heavy with dissolves.
- Optical flow (Farnebäck) replacing frame-diff if motion energy underperforms.
- Beat-synced cut boundaries from audio onset detection.
- Auto-tuned per-film weights from a small set of manual keep/drop labels.

**New signal sources**
- Chapter markers as coarse structural priors.
- Trailer alignment — shots appearing in the official trailer as positive training signal.
- Dolby Atmos object metadata where present.

**Modes**
- "Supercut" mode that preserves some narrative flow.
- Interactive GUI with a scrubbable timeline over the preview listing.
- One-shot autonomous mode — sensible defaults, no preview loop.

**Infra**
- GPU batching for CLIP across multiple films.
- Shared annotation-classification cache across a film library.
- Resume-from-stage if analysis crashes mid-pipeline.
