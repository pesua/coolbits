"""Manifest schema, IO, and helpers.

The manifest is the durable, processed view of one source file. Render
consumes it; analyze writes it. Anything expensive enough to want to
keep around (CLIP embeddings, sampled frames) lives in the cache, not
the manifest.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclasses.dataclass
class Shot:
    index: int
    start_tc: str
    end_tc: str
    start_frame: int
    end_frame: int
    duration_s: float
    features: dict[str, float] = dataclasses.field(default_factory=dict)
    annotations: list[str] = dataclasses.field(default_factory=list)
    dialogue: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Manifest:
    schema_version: int
    source_hash: str
    source_path: str
    source_duration: float | None
    analyzed_at: str
    tool_versions: dict[str, str]
    subtitle_track: dict[str, Any] | None
    features_present: list[str]
    shots: list[Shot]

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        shots = [Shot(**s) for s in d.get("shots", [])]
        return cls(
            schema_version=d["schema_version"],
            source_hash=d["source_hash"],
            source_path=d["source_path"],
            source_duration=d.get("source_duration"),
            analyzed_at=d["analyzed_at"],
            tool_versions=d.get("tool_versions", {}),
            subtitle_track=d.get("subtitle_track"),
            features_present=d.get("features_present", []),
            shots=shots,
        )


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def partial_hash(path: Path, chunk: int = 1 << 20) -> str:
    """First+middle+last 1 MiB chunks. Cheap and stable for big MKVs."""
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(size.to_bytes(8, "little"))
    with path.open("rb") as f:
        f.seek(0)
        h.update(f.read(min(chunk, size)))
        if size > 2 * chunk:
            mid = (size // 2) - (chunk // 2)
            f.seek(max(0, mid))
            h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(0, size - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()[:32]


def manifest_dir(workspace: Path) -> Path:
    return workspace / "manifests"


def cache_dir(workspace: Path, source_hash: str) -> Path:
    return workspace / "cache" / source_hash


def manifest_path(workspace: Path, source_hash: str) -> Path:
    return manifest_dir(workspace) / f"{source_hash}.json"


def save(manifest: Manifest, path: Path) -> None:
    """Atomic write: temp file in same dir, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2, sort_keys=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def load(path: Path) -> Manifest:
    with path.open() as f:
        return Manifest.from_dict(json.load(f))
