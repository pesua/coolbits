"""Phase 1 — analysis stages.

Each stage takes the manifest plus extra inputs and mutates it in place
(or writes to the cache dir alongside it). Stages are idempotent on a
per-source-hash basis: re-running a stage replaces its outputs.
"""
