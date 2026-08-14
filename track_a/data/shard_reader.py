"""Streaming JSONL shard reader (contract C6, no full-corpus load).

Iterates records line-by-line with pathlib + UTF-8. ``from_json`` raises on
unknown ``schema_version``, so a bad/foreign shard fails fast rather than
silently feeding wrong data into training.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from track_a.shard_schema import ShardRecord, from_json


def iter_shards(path: str | Path) -> Iterator[ShardRecord]:
    """Yield ShardRecords from one JSONL shard, skipping blank lines."""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield from_json(line)


def iter_shards_many(paths) -> Iterator[ShardRecord]:
    """Concatenate records across multiple shard files in order."""
    for path in paths:
        yield from iter_shards(path)
