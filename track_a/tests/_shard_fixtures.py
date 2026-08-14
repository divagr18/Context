"""Hand-built C6 shard fixtures for dataset-layer tests (T5).

Builds minimal valid ``ShardRecord``s in code and serialises them via
``to_json`` so the reader tests exercise the real (de)serialisation path
without slow document generation.
"""
from __future__ import annotations

from pathlib import Path

from track_a.shard_schema import (
    CRender, KIND_SINGLE_SHOT, KIND_STREAMING, ShardRecord, StreamingWindow,
    to_json,
)
from track_a.needle_gen.types import Domain, Split


def make_single_shot_record(doc_ids=(1, 2, 3, 4, 5, 6, 7, 8),
                            doc_len_tokens: int = 2048,
                            ratios=("2", "4", "8", "16", "32"),
                            doc_id: str = "ss-0") -> ShardRecord:
    c_renders = {
        r: CRender(target_ids=(200 + int(r), 201, 202),
                   info_pressure_fact_count=3)
        for r in ratios
    }
    return ShardRecord(
        kind=KIND_SINGLE_SHOT, doc_id=doc_id, split=Split.TRAIN,
        domain=Domain.PROJECT_UPDATES, doc_len_name="short",
        doc_ids=tuple(doc_ids), entities=(), facts=(), questions=(),
        mention_spans=(), value_spans=(), cue_spans=(),
        scene_boundaries=(0,), doc_len_tokens=doc_len_tokens,
        c_renders=c_renders, windows=None,
    )


def make_streaming_record(doc_id: str = "st-0",
                          windows=None) -> ShardRecord:
    if windows is None:
        windows = (
            StreamingWindow(budget=256, state_ids=(), window_ids=(1, 2, 3),
                            ops_text="UPSERT ENTITY E0001 type=project name=Alpha"),
            StreamingWindow(budget=256, state_ids=(10, 11), window_ids=(4, 5),
                            ops_text=""),
            StreamingWindow(budget=256, state_ids=(10, 11), window_ids=(6, 7),
                            ops_text="UPSERT FACT E0001.port = 8080 pos=2"),
        )
    return ShardRecord(
        kind=KIND_STREAMING, doc_id=doc_id, split=Split.TRAIN,
        domain=Domain.PROJECT_UPDATES, doc_len_name="short",
        doc_ids=(1, 2, 3, 4, 5, 6, 7), entities=(), facts=(), questions=(),
        mention_spans=(), value_spans=(), cue_spans=(),
        scene_boundaries=(0,), doc_len_tokens=7, c_renders=None,
        windows=tuple(windows),
    )


def write_shards(records, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(to_json(r) for r in records) + "\n", encoding="utf-8")
    return p
