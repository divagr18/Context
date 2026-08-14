"""Shard schema (contract C6): round-trip + version/kind validation."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from track_a import shard_schema
from track_a.needle_gen import fixtures
from track_a.needle_gen.types import CueKind, CueSpan, Domain, MentionSpan, Split, ValueSpan
from track_a.shard_schema import CRender, ShardRecord, StreamingWindow


def _record(kind: str) -> ShardRecord:
    db = fixtures.make_fixture_factdb(seed=3, n_queried=7, n_decoys=2, with_chain=True)
    return ShardRecord(
        kind=kind,
        doc_id=db.doc_id,
        split=Split.TRAIN,
        domain=Domain.PROJECT_UPDATES,
        doc_len_name="short",
        doc_ids=tuple(range(64)),
        entities=db.entities,
        facts=db.facts,
        questions=db.questions,
        mention_spans=(MentionSpan(entity_id=db.entities[0].id, start_tok=3, end_tok=5),),
        value_spans=(ValueSpan(fact_id=db.facts[0].id, value_idx=0, start_tok=9, end_tok=10),),
        cue_spans=(CueSpan(kind=CueKind.HEDGE, start_tok=11, end_tok=12),),
        scene_boundaries=(0, 32),
        doc_len_tokens=64,
        c_renders={"64": CRender(target_ids=(5, 6, 7), info_pressure_fact_count=2)}
        if kind == shard_schema.KIND_SINGLE_SHOT
        else None,
        windows=(StreamingWindow(budget=128, state_ids=(1, 2), window_ids=(3, 4),
                                 ops_text="SUPERSEDE FACT E0001.attr : a => b pos=5"),)
        if kind == shard_schema.KIND_STREAMING
        else None,
    )


def test_single_shot_round_trip() -> None:
    record = _record(shard_schema.KIND_SINGLE_SHOT)
    line = shard_schema.to_json(record)
    assert "\n" not in line
    assert shard_schema.from_json(line) == record


def test_streaming_round_trip() -> None:
    record = _record(shard_schema.KIND_STREAMING)
    assert shard_schema.from_json(shard_schema.to_json(record)) == record


def test_unknown_schema_version_raises() -> None:
    tampered = json.loads(shard_schema.to_json(_record(shard_schema.KIND_SINGLE_SHOT)))
    tampered["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        shard_schema.from_json(json.dumps(tampered))


def test_kind_field_mismatch_rejected() -> None:
    good = _record(shard_schema.KIND_SINGLE_SHOT)
    with pytest.raises(ValueError):
        shard_schema.to_json(replace(good, c_renders=None))
    stray = replace(good, windows=(StreamingWindow(1, (), (), ""),))
    with pytest.raises(ValueError):
        shard_schema.to_json(stray)
    with pytest.raises(ValueError):
        shard_schema.to_json(replace(good, kind="bogus"))
