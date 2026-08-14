"""T5: shard reader invariants (streaming read, schema guard, multi-file)."""
from __future__ import annotations

import json

import pytest

from track_a.data.shard_reader import iter_shards, iter_shards_many
from track_a.shard_schema import KIND_SINGLE_SHOT, KIND_STREAMING, to_json
from track_a.tests._shard_fixtures import (
    make_single_shot_record, make_streaming_record, write_shards,
)


def test_round_trip(tmp_path):
    ss = make_single_shot_record(doc_id="a")
    st = make_streaming_record(doc_id="b")
    path = write_shards([ss, st], tmp_path / "s.jsonl")
    records = list(iter_shards(path))
    assert [r.doc_id for r in records] == ["a", "b"]
    assert records[0].kind == KIND_SINGLE_SHOT
    assert records[1].kind == KIND_STREAMING
    assert records[0].doc_ids == ss.doc_ids
    assert set(records[0].c_renders.keys()) == {"2", "4", "8", "16", "32"}
    assert records[0].c_renders["2"].target_ids == (202, 201, 202)
    assert len(records[1].windows) == 3


def test_skips_blank_lines(tmp_path):
    rec = make_single_shot_record()
    path = tmp_path / "s.jsonl"
    path.write_text(to_json(rec) + "\n\n" + to_json(rec) + "\n",
                    encoding="utf-8")
    assert len(list(iter_shards(path))) == 2


def test_unknown_schema_version_raises(tmp_path):
    rec = make_single_shot_record()
    obj = json.loads(to_json(rec))
    obj["schema_version"] = 999
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(obj) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_shards(path))


def test_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        list(iter_shards(tmp_path / "nope.jsonl"))


def test_iter_shards_many_concatenates(tmp_path):
    p1 = write_shards([make_single_shot_record(doc_id="x")],
                      tmp_path / "1.jsonl")
    p2 = write_shards([make_streaming_record(doc_id="y")],
                      tmp_path / "2.jsonl")
    records = list(iter_shards_many([p1, p2]))
    assert [r.doc_id for r in records] == ["x", "y"]


def test_iter_shards_many_empty_list():
    assert list(iter_shards_many([])) == []
