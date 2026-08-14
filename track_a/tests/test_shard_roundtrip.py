"""T-INT: corpus shards round-trip (write -> read -> pack consistency)."""
from __future__ import annotations

import pytest

from track_a.data import IGNORE_INDEX, TrainingDataset, special_id, window_samples
from track_a.data.shard_reader import iter_shards
from track_a.needle_gen.corpus_writer import write_corpus
from track_a.shard_schema import (
    KIND_SINGLE_SHOT, KIND_STREAMING, from_json, to_json,
)
from track_a.tokenize import get_tokenizer

TRAIN_YAML = "track_a/needle_gen/splits/train.yaml"


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


def test_single_shot_shard_roundtrip(tmp_path, tok):
    out = tmp_path / "ss.jsonl"
    n = write_corpus(TRAIN_YAML, out, limit_docs=3, kind=KIND_SINGLE_SHOT,
                     tokenizer=tok)
    assert n == 3
    records = list(iter_shards(out))
    assert len(records) == 3
    for rec in records:
        assert rec.kind == KIND_SINGLE_SHOT
        assert rec.windows is None
        assert set(rec.c_renders.keys()) == {"2", "4", "8", "16", "32"}
        assert len(rec.doc_ids) == rec.doc_len_tokens > 0
        assert len(rec.facts) > 0
        # exact record-level round-trip through the shard schema
        assert from_json(to_json(rec)) == rec


def test_single_shot_pack_targets_match_renders(tmp_path, tok):
    out = tmp_path / "ss.jsonl"
    write_corpus(TRAIN_YAML, out, limit_docs=3, kind=KIND_SINGLE_SHOT,
                 tokenizer=tok)
    records = list(iter_shards(out))
    close_id = special_id(tok, "</C>")
    doc_open_id = special_id(tok, "<doc>")

    expected_targets = []
    for rec in records:
        for render in rec.c_renders.values():
            expected_targets.append(tuple(render.target_ids) + (close_id,))

    samples = list(TrainingDataset([out], tok, kind=KIND_SINGLE_SHOT))
    assert len(samples) == len(expected_targets)  # all ratios, all docs
    got_targets = []
    for s in samples:
        assert len(s.input_ids) == len(s.labels)
        assert s.input_ids[0] == doc_open_id
        assert s.input_ids[-1] == close_id
        unmasked = tuple(l for l in s.labels if l != IGNORE_INDEX)
        assert unmasked and unmasked[-1] == close_id
        got_targets.append(unmasked)
    assert sorted(got_targets) == sorted(expected_targets)


def test_streaming_shard_roundtrip(tmp_path, tok):
    out = tmp_path / "st.jsonl"
    n = write_corpus(TRAIN_YAML, out, limit_docs=1, kind=KIND_STREAMING,
                     tokenizer=tok)
    assert n == 1
    records = list(iter_shards(out))
    assert len(records) == 1
    rec = records[0]
    assert rec.kind == KIND_STREAMING
    assert rec.c_renders is None
    assert rec.windows and len(rec.windows) >= 1
    for w in rec.windows:
        assert w.budget > 0
        assert len(w.window_ids) > 0
    assert from_json(to_json(rec)) == rec


def test_streaming_windows_pack_to_ops_samples(tmp_path, tok):
    out = tmp_path / "st.jsonl"
    write_corpus(TRAIN_YAML, out, limit_docs=1, kind=KIND_STREAMING,
                 tokenizer=tok)
    rec = list(iter_shards(out))[0]
    samples = list(window_samples(rec, tok))
    n_ops_windows = sum(1 for w in rec.windows if w.ops_text)
    assert len(samples) == n_ops_windows >= 1
    ops_close_id = special_id(tok, "</OPS>")
    for s in samples:
        assert len(s.input_ids) == len(s.labels)
        assert s.input_ids[-1] == ops_close_id
        unmasked = [l for l in s.labels if l != IGNORE_INDEX]
        assert unmasked  # ops + </OPS> are the supervised span
