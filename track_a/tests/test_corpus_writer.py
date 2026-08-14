"""T3C: corpus_writer CLI invariants (contract C8)."""
from __future__ import annotations

from pathlib import Path

import pytest

from track_a.needle_gen.corpus_writer import (
    COMPRESSION_RATIOS, load_split_config, make_gen_config, write_corpus,
)
from track_a.shard_schema import KIND_SINGLE_SHOT, KIND_STREAMING, from_json
from track_a.tokenize import get_tokenizer

TRAIN_CFG = "track_a/needle_gen/splits/train.yaml"


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


def test_single_shot_shards(tmp_path: Path, tok):
    out = tmp_path / "ss.jsonl"
    n = write_corpus(TRAIN_CFG, out, limit_docs=2, kind=KIND_SINGLE_SHOT,
                     tokenizer=tok)
    assert n == 2
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        rec = from_json(line)
        assert rec.kind == KIND_SINGLE_SHOT
        assert rec.split.value == "train"
        assert rec.domain.value == "project_updates"
        assert rec.windows is None
        assert rec.c_renders is not None
        assert set(rec.c_renders.keys()) == {str(r) for r in COMPRESSION_RATIOS}
        assert len(rec.doc_ids) == rec.doc_len_tokens
        # renders shrink monotonically with compression ratio
        lens = [len(rec.c_renders[str(r)].target_ids) for r in COMPRESSION_RATIOS]
        assert lens == sorted(lens, reverse=True)
        assert rec.c_renders["32"].info_pressure_fact_count <= \
            rec.c_renders["2"].info_pressure_fact_count


def test_streaming_shards(tmp_path: Path, tok):
    out = tmp_path / "st.jsonl"
    n = write_corpus(TRAIN_CFG, out, limit_docs=1, kind=KIND_STREAMING,
                     tokenizer=tok)
    assert n == 1
    rec = from_json(out.read_text(encoding="utf-8").strip().split("\n")[0])
    assert rec.kind == KIND_STREAMING
    assert rec.c_renders is None
    assert rec.windows is not None and len(rec.windows) >= 1
    w0 = rec.windows[0]
    assert w0.state_ids == ()  # empty state before the first window
    assert len(w0.window_ids) > 0
    all_ops = "\n".join(w.ops_text for w in rec.windows if w.ops_text).split("\n")
    from track_a.parse_compact import parse_ops_lines
    _ops, failures = parse_ops_lines([l for l in all_ops if l])
    assert failures == []
    assert len(_ops) > 0


def test_limit_docs_caps_output(tmp_path: Path, tok):
    cfg = load_split_config(TRAIN_CFG)
    assert cfg["n_docs"] > 3
    out = tmp_path / "cap.jsonl"
    n = write_corpus(TRAIN_CFG, out, limit_docs=3, tokenizer=tok)
    assert n == 3


def test_shard_roundtrip_preserves_facts(tmp_path: Path, tok):
    out = tmp_path / "rt.jsonl"
    write_corpus(TRAIN_CFG, out, limit_docs=1, kind=KIND_SINGLE_SHOT,
                 tokenizer=tok)
    rec = from_json(out.read_text(encoding="utf-8").strip().split("\n")[0])
    assert len(rec.facts) > 0
    assert all(f.id for f in rec.facts)
    queried = [f for f in rec.facts if f.is_queried]
    assert len(queried) >= 8
    assert len(rec.entities) > 0
    assert len(rec.questions) > 0
    assert rec.scene_boundaries[0] == 0
    assert all(a < b for a, b in zip(rec.scene_boundaries,
                                     rec.scene_boundaries[1:]))
