"""T5: TrainingDataset end-to-end over fixture shards."""
from __future__ import annotations

import pytest

from track_a.data.dataset import TrainingDataset
from track_a.data.pack import IGNORE_INDEX, special_id
from track_a.shard_schema import KIND_SINGLE_SHOT, KIND_STREAMING
from track_a.tests._shard_fixtures import (
    make_single_shot_record, make_streaming_record, write_shards,
)
from track_a.tokenize import get_tokenizer


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


def test_single_shot_yields_all_ratios(tmp_path, tok):
    path = write_shards([make_single_shot_record(doc_id="a")],
                        tmp_path / "ss.jsonl")
    samples = list(TrainingDataset([path], tok, kind=KIND_SINGLE_SHOT))
    assert len(samples) == 5
    c_close = special_id(tok, "</C>")
    assert all(s.input_ids[-1] == c_close for s in samples)
    assert all(len(s.input_ids) == len(s.labels) for s in samples)


def test_single_shot_sampled_ratio(tmp_path, tok):
    path = write_shards([make_single_shot_record()], tmp_path / "ss.jsonl")
    samples = list(TrainingDataset([path], tok, kind=KIND_SINGLE_SHOT,
                                   all_ratios=False, seed=5))
    assert len(samples) == 1


def test_streaming_yields_nonempty_windows(tmp_path, tok):
    path = write_shards([make_streaming_record()], tmp_path / "st.jsonl")
    samples = list(TrainingDataset([path], tok, kind=KIND_STREAMING))
    assert len(samples) == 2
    ops_close = special_id(tok, "</OPS>")
    assert all(s.input_ids[-1] == ops_close for s in samples)


def test_kind_filter_skips_foreign_records(tmp_path, tok):
    path = write_shards([make_streaming_record()], tmp_path / "st.jsonl")
    assert list(TrainingDataset([path], tok, kind=KIND_SINGLE_SHOT)) == []


def test_invalid_kind_raises(tok):
    with pytest.raises(ValueError):
        TrainingDataset([], tok, kind="bogus")


def test_shuffle_deterministic_and_complete(tmp_path, tok):
    records = [make_single_shot_record(doc_id=f"d{i}") for i in range(8)]
    path = write_shards(records, tmp_path / "ss.jsonl")
    s1 = [x.input_ids for x in
          TrainingDataset([path], tok, seed=11, shuffle_buffer=6)]
    s2 = [x.input_ids for x in
          TrainingDataset([path], tok, seed=11, shuffle_buffer=6)]
    s3 = [x.input_ids for x in
          TrainingDataset([path], tok, seed=12, shuffle_buffer=6)]
    assert s1 == s2                       # deterministic per seed
    assert len(s1) == 8 * 5               # every doc x ratio survives
    assert sorted(map(len, s1)) == sorted(map(len, s3))


def test_every_sample_has_unmasked_targets(tmp_path, tok):
    path = write_shards([make_single_shot_record()], tmp_path / "ss.jsonl")
    for s in TrainingDataset([path], tok):
        assert any(l != IGNORE_INDEX for l in s.labels)
