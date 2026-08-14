"""T5: streaming window sample extraction."""
from __future__ import annotations

import pytest

from track_a.data.pack import IGNORE_INDEX, PackSample, special_id
from track_a.data.window_samples import window_samples
from track_a.shard_schema import StreamingWindow
from track_a.tests._shard_fixtures import make_streaming_record
from track_a.tokenize import get_tokenizer


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


def test_yields_only_nonempty_ops_windows(tok):
    rec = make_streaming_record()  # 3 windows, middle one has empty ops_text
    samples = list(window_samples(rec, tok))
    assert len(samples) == 2
    assert all(isinstance(s, PackSample) for s in samples)


def test_samples_end_with_ops_close_token(tok):
    rec = make_streaming_record()
    ops_close = special_id(tok, "</OPS>")
    for s in window_samples(rec, tok):
        assert s.input_ids[-1] == ops_close
        assert s.labels[-1] == IGNORE_INDEX
        assert len(s.input_ids) == len(s.labels)


def test_no_windows_yields_nothing(tok):
    rec = make_streaming_record(windows=())
    assert list(window_samples(rec, tok)) == []


def test_all_empty_ops_yields_nothing(tok):
    rec = make_streaming_record(windows=(
        StreamingWindow(budget=256, state_ids=(), window_ids=(1,), ops_text=""),
    ))
    assert list(window_samples(rec, tok)) == []


def test_sample_contains_window_marker(tok):
    rec = make_streaming_record()
    samples = list(window_samples(rec, tok))
    assert special_id(tok, "<window>") in samples[0].input_ids
    assert special_id(tok, "<state>") in samples[0].input_ids
