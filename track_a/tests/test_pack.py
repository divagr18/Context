"""T5: pack format + loss-mask contract (loss only on the C*/ops/answer span)."""
from __future__ import annotations

import pytest

from track_a.data.pack import (
    IGNORE_INDEX, PackSample, pack_qa, pack_single_shot,
    pack_streaming_window, special_id,
)
from track_a.shard_schema import StreamingWindow
from track_a.tokenize import get_tokenizer


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


def test_special_id_atomic_for_all_struct_tokens(tok):
    for s in ("<doc>", "</doc>", "</C>", "<state>", "</state>", "<window>",
              "</window>", "</OPS>", "<Q>", "</Q>", "<A>", "</A>",
              "<budget=1024>"):
        special_id(tok, s)  # raises if not exactly one token


def test_special_id_rejects_multi_token(tok):
    with pytest.raises(ValueError):
        special_id(tok, "hello world definitely")


def test_single_shot_format_and_mask(tok):
    doc_ids = (5, 6, 7, 8)
    c_ids = (200, 201, 202)
    sample = pack_single_shot(doc_ids, c_ids, doc_len_tokens=2048, ratio=2,
                              tok=tok)
    ids = sample.input_ids
    doc_open = special_id(tok, "<doc>")
    doc_close = special_id(tok, "</doc>")
    c_close = special_id(tok, "</C>")
    assert ids[0] == doc_open
    assert ids[1:5] == doc_ids
    assert ids[5] == doc_close
    assert ids[6] != doc_close  # budget token follows </doc>
    target = list(c_ids) + [c_close]
    target_start = 1 + len(doc_ids) + 2  # <doc> + doc + </doc> + budget
    assert list(ids[target_start:]) == target
    assert all(l == IGNORE_INDEX for l in sample.labels[:target_start - 1])
    assert list(sample.labels[target_start - 1:-1]) == target
    assert sample.labels[-1] == IGNORE_INDEX
    assert sum(1 for l in sample.labels if l != IGNORE_INDEX) == len(target)
    assert len(sample.input_ids) == len(sample.labels)


def test_labels_always_predict_next_token(tok):
    sample = pack_single_shot((1, 2), (300, 301), doc_len_tokens=1024,
                              ratio=4, tok=tok)
    for i, lbl in enumerate(sample.labels):
        if lbl != IGNORE_INDEX:
            assert lbl == sample.input_ids[i + 1]


def test_streaming_format_and_mask(tok):
    ops = "UPSERT ENTITY E0001 type=project name=Alpha"
    win = StreamingWindow(budget=256, state_ids=(10, 11), window_ids=(1, 2, 3),
                          ops_text=ops)
    sample = pack_streaming_window(win, tok)
    ids = sample.input_ids
    assert ids[0] == special_id(tok, "<state>")
    assert ids[1:3] == (10, 11)
    assert ids[3] == special_id(tok, "</state>")
    assert ids[4] == special_id(tok, "<window>")
    assert ids[5:8] == (1, 2, 3)
    assert ids[8] == special_id(tok, "</window>")
    ops_close = special_id(tok, "</OPS>")
    assert ids[-1] == ops_close
    ops_ids = list(tok.encode(ops, add_special_tokens=False))
    target = ops_ids + [ops_close]
    assert list(ids[len(ids) - len(target):]) == target
    assert sum(1 for l in sample.labels if l != IGNORE_INDEX) == len(target)


def test_streaming_empty_ops_still_teaches_close(tok):
    win = StreamingWindow(budget=256, state_ids=(), window_ids=(1,), ops_text="")
    sample = pack_streaming_window(win, tok)
    assert sample.input_ids[-1] == special_id(tok, "</OPS>")
    assert sum(1 for l in sample.labels if l != IGNORE_INDEX) == 1


def test_qa_format_and_mask(tok):
    sample = pack_qa((400, 401), "What is the port?", "8080", tok)
    ids = sample.input_ids
    assert ids[0] == special_id(tok, "<doc>")
    assert special_id(tok, "<Q>") in ids
    assert special_id(tok, "<A>") in ids
    a_close = special_id(tok, "</A>")
    assert ids[-1] == a_close
    a_ids = list(tok.encode("8080", add_special_tokens=False))
    target = a_ids + [a_close]
    assert list(ids[len(ids) - len(target):]) == target
    assert sum(1 for l in sample.labels if l != IGNORE_INDEX) == len(target)


def test_pack_sample_length_mismatch_rejected():
    with pytest.raises(ValueError):
        PackSample(input_ids=(1, 2, 3), labels=(1, 2))
