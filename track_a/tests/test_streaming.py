"""T-INT / S7: streaming ground truth on REAL generated documents.

Folds every window's edit-ops through the real T4 OpLog and asserts the
folded state equals the independent budget-truncated canonical state, that
SUPERSEDE keeps prior values flagged in history, and that windows tile the
document.
"""
from __future__ import annotations

import pytest

from track_a.needle_gen.core.docgen import build_document
from track_a.needle_gen.streaming import (
    StreamState, apply_event, build_streaming_windows, extract_events,
    split_windows, truncate_state,
)
from track_a.needle_gen.types import (
    Domain, GenConfig, Split, canonicalize_value,
)
from track_a.parse_compact import OpLog
from track_a.tokenize import get_tokenizer

WINDOW_TOKENS = 1024
OVERLAP = 0.10
BUDGET = 1500


def make_config(seed: int = 5) -> GenConfig:
    return GenConfig(seed=seed, split=Split.TRAIN,
                     domain=Domain.PROJECT_UPDATES, doc_len_name="medium",
                     n_docs=1, family_ids=(), paraphrase_idx_range=(0, 5))


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


@pytest.fixture(scope="module")
def built(tok):
    return build_document(make_config(), 0, tok)


@pytest.fixture(scope="module")
def setup(built, tok):
    db = built.fact_db
    entity_map = {e.id: e for e in db.entities}
    windows = build_streaming_windows(
        built.doc_ids, db.facts, db.scene_boundaries, entity_map, BUDGET,
        WINDOW_TOKENS, OVERLAP, tok)
    return db, entity_map, windows


def _name_of(entity_map, eid: str) -> str:
    return canonicalize_value(entity_map[eid].name)


def test_windows_tile_the_document(setup, built):
    _db, _em, windows = setup
    ranges = split_windows(len(built.doc_ids), WINDOW_TOKENS, OVERLAP)
    assert len(windows) == len(ranges) >= 4
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(built.doc_ids)
    assert windows[0].state_ids == ()  # empty state before the first window
    for w in windows:
        assert len(w.window_ids) > 0
        assert w.budget == BUDGET


def test_all_window_ops_parse_cleanly(setup):
    _db, _em, windows = setup
    total_ops = 0
    for w in windows:
        if not w.ops_text:
            continue
        log = OpLog()
        failures = log.append(w.ops_text.split("\n"))
        assert failures == []
        total_ops += len(log.ops)
    assert total_ops > 50  # substantive edit traffic across the document


def test_fold_of_all_windows_equals_final_truncated_state(setup, tok):
    db, entity_map, windows = setup
    log = OpLog()
    for w in windows:
        if w.ops_text:
            assert log.append(w.ops_text.split("\n")) == []
    folded = log.fold()

    full = StreamState()
    for ev in extract_events(db.facts):
        apply_event(full, ev)
    expected = truncate_state(full, BUDGET, entity_map, tok)

    expect_facts = {
        (_name_of(entity_map, eid), attr): (value, pos)
        for (eid, attr), (value, pos, _rank, _h) in expected.facts.items()
    }
    assert dict(folded.current) == expect_facts

    expect_rels = {(_name_of(entity_map, a), rel, _name_of(entity_map, b))
                   for (a, rel, b) in expected.relations}
    assert folded.relations == expect_rels

    expect_negs = {(_name_of(entity_map, eid), attr)
                   for (eid, attr) in expected.negations}
    assert folded.negations == expect_negs

    expect_unres = {
        (_name_of(entity_map, eid), attr):
            tuple(sorted((canonicalize_value(v), p) for v, p in cands))
        for (eid, attr), (cands, _rank) in expected.unresolved.items()
    }
    folded_unres = {key: tuple(sorted(cands))
                    for key, cands in folded.unresolved.items()}
    assert folded_unres == expect_unres


def test_truncation_is_exercised(setup, tok):
    """Budget must actually force drops, else this test suite is vacuous."""
    db, entity_map, _windows = setup
    full = StreamState()
    for ev in extract_events(db.facts):
        apply_event(full, ev)
    untruncated_entries = (len(full.facts) + len(full.relations)
                           + len(full.negations) + len(full.unresolved))
    truncated = truncate_state(full, BUDGET, entity_map, tok)
    truncated_entries = (len(truncated.facts) + len(truncated.relations)
                         + len(truncated.negations) + len(truncated.unresolved))
    assert truncated_entries < untruncated_entries
    # state-transition facts (rank 0) survive truncation preferentially
    assert len(truncated.facts) >= 1


SUP_WINDOW_TOKENS = 512  # stride int(512*0.9)=460 < MID pad target (600)
SUP_BUDGET = 1500


def test_supersession_history_survives_with_correct_flags(tok):
    """MID-distance chains (>=600-token span) must cross 512-token windows.

    Chains only ever receive NEAR or MID buckets (docgen._chain_bucket), so a
    MID chain's padded final transition (>=600 tokens from the intro) is
    guaranteed to cross at least one 460-token window stride, forcing the
    value change to cross a window boundary and emit a real SUPERSEDE op.
    Chain bucket assignment is randomised, so search seeds 0-7 (deterministic
    seeds -> deterministic outcome; P(all 8 docs lack MID chains) < 1e-6).
    """
    found = 0
    for seed in range(8):
        built = build_document(make_config(seed=seed), 0, tok)
        db = built.fact_db
        entity_map = {e.id: e for e in db.entities}
        windows = build_streaming_windows(
            built.doc_ids, db.facts, db.scene_boundaries, entity_map,
            SUP_BUDGET, SUP_WINDOW_TOKENS, OVERLAP, tok)
        log = OpLog()
        for w in windows:
            if w.ops_text:
                assert log.append(w.ops_text.split("\n")) == []
        folded = log.fold()
        for (name, attr) in folded.current:
            hist = folded.history(name, attr)
            if len(hist) < 2:
                continue
            found += 1
            assert folded.current[(name, attr)] == (hist[-1][0], hist[-1][1])
            assert hist[-1][2] is False  # current value not superseded
            assert all(h[2] is True for h in hist[:-1]), \
                f"prior values of ({name}, {attr}) must be flagged superseded"
        if found:
            break
    assert found >= 1, (
        "expected at least one multi-entry supersession history across seeds "
        "0-7 (a MID-distance chain crossing a window boundary)"
    )
