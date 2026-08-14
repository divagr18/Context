"""T3C: streaming ground-truth invariants (S7 enabler).

Core property (S7): folding ALL window edit-ops from an empty OpLog must
reproduce the final budget-truncated canonical state, and SUPERSEDE must keep
prior values in history. Also checks op-parser cleanliness and state/budget
consistency.
"""
from __future__ import annotations

import pytest

from track_a.needle_gen.core.docgen import build_document
from track_a.needle_gen.streaming import (
    StreamState, apply_event, build_streaming_windows, extract_events,
    render_state_lines, truncate_state,
)
from track_a.needle_gen.types import (
    Domain, GenConfig, Split, canonicalize_value,
)
from track_a.parse_compact import OpLog, parse_ops_lines
from track_a.tokenize import get_tokenizer

BUDGET = 600
WINDOW_TOKENS = 512
OVERLAP = 0.10


def make_config(seed: int = 3) -> GenConfig:
    return GenConfig(seed=seed, split=Split.TRAIN, domain=Domain.PROJECT_UPDATES,
                     doc_len_name="short", n_docs=1, family_ids=(),
                     paraphrase_idx_range=(0, 5))


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


@pytest.fixture(scope="module")
def built(tok):
    return build_document(make_config(), 0, tok)


@pytest.fixture(scope="module")
def entity_map(built):
    return {e.id: e for e in built.fact_db.entities}


@pytest.fixture(scope="module")
def windows(built, entity_map, tok):
    return build_streaming_windows(
        built.doc_ids, built.fact_db.facts, built.fact_db.scene_boundaries,
        entity_map, BUDGET, WINDOW_TOKENS, OVERLAP, tok)


def _expected_final_state(built, entity_map, tok):
    events = extract_events(built.fact_db.facts)
    full = StreamState()
    for ev in events:
        apply_event(full, ev)
    return truncate_state(full, BUDGET, entity_map, tok)


def test_multiple_windows(built, tok):
    from track_a.needle_gen.streaming import split_windows
    wins = split_windows(len(built.doc_ids), WINDOW_TOKENS, OVERLAP)
    assert len(wins) >= 3, "short doc at 512-token windows should span >=3 windows"


def test_ops_parse_cleanly(windows):
    total_ops = 0
    for w in windows:
        if not w.ops_text:
            continue
        _ops, failures = parse_ops_lines(w.ops_text.split("\n"))
        assert failures == [], f"malformed ops: {failures}"
        total_ops += len(_ops)
    assert total_ops > 0


def test_window_state_matches_pre_state(windows, entity_map, built, tok):
    # Reconstruct running state and confirm each window's state_ids render the
    # budget-truncated state that existed BEFORE reading that window.
    events = extract_events(built.fact_db.facts)
    from track_a.needle_gen.streaming import scene_cutoff, split_windows
    win_ranges = split_windows(len(built.doc_ids), WINDOW_TOKENS, OVERLAP)
    full = StreamState()
    ptr = 0
    prev_trunc = StreamState()
    for (ws, we), w in zip(win_ranges, windows):
        cutoff = scene_cutoff(built.fact_db.scene_boundaries, we)
        while ptr < len(events) and events[ptr].scene_idx < cutoff:
            apply_event(full, events[ptr])
            ptr += 1
        expect_text = "\n".join(render_state_lines(prev_trunc, entity_map))
        got_text = tok.decode(list(w.state_ids)) if w.state_ids else ""
        assert got_text == expect_text
        curr = truncate_state(full, BUDGET, entity_map, tok)
        prev_trunc = curr


def test_fold_ops_reproduces_final_state(windows, built, entity_map, tok):
    """S7: OpLog.fold over all ops == final budget-truncated canonical state."""
    log = OpLog()
    for w in windows:
        if w.ops_text:
            failures = log.append(w.ops_text.split("\n"))
            assert failures == []
    folded = log.fold()
    expected = _expected_final_state(built, entity_map, tok)

    def name(eid):
        return entity_map[eid].name

    # Current facts keyed by resolved entity name.
    expect_facts = {}
    for (eid, attr), (value, pos, _r, _h) in expected.facts.items():
        expect_facts[(name(eid), attr)] = (canonicalize_value(value), pos)
    assert dict(folded.current) == expect_facts

    expect_rels = {(name(a), rel, name(b)) for (a, rel, b) in expected.relations}
    assert folded.relations == expect_rels

    expect_negs = {(name(eid), attr) for (eid, attr) in expected.negations}
    assert folded.negations == expect_negs

    expect_unres = {}
    for (eid, attr), (cands, _r) in expected.unresolved.items():
        expect_unres[(name(eid), attr)] = tuple(
            (canonicalize_value(v), p) for v, p in cands)
    assert dict(folded.unresolved) == expect_unres


def test_supersede_keeps_prior_in_history(entity_map):
    """SUPERSEDE across states retains the prior value flagged superseded.

    Uses a direct three-state sequence (empty -> v0 -> v1) so the check does
    not depend on whether a docgen chain happens to span window boundaries
    (within a single window transitions net to one UPSERT by design).
    """
    from track_a.needle_gen.streaming import StreamState, diff_states

    eid = next(iter(sorted(entity_map)))
    attr = "port"
    empty = StreamState()
    s0 = StreamState()
    s0.facts[(eid, attr)] = ["8080", 5, 0, False]
    s0.entities_seen.add(eid)
    s1 = StreamState()
    s1.facts[(eid, attr)] = ["9090", 12, 0, False]
    s1.entities_seen.add(eid)

    ops_a = diff_states(empty, s0, entity_map)
    ops_b = diff_states(s0, s1, entity_map)
    assert any(l.startswith("UPSERT FACT") for l in ops_a)
    assert any(l.startswith("SUPERSEDE FACT") for l in ops_b)

    log = OpLog()
    assert log.append(ops_a) == []
    assert log.append(ops_b) == []
    folded = log.fold()
    name = entity_map[eid].name
    hist = folded.history(name, attr)
    assert [h[0] for h in hist] == ["8080", "9090"]
    assert hist[0][2] is True and hist[1][2] is False
    assert folded.current[(name, attr)] == ("9090", 12)


def test_state_render_within_budget(windows, tok):
    for w in windows:
        if not w.state_ids:
            continue
        assert len(w.state_ids) <= BUDGET, (len(w.state_ids), BUDGET)
