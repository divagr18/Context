"""T-INT / S1: generator -> canonical render -> parser round-trip.

Asserts, across many generated documents (both domains):
  * generator output parses with ZERO ParseFailures at every budget,
  * at full budget the parsed records match ground-truth facts exactly
    (entities, attrs, values, chains incl. supersedes, hedged flags,
    negatives, conflicts, relations),
  * budget renders are nested (longest-prefix-no-skip property).
"""
from __future__ import annotations

import pytest

from track_a.needle_gen.core.docgen import build_document
from track_a.needle_gen.generate import canonical_order_and_render
from track_a.needle_gen.types import (
    Domain, FactDB, FactType, GenConfig, Split, UncertaintyKind,
    canonicalize_value,
)
from track_a.parse_compact import (
    EntityRec, FactRec, NegRec, RelRec, UnresolvedRec, parse_target_lines,
)
from track_a.tokenize import get_tokenizer

FULL_BUDGET = 10**6


def make_config(seed: int = 7, domain: Domain = Domain.PROJECT_UPDATES,
                doc_len: str = "short", **overrides) -> GenConfig:
    kwargs = dict(seed=seed, split=Split.TRAIN, domain=domain,
                  doc_len_name=doc_len, n_docs=1, family_ids=(),
                  paraphrase_idx_range=(0, 5))
    kwargs.update(overrides)
    return GenConfig(**kwargs)


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


@pytest.fixture(scope="module")
def built_docs(tok):
    docs = []
    for seed in range(6):
        docs.append(build_document(make_config(seed=seed), 0, tok).fact_db)
    for seed in range(2):  # OOD domain documents
        docs.append(
            build_document(make_config(seed=seed, domain=Domain.LOGISTICS_OPS),
                           0, tok).fact_db)
    return docs


def _rec_key(r) -> tuple:
    if isinstance(r, EntityRec):
        return ("ENTITY", r.eid, r.type, r.name)
    if isinstance(r, FactRec):
        return ("FACT", r.eid, r.attr, r.value, r.pos, r.supersedes, r.hedged)
    if isinstance(r, RelRec):
        return ("REL", r.eid_a, r.rel, r.eid_b, r.pos)
    if isinstance(r, NegRec):
        return ("NEG", r.eid, r.attr, r.pos)
    if isinstance(r, UnresolvedRec):
        return ("UNRESOLVED", r.eid, r.attr, tuple(sorted(r.candidates)))
    raise AssertionError(f"unexpected record type {type(r)}")


def _expected_records(db: FactDB) -> list:
    """Ground-truth records, built straight from the FactDB (grammar C2)."""
    recs: list = []
    eids = sorted({eid for f in db.facts for eid in f.entity_ids})
    for eid in eids:
        e = db.entity_by_id(eid)
        recs.append(EntityRec(eid, e.type, canonicalize_value(e.name)))
    for f in db.facts:
        eid = f.entity_ids[0]
        attr = f.attribute or ""
        pos0 = f.scene_positions[0] if f.scene_positions else 0
        if f.type is FactType.RELATIONAL:
            recs.append(RelRec(eid, f.relation, f.entity_ids[1], pos0))
        elif f.type is FactType.NEGATIVE:
            recs.append(NegRec(eid, attr, pos0))
        elif f.type is FactType.UNCERTAINTY:
            if f.uncertainty_kind is UncertaintyKind.CONFLICT:
                cands = tuple((canonicalize_value(v), p)
                              for v, p in zip(f.values, f.scene_positions))
                recs.append(UnresolvedRec(eid, attr, cands))
            else:
                recs.append(FactRec(eid, attr, canonicalize_value(f.values[0]),
                                    pos0, None, True))
        elif f.is_chain:
            vals = [canonicalize_value(v) for v in f.values]
            poses = f.scene_positions
            recs.append(FactRec(eid, attr, vals[-1], poses[-1], vals[0], False))
            for i in range(len(vals) - 2, 0, -1):
                recs.append(FactRec(eid, attr, vals[i], poses[i],
                                    vals[i - 1], False))
        else:
            recs.append(FactRec(eid, attr, canonicalize_value(f.values[-1]),
                                pos0, None, False))
    return recs


def test_generator_output_parses_with_zero_failures(built_docs, tok):
    total_lines = 0
    for db in built_docs:
        text = canonical_order_and_render(db, FULL_BUDGET, tok)
        records, failures = parse_target_lines(text.split("\n"))
        assert failures == [], f"doc {db.doc_id}: {failures[:3]}"
        assert records, f"doc {db.doc_id}: empty render"
        total_lines += len(records)
    assert total_lines > 500  # sanity: real volume across 8 docs


def test_full_budget_round_trip_matches_ground_truth(built_docs, tok):
    for db in built_docs:
        text = canonical_order_and_render(db, FULL_BUDGET, tok)
        records, failures = parse_target_lines(text.split("\n"))
        assert failures == []
        parsed = sorted(map(_rec_key, records))
        expected = sorted(map(_rec_key, _expected_records(db)))
        assert parsed == expected, f"doc {db.doc_id}"


def test_chain_supersession_recovered(built_docs, tok):
    """Every ground-truth chain's final line carries supersedes=<original>."""
    checked = 0
    for db in built_docs:
        text = canonical_order_and_render(db, FULL_BUDGET, tok)
        records, _ = parse_target_lines(text.split("\n"))
        fact_recs = {}
        for r in records:
            if isinstance(r, FactRec):
                fact_recs.setdefault((r.eid, r.attr), []).append(r)
        for f in db.facts:
            if not f.is_chain:
                continue
            recs = fact_recs[(f.entity_ids[0], f.attribute)]
            final = max(recs, key=lambda r: r.pos)
            assert final.value == canonicalize_value(f.values[-1])
            assert final.supersedes == canonicalize_value(f.values[0])
            assert len(recs) == len(f.values) - 1
            checked += 1
    assert checked >= 8, "need chains across the generated docs"


def test_hedged_and_conflict_flags_survive(built_docs, tok):
    for db in built_docs:
        text = canonical_order_and_render(db, FULL_BUDGET, tok)
        records, _ = parse_target_lines(text.split("\n"))
        hedged = {(r.eid, r.attr) for r in records
                  if isinstance(r, FactRec) and r.hedged}
        conflicts = {(r.eid, r.attr) for r in records
                     if isinstance(r, UnresolvedRec)}
        for f in db.facts:
            key = (f.entity_ids[0], f.attribute)
            if f.uncertainty_kind is UncertaintyKind.HEDGE:
                assert key in hedged
            elif f.uncertainty_kind is UncertaintyKind.CONFLICT:
                assert key in conflicts


def test_budget_renders_are_nested_prefixes(built_docs, tok):
    budgets = (64, 256, 1024, FULL_BUDGET)
    for db in built_docs:
        line_sets = []
        for budget in budgets:
            text = canonical_order_and_render(db, budget, tok)
            records, failures = parse_target_lines(text.split("\n")) \
                if text else ([], [])
            assert failures == []
            line_sets.append({str(k) for k in map(_rec_key, records)})
        for smaller, larger in zip(line_sets, line_sets[1:]):
            assert smaller <= larger, f"doc {db.doc_id}: prefix property"


def test_empty_render_parses_to_nothing(tok):
    db = build_document(make_config(seed=3), 0, tok).fact_db
    text = canonical_order_and_render(db, 1, tok)  # budget too small for anything
    if text == "":
        return  # nothing to parse; guarded path in canonical_order_and_render
    records, failures = parse_target_lines(text.split("\n"))
    assert failures == []
    assert records == []
