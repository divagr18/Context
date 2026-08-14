"""T3B: question-generation invariants (phrasings, answers, prior-value, multihop)."""
from __future__ import annotations

import re

import pytest

from track_a.needle_gen.core.docgen import build_document
from track_a.needle_gen.types import (
    Domain, GenConfig, Split, FactType, UncertaintyKind,
)
from track_a.tokenize import get_tokenizer

SLOT_RE = re.compile(r"\{(\w+)\}")


def make_config(seed: int = 7, doc_len: str = "short", **overrides) -> GenConfig:
    kwargs = dict(seed=seed, split=Split.TRAIN, domain=Domain.PROJECT_UPDATES,
                  doc_len_name=doc_len, n_docs=1, family_ids=(),
                  paraphrase_idx_range=(0, 5))
    kwargs.update(overrides)
    return GenConfig(**kwargs)


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


@pytest.fixture(scope="module")
def doc(tok):
    return build_document(make_config(seed=7), 0, tok)


@pytest.fixture(scope="module")
def forced_mh_docs(tok):
    return [build_document(make_config(seed=5, multihop_doc_fraction=1.0), i, tok)
            for i in range(6)]


def _fact_map(doc):
    return {f.id: f for f in doc.fact_db.facts}


def test_question_ids_unique(doc):
    ids = [q.id for q in doc.fact_db.questions]
    assert len(ids) == len(set(ids))


def test_fact_ids_reference_existing_queried_facts(doc):
    fmap = _fact_map(doc)
    for q in doc.fact_db.questions:
        assert q.fact_ids
        for fid in q.fact_ids:
            assert fid in fmap
            assert fmap[fid].is_queried


def test_no_unfilled_slots(doc):
    for q in doc.fact_db.questions:
        assert not SLOT_RE.search(q.text), q.text


def test_every_queried_fact_has_two_phrasings(doc):
    from collections import defaultdict

    counts = defaultdict(set)
    for q in doc.fact_db.questions:
        if q.is_multihop or q.probes_prior_value:
            continue
        counts[q.fact_ids[0]].add(q.phrasing_idx)
    for fid, fact in _fact_map(doc).items():
        if fact.is_queried:
            assert len(counts[fid]) >= 2, (fid, counts[fid])


def test_answers_match_fact_type(doc):
    fmap = _fact_map(doc)
    entity_by_id = {e.id: e for e in doc.fact_db.entities}
    for q in doc.fact_db.questions:
        if q.is_multihop:
            continue
        fact = fmap[q.fact_ids[0]]
        if fact.type is FactType.NEGATIVE:
            assert q.answer == "denied"
        elif fact.uncertainty_kind is UncertaintyKind.CONFLICT:
            assert q.answer == "unresolved"
        elif fact.uncertainty_kind is UncertaintyKind.HEDGE:
            assert q.answer == f"unconfirmed:{fact.values[0]}"
        elif fact.type is FactType.RELATIONAL:
            assert q.answer == entity_by_id[fact.entity_ids[0]].name
        elif q.probes_prior_value:
            assert q.answer == fact.values[-2]
        elif fact.type is FactType.STATE_TRANSITION:
            assert q.answer == fact.values[-1]
        else:  # EXACT_VALUE or BINDING
            assert q.answer in fact.values


def test_deep_chains_always_have_prior_question(doc):
    fmap = _fact_map(doc)
    prior_fact_ids = {q.fact_ids[0] for q in doc.fact_db.questions
                      if q.probes_prior_value}
    for fact in doc.fact_db.facts:
        if fact.is_chain and len(fact.values) >= 3 and fact.is_queried:
            assert fact.id in prior_fact_ids


def test_depth2_chain_prior_fraction_statistical(tok):
    with_prior = 0
    total = 0
    for i in range(12):
        d = build_document(make_config(seed=i), 0, tok)
        prior_ids = {q.fact_ids[0] for q in d.fact_db.questions
                     if q.probes_prior_value}
        for f in d.fact_db.facts:
            if f.is_chain and len(f.values) == 2 and f.is_queried:
                total += 1
                if f.id in prior_ids:
                    with_prior += 1
    assert total >= 4
    ratio = with_prior / total
    assert 0.1 <= ratio <= 0.9, ratio


def test_forced_multihop_documents(forced_mh_docs):
    entity_ok = 0
    for d in forced_mh_docs:
        fmap = _fact_map(d)
        entity_by_id = {e.id: e for e in d.fact_db.entities}
        mqs = [q for q in d.fact_db.questions if q.is_multihop]
        assert mqs, "forced multihop doc missing multihop questions"
        for q in mqs:
            assert len(q.fact_ids) == 2
            rel_fact = fmap[q.fact_ids[0]]
            assert rel_fact.type is FactType.RELATIONAL
            assert q.answer == entity_by_id[rel_fact.entity_ids[0]].name
            entity_ok += 1
    assert entity_ok >= 6


def test_natural_multihop_fraction(tok):
    n_mh_docs = 0
    for i in range(40):
        d = build_document(make_config(seed=i), 0, tok)
        if any(q.is_multihop for q in d.fact_db.questions):
            n_mh_docs += 1
    ratio = n_mh_docs / 40
    assert 0.02 <= ratio <= 0.5, ratio
