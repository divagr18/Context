"""T3B: full bank-driven FactDB generation invariants (S1/S7 enablers)."""
from __future__ import annotations

import pytest

from track_a.needle_gen.core.docgen import build_document
from track_a.needle_gen.types import (
    DOC_LEN_TARGETS, CueKind, DistanceBucket, Domain, FactType, GenConfig,
    Split, UncertaintyKind, canonicalize_value,
)
from track_a.tokenize import get_tokenizer


def make_config(seed: int = 7, split: Split = Split.TRAIN,
                domain: Domain = Domain.PROJECT_UPDATES,
                doc_len: str = "short", **overrides) -> GenConfig:
    kwargs = dict(seed=seed, split=split, domain=domain, doc_len_name=doc_len,
                  n_docs=1, family_ids=(), paraphrase_idx_range=(0, 5))
    kwargs.update(overrides)
    return GenConfig(**kwargs)


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


@pytest.fixture(scope="module")
def doc(tok):
    return build_document(make_config(seed=7), 0, tok)


@pytest.fixture(scope="module")
def doc_b(tok):
    return build_document(make_config(seed=7), 1, tok)


@pytest.fixture(scope="module")
def medium_doc(tok):
    return build_document(make_config(seed=11, doc_len="medium"), 0, tok)


def test_deterministic_same_seed(tok):
    a = build_document(make_config(seed=99), 0, tok)
    b = build_document(make_config(seed=99), 0, tok)
    assert a.doc_ids == b.doc_ids
    assert [f.id for f in a.fact_db.facts] == [f.id for f in b.fact_db.facts]
    assert [f.values for f in a.fact_db.facts] == [f.values for f in b.fact_db.facts]


def test_different_doc_indices_differ(doc, doc_b):
    assert doc.doc_ids != doc_b.doc_ids


def test_all_six_fact_types_present(doc):
    types = {f.type for f in doc.fact_db.facts}
    assert types == set(FactType)


def test_guaranteed_queried_coverage(doc):
    queried_types = {f.type for f in doc.fact_db.facts if f.is_queried}
    assert queried_types == set(FactType)


def test_decoys_present(doc):
    n_queried = len(doc.fact_db.queried_facts())
    n_decoys = len(doc.fact_db.decoy_facts())
    assert n_queried >= 8
    assert n_decoys >= n_queried


def test_fresh_entity_registry_per_doc(doc, doc_b):
    ids_a = [e.id for e in doc.fact_db.entities]
    ids_b = [e.id for e in doc_b.fact_db.entities]
    assert ids_a and ids_b
    assert len(set(ids_a)) == len(ids_a)
    assert len(set(ids_b)) == len(ids_b)
    assert ids_a[0] == "E0001" and ids_b[0] == "E0001"
    assert ids_a == [f"E{i + 1:04d}" for i in range(len(ids_a))]
    names_a = [e.name for e in doc.fact_db.entities]
    assert len(set(names_a)) == len(names_a)


def test_entity_names_and_fact_values_match_value_pattern(doc):
    from track_a.needle_gen.types import VALUE_PATTERN

    for e in doc.fact_db.entities:
        assert VALUE_PATTERN.match(e.name)
    for f in doc.fact_db.facts:
        for v in f.values:
            assert VALUE_PATTERN.match(v)


def test_chain_structure(doc):
    chains = [f for f in doc.fact_db.facts if f.is_chain]
    assert chains, "expected at least one state-transition chain"
    for c in chains:
        assert c.type is FactType.STATE_TRANSITION
        assert len(c.values) >= 2
        assert len(c.values) <= 3
        assert len(c.values) == len(c.scene_positions)
        assert len(set(c.values)) == len(c.values), "chain values must be distinct"
        assert list(c.scene_positions) == sorted(c.scene_positions)


def test_conflict_has_two_values(doc):
    conflicts = [f for f in doc.fact_db.facts
                 if f.uncertainty_kind is UncertaintyKind.CONFLICT]
    assert conflicts
    for c in conflicts:
        assert len(c.values) == 2
        assert c.values[0] != c.values[1]


def test_binding_distance_bucket_consistent(doc):
    from track_a.needle_gen.types import distance_bucket_for

    bindings = [f for f in doc.fact_db.facts if f.type is FactType.BINDING]
    assert bindings
    bounds = doc.fact_db.scene_boundaries
    for b in bindings:
        assert b.distance_bucket is not None
        intro_scene, ref_scene = b.scene_positions[0], b.scene_positions[1]
        assert ref_scene > intro_scene
        measured = bounds[ref_scene] - bounds[intro_scene]
        assert distance_bucket_for(measured) is b.distance_bucket


def test_medium_doc_binding_buckets_valid(medium_doc):
    from track_a.needle_gen.types import distance_bucket_for

    bounds = medium_doc.fact_db.scene_boundaries
    for b in (f for f in medium_doc.fact_db.facts if f.type is FactType.BINDING):
        intro_scene, ref_scene = b.scene_positions[0], b.scene_positions[1]
        measured = bounds[ref_scene] - bounds[intro_scene]
        assert distance_bucket_for(measured) is b.distance_bucket


def test_scene_positions_valid_range(doc):
    n_scenes = len(doc.fact_db.scene_boundaries)
    for f in doc.fact_db.facts:
        for pos in f.scene_positions:
            assert 0 <= pos < n_scenes


def test_doc_length_within_tolerance(doc):
    target = DOC_LEN_TARGETS["short"]
    assert doc.fact_db.doc_len_tokens >= target
    assert doc.fact_db.doc_len_tokens <= int(target * 1.30)
    assert len(doc.doc_ids) == doc.fact_db.doc_len_tokens


def test_queried_density(doc):
    target = DOC_LEN_TARGETS["short"]
    expected = target * 1.0 / 100
    n_queried = len(doc.fact_db.queried_facts())
    assert n_queried >= 8
    assert n_queried <= expected * 2.5


def test_mentions_decode_to_entity(doc, tok):
    assert doc.fact_db.mentions
    entity_by_id = {e.id: e for e in doc.fact_db.entities}
    for m in doc.fact_db.mentions:
        entity = entity_by_id[m.entity_id]
        text = tok.decode(doc.doc_ids[m.start_tok:m.end_tok]).strip()
        canon = canonicalize_value(text)
        assert canon == entity.name or canon in entity.aliases
        assert m.start_tok < m.end_tok


def test_value_spans_decode_to_fact_value(doc, tok):
    assert doc.fact_db.value_spans
    fact_by_id = {f.id: f for f in doc.fact_db.facts}
    for vs in doc.fact_db.value_spans:
        fact = fact_by_id[vs.fact_id]
        text = tok.decode(doc.doc_ids[vs.start_tok:vs.end_tok]).strip()
        assert text == fact.values[vs.value_idx]
        assert vs.start_tok < vs.end_tok


def test_cue_spans_cover_negation_and_hedge(doc):
    kinds = {c.kind for c in doc.fact_db.cue_spans}
    n_neg_facts = sum(1 for f in doc.fact_db.facts if f.type is FactType.NEGATIVE)
    n_hedge = sum(1 for f in doc.fact_db.facts
                  if f.uncertainty_kind is UncertaintyKind.HEDGE)
    assert CueKind.NEGATION in kinds and n_neg_facts > 0
    assert CueKind.HEDGE in kinds and n_hedge > 0
    n_neg_cues = sum(1 for c in doc.fact_db.cue_spans if c.kind is CueKind.NEGATION)
    assert n_neg_cues == n_neg_facts


def test_ood_domain_builds(tok):
    ood = build_document(make_config(seed=7, domain=Domain.LOGISTICS_OPS), 0, tok)
    types = {f.type for f in ood.fact_db.facts}
    assert types == set(FactType)
    assert ood.fact_db.doc_len_tokens >= DOC_LEN_TARGETS["short"]
