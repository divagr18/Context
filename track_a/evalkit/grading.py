"""Compact fact grading: compare a parsed C against ground-truth FactDB.

Focuses on exact-value facts (the primary needle) at the (name, attr, value)
level. Relational / negative / uncertainty facts are graded by the full eval
battery (T8); recall-shaping (T7) uses this exact-value proxy. Names are
canonicalised (spaces -> underscores) to match the render/parse grammar.
"""

from __future__ import annotations

from track_a.needle_gen.types import FactType, canonicalize_value
from track_a.parse_compact import EntityRec, FactRec, parse_target_lines

_VALUE_FACT_TYPES = (FactType.EXACT_VALUE, FactType.BINDING,
                     FactType.STATE_TRANSITION)


def parse_c_text(text: str):
    """Parse rendered C text into (records, failures)."""
    return parse_target_lines(text.split("\n"))


def eid_name_map(records) -> dict[str, str]:
    """Map entity id -> canonical name from ENTITY records."""
    m: dict[str, str] = {}
    for r in records:
        if isinstance(r, EntityRec):
            m[r.eid] = canonicalize_value(r.name)
    return m


def fact_triples_from_records(records) -> set[tuple[str, str, str]]:
    """(name, attr, value) triples recovered from FACT records."""
    names = eid_name_map(records)
    triples: set[tuple[str, str, str]] = set()
    for r in records:
        if not isinstance(r, FactRec):
            continue
        name = names.get(r.eid)
        if name is None:
            continue  # unbound eid -> dropped, not recoverable
        triples.add((name, r.attr, r.value))
    return triples


def _fact_db_triples(fact_db, want_queried: bool) -> set[tuple[str, str, str]]:
    name_by_eid = {e.id: canonicalize_value(e.name) for e in fact_db.entities}
    triples: set[tuple[str, str, str]] = set()
    for f in fact_db.facts:
        if f.is_queried != want_queried:
            continue
        if f.type not in _VALUE_FACT_TYPES:
            continue
        if not f.values:
            continue
        name = name_by_eid.get(f.entity_ids[0])
        if name is None:
            continue
        triples.add((name, f.attribute or "", canonicalize_value(f.values[-1])))
    return triples


def ground_truth_triples(fact_db) -> set[tuple[str, str, str]]:
    """(name, attr, value) triples for QUERIED value facts."""
    return _fact_db_triples(fact_db, want_queried=True)


def decoy_triples(fact_db) -> set[tuple[str, str, str]]:
    """(name, attr, value) triples for DECOY value facts (tolerated)."""
    return _fact_db_triples(fact_db, want_queried=False)


def recall_and_hallucination(parsed_triples, gt_triples, decoy_triples):
    """Return (recall, hallucination_rate).

    recall = |parsed ∩ gt| / |gt| (1.0 if gt empty).
    hallucination_rate = |parsed − gt − decoy| / max(1, |parsed|).
    """
    if gt_triples:
        recall = len(parsed_triples & gt_triples) / len(gt_triples)
    else:
        recall = 1.0
    hallucinated = parsed_triples - gt_triples - decoy_triples
    hall_rate = len(hallucinated) / max(1, len(parsed_triples))
    return recall, hall_rate
