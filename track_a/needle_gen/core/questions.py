"""Question generation from grounded facts (PLAN T3B / questions.json banks)."""
from __future__ import annotations

import random
import re

from track_a.needle_gen.assets_loader import QUESTION_ANSWERS, SLOT_RE
from track_a.needle_gen.core.entities import EntitySpec
from track_a.needle_gen.types import Fact, FactType, Question, UncertaintyKind


def _fill(template: str, slots: dict[str, str], ctx: str) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key not in slots:
            raise KeyError(f"{ctx}: question template slot {key!r} missing")
        return slots[key]

    return re.sub(SLOT_RE.pattern, repl, template)


def _pick_phrasings(rng: random.Random, bank: list[str], n: int = 2) -> list[int]:
    """Distinct phrasing indices into the bank (>=2 phrasings per fact)."""
    k = min(n, len(bank))
    return sorted(rng.sample(range(len(bank)), k))


def _question(qid: int, fact_ids: tuple[str, ...], text: str, answer: str,
              phrasing_idx: int, is_multihop: bool = False,
              probes_prior: bool = False) -> Question:
    return Question(
        id=qid, fact_ids=fact_ids, text=text, answer=answer,
        is_multihop=is_multihop, probes_prior_value=probes_prior,
        phrasing_idx=phrasing_idx,
    )


def questions_for_fact(rng: random.Random, bank: dict, fact: Fact,
                       entity_map: dict[str, EntitySpec],
                       ) -> list[tuple[str, str, str, bool, bool, int]]:
    """Emit (question_key, text, answer, is_multihop, probes_prior, phrasing_idx).

    Two distinct phrasings per question role; chains add prior-value probes
    (always for >=3 values, ~50% for 2-value chains).
    """
    out: list[tuple[str, str, str, bool, bool, int]] = []
    e_surf = entity_map[fact.entity_ids[0]].surface

    def emit(key: str, slots: dict[str, str], probes_prior: bool = False) -> None:
        answer = _render_answer(key, fact, entity_map, slots)
        for idx in _pick_phrasings(rng, bank[key]):
            text = _fill(bank[key][idx], slots, key)
            out.append((key, text, answer, False, probes_prior, idx))

    if fact.type is FactType.EXACT_VALUE or fact.type is FactType.BINDING:
        key = "exact_value" if fact.type is FactType.EXACT_VALUE else "binding"
        emit(key, {"entity": e_surf, "attribute": fact.attribute or "",
                   "value": fact.values[-1] if fact.values else "",
                   "entity_type": entity_map[fact.entity_ids[0]].type})
    elif fact.type is FactType.RELATIONAL:
        e_b = entity_map[fact.entity_ids[1]].surface
        emit("relational", {"entity_a": e_surf, "entity_b": e_b,
                            "relation": fact.relation or ""})
    elif fact.type is FactType.STATE_TRANSITION:
        cur = fact.values[-1] if fact.values else ""
        emit("state_transition_current",
             {"entity": e_surf, "attribute": fact.attribute or "", "value": cur})
        if len(fact.values) >= 2:
            if len(fact.values) >= 3 or rng.random() < 0.5:
                emit("state_transition_prior",
                     {"entity": e_surf, "attribute": fact.attribute or "",
                      "old_value": fact.values[-2], "value": cur},
                     probes_prior=True)
    elif fact.type is FactType.NEGATIVE:
        emit("negative", {"entity": e_surf, "attribute": fact.attribute or ""})
    elif fact.type is FactType.UNCERTAINTY:
        if fact.uncertainty_kind is UncertaintyKind.CONFLICT:
            emit("uncertainty_conflict",
                 {"entity": e_surf, "attribute": fact.attribute or ""})
        else:
            emit("uncertainty_hedge",
                 {"entity": e_surf, "attribute": fact.attribute or "",
                  "value": fact.values[0] if fact.values else ""})
    return out


def _render_answer(key: str, fact: Fact, entity_map: dict[str, EntitySpec],
                   slots: dict[str, str]) -> str:
    template = QUESTION_ANSWERS[key]
    if key == "relational":
        return entity_map[fact.entity_ids[0]].canonical
    return _fill(template, slots, f"answer:{key}")


def multihop_questions(rng: random.Random, bank: dict, rel_fact: Fact,
                       other_fact: Fact, entity_map: dict[str, EntitySpec],
                       ) -> list[tuple[str, str, str, bool, bool, int]]:
    """Compositional questions over a relational fact + a fact on entity_b."""
    owner = entity_map[rel_fact.entity_ids[0]].canonical
    out: list[tuple[str, str, str, bool, bool, int]] = []
    if other_fact.is_chain and len(other_fact.values) >= 2:
        key = "multihop_superseded_owner"
        slots = {"entity": entity_map[other_fact.entity_ids[0]].surface,
                 "attribute": other_fact.attribute or "",
                 "old_value": other_fact.values[0]}
    else:
        key = "multihop_owns_attr"
        slots = {"entity": entity_map[other_fact.entity_ids[0]].surface,
                 "attribute": other_fact.attribute or "",
                 "value": other_fact.values[-1] if other_fact.values else ""}
    for idx in _pick_phrasings(rng, bank[key], n=1):
        text = _fill(bank[key][idx], slots, key)
        out.append((key, text, owner, True, False, idx))
    return out
