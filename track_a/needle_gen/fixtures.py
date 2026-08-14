"""Inline fixture family + FactDB factory.

Lets every agent's tests run before authored W2 assets land (PLAN.md 10).
One minimal 2-paraphrase filler family; all content deterministic from seed.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from track_a.needle_gen.types import Entity, Fact, FactDB, FactType, Question, UncertaintyKind


@dataclass(frozen=True)
class FixtureFamily:
    """Inline filler template family (stand-in for the W2 asset families)."""

    family_id: str
    slots: tuple[str, ...]
    paraphrases: tuple[str, ...]

    def render(self, paraphrase_idx: int, slot_values: dict[str, str]) -> str:
        """Fill one paraphrase template with slot values (deterministic)."""
        text = self.paraphrases[paraphrase_idx % len(self.paraphrases)]
        for slot in self.slots:
            text = text.replace("{" + slot + "}", slot_values[slot])
        return text


FIXTURE_FAMILY = FixtureFamily(
    family_id="FIX-FILL-01",
    slots=("role", "project", "system", "depot", "date"),
    paraphrases=(
        "The {role} reviewed {project} alongside {system}. Notes about {depot} "
        "circulated before the sync on {date}. Follow-ups were assigned quietly.",
        "During the {date} sync the {role} discussed {project} and {system}. "
        "Updates about {depot} were recorded for the weekly digest.",
    ),
)

_POOLS: dict[str, tuple[str, ...]] = {
    "role": ("engineer", "planner", "auditor", "dispatcher"),
    "project": ("atlas", "borealis", "cascade", "driftline", "emberline", "falconridge"),
    "system": ("billing-core", "routing-engine", "ledger-svc", "manifest-api"),
    "depot": ("depot-north", "depot-south", "gateway-east", "gateway-west"),
    "date": ("2026-03-01", "2026-03-08", "2026-03-15", "2026-03-22"),
}

_ENTITY_TYPES = ("person", "project", "system", "depot")
_NAMES: dict[str, tuple[str, ...]] = {
    "person": ("Aria_Vale", "Bram_Cottle", "Cora_Drin", "Davi_Osk", "Elma_Rune", "Faro_Ines"),
    "project": ("atlas", "borealis", "cascade", "driftline", "emberline"),
    "system": ("billing-core", "routing-engine", "ledger-svc", "manifest-api"),
    "depot": ("depot-north", "depot-south", "gateway-east", "gateway-west"),
}


def make_scene_text_maker(
    seed: int, family: FixtureFamily = FIXTURE_FAMILY
) -> Callable[[int], str]:
    """Deterministic `scene_text(scene_idx) -> str` callable for the assembler."""
    rng = random.Random(seed)

    def scene_text(scene_idx: int) -> str:
        values = {slot: rng.choice(_POOLS[slot]) for slot in family.slots}
        return family.render(scene_idx, values)

    return scene_text


def make_fixture_factdb(
    seed: int, n_queried: int = 8, n_decoys: int = 4, with_chain: bool = True
) -> FactDB:
    """FactDB covering all six FactTypes plus decoys; deterministic from seed.

    Core queried set (7): a 3-value state chain (single state fact when
    with_chain=False), binding, exact, relational, negative, hedge, conflict.
    Extra queried facts are exact-value; decoys are true-but-unqueried facts.
    """
    if n_queried < 7:
        raise ValueError("fixture needs n_queried >= 7 to cover all fact types")
    rng = random.Random(seed)
    entities: list[Entity] = []
    facts: list[Fact] = []
    pos = 1

    def new_entity() -> Entity:
        etype = _ENTITY_TYPES[len(entities) % len(_ENTITY_TYPES)]
        entity = Entity(
            id=f"E{len(entities) + 1:04d}",
            type=etype,
            name=f"{_NAMES[etype][len(entities) % len(_NAMES[etype])]}_{len(entities) + 1:03d}",
        )
        entities.append(entity)
        return entity

    def new_fact(
        ftype: FactType, n_entities: int, n_values: int, queried: bool, **kwargs: object
    ) -> Fact:
        nonlocal pos
        fid = f"F{len(facts) + 1:04d}"
        eids = tuple(new_entity().id for _ in range(n_entities))
        positions = tuple(pos + i * 3 for i in range(n_values)) or (pos,)
        pos += max(n_values, 1) * 3 + rng.randint(0, 2)
        fact = Fact(
            id=fid,
            type=ftype,
            entity_ids=eids,
            attribute=None if ftype is FactType.RELATIONAL else f"attr_{fid}",
            values=tuple(f"v_{fid}_{i}" for i in range(n_values)),
            scene_positions=positions,
            relation=f"rel_{fid}" if ftype is FactType.RELATIONAL else None,
            is_queried=queried,
            **kwargs,
        )
        facts.append(fact)
        return fact

    if with_chain:
        new_fact(FactType.STATE_TRANSITION, 1, 3, True)
    else:
        new_fact(FactType.STATE_TRANSITION, 1, 1, True)
    new_fact(FactType.BINDING, 1, 1, True)
    new_fact(FactType.EXACT_VALUE, 1, 1, True)
    new_fact(FactType.RELATIONAL, 2, 0, True)
    new_fact(FactType.NEGATIVE, 1, 0, True)
    new_fact(FactType.UNCERTAINTY, 1, 1, True, uncertainty_kind=UncertaintyKind.HEDGE)
    new_fact(FactType.UNCERTAINTY, 1, 2, True, uncertainty_kind=UncertaintyKind.CONFLICT)
    for _ in range(n_queried - 7):
        new_fact(FactType.EXACT_VALUE, 1, 1, True)
    for i in range(n_decoys):
        dtype = (FactType.EXACT_VALUE, FactType.RELATIONAL, FactType.STATE_TRANSITION)[i % 3]
        new_fact(dtype, 2 if dtype is FactType.RELATIONAL else 1,
                 0 if dtype is FactType.RELATIONAL else 1, False)

    questions: list[Question] = []
    for fact in facts:
        if not fact.is_queried:
            continue
        if fact.uncertainty_kind is UncertaintyKind.CONFLICT:
            answer = "unresolved"
        elif fact.type is FactType.NEGATIVE:
            answer = "denied"
        else:
            answer = fact.values[-1] if fact.values else ""
        questions.append(
            Question(
                id=f"Q{len(questions) + 1:04d}",
                fact_ids=(fact.id,),
                text=f"What is {fact.attribute or fact.relation} for {fact.entity_ids[0]}?",
                answer=answer,
            )
        )
    return FactDB(
        doc_id=f"fixture-{seed}",
        entities=tuple(entities),
        facts=tuple(facts),
        questions=tuple(questions),
    )
