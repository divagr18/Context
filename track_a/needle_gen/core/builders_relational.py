"""Builder for relational facts (two entities)."""
from __future__ import annotations

import random

from track_a.needle_gen.core.entities import EntitySpec
from track_a.needle_gen.core.factspec import FactSpec, SceneSpec
from track_a.needle_gen.types import FactType

PERSON_RELATIONS = {"manages", "reports_to", "delegates_to"}


def build_relational(rng: random.Random, pools: dict, entity_a: EntitySpec,
                     entity_b: EntitySpec, queried: bool,
                     relation_hint: str | None = None) -> FactSpec:
    fam = rng.choice(["REL-01", "REL-02"])
    pool_rels = pools["relations"][fam]
    both_person = entity_a.type == "person" and entity_b.type == "person"
    if both_person:
        candidates = [r for r in pool_rels if r in PERSON_RELATIONS]
    else:
        candidates = [r for r in pool_rels if r not in PERSON_RELATIONS]
    if not candidates:
        candidates = list(pool_rels)
    if relation_hint is not None and relation_hint in candidates:
        relation = relation_hint
    else:
        relation = rng.choice(candidates)
    scene = SceneSpec(
        family_id=fam,
        slot_values={"entity_a": entity_a.surface, "relation": relation,
                     "entity_b": entity_b.surface},
        entity_slot_map={"entity_a": entity_a.eid, "entity_b": entity_b.eid},
        value_slot_map={},
    )
    return FactSpec(
        type=FactType.RELATIONAL, entity_ids=(entity_a.eid, entity_b.eid),
        attribute=None, values=(), relation=relation, uncertainty_kind=None,
        is_queried=queried, distance_bucket=None, scenes=[scene],
        value_scene_idx={},
    )
