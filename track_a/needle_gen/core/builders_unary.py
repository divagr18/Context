"""Builders for unary facts: exact_value, negative, hedge, conflict, binding."""
from __future__ import annotations

import random

from track_a.needle_gen.core.entities import EntitySpec
from track_a.needle_gen.core.factspec import FactSpec, SceneSpec
from track_a.needle_gen.types import (
    CueKind, DistanceBucket, FactType, UncertaintyKind,
)


def _pick_attr_values(rng: random.Random, pools: dict, n: int) -> tuple[str, str, list[str]]:
    """Choose one attribute and *n* DISTINCT values from that attribute's pool.

    Keeping all values in a single pool is what makes chains/conflicts coherent
    (a port chain stays ports; a conflict compares two plausible ports).
    """
    fam = rng.choice(["EXACT-01", "EXACT-02"])
    attrs = pools["attributes_by_family"][fam]
    attr = rng.choice(sorted(attrs.keys()))
    pool_vals = attrs[attr]["values"]
    k = min(n, len(pool_vals))
    vals = rng.sample(pool_vals, k)
    while len(vals) < n:
        vals.append(vals[0] + f"-v{len(vals)}")
    return fam, attr, vals


def _pick_attr_value(rng: random.Random, pools: dict) -> tuple[str, str, str]:
    fam, attr, vals = _pick_attr_values(rng, pools, 1)
    return fam, attr, vals[0]


def build_exact(rng: random.Random, pools: dict, entity: EntitySpec,
                queried: bool) -> FactSpec:
    fam, attr, value = _pick_attr_value(rng, pools)
    scene = SceneSpec(
        family_id=fam,
        slot_values={"entity": entity.surface, "attribute": attr, "value": value},
        entity_slot_map={"entity": entity.eid},
        value_slot_map={"value": 0},
    )
    return FactSpec(
        type=FactType.EXACT_VALUE, entity_ids=(entity.eid,), attribute=attr,
        values=(value,), relation=None, uncertainty_kind=None, is_queried=queried,
        distance_bucket=None, scenes=[scene], value_scene_idx={0: 0},
    )


def build_negative(rng: random.Random, pools: dict, entity: EntitySpec,
                   queried: bool) -> FactSpec:
    fam, attr, _ = _pick_attr_value(rng, pools)
    scene = SceneSpec(
        family_id="NEG-01",
        slot_values={"entity": entity.surface, "attribute": attr},
        entity_slot_map={"entity": entity.eid},
        cue_kind=CueKind.NEGATION,
    )
    return FactSpec(
        type=FactType.NEGATIVE, entity_ids=(entity.eid,), attribute=attr,
        values=(), relation=None, uncertainty_kind=None, is_queried=queried,
        distance_bucket=None, scenes=[scene], value_scene_idx={},
    )


def build_hedge(rng: random.Random, pools: dict, entity: EntitySpec,
                queried: bool) -> FactSpec:
    fam, attr, value = _pick_attr_value(rng, pools)
    source = rng.choice(pools["sources"])
    scene = SceneSpec(
        family_id="UNRES-01",
        slot_values={"source": source, "entity": entity.surface,
                     "attribute": attr, "value": value},
        entity_slot_map={"entity": entity.eid},
        value_slot_map={"value": 0},
        cue_kind=CueKind.HEDGE,
    )
    return FactSpec(
        type=FactType.UNCERTAINTY, entity_ids=(entity.eid,), attribute=attr,
        values=(value,), relation=None, uncertainty_kind=UncertaintyKind.HEDGE,
        is_queried=queried, distance_bucket=None, scenes=[scene],
        value_scene_idx={0: 0},
    )


def build_conflict(rng: random.Random, pools: dict, entity: EntitySpec,
                   queried: bool) -> FactSpec:
    _, attr, vals = _pick_attr_values(rng, pools, 2)
    v1, v2 = vals[0], vals[1]
    s1, s2 = rng.choice(pools["sources"]), rng.choice(pools["sources"])
    scenes = [
        SceneSpec(
            family_id="UNRES-02",
            slot_values={"source": s1, "entity": entity.surface,
                         "attribute": attr, "value": v1},
            entity_slot_map={"entity": entity.eid},
            value_slot_map={"value": 0},
        ),
        SceneSpec(
            family_id="UNRES-02",
            slot_values={"source": s2, "entity": entity.surface,
                         "attribute": attr, "value": v2},
            entity_slot_map={"entity": entity.eid},
            value_slot_map={"value": 1},
        ),
    ]
    return FactSpec(
        type=FactType.UNCERTAINTY, entity_ids=(entity.eid,), attribute=attr,
        values=(v1, v2), relation=None, uncertainty_kind=UncertaintyKind.CONFLICT,
        is_queried=queried, distance_bucket=None, scenes=scenes,
        value_scene_idx={0: 0, 1: 1},
    )


def build_binding(rng: random.Random, pools: dict, entity: EntitySpec,
                  queried: bool, bucket: DistanceBucket) -> FactSpec:
    """Binding needle: intro scene + deferred reference scene.

    Uses ONE attribute with two distinct values so the deferred reference
    genuinely refers back to the introduced attribute: values = (intro_value,
    reference_value) under the same attribute; the two scenes are laid far
    apart by docgen according to *bucket*.
    """
    _, attr, vals = _pick_attr_values(rng, pools, 2)
    v_intro, v_ref = vals[0], vals[1]
    intro = SceneSpec(
        family_id="BIND-01",
        slot_values={"entity": entity.surface, "entity_type": entity.type,
                     "attribute": attr, "value": v_intro},
        entity_slot_map={"entity": entity.eid},
        value_slot_map={"value": 0},
    )
    ref_surface = entity.alias or entity.surface
    ref = SceneSpec(
        family_id="BIND-02",
        slot_values={"entity": ref_surface, "attribute": attr,
                     "new_value": v_ref},
        entity_slot_map={"entity": entity.eid},
        value_slot_map={"new_value": 1},
    )
    return FactSpec(
        type=FactType.BINDING, entity_ids=(entity.eid,), attribute=attr,
        values=(v_intro, v_ref), relation=None, uncertainty_kind=None,
        is_queried=queried, distance_bucket=bucket, scenes=[intro, ref],
        value_scene_idx={0: 0, 1: 1},
    )
