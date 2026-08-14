"""Builder for state-transition chains (2-hop majority, deep subset).

A chain of k values needs k-1 STATE-01 transition scenes; the first value is
established by an EXACT intro scene, so ``scene_positions`` align 1:1 with
``values`` (PLAN types.py Fact contract; matches render_unit's expectations:
final line at poses[-1], intermediates newest-first at poses[1..k-2]).
"""
from __future__ import annotations

import random

from track_a.needle_gen.core.builders_unary import _pick_attr_values
from track_a.needle_gen.core.entities import EntitySpec
from track_a.needle_gen.core.factspec import FactSpec, SceneSpec
from track_a.needle_gen.types import DistanceBucket, FactType


def build_chain(rng: random.Random, pools: dict, entity: EntitySpec,
                queried: bool, depth: int,
                deep_bucket: DistanceBucket | None) -> FactSpec:
    """Chain with *depth* values (2 = shallow 1-hop, 3 = 2-hop).

    Scenes: [EXACT intro for v0, STATE t1 (v0->v1), STATE t2 (v1->v2), ...].
    All values come from ONE attribute pool so the transition stays coherent.
    For deep chains (*deep_bucket* set), docgen pads before the LAST scene so
    the first->last scene distance lands in that bucket.
    """
    fam, attr, sampled = _pick_attr_values(rng, pools, depth)
    values: list[str] = list(sampled)

    intro = SceneSpec(
        family_id=fam,
        slot_values={"entity": entity.surface, "attribute": attr, "value": values[0]},
        entity_slot_map={"entity": entity.eid},
        value_slot_map={"value": 0},
    )
    scenes = [intro]
    value_scene_idx = {0: 0}
    for i in range(1, depth):
        scenes.append(SceneSpec(
            family_id="STATE-01",
            slot_values={"entity": entity.surface, "attribute": attr,
                         "old_value": values[i - 1], "new_value": values[i]},
            entity_slot_map={"entity": entity.eid},
            value_slot_map={"old_value": i - 1, "new_value": i},
        ))
        value_scene_idx[i] = i
    return FactSpec(
        type=FactType.STATE_TRANSITION, entity_ids=(entity.eid,), attribute=attr,
        values=tuple(values), relation=None, uncertainty_kind=None,
        is_queried=queried, distance_bucket=deep_bucket, scenes=scenes,
        value_scene_idx=value_scene_idx,
    )
