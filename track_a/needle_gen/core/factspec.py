"""Planning structures linking facts to the scenes that express them (T3B).

A FactSpec carries everything needed to materialize a ``types.Fact`` once
docgen has laid out scenes; SceneSpecs carry rendering hints and annotation
maps (entity slots, value slots, cue kind) so annotations.py can compute
token spans after assembly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from track_a.needle_gen.types import (
    CueKind, DistanceBucket, FactType, UncertaintyKind,
)


@dataclass
class SceneSpec:
    """One prose scene expressing (part of) a fact, or filler."""

    family_id: str
    slot_values: dict[str, str]
    entity_slot_map: dict[str, str] = field(default_factory=dict)  # slot -> entity id
    value_slot_map: dict[str, int] = field(default_factory=dict)  # slot -> value index
    cue_kind: CueKind | None = None
    is_filler: bool = False


@dataclass
class FactSpec:
    """A planned fact plus the ordered scenes that express it."""

    type: FactType
    entity_ids: tuple[str, ...]
    attribute: str | None
    values: tuple[str, ...]
    relation: str | None
    uncertainty_kind: UncertaintyKind | None
    is_queried: bool
    distance_bucket: DistanceBucket | None
    scenes: list[SceneSpec] = field(default_factory=list)
    # value index -> scene index (within this fact's scenes) where it appears
    value_scene_idx: dict[int, int] = field(default_factory=dict)
    is_multihop_anchor: bool = False
