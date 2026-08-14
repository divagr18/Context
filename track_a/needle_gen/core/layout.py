"""Scene layout: order fact scenes, pad distances with fillers, hit length.

Produces an ordered list of ``LaidScene`` records. Distance-requiring facts
(binding needles, deep chains) have their far endpoint placed inline — right
after enough filler padding accumulates from their OWN anchor — so the
endpoint lands in the target distance bucket regardless of other facts.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from track_a.needle_gen.core.factspec import SceneSpec
from track_a.needle_gen.types import CueKind


@dataclass
class PreparedScene:
    """A rendered scene ready for layout."""

    text: str
    family_id: str
    fact_key: str | None
    is_filler: bool
    entity_slot_map: dict[str, str] = field(default_factory=dict)
    value_slot_map: dict[str, int] = field(default_factory=dict)
    cue_kind: CueKind | None = None
    char_spans: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass
class LaidScene:
    """A scene placed in the document with layout metadata."""

    scene: PreparedScene
    token_count: int
    distance_from_anchor: int | None = None


def lay_out(
    rng: random.Random,
    count_tokens: Callable[[str], int],
    facts_with_scenes: list[tuple[str, list[PreparedScene], int | None]],
    make_filler: Callable[[], PreparedScene],
    target_tokens: int,
) -> list[LaidScene]:
    """Order all scenes, pad distance buckets inline, fill to *target_tokens*.

    Args:
        facts_with_scenes: list of (fact_key, prepared scenes, target_distance
            or None). target_distance != None marks a distance fact whose last
            scene is placed right after padding reaches the target distance.
        count_tokens: token counter for a scene text.
        make_filler: returns a fresh filler PreparedScene.
        target_tokens: document length goal (packing stops at >= goal).

    Returns:
        Ordered LaidScene list; distance endpoints carry their distance.
    """
    laid: list[LaidScene] = []
    running = 0

    def emit(scene: PreparedScene, dist: int | None = None) -> None:
        nonlocal running
        laid.append(LaidScene(scene=scene, token_count=count_tokens(scene.text),
                              distance_from_anchor=dist))
        running += laid[-1].token_count

    for fact_key, scenes, target_dist in facts_with_scenes:
        if not scenes:
            continue
        if target_dist is None or len(scenes) < 2:
            for sc in scenes:
                emit(sc)
            continue
        anchor_start = running
        for sc in scenes[:-1]:
            emit(sc)
        guard = 0
        while running - anchor_start < target_dist:
            emit(make_filler())
            guard += 1
            if guard > 100000:
                raise RuntimeError("distance padding failed to converge")
        dist = running - anchor_start
        for sc in scenes[-1:]:
            emit(sc, dist)

    while running < target_tokens:
        emit(make_filler())

    return laid
