"""Scene-level event extraction + incremental streaming state fold.

Each fact expands to one event per scene it touches (chains/binding yield
several, so later windows produce SUPERSEDE updates rather than whole-fact
inserts). Events fold into a ``StreamState``: a mutable belief state keyed by
(eid, attr) with current values, relations, negations, and unresolved
conflicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from track_a.needle_gen.types import (
    TYPE_RANK, Fact, FactType, UncertaintyKind,
)


@dataclass(frozen=True)
class Event:
    """One state mutation at a scene boundary."""

    scene_idx: int
    kind: str  # "fact" | "rel" | "neg" | "conflict_side"
    rank: int = 5
    eid: str = ""
    attr: str = ""
    value: str = ""
    pos: int = 0
    hedged: bool = False
    eid_a: str = ""
    rel: str = ""
    eid_b: str = ""


@dataclass
class StreamState:
    """Mutable folded belief state (never truncated here)."""

    facts: dict = field(default_factory=dict)          # (eid,attr)->[value,pos,rank,hedged]
    relations: dict = field(default_factory=dict)      # (a,rel,b)->[pos,rank]
    negations: dict = field(default_factory=dict)      # (eid,attr)->[pos,rank]
    unresolved: dict = field(default_factory=dict)     # (eid,attr)->[candidates,rank]
    conflict_claims: dict = field(default_factory=dict)  # (eid,attr)->[(value,pos)]
    entities_seen: set = field(default_factory=set)


def events_for_fact(fact: Fact) -> list[Event]:
    """Expand a fact into scene-ordered events."""
    rank = TYPE_RANK.get(fact.type, 5)
    t = fact.type
    evs: list[Event] = []
    if t is FactType.RELATIONAL:
        p = fact.scene_positions[0]
        evs.append(Event(p, "rel", rank=rank, eid_a=fact.entity_ids[0],
                         rel=fact.relation or "", eid_b=fact.entity_ids[1], pos=p))
    elif t is FactType.NEGATIVE:
        p = fact.scene_positions[0]
        evs.append(Event(p, "neg", rank=rank, eid=fact.entity_ids[0],
                         attr=fact.attribute or "", pos=p))
    elif t is FactType.UNCERTAINTY and fact.uncertainty_kind is UncertaintyKind.CONFLICT:
        for v, p in zip(fact.values, fact.scene_positions):
            evs.append(Event(p, "conflict_side", rank=rank, eid=fact.entity_ids[0],
                             attr=fact.attribute or "", value=v, pos=p))
    elif t is FactType.UNCERTAINTY and fact.uncertainty_kind is UncertaintyKind.HEDGE:
        p = fact.scene_positions[0]
        evs.append(Event(p, "fact", rank=rank, eid=fact.entity_ids[0],
                         attr=fact.attribute or "", value=fact.values[0],
                         pos=p, hedged=True))
    else:  # EXACT_VALUE, BINDING, STATE_TRANSITION (chains -> updates)
        for v, p in zip(fact.values, fact.scene_positions):
            evs.append(Event(p, "fact", rank=rank, eid=fact.entity_ids[0],
                             attr=fact.attribute or "", value=v, pos=p))
    return evs


def extract_events(facts: tuple[Fact, ...]) -> list[Event]:
    """All events from all facts, sorted by scene_idx (stable)."""
    evs: list[Event] = []
    for f in facts:
        evs.extend(events_for_fact(f))
    evs.sort(key=lambda e: e.scene_idx)
    return evs


def apply_event(state: StreamState, ev: Event) -> None:
    """Mutate *state* with one event."""
    if ev.kind == "fact":
        state.facts[(ev.eid, ev.attr)] = [ev.value, ev.pos, ev.rank, ev.hedged]
        state.entities_seen.add(ev.eid)
    elif ev.kind == "rel":
        state.relations[(ev.eid_a, ev.rel, ev.eid_b)] = [ev.pos, ev.rank]
        state.entities_seen.add(ev.eid_a)
        state.entities_seen.add(ev.eid_b)
    elif ev.kind == "neg":
        state.negations[(ev.eid, ev.attr)] = [ev.pos, ev.rank]
        state.entities_seen.add(ev.eid)
    elif ev.kind == "conflict_side":
        key = (ev.eid, ev.attr)
        claims = state.conflict_claims.setdefault(key, [])
        claims.append((ev.value, ev.pos))
        state.entities_seen.add(ev.eid)
        if len(claims) == 1:
            state.facts[key] = [ev.value, ev.pos, ev.rank, False]
        else:
            state.facts.pop(key, None)
            state.unresolved[key] = [list(claims), ev.rank]
