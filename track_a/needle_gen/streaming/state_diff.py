"""Streaming state rendering, budget truncation, and diff -> edit-ops.

The streaming belief state (``StreamState``) is rendered as full-state record
lines, truncated to a token budget, and consecutive truncated states are
diffed into the edit-op grammar (C2). Ops are byte-consistent with the T4
parser: entities are emitted before facts that reference them, fact->unresolved
transitions produce ``DROP FACT`` + ``MARK UNRESOLVED``, and unresolved/entities
are never dropped (no such op exists in the grammar).
"""
from __future__ import annotations

import copy

from track_a.needle_gen.types import canonicalize_value
from track_a.shard_schema import StreamingWindow

from .events import StreamState, apply_event, extract_events
from .windows import scene_cutoff, split_windows


def _referenced_eids(state: StreamState) -> set:
    """Eids referenced by current facts/relations/negations/unresolved."""
    referenced: set = set()
    for (eid, _attr) in state.facts:
        referenced.add(eid)
    for (a, _rel, b) in state.relations:
        referenced.add(a)
        referenced.add(b)
    for (eid, _attr) in state.negations:
        referenced.add(eid)
    for (eid, _attr) in state.unresolved:
        referenced.add(eid)
    return referenced


def render_state_lines(state: StreamState, entity_map: dict) -> list[str]:
    """Canonical full-state record lines for the current belief state.

    Only entities referenced by a current entry are rendered, so budget
    truncation that drops a fact also drops its (now-unreferenced) entity and
    the state can actually reach the token budget (there is no DROP ENTITY op).
    """
    lines: list[str] = []
    for eid in sorted(_referenced_eids(state)):
        e = entity_map.get(eid)
        if e is None:
            continue
        lines.append(f"ENTITY {eid} type={e.type} name={canonicalize_value(e.name)}")
    facts = sorted(state.facts.items(), key=lambda kv: (kv[1][2], kv[1][1], kv[0]))
    for (eid, attr), (value, pos, _rank, _h) in facts:
        lines.append(f"FACT {eid}.{attr} = {canonicalize_value(value)} pos={pos}")
    rels = sorted(state.relations.items(), key=lambda kv: (kv[1][1], kv[1][0], kv[0]))
    for (a, rel, b), (pos, _rank) in rels:
        lines.append(f"REL {a} {rel} {b} pos={pos}")
    negs = sorted(state.negations.items(), key=lambda kv: (kv[1][1], kv[1][0], kv[0]))
    for (eid, attr), (pos, _rank) in negs:
        lines.append(f"NEG {eid}.{attr} denied pos={pos}")
    for (eid, attr) in sorted(state.unresolved):
        cands, _rank = state.unresolved[(eid, attr)]
        parts = ", ".join(f"{canonicalize_value(v)}@{p}" for v, p in cands)
        lines.append(f"UNRESOLVED {eid}.{attr} candidates=[{parts}]")
    return lines


def _token_count(state: StreamState, entity_map: dict, tokenizer) -> int:
    lines = render_state_lines(state, entity_map)
    if not lines:
        return 0
    return len(tokenizer.encode("\n".join(lines), add_special_tokens=False))


def truncate_state(state: StreamState, budget: int, entity_map: dict,
                   tokenizer) -> StreamState:
    """Budget-truncated copy: drops lowest-priority fact/rel/neg entries.

    Unresolved conflicts and entity bindings are never dropped (the grammar
    has no DROP op for them). Priority = (rank desc, pos desc) dropped first.
    """
    working = copy.deepcopy(state)
    while _token_count(working, entity_map, tokenizer) > budget:
        droppable: list[tuple[int, int, str, object]] = []
        for k, (_v, pos, rank, _h) in working.facts.items():
            droppable.append((rank, pos, "fact", k))
        for k, (pos, rank) in working.relations.items():
            droppable.append((rank, pos, "rel", k))
        for k, (pos, rank) in working.negations.items():
            droppable.append((rank, pos, "neg", k))
        if not droppable:
            break
        droppable.sort(key=lambda x: (-x[0], -x[1]))
        _rank, _pos, kind, key = droppable[0]
        if kind == "fact":
            working.facts.pop(key, None)
        elif kind == "rel":
            working.relations.pop(key, None)
        else:
            working.negations.pop(key, None)
    return working


def diff_states(prev: StreamState, nxt: StreamState, entity_map: dict) -> list[str]:
    """Edit-ops converting *prev* truncated state into *next*."""
    ops: list[str] = []
    for eid in sorted(nxt.entities_seen - prev.entities_seen):
        e = entity_map.get(eid)
        if e is None:
            continue
        ops.append(f"UPSERT ENTITY {eid} type={e.type} name={canonicalize_value(e.name)}")

    for key in sorted(set(nxt.facts) - set(prev.facts)):
        value, pos, _r, _h = nxt.facts[key]
        ops.append(f"UPSERT FACT {key[0]}.{key[1]} = {canonicalize_value(value)} pos={pos}")
    for key in sorted(set(nxt.facts) & set(prev.facts)):
        ov, _op, _r1, _h1 = prev.facts[key]
        nv, np_, _r2, _h2 = nxt.facts[key]
        if ov != nv:
            ops.append(
                f"SUPERSEDE FACT {key[0]}.{key[1]} : "
                f"{canonicalize_value(ov)} => {canonicalize_value(nv)} pos={np_}"
            )
    for key in sorted(set(prev.facts) - set(nxt.facts)):
        _v, pos, _r, _h = prev.facts[key]
        ops.append(f"DROP FACT {key[0]}.{key[1]} pos={pos}")

    for key in sorted(set(nxt.relations) - set(prev.relations)):
        pos, _r = nxt.relations[key]
        ops.append(f"UPSERT REL {key[0]} {key[1]} {key[2]} pos={pos}")
    for key in sorted(set(prev.relations) - set(nxt.relations)):
        pos, _r = prev.relations[key]
        ops.append(f"DROP REL {key[0]} {key[1]} {key[2]} pos={pos}")

    for key in sorted(set(nxt.negations) - set(prev.negations)):
        pos, _r = nxt.negations[key]
        ops.append(f"UPSERT NEG {key[0]}.{key[1]} pos={pos}")
    for key in sorted(set(prev.negations) - set(nxt.negations)):
        pos, _r = prev.negations[key]
        ops.append(f"DROP NEG {key[0]}.{key[1]} pos={pos}")

    for key in sorted(set(nxt.unresolved) - set(prev.unresolved)):
        cands, _r = nxt.unresolved[key]
        parts = ", ".join(f"{canonicalize_value(v)}@{p}" for v, p in cands)
        ops.append(f"MARK UNRESOLVED {key[0]}.{key[1]} candidates=[{parts}]")
    return ops


def build_streaming_windows(doc_ids, facts, scene_boundaries, entity_map,
                            budget, window_tokens, overlap_frac,
                            tokenizer) -> list[StreamingWindow]:
    """Streaming samples: (state before window, window, ops) per window."""
    win_ranges = split_windows(len(doc_ids), window_tokens, overlap_frac)
    events = extract_events(facts)
    samples: list[StreamingWindow] = []
    full_state = StreamState()
    event_ptr = 0
    prev_trunc = StreamState()
    for (ws, we) in win_ranges:
        cutoff = scene_cutoff(scene_boundaries, we)
        while event_ptr < len(events) and events[event_ptr].scene_idx < cutoff:
            apply_event(full_state, events[event_ptr])
            event_ptr += 1
        curr_trunc = truncate_state(full_state, budget, entity_map, tokenizer)
        state_lines = render_state_lines(prev_trunc, entity_map)
        state_ids = (tuple(tokenizer.encode("\n".join(state_lines),
                                            add_special_tokens=False))
                     if state_lines else ())
        window_ids = tuple(doc_ids[ws:we])
        ops = diff_states(prev_trunc, curr_trunc, entity_map)
        samples.append(StreamingWindow(budget, state_ids, window_ids, "\n".join(ops)))
        prev_trunc = curr_trunc
    return samples
