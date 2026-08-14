"""Parser for streaming edit-ops (grammar C2, PLAN.md sections 3-4).

Same never-raise discipline as ``full_state.py``: explicit token checks map
every malformed line to a ParseFailure, strictly separated from parsed ops.
Token-level grammar rules (_field/_pos/_eid_attr/_candidates) are imported
from ``full_state`` -- one source of truth for the grammar.

Note on UPSERT NEG: PLAN.md section 4 lists ``UPSERT ENTITY|FACT|REL|NEG``
among the streaming edit-ops (mirroring DROP FACT|REL|NEG); it is supported
here as the only path that populates the folded negation set.
"""

from __future__ import annotations

from track_a.needle_gen.types import canonicalize_value

from .full_state import _candidates, _eid_attr, _field, _pos
from .records import (
    DropFactOp,
    DropNegOp,
    DropRelOp,
    MarkUnresolvedOp,
    OpRecord,
    ParseFailure,
    ResolveOp,
    SuperseFactOp,
    UpsertEntityOp,
    UpsertFactOp,
    UpsertNegOp,
    UpsertRelOp,
)

_TYPE_PREFIX = "type="
_NAME_PREFIX = "name="


def _kv(token: str, prefix: str) -> str | None:
    """Value of a ``<prefix><field>`` token; None if prefix missing/empty/bad."""
    if not token.startswith(prefix):
        return None
    raw = token[len(prefix):]
    if not _field(raw):
        return None
    return raw


def _upsert(t: list[str]) -> tuple[OpRecord | None, str]:
    if len(t) < 2:
        return None, "upsert: missing subject"
    subject = t[1]
    if subject == "ENTITY":
        if len(t) != 5:
            return None, "upsert entity: expected 5 tokens"
        if not _field(t[2]):
            return None, "upsert entity: invalid eid"
        type_raw = _kv(t[3], _TYPE_PREFIX)
        if type_raw is None:
            return None, "upsert entity: invalid type= field"
        name_raw = _kv(t[4], _NAME_PREFIX)
        if name_raw is None:
            return None, "upsert entity: invalid name= field"
        return UpsertEntityOp(t[2], type_raw, canonicalize_value(name_raw)), ""
    if subject == "FACT":
        if len(t) != 6:
            return None, "upsert fact: expected 6 tokens"
        ea = _eid_attr(t[2])
        if ea is None:
            return None, "upsert fact: invalid <eid>.<attr>"
        if t[3] != "=":
            return None, "upsert fact: expected '='"
        if not _field(t[4]):
            return None, "upsert fact: invalid value"
        pos = _pos(t[5])
        if pos is None:
            return None, "upsert fact: invalid pos= field"
        eid, attr = ea
        return UpsertFactOp(eid, attr, canonicalize_value(t[4]), pos), ""
    if subject == "REL":
        if len(t) != 6:
            return None, "upsert rel: expected 6 tokens"
        if not _field(t[2]) or not _field(t[4]):
            return None, "upsert rel: invalid eid"
        if not _field(t[3]):
            return None, "upsert rel: invalid relation"
        pos = _pos(t[5])
        if pos is None:
            return None, "upsert rel: invalid pos= field"
        return UpsertRelOp(t[2], t[3], t[4], pos), ""
    if subject == "NEG":
        if len(t) != 4:
            return None, "upsert neg: expected 4 tokens"
        ea = _eid_attr(t[2])
        if ea is None:
            return None, "upsert neg: invalid <eid>.<attr>"
        pos = _pos(t[3])
        if pos is None:
            return None, "upsert neg: invalid pos= field"
        eid, attr = ea
        return UpsertNegOp(eid, attr, pos), ""
    return None, f"upsert: unknown subject: {subject}"


def _superse(t: list[str]) -> tuple[OpRecord | None, str]:
    if len(t) < 2:
        return None, "superse: missing subject"
    if t[1] != "FACT":
        return None, "superse: only FACT is supersedeable"
    if len(t) != 8:
        return None, "superse fact: expected 8 tokens"
    ea = _eid_attr(t[2])
    if ea is None:
        return None, "superse fact: invalid <eid>.<attr>"
    if t[3] != ":":
        return None, "superse fact: expected ':'"
    if not _field(t[4]):
        return None, "superse fact: invalid old value"
    if t[5] != "=>":
        return None, "superse fact: expected '=>'"
    if not _field(t[6]):
        return None, "superse fact: invalid new value"
    pos = _pos(t[7])
    if pos is None:
        return None, "superse fact: invalid pos= field"
    eid, attr = ea
    return SuperseFactOp(eid, attr, canonicalize_value(t[4]), canonicalize_value(t[6]), pos), ""


def _drop(t: list[str]) -> tuple[OpRecord | None, str]:
    if len(t) < 2:
        return None, "drop: missing subject"
    subject = t[1]
    if subject in ("FACT", "NEG"):
        if len(t) != 4:
            return None, "drop: expected 4 tokens"
        ea = _eid_attr(t[2])
        if ea is None:
            return None, "drop: invalid <eid>.<attr>"
        pos = _pos(t[3])
        if pos is None:
            return None, "drop: invalid pos= field"
        eid, attr = ea
        if subject == "FACT":
            return DropFactOp(eid, attr, pos), ""
        return DropNegOp(eid, attr, pos), ""
    if subject == "REL":
        if len(t) != 6:
            return None, "drop rel: expected 6 tokens"
        if not _field(t[2]) or not _field(t[4]):
            return None, "drop rel: invalid eid"
        if not _field(t[3]):
            return None, "drop rel: invalid relation"
        pos = _pos(t[5])
        if pos is None:
            return None, "drop rel: invalid pos= field"
        return DropRelOp(t[2], t[3], t[4], pos), ""
    return None, f"drop: unknown subject: {subject}"


def _mark(t: list[str]) -> tuple[OpRecord | None, str]:
    if len(t) < 4:
        return None, "mark: expected UNRESOLVED <eid>.<attr> candidates=[...]"
    if t[1] != "UNRESOLVED":
        return None, "mark: only UNRESOLVED is markable"
    ea = _eid_attr(t[2])
    if ea is None:
        return None, "mark unresolved: invalid <eid>.<attr>"
    candidates, reason, next_idx = _candidates(t, 3)
    if candidates is None:
        return None, f"mark unresolved: {reason}"
    if next_idx != len(t):
        return None, "mark unresolved: trailing tokens"
    eid, attr = ea
    return MarkUnresolvedOp(eid, attr, candidates), ""


def _resolve(t: list[str]) -> tuple[OpRecord | None, str]:
    if len(t) != 5:
        return None, "resolve: expected 5 tokens"
    ea = _eid_attr(t[1])
    if ea is None:
        return None, "resolve: invalid <eid>.<attr>"
    if t[2] != "=":
        return None, "resolve: expected '='"
    if not _field(t[3]):
        return None, "resolve: invalid value"
    pos = _pos(t[4])
    if pos is None:
        return None, "resolve: invalid pos= field"
    eid, attr = ea
    return ResolveOp(eid, attr, canonicalize_value(t[3]), pos), ""


_VERB_PARSERS = {
    "UPSERT": _upsert,
    "SUPERSEDE": _superse,
    "DROP": _drop,
    "MARK": _mark,
    "RESOLVE": _resolve,
}


def _parse_tokens(tokens: list[str]) -> tuple[OpRecord | None, str]:
    if not tokens:
        return None, "empty line"
    parser = _VERB_PARSERS.get(tokens[0])
    if parser is None:
        return None, f"unknown op verb: {tokens[0]}"
    return parser(tokens)


def parse_ops_lines(lines: list[str]) -> tuple[list[OpRecord], list[ParseFailure]]:
    """Parse streaming edit-ops; malformed lines never raise, only count."""
    ops: list[OpRecord] = []
    failures: list[ParseFailure] = []
    for line in lines:
        try:
            op, reason = _parse_tokens(line.strip().split())
        except Exception:  # contract backstop, never expected to fire
            failures.append(ParseFailure(line, "internal parser error"))
            continue
        if op is None:
            failures.append(ParseFailure(line, reason))
        else:
            ops.append(op)
    return ops, failures
