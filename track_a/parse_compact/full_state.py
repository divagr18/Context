"""Parser for full-state C records (grammar C2, PLAN.md section 4).

Never-raise discipline: every malformed line (truncated, garbage tokens,
missing fields, unterminated candidates list, whitespace inside a field,
unknown verb) yields a ParseFailure -- malformed input is handled by
explicit token checks, never by catching exceptions. Malformed lines are
strictly separated from parsed records. All field tokens must match the
space-free grammar charset; values and names are canonicalized via
``canonicalize_value`` (contract C1).

The token-level helpers here (_field/_pos/_eid_attr/_candidates) are the
single source of grammar-token rules; ``ops.py`` reuses them.
"""

from __future__ import annotations

import re

from track_a.needle_gen.types import canonicalize_value

from .records import (
    EntityRec,
    FactRec,
    NegRec,
    ParseFailure,
    Record,
    RelRec,
    UnresolvedRec,
)

# Grammar charset for every field token (PLAN.md section 4: value pools match
# [A-Za-z0-9_$./:%-]+, no spaces). Tokens are whitespace-split already, so
# this charset check plus the structural checks below carry the full grammar.
FIELD_RE = re.compile(r"^[A-Za-z0-9_$./:%-]+$")
_DIGITS_RE = re.compile(r"^[0-9]+$")

_TYPE_PREFIX = "type="
_NAME_PREFIX = "name="
_POS_PREFIX = "pos="
_SUPERSEDES_PREFIX = "supersedes="
_CANDIDATES_PREFIX = "candidates=["


def _field(token: str) -> bool:
    return FIELD_RE.match(token) is not None


def _pos(token: str) -> int | None:
    """Scene index from a ``pos=<int>`` token; None if malformed."""
    if not token.startswith(_POS_PREFIX):
        return None
    digits = token[len(_POS_PREFIX):]
    if _DIGITS_RE.match(digits) is None:
        return None
    return int(digits)


def _eid_attr(token: str) -> tuple[str, str] | None:
    """Split ``<eid>.<attr>``; None if the dot or either side is missing."""
    eid, sep, attr = token.partition(".")
    if not sep or not eid or not attr or not _field(eid) or not _field(attr):
        return None
    return eid, attr


def _candidates(tokens: list[str], start: int) -> tuple[tuple[tuple[str, int], ...] | None, str, int]:
    """Parse ``candidates=[v@p, ...]`` spanning ``tokens[start:]``.

    Returns ``(candidates, reason, next_index)``; candidates is None on
    failure (unterminated list, empty list, malformed entry) and ``reason``
    is the deterministic failure cause.
    """
    if start >= len(tokens):
        return None, "missing candidates list", start
    if not tokens[start].startswith(_CANDIDATES_PREFIX):
        return None, "expected candidates=[...]", start
    end = start
    while end < len(tokens) and not tokens[end].endswith("]"):
        end += 1
    if end >= len(tokens):
        return None, "unterminated candidates list", start
    blob = " ".join(tokens[start : end + 1])
    inner = blob[len(_CANDIDATES_PREFIX) : -1]
    if inner.strip() == "":
        return None, "empty candidates list", end + 1
    parsed: list[tuple[str, int]] = []
    for part in inner.split(","):
        part = part.strip()
        if part == "":
            return None, "empty candidate entry", end + 1
        value, sep, pos_text = part.partition("@")
        if not sep or not value or _DIGITS_RE.match(pos_text) is None:
            return None, "candidate must be <value>@<pos>", end + 1
        if not _field(value):
            return None, "invalid candidate value", end + 1
        parsed.append((canonicalize_value(value), int(pos_text)))
    return tuple(parsed), "", end + 1


def _entity(t: list[str]) -> tuple[Record | None, str]:
    if len(t) != 4:
        return None, "entity: expected 4 tokens"
    eid, type_tok, name_tok = t[1], t[2], t[3]
    if not _field(eid):
        return None, "entity: invalid eid"
    if not type_tok.startswith(_TYPE_PREFIX) or not _field(type_tok[len(_TYPE_PREFIX):]):
        return None, "entity: invalid type= field"
    if not name_tok.startswith(_NAME_PREFIX) or not _field(name_tok[len(_NAME_PREFIX):]):
        return None, "entity: invalid name= field"
    return EntityRec(eid, type_tok[len(_TYPE_PREFIX):], canonicalize_value(name_tok[len(_NAME_PREFIX):])), ""


def _fact(t: list[str]) -> tuple[Record | None, str]:
    # FACT <eid>.<attr> = <value> [supersedes=<value>] [hedged] pos=<pos>
    if len(t) < 5:
        return None, "fact: expected at least 5 tokens"
    ea = _eid_attr(t[1])
    if ea is None:
        return None, "fact: invalid <eid>.<attr>"
    if t[2] != "=":
        return None, "fact: expected '=' after <eid>.<attr>"
    if not _field(t[3]):
        return None, "fact: invalid value"
    i = 4
    supersedes: str | None = None
    if t[i].startswith(_SUPERSEDES_PREFIX):
        raw = t[i][len(_SUPERSEDES_PREFIX):]
        if not _field(raw):
            return None, "fact: invalid supersedes value"
        supersedes = canonicalize_value(raw)
        i += 1
    hedged = False
    if i < len(t) and t[i] == "hedged":
        hedged = True
        i += 1
    if i >= len(t):
        return None, "fact: missing pos= field"
    pos = _pos(t[i])
    if pos is None:
        return None, "fact: invalid pos= field"
    if i != len(t) - 1:
        return None, "fact: trailing tokens"
    eid, attr = ea
    return FactRec(eid, attr, canonicalize_value(t[3]), pos, supersedes, hedged), ""


def _rel(t: list[str]) -> tuple[Record | None, str]:
    if len(t) != 5:
        return None, "rel: expected 5 tokens"
    if not _field(t[1]) or not _field(t[3]):
        return None, "rel: invalid eid"
    if not _field(t[2]):
        return None, "rel: invalid relation"
    pos = _pos(t[4])
    if pos is None:
        return None, "rel: invalid pos= field"
    return RelRec(t[1], t[2], t[3], pos), ""


def _neg(t: list[str]) -> tuple[Record | None, str]:
    if len(t) != 4:
        return None, "neg: expected 4 tokens"
    ea = _eid_attr(t[1])
    if ea is None:
        return None, "neg: invalid <eid>.<attr>"
    if t[2] != "denied":
        return None, "neg: expected literal 'denied'"
    pos = _pos(t[3])
    if pos is None:
        return None, "neg: invalid pos= field"
    eid, attr = ea
    return NegRec(eid, attr, pos), ""


def _unresolved(t: list[str]) -> tuple[Record | None, str]:
    if len(t) < 3:
        return None, "unresolved: expected <eid>.<attr> candidates=[...]"
    ea = _eid_attr(t[1])
    if ea is None:
        return None, "unresolved: invalid <eid>.<attr>"
    candidates, reason, next_idx = _candidates(t, 2)
    if candidates is None:
        return None, f"unresolved: {reason}"
    if next_idx != len(t):
        return None, "unresolved: trailing tokens"
    eid, attr = ea
    return UnresolvedRec(eid, attr, candidates), ""


_VERB_PARSERS = {
    "ENTITY": _entity,
    "FACT": _fact,
    "REL": _rel,
    "NEG": _neg,
    "UNRESOLVED": _unresolved,
}


def _parse_tokens(tokens: list[str]) -> tuple[Record | None, str]:
    if not tokens:
        return None, "empty line"
    parser = _VERB_PARSERS.get(tokens[0])
    if parser is None:
        return None, f"unknown record verb: {tokens[0]}"
    return parser(tokens)


def parse_target_lines(lines: list[str]) -> tuple[list[Record], list[ParseFailure]]:
    """Parse full-state records; malformed lines never raise, only count."""
    records: list[Record] = []
    failures: list[ParseFailure] = []
    for line in lines:
        try:
            record, reason = _parse_tokens(line.strip().split())
        except Exception:  # contract backstop, never expected to fire
            failures.append(ParseFailure(line, "internal parser error"))
            continue
        if record is None:
            failures.append(ParseFailure(line, reason))
        else:
            records.append(record)
    return records, failures
