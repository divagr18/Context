"""Frozen record/op dataclasses for the compact representation C (PLAN.md S4).

Two families mirror the grammar C2 line-for-line:

* full-state records: EntityRec, FactRec, RelRec, NegRec, UnresolvedRec
* streaming edit-ops: Upsert*/Superse*/Drop*/MarkUnresolved/Resolve ops

``kind`` class attributes are the golden-vector serialization contract
(tests/golden/grammar_vectors.jsonl): a record serializes to
``{"kind": <kind>, **dataclass_fields}`` with tuples rendered as JSON lists.
All classes are frozen so an appended OpLog can never be mutated in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

#: One UNRESOLVED/MARK-UNRESOLVED candidate: (value, pos).
Candidate = tuple[str, int]

# ---------------------------------------------------------------------------
# Full-state records (single-shot targets / bootstrap content)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityRec:
    """ENTITY <eid> type=<t> name=<name>"""

    kind: ClassVar[str] = "entity"
    eid: str
    type: str
    name: str


@dataclass(frozen=True)
class FactRec:
    """FACT <eid>.<attr> = <value> [supersedes=<value>] [hedged] pos=<pos>"""

    kind: ClassVar[str] = "fact"
    eid: str
    attr: str
    value: str
    pos: int
    supersedes: str | None = None
    hedged: bool = False


@dataclass(frozen=True)
class RelRec:
    """REL <eid_a> <rel> <eid_b> pos=<pos>"""

    kind: ClassVar[str] = "rel"
    eid_a: str
    rel: str
    eid_b: str
    pos: int


@dataclass(frozen=True)
class NegRec:
    """NEG <eid>.<attr> denied pos=<pos>"""

    kind: ClassVar[str] = "neg"
    eid: str
    attr: str
    pos: int


@dataclass(frozen=True)
class UnresolvedRec:
    """UNRESOLVED <eid>.<attr> candidates=[<v1>@<pos1>, <v2>@<pos2>]"""

    kind: ClassVar[str] = "unresolved"
    eid: str
    attr: str
    candidates: tuple[Candidate, ...]


# ---------------------------------------------------------------------------
# Streaming edit-ops (delta per window, PLAN.md section 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UpsertEntityOp:
    """UPSERT ENTITY <eid> type=<t> name=<name>"""

    kind: ClassVar[str] = "upsert_entity"
    eid: str
    type: str
    name: str


@dataclass(frozen=True)
class UpsertFactOp:
    """UPSERT FACT <eid>.<attr> = <value> pos=<pos>"""

    kind: ClassVar[str] = "upsert_fact"
    eid: str
    attr: str
    value: str
    pos: int


@dataclass(frozen=True)
class SuperseFactOp:
    """SUPERSEDE FACT <eid>.<attr> : <old_value> => <new_value> pos=<pos>"""

    kind: ClassVar[str] = "superse_fact"
    eid: str
    attr: str
    old_value: str
    new_value: str
    pos: int


@dataclass(frozen=True)
class UpsertRelOp:
    """UPSERT REL <eid_a> <rel> <eid_b> pos=<pos>"""

    kind: ClassVar[str] = "upsert_rel"
    eid_a: str
    rel: str
    eid_b: str
    pos: int


@dataclass(frozen=True)
class UpsertNegOp:
    """UPSERT NEG <eid>.<attr> pos=<pos>"""

    kind: ClassVar[str] = "upsert_neg"
    eid: str
    attr: str
    pos: int


@dataclass(frozen=True)
class DropFactOp:
    """DROP FACT <eid>.<attr> pos=<pos>"""

    kind: ClassVar[str] = "drop_fact"
    eid: str
    attr: str
    pos: int


@dataclass(frozen=True)
class DropRelOp:
    """DROP REL <eid_a> <rel> <eid_b> pos=<pos>"""

    kind: ClassVar[str] = "drop_rel"
    eid_a: str
    rel: str
    eid_b: str
    pos: int


@dataclass(frozen=True)
class DropNegOp:
    """DROP NEG <eid>.<attr> pos=<pos>"""

    kind: ClassVar[str] = "drop_neg"
    eid: str
    attr: str
    pos: int


@dataclass(frozen=True)
class MarkUnresolvedOp:
    """MARK UNRESOLVED <eid>.<attr> candidates=[<v1>@<pos1>, <v2>@<pos2>]"""

    kind: ClassVar[str] = "mark_unresolved"
    eid: str
    attr: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class ResolveOp:
    """RESOLVE <eid>.<attr> = <value> pos=<pos>"""

    kind: ClassVar[str] = "resolve"
    eid: str
    attr: str
    value: str
    pos: int


# ---------------------------------------------------------------------------
# Parse failure + union aliases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParseFailure:
    """One malformed line. ``reason`` is a deterministic string."""

    line: str
    reason: str


#: Any full-state record.
Record = EntityRec | FactRec | RelRec | NegRec | UnresolvedRec
#: Any streaming edit-op.
OpRecord = (
    UpsertEntityOp
    | UpsertFactOp
    | SuperseFactOp
    | UpsertRelOp
    | UpsertNegOp
    | DropFactOp
    | DropRelOp
    | DropNegOp
    | MarkUnresolvedOp
    | ResolveOp
)
