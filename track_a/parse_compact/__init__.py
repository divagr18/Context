"""Parser for the compact representation C and the streaming edit-op log.

Public contract C4:

    parse_target_lines(lines) -> (list[Record], list[ParseFailure])
    parse_ops_lines(lines)    -> (list[OpRecord], list[ParseFailure])
    OpLog().append(lines)     -> list[ParseFailure]
    OpLog().fold()            -> FoldedState

FoldedState exposes ``current`` ((entity_name, attr) -> (value, pos)),
``relations``, ``negations``, ``unresolved``, ``name_of(eid)`` and
``history(name, attr)`` -> chronological [(value, pos, superseded), ...].

Both parsers never raise on malformed input; every bad line yields a
ParseFailure with a deterministic reason (grammar C2, PLAN.md section 4).
Golden-vector serialization kinds are the ``kind`` class attributes in
``records.py`` (see tests/golden/grammar_vectors.jsonl).
"""

from .full_state import parse_target_lines
from .oplog import FoldedState, OpLog
from .ops import parse_ops_lines
from .records import (
    DropFactOp,
    DropNegOp,
    DropRelOp,
    EntityRec,
    FactRec,
    MarkUnresolvedOp,
    NegRec,
    OpRecord,
    ParseFailure,
    Record,
    RelRec,
    ResolveOp,
    SuperseFactOp,
    UnresolvedRec,
    UpsertEntityOp,
    UpsertFactOp,
    UpsertNegOp,
    UpsertRelOp,
)

__all__ = [
    "DropFactOp",
    "DropNegOp",
    "DropRelOp",
    "EntityRec",
    "FactRec",
    "FoldedState",
    "MarkUnresolvedOp",
    "NegRec",
    "OpLog",
    "OpRecord",
    "ParseFailure",
    "Record",
    "RelRec",
    "ResolveOp",
    "SuperseFactOp",
    "UnresolvedRec",
    "UpsertEntityOp",
    "UpsertFactOp",
    "UpsertNegOp",
    "UpsertRelOp",
    "parse_ops_lines",
    "parse_target_lines",
]
