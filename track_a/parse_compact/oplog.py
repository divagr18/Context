"""Append-only op log and its folded state (contract C4, PLAN.md section 3).

Locked semantics (entity resolution is two-phase: eid->name bindings are
collected from the WHOLE log first -- PLAN.md section 4: "the grader resolves
eid->name via ENTITY records" -- then ops fold in log order):

* SUPERSEDE appends the new value as current and flags the prior current
  value ``superseded=True`` WHILE RETAINING it in history.
* UPSERT FACT on a key already current behaves exactly like SUPERSEDE with
  the stored current as old value; on a key not current it just appends.
* DROP removes the key from the current view only; history/log untouched.
* MARK UNRESOLVED sets unresolved candidates; RESOLVE clears them and sets
  the current value (appended to history like UPSERT FACT).
* An op citing an eid with no ENTITY binding in the log is unbound = dropped
  (never reaches the folded view); ``name_of`` returns None for it.
* ``fold()`` is pure: stored log entries are frozen and never mutated, and
  folding twice yields the same state.
"""

from __future__ import annotations

from typing import assert_never

from .ops import parse_ops_lines
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

FactKey = tuple[str, str]
HistoryEntry = tuple[str, int, bool]  # (value, pos, superseded)
RelationKey = tuple[str, str, str]  # (name_a, rel, name_b)


class FoldedState:
    """Immutable folded view over one OpLog, built fresh by each fold()."""

    def __init__(
        self,
        names: dict[str, str],
        current: dict[FactKey, tuple[str, int]],
        relations: set[RelationKey],
        negations: set[FactKey],
        unresolved: dict[FactKey, tuple[tuple[str, int], ...]],
        histories: dict[FactKey, list[HistoryEntry]],
    ) -> None:
        self._names = names
        #: (entity_name, attr) -> (value, pos) of the current fact view.
        self.current = current
        #: Present-set of relations, keyed (name_a, rel, name_b).
        self.relations = relations
        #: Present-set of negations, keyed (name, attr).
        self.negations = negations
        #: (name, attr) -> candidate tuple for still-open conflicts.
        self.unresolved = unresolved
        self._histories = histories

    def name_of(self, eid: str) -> str | None:
        """Resolved canonical name; None = unbound (graders count dropped)."""
        return self._names.get(eid)

    def history(self, name: str, attr: str) -> list[HistoryEntry]:
        """Chronological (value, pos, superseded) list; fresh copy per call."""
        return list(self._histories.get((name, attr), ()))


class OpLog:
    """Append-only log of streaming edit-ops; state is derived by fold()."""

    def __init__(self) -> None:
        self._log: list[OpRecord] = []

    @property
    def ops(self) -> tuple[OpRecord, ...]:
        """Snapshot of the appended log (frozen entries, safe to hold)."""
        return tuple(self._log)

    def __len__(self) -> int:
        return len(self._log)

    def append(self, lines: list[str]) -> list[ParseFailure]:
        """Parse ``lines`` and extend the log; returns this call's failures."""
        ops, failures = parse_ops_lines(lines)
        self._log.extend(ops)
        return failures

    def fold(self) -> FoldedState:
        """Pure fold of the whole log into a fresh FoldedState."""
        names: dict[str, str] = {}
        for op in self._log:
            if isinstance(op, UpsertEntityOp):
                names[op.eid] = op.name

        current: dict[FactKey, tuple[str, int]] = {}
        relations: set[RelationKey] = set()
        negations: set[FactKey] = set()
        unresolved: dict[FactKey, tuple[tuple[str, int], ...]] = {}
        histories: dict[FactKey, list[HistoryEntry]] = {}

        def set_fact(key: FactKey, value: str, pos: int) -> None:
            """Append ``value`` as current, flagging any current entry."""
            entries = histories.setdefault(key, [])
            if key in current:
                prev_value, prev_pos, _ = entries[-1]
                entries[-1] = (prev_value, prev_pos, True)
            entries.append((value, pos, False))
            current[key] = (value, pos)

        for op in self._log:
            match op:
                case UpsertEntityOp():
                    pass  # bindings resolved in phase one above
                case UpsertFactOp():
                    name = names.get(op.eid)
                    if name is not None:
                        set_fact((name, op.attr), op.value, op.pos)
                case SuperseFactOp():
                    name = names.get(op.eid)
                    if name is not None:
                        # ``old_value`` is provenance; the fold applies
                        # ``new_value`` against whatever is current.
                        set_fact((name, op.attr), op.new_value, op.pos)
                case UpsertRelOp():
                    name_a, name_b = names.get(op.eid_a), names.get(op.eid_b)
                    if name_a is not None and name_b is not None:
                        relations.add((name_a, op.rel, name_b))
                case DropRelOp():
                    name_a, name_b = names.get(op.eid_a), names.get(op.eid_b)
                    if name_a is not None and name_b is not None:
                        relations.discard((name_a, op.rel, name_b))
                case UpsertNegOp():
                    name = names.get(op.eid)
                    if name is not None:
                        negations.add((name, op.attr))
                case DropNegOp():
                    name = names.get(op.eid)
                    if name is not None:
                        negations.discard((name, op.attr))
                case DropFactOp():
                    name = names.get(op.eid)
                    if name is not None:
                        current.pop((name, op.attr), None)
                case MarkUnresolvedOp():
                    name = names.get(op.eid)
                    if name is not None:
                        unresolved[(name, op.attr)] = op.candidates
                case ResolveOp():
                    name = names.get(op.eid)
                    if name is not None:
                        unresolved.pop((name, op.attr), None)
                        set_fact((name, op.attr), op.value, op.pos)
                case _:
                    assert_never(op)

        return FoldedState(names, current, relations, negations, unresolved, histories)
