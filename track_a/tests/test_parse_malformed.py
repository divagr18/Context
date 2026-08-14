"""Scenario S2: malformed input never raises, always counted, never mixed.

Every line below violates grammar C2 (truncation, garbage tokens, missing
fields, wrong order, unterminated/ill-formed candidates lists, whitespace
splitting a field). The parser must return a ParseFailure for each -- no
exceptions, no silent acceptance, no mixing into the record list.
"""

from __future__ import annotations

import pytest

from track_a.parse_compact import (
    ParseFailure,
    Record,
    OpRecord,
    parse_ops_lines,
    parse_target_lines,
)

# Malformed full-state record lines (14 distinct inputs).
MALFORMED_TARGET_LINES: list[str] = [
    "",                                              # empty line
    "   ",                                           # whitespace only
    "UPSERT FACT E1.a = v pos=1",                    # op verb in a target stream
    "ENTITY E1 type=person",                         # truncated (missing name=)
    "ENTITY E1 type=person name=",                   # empty name value
    "FACT E1.a =",                                   # truncated (missing value/pos)
    "FACT E1.a = v pos=three",                       # non-integer pos
    "FACT E1.a = v hedged supersedes=old pos=1",     # optionals in wrong order
    "REL E1 manages pos=2",                          # missing eid_b
    "NEG E1.a pos=2",                                # missing literal 'denied'
    "UNRESOLVED E1.a candidates=[v1@1, v2@2",        # unterminated candidates
    "UNRESOLVED E1.a candidates=[]",                 # empty candidates list
    "UNRESOLVED E1.a candidates=[v1@one, v2@2]",     # candidate pos not an integer
    "FACT E 1.a = v pos=1",                          # whitespace inside the eid field
]

# Malformed streaming edit-op lines (10 distinct inputs).
MALFORMED_OP_LINES: list[str] = [
    "SUPERSEDE FACT E1.a : v1 => v2",                # missing pos=
    "SUPERSEDE FACT E1.a : v1 = > v2 pos=1",         # '=>' split by whitespace
    "DROP ENTITY E1 pos=0",                          # DROP has no ENTITY subject
    "MARK RESOLVED E1.a",                            # MARK only supports UNRESOLVED
    "RESOLVE E1.a = v",                              # missing pos=
    "UPSERT FACT E1.a v pos=1",                      # missing '=' separator
    "UPSERT FACT E1.a = v pos=1 extra",              # trailing tokens
    "MARK UNRESOLVED E1.a candidates=[v1]",          # candidate missing @pos
    "DROP FACT E1.a",                                # missing pos=
    "frobnicate the widget",                         # pure garbage verb
]

VALID_TARGET_LINE = "FACT E1.role = lead pos=3"
VALID_OP_LINE = "UPSERT FACT E1.role = lead pos=3"


def test_distinct_malformed_corpus_is_large_enough() -> None:
    corpus = MALFORMED_TARGET_LINES + MALFORMED_OP_LINES
    assert len(set(corpus)) >= 12


@pytest.mark.parametrize("line", MALFORMED_TARGET_LINES)
def test_malformed_full_state_line_yields_only_a_failure(line: str) -> None:
    records, failures = parse_target_lines([line])
    assert records == []
    assert len(failures) == 1
    assert isinstance(failures[0], ParseFailure)
    assert failures[0].line == line
    assert failures[0].reason  # deterministic, non-empty


@pytest.mark.parametrize("line", MALFORMED_OP_LINES)
def test_malformed_op_line_yields_only_a_failure(line: str) -> None:
    ops, failures = parse_ops_lines([line])
    assert ops == []
    assert len(failures) == 1
    assert isinstance(failures[0], ParseFailure)
    assert failures[0].line == line
    assert failures[0].reason


def test_full_malformed_sweep_raises_nothing_and_counts_all() -> None:
    records, failures = parse_target_lines(MALFORMED_TARGET_LINES)
    assert records == []
    assert len(failures) == len(MALFORMED_TARGET_LINES)
    ops, op_failures = parse_ops_lines(MALFORMED_OP_LINES)
    assert ops == []
    assert len(op_failures) == len(MALFORMED_OP_LINES)


def test_failure_reasons_are_deterministic() -> None:
    for line in MALFORMED_TARGET_LINES:
        _, first = parse_target_lines([line])
        _, second = parse_target_lines([line])
        assert first[0].reason == second[0].reason
    for line in MALFORMED_OP_LINES:
        _, first = parse_ops_lines([line])
        _, second = parse_ops_lines([line])
        assert first[0].reason == second[0].reason


def test_failures_strictly_separated_from_records_in_mixed_target_stream() -> None:
    lines = [VALID_TARGET_LINE, "FACT E1.a = v pos=x", VALID_TARGET_LINE, "garbage"]
    records, failures = parse_target_lines(lines)
    assert len(records) + len(failures) == len(lines)
    assert len(records) == 2
    assert len(failures) == 2
    assert all(isinstance(r, Record.__args__) for r in records)
    assert all(isinstance(f, ParseFailure) for f in failures)
    assert all(f.line in lines for f in failures)


def test_failures_strictly_separated_from_ops_in_mixed_op_stream() -> None:
    lines = [VALID_OP_LINE, "", VALID_OP_LINE, "DROP ENTITY E1 pos=0"]
    ops, failures = parse_ops_lines(lines)
    assert len(ops) + len(failures) == len(lines)
    assert len(ops) == 2
    assert len(failures) == 2
    assert all(isinstance(o, OpRecord.__args__) for o in ops)
    assert all(isinstance(f, ParseFailure) for f in failures)
