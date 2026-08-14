"""Golden grammar vectors parsed exactly (grammar C2, contracts W4/C4).

Vector schema -- one JSON object per line in ``golden/grammar_vectors.jsonl``,
a cross-module contract the generator asserts against:

    {"lines": [...],
     "expected_records": [...]  OR  "expected_ops": [...],
     "expected_failure_count": int}

``expected_records`` vectors are fed to ``parse_target_lines`` (full-state
grammar), ``expected_ops`` vectors to ``parse_ops_lines`` (streaming edit-ops).
Each expected object carries ``"kind"`` -- the ``kind`` class attribute of the
matching ``parse_compact.records`` dataclass -- plus exactly that dataclass's
field names; tuple fields serialize as JSON lists. Only the failure COUNT is
contract here; ``ParseFailure.reason`` strings are deterministic but internal.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from track_a.parse_compact import ParseFailure, parse_ops_lines, parse_target_lines

_GOLDEN = Path(__file__).resolve().parent / "golden" / "grammar_vectors.jsonl"


def _jsonify(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    return value


def _to_dict(record: Any) -> dict[str, Any]:
    fields = {f.name: _jsonify(getattr(record, f.name)) for f in dataclasses.fields(record)}
    return {"kind": type(record).kind, **fields}


def _load_vectors() -> list[dict[str, Any]]:
    with _GOLDEN.open(encoding="utf-8") as fh:
        return [json.loads(raw) for raw in fh if raw.strip()]


VECTORS = _load_vectors()


def test_golden_file_exists_with_minimum_coverage() -> None:
    assert len(VECTORS) >= 10
    assert sum(v["expected_failure_count"] for v in VECTORS) >= 4
    # Every vector declares exactly one expectation channel.
    for vector in VECTORS:
        assert ("expected_records" in vector) != ("expected_ops" in vector)


@pytest.mark.parametrize("vector", VECTORS, ids=[f"v{i:02d}" for i in range(len(VECTORS))])
def test_golden_vector_parses_exactly(vector: dict[str, Any]) -> None:
    lines = vector["lines"]
    if "expected_records" in vector:
        parsed, failures = parse_target_lines(lines)
        assert [_to_dict(r) for r in parsed] == vector["expected_records"]
    else:
        parsed, failures = parse_ops_lines(lines)
        assert [_to_dict(o) for o in parsed] == vector["expected_ops"]
    assert len(failures) == vector["expected_failure_count"]


@pytest.mark.parametrize("vector", VECTORS, ids=[f"v{i:02d}" for i in range(len(VECTORS))])
def test_every_line_lands_in_exactly_one_bucket(vector: dict[str, Any]) -> None:
    """Records/ops and failures strictly partition the input lines."""
    lines = vector["lines"]
    if "expected_records" in vector:
        parsed, failures = parse_target_lines(lines)
    else:
        parsed, failures = parse_ops_lines(lines)
    assert len(parsed) + len(failures) == len(lines)
    for failure in failures:
        assert isinstance(failure, ParseFailure)
        assert failure.line in lines
        assert failure.reason
