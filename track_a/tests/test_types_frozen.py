"""T-INT: types.py is the keystone contract (C1) — pin its exact bytes."""
from __future__ import annotations

import hashlib
from pathlib import Path

TYPES_SHA256 = "dc64347b58a84c779a4603741defea386e68e0a540c1cc8a9b4fea3ab95d6e7a"


def test_types_frozen_hash() -> None:
    path = Path(__file__).resolve().parent.parent / "needle_gen" / "types.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == TYPES_SHA256, (
        "track_a/needle_gen/types.py changed. It is the frozen keystone "
        "contract (PLAN C1); any change requires an explicit PLAN.md "
        "contract decision and this pin updated deliberately."
    )
