"""Per-sample compression-budget sampling (uniform over 2x..32x ratios).

The single-shot shards store C* renders keyed by compression-ratio string
("2","4","8","16","32"). Training samples draw a ratio so the model learns
compaction across the whole rate-distortion curve from one checkpoint.
"""
from __future__ import annotations

import random

RATIOS: tuple[int, ...] = (2, 4, 8, 16, 32)


def available_ratios(c_renders: dict) -> tuple[int, ...]:
    """Ratio ints actually present in a record's c_renders."""
    present = []
    for r in RATIOS:
        if str(r) in c_renders and len(c_renders[str(r)].target_ids) > 0:
            present.append(r)
    return tuple(present)


def sample_ratio(rng: random.Random, c_renders: dict) -> int:
    """Uniform pick among the record's available ratios."""
    options = available_ratios(c_renders)
    if not options:
        raise ValueError("record has no non-empty c_renders to sample from")
    return rng.choice(options)
