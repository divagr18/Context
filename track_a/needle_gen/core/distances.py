"""Distance-bucket placement logic for binding needles and deep chains."""
from __future__ import annotations

import random

from track_a.needle_gen.types import (
    DISTANCE_BUCKET_MAX, DOC_LEN_TARGETS, DistanceBucket, GenConfig,
)

# Target distances (tokens between endpoints) used when padding. Each is the
# bucket minimum plus a small margin so greedy filler padding lands inside.
_PAD_TARGETS: dict[DistanceBucket, int] = {
    DistanceBucket.NEAR: 0,      # placed adjacent; always < 500
    DistanceBucket.MID: 600,     # in [500, 4000)
    DistanceBucket.FAR: 4500,    # in [4000, 20000)
    DistanceBucket.EXTREME: 21000,  # > 20000
}


def feasible_buckets(config: GenConfig) -> tuple[DistanceBucket, ...]:
    """Buckets achievable within this config's document length.

    A bucket is feasible when its pad target plus a 300-token margin for the
    endpoint scenes fits inside the doc target.
    """
    target = DOC_LEN_TARGETS[config.doc_len_name]
    out: list[DistanceBucket] = []
    for bucket in (DistanceBucket.NEAR, DistanceBucket.MID, DistanceBucket.FAR, DistanceBucket.EXTREME):
        if _PAD_TARGETS[bucket] + 300 <= target:
            out.append(bucket)
    return tuple(out)


def choose_bucket(rng: random.Random, config: GenConfig) -> DistanceBucket:
    """Uniform pick among feasible buckets (deterministic from rng)."""
    options = feasible_buckets(config)
    if not options:
        raise ValueError("no feasible distance bucket for this doc length")
    return rng.choice(options)


def pad_target(bucket: DistanceBucket) -> int:
    """Tokens of padding to emit before the far endpoint scene."""
    return _PAD_TARGETS[bucket]


def bucket_max(bucket: DistanceBucket) -> int:
    """Exclusive upper bound of the bucket (for verification)."""
    if bucket is DistanceBucket.EXTREME:
        return 10**9
    return DISTANCE_BUCKET_MAX[bucket]
