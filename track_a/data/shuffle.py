"""Seeded deterministic shuffle buffer.

Classic reservoir-buffer shuffle: fill a buffer of size ``k``, then for each
incoming item append it and pop a uniformly random element. Deterministic for
a given seed, bounded memory, single pass — suitable for large JSONL shards.
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Iterator


class ShuffleBuffer:
    """Bounded-memory deterministic shuffle over a stream."""

    def __init__(self, seed: int, buffer_size: int):
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        self.seed = seed
        self.buffer_size = buffer_size

    def __call__(self, iterable: Iterable) -> Iterator:
        rng = random.Random(self.seed)
        buf: list = []
        it = iter(iterable)
        for item in it:
            buf.append(item)
            if len(buf) >= self.buffer_size:
                break
        yield from _drain_and_mix(buf, it, rng)


def _drain_and_mix(buf: list, it, rng: random.Random) -> Iterator:
    for item in it:
        buf.append(item)
        idx = rng.randrange(len(buf))
        yield buf[idx]
        buf[idx] = buf[-1]
        buf.pop()
    rng.shuffle(buf)
    yield from buf
