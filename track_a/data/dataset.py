"""Iterable training dataset over generated shards.

Ties together the shard reader, budget sampler, packer, and optional shuffle
buffer. Produces ``PackSample``s for one shard kind:

* ``single_shot``: one sample per (doc, available compression ratio).
* ``streaming``: one sample per non-empty edit-op window.

C*/ops/answer targets are read verbatim from the shard (never re-rendered),
and loss masking keeps gradients only on the target span (see pack.py).
"""
from __future__ import annotations

import random
from collections.abc import Iterator

from track_a.data.budget_sampler import available_ratios, sample_ratio
from track_a.data.pack import PackSample, pack_single_shot
from track_a.data.shard_reader import iter_shards_many
from track_a.data.shuffle import ShuffleBuffer
from track_a.data.window_samples import window_samples
from track_a.shard_schema import KIND_SINGLE_SHOT, KIND_STREAMING


class TrainingDataset:
    """Lazy, single-pass, optionally shuffled sample stream."""

    def __init__(self, shard_paths, tok, kind: str = KIND_SINGLE_SHOT,
                 seed: int = 0, shuffle_buffer: int = 0,
                 all_ratios: bool = True):
        if kind not in (KIND_SINGLE_SHOT, KIND_STREAMING):
            raise ValueError(f"unknown kind: {kind}")
        self.shard_paths = list(shard_paths)
        self.tok = tok
        self.kind = kind
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.all_ratios = all_ratios

    def _records(self):
        records = iter_shards_many(self.shard_paths)
        if self.shuffle_buffer > 0:
            records = ShuffleBuffer(self.seed, self.shuffle_buffer)(records)
        return records

    def __iter__(self) -> Iterator[PackSample]:
        rng = random.Random(self.seed + 1)
        for rec in self._records():
            if rec.kind != self.kind:
                continue
            if rec.kind == KIND_SINGLE_SHOT:
                yield from self._single_shot(rec, rng)
            else:
                yield from window_samples(rec, self.tok)

    def _single_shot(self, rec, rng) -> Iterator[PackSample]:
        ratios = (list(available_ratios(rec.c_renders)) if self.all_ratios
                  else [sample_ratio(rng, rec.c_renders)])
        for ratio in ratios:
            render = rec.c_renders[str(ratio)]
            yield pack_single_shot(rec.doc_ids, render.target_ids,
                                   rec.doc_len_tokens, ratio, self.tok)
