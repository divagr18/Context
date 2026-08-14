"""Track A dataset layer (T5).

Reads generated JSONL shards (C6) and yields packed training samples with
loss masking restricted to the C*/ops/answer span.

Public entry points:
    TrainingDataset      -- iterable sample stream (single_shot / streaming)
    PackSample           -- packed (input_ids, labels) pair
    pack_single_shot / pack_streaming_window / pack_qa
    iter_shards / iter_shards_many
    ShuffleBuffer, available_ratios / sample_ratio, window_samples
"""

from track_a.data.budget_sampler import available_ratios, sample_ratio
from track_a.data.dataset import TrainingDataset
from track_a.data.pack import (
    IGNORE_INDEX, PackSample, budget_token_id, pack_qa, pack_single_shot,
    pack_streaming_window, special_id,
)
from track_a.data.shard_reader import iter_shards, iter_shards_many
from track_a.data.shuffle import ShuffleBuffer
from track_a.data.window_samples import window_samples

__all__ = [
    "IGNORE_INDEX",
    "PackSample",
    "ShuffleBuffer",
    "TrainingDataset",
    "available_ratios",
    "budget_token_id",
    "iter_shards",
    "iter_shards_many",
    "pack_qa",
    "pack_single_shot",
    "pack_streaming_window",
    "sample_ratio",
    "special_id",
    "window_samples",
]
