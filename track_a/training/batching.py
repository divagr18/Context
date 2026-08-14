"""Batch collation: pad PackSamples into padded (input_ids, labels) batches.

Source positions are already IGNORE_INDEX in the per-sample labels; padding
adds IGNORE_INDEX labels and a False attention-mask column. ``n_tokens``
counts the non-pad tokens (the training-token budget accounting unit).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from track_a.data.pack import IGNORE_INDEX


@dataclass
class Batch:
    input_ids: Tensor
    labels: Tensor
    attention_mask: Tensor
    n_tokens: int


def collate(samples, pad_id: int, device=None) -> Batch:
    max_len = max(len(s.input_ids) for s in samples)
    B = len(samples)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.bool)
    n_tokens = 0
    for i, s in enumerate(samples):
        L = len(s.input_ids)
        input_ids[i, :L] = torch.tensor(s.input_ids, dtype=torch.long)
        labels[i, :L] = torch.tensor(s.labels, dtype=torch.long)
        attention_mask[i, :L] = True
        n_tokens += L
    if device is not None:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
    return Batch(input_ids, labels, attention_mask, n_tokens)
