"""Temporal/state-order aux loss (PLAN 6.3, loss 2).

Probe classifier over the final representation predicting relative order of
two positions: for consecutive (earlier, later) positions in story time,
probe(h_later - h_earlier) should be positive; the reversed difference
should be negative. Consumes AuxBundle.aux_hidden.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


def temporal_loss(
    probe: nn.Linear,
    hidden: Tensor,
    order_positions: Tensor | None,
    order_mask: Tensor | None,
) -> Tensor:
    """BCE over ordered position-pair representation differences."""
    if order_positions is None or order_mask is None:
        raise ValueError(
            "temporal_loss requires supervision.order_positions/order_mask"
        )
    B, _K = order_positions.shape
    logits: list[Tensor] = []
    targets: list[Tensor] = []
    for b in range(B):
        valid = torch.nonzero(order_mask[b], as_tuple=False).squeeze(-1)
        if valid.numel() < 2:
            continue
        pos_seq = order_positions[b][valid]
        for j in range(pos_seq.numel() - 1):
            p0 = int(pos_seq[j])
            p1 = int(pos_seq[j + 1])
            if p1 <= p0:
                continue  # positions must ascend in story time
            h0 = hidden[b, p0]
            h1 = hidden[b, p1]
            logits.append(probe(h1 - h0))
            targets.append(torch.ones((), device=hidden.device))
            logits.append(probe(h0 - h1))
            targets.append(torch.zeros((), device=hidden.device))

    if not logits:
        return torch.zeros((), device=hidden.device)
    logits_t = torch.cat([l.reshape(1) for l in logits])
    targets_t = torch.stack(targets)
    return F.binary_cross_entropy_with_logits(logits_t, targets_t)
