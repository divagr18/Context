"""Needle-salience aux loss (PLAN 6.3, loss 5).

Per-token binary classifier on the final representation: is this token part
of a later-queried (important) fact span, or a decoy? Labels come from the
generator's queried/decoy fact flags; -1 marks unsupervised tokens. This is
the closest analog to a learned salience/attention-steering signal. Consumes
AuxBundle.aux_hidden.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def salience_loss(
    probe: nn.Linear,
    hidden: Tensor,
    token_salient: Tensor | None,
) -> Tensor:
    """Masked per-token BCE over queried-vs-decoy labels."""
    if token_salient is None:
        raise ValueError("salience_loss requires supervision.token_salient")
    logits = probe(hidden).squeeze(-1)  # (B, T)
    mask = token_salient >= 0
    if not mask.any():
        return torch.zeros((), device=hidden.device)
    targets = token_salient[mask].to(hidden.dtype)
    return F.binary_cross_entropy_with_logits(logits[mask], targets)
