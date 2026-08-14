"""Negation aux loss (PLAN 6.3, loss 3).

Per-token binary classifier on the final representation: does this token
belong to a negated/denied-claim span? Labels come from the generator's
CueSpan(kind=NEGATION) annotations; -1 marks unsupervised tokens. Consumes
AuxBundle.aux_hidden.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def negation_loss(
    probe: nn.Linear,
    hidden: Tensor,
    token_negated: Tensor | None,
) -> Tensor:
    """Masked per-token BCE over negation-cue labels."""
    if token_negated is None:
        raise ValueError("negation_loss requires supervision.token_negated")
    logits = probe(hidden).squeeze(-1)  # (B, T)
    mask = token_negated >= 0
    if not mask.any():
        return torch.zeros((), device=hidden.device)
    targets = token_negated[mask].to(hidden.dtype)
    return F.binary_cross_entropy_with_logits(logits[mask], targets)
