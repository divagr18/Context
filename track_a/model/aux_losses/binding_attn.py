"""Numeric/entity-binding aux loss (PLAN 6.3, loss 4).

Encourages attention from a value token back to its owning-entity token.
Supervised by ``value_owner_pos`` (owning token index for each value token).
Loss is the negative log attention from value token to owner, averaged over
designated heads and valid value tokens (owner must precede the value token
so the causal mask permits it).
"""
from __future__ import annotations

import torch
from torch import Tensor

from track_a.model.aux_losses.heads import gather_head_attn

_EPS = 1e-8


def binding_attn_loss(
    layer_attn: tuple[Tensor, ...],
    aux_layers: tuple[int, ...],
    heads: tuple[tuple[int, int], ...],
    value_owner_pos: Tensor | None,
) -> Tensor:
    """Mean -log attention from value tokens to their owning entity token."""
    if value_owner_pos is None:
        raise ValueError("binding_attn_loss requires supervision.value_owner_pos")
    attn = gather_head_attn(layer_attn, aux_layers, heads).mean(dim=0)  # (B,T,T)
    B, T, _ = attn.shape
    t_idx = torch.arange(T, device=attn.device)

    per_sample: list[Tensor] = []
    for b in range(B):
        owner = value_owner_pos[b]
        valid = (owner >= 0) & (owner < t_idx)  # causal-valid owners only
        if not valid.any():
            continue
        att = attn[b][t_idx[valid], owner[valid]]
        per_sample.append(-torch.log(att + _EPS).mean())

    if not per_sample:
        return torch.zeros((), device=attn.device)
    return torch.stack(per_sample).mean()
