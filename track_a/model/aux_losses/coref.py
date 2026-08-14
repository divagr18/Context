"""Entity-coreference aux loss (PLAN 6.3, loss 1).

InfoNCE over attention patterns: for designated heads, attention mass from a
token of entity E should concentrate on the OTHER tokens of E rather than on
tokens of other entities. Operates purely on AuxBundle.layer_attn.
"""
from __future__ import annotations

import torch
from torch import Tensor

from track_a.model.aux_losses.heads import gather_head_attn

_EPS = 1e-8


def coref_loss(
    layer_attn: tuple[Tensor, ...],
    aux_layers: tuple[int, ...],
    heads: tuple[tuple[int, int], ...],
    token_entity: Tensor | None,
) -> Tensor:
    """Mean InfoNCE over tokens belonging to entities with >=1 peer token."""
    if token_entity is None:
        raise ValueError("coref_loss requires supervision.token_entity")
    attn = gather_head_attn(layer_attn, aux_layers, heads).mean(dim=0)  # (B,T,T)
    B, T, _ = attn.shape
    eye = torch.eye(T, dtype=torch.bool, device=attn.device)
    attn = attn.masked_fill(eye, 0.0)

    per_sample: list[Tensor] = []
    for b in range(B):
        ent = token_entity[b]
        ids = torch.unique(ent)
        ids = ids[ids >= 0]
        if ids.numel() < 2:
            continue  # need at least two entities to be informative
        E = ids.numel()
        onehot = torch.zeros(T, E, device=attn.device)
        for k in range(E):
            onehot[:, k] = (ent == ids[k]).to(attn.dtype)
        mass = attn[b] @ onehot  # (T, E): attention mass to each entity
        match = ent.unsqueeze(1) == ids.unsqueeze(0)  # (T, E)
        col = match.to(attn.dtype).argmax(dim=1)  # own-entity column per token
        valid = ent >= 0
        num = mass[valid, col[valid]]
        den = mass[valid].sum(dim=-1)
        per_sample.append((torch.log(den + _EPS) - torch.log(num + _EPS)).mean())

    if not per_sample:
        return torch.zeros((), device=attn.device)
    return torch.stack(per_sample).mean()
