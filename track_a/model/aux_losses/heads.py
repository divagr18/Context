"""Aux head assignment + probe modules + attention gather (PLAN 6.3).

Heads are designated as disjoint (layer, head) pairs drawn from the LAST
ceil(n_layers/3) layers (``ModelConfig.aux_layer_indices``), round-robin
across enabled losses. Attention-pattern losses consume the gathered matrices;
probe losses use small linear probes over ``AuxBundle.aux_hidden``.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from track_a.model.aux_losses.config import AuxLossConfig
from track_a.model.config import ModelConfig


def assign_heads(
    cfg: ModelConfig, aux_cfg: AuxLossConfig,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Disjoint (layer, head) assignments per enabled aux loss.

    Raises if the aux-layer head pool cannot supply heads_per_loss to every
    enabled loss without overlap.
    """
    enabled = aux_cfg.enabled()
    if not enabled:
        return {}
    aux_layers = cfg.aux_layer_indices()
    pool = [(l, h) for l in aux_layers for h in range(cfg.n_heads)]
    need = len(enabled) * aux_cfg.heads_per_loss
    if need > len(pool):
        raise ValueError(
            f"aux head pool too small: need {need} (heads_per_loss="
            f"{aux_cfg.heads_per_loss} x {len(enabled)} losses), have {len(pool)}"
        )
    out: dict[str, tuple[tuple[int, int], ...]] = {}
    i = 0
    for name in enabled:
        out[name] = tuple(tuple(pool[i + k]) for k in range(aux_cfg.heads_per_loss))
        i += aux_cfg.heads_per_loss
    return out


def gather_head_attn(
    layer_attn: tuple[Tensor, ...],
    aux_layers: tuple[int, ...],
    heads: tuple[tuple[int, int], ...],
) -> Tensor:
    """Stack attention matrices for the given (layer, head) pairs -> (N,B,T,T).

    ``layer_attn`` is ordered to match ``aux_layers`` (ascending), so the layer
    index within the tuple is ``aux_layers.index(layer)``.
    """
    if not heads:
        raise ValueError("no heads assigned")
    layer_index = {l: k for k, l in enumerate(aux_layers)}
    mats = []
    for layer, head in heads:
        mats.append(layer_attn[layer_index[layer]][:, head])
    return torch.stack(mats, dim=0)


class AuxModules(nn.Module):
    """Trainable probe classifiers for the representation-based aux losses."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.temporal_probe = nn.Linear(d_model, 1)
        self.negation_probe = nn.Linear(d_model, 1)
        self.salience_probe = nn.Linear(d_model, 1)
