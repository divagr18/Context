"""Track A auxiliary-loss package (PLAN 6.3).

Five toggleable losses, each weighted (default 0.1):
    coref         attention-mass InfoNCE across same-entity tokens
    binding_attn  attention from value tokens to owning-entity tokens
    temporal      probe over representation diffs predicting story order
    negation      probe classifying negation-cue tokens
    salience      probe classifying queried-vs-decoy tokens

``compute_aux_losses`` evaluates the enabled losses from one ``AuxBundle`` and
returns (weighted_total, per-loss detached components) for logging.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from track_a.model.aux_losses.binding_attn import binding_attn_loss
from track_a.model.aux_losses.config import (
    ATTENTION_LOSSES, AuxLossConfig, AuxSupervision, LOSS_NAMES,
)
from track_a.model.aux_losses.coref import coref_loss
from track_a.model.aux_losses.heads import AuxModules, assign_heads, gather_head_attn
from track_a.model.aux_losses.negation import negation_loss
from track_a.model.aux_losses.salience import salience_loss
from track_a.model.aux_losses.temporal import temporal_loss
from track_a.model.config import AuxBundle, ModelConfig

__all__ = [
    "ATTENTION_LOSSES",
    "AuxBundle",
    "AuxLossConfig",
    "AuxModules",
    "AuxSupervision",
    "LOSS_NAMES",
    "assign_heads",
    "binding_attn_loss",
    "compute_aux_losses",
    "coref_loss",
    "gather_head_attn",
    "negation_loss",
    "salience_loss",
    "temporal_loss",
]


def _bundle_device(bundle: AuxBundle) -> torch.device:
    if bundle.aux_hidden is not None:
        return bundle.aux_hidden.device
    if bundle.layer_attn:
        return bundle.layer_attn[0].device
    return torch.device("cpu")


def _require_bundle(bundle: AuxBundle) -> None:
    if bundle.layer_attn is None or bundle.aux_hidden is None:
        raise ValueError(
            "aux losses require a populated AuxBundle (set aux_enabled=True)"
        )


def compute_aux_losses(
    modules: AuxModules,
    bundle: AuxBundle,
    cfg: ModelConfig,
    aux_cfg: AuxLossConfig,
    supervision: AuxSupervision,
    assignments: Optional[dict] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Weighted sum of enabled aux losses + detached per-loss components.

    ``components`` keys are ``aux_<name>`` (detached) for tracker logging.
    The returned total stays in the autograd graph (weights CE + aux sums).
    """
    enabled = aux_cfg.enabled()
    device = _bundle_device(bundle)
    if not enabled:
        return torch.zeros((), device=device), {}
    _require_bundle(bundle)
    if assignments is None:
        assignments = assign_heads(cfg, aux_cfg)
    aux_layers = cfg.aux_layer_indices()

    total: Optional[Tensor] = None
    components: dict[str, Tensor] = {}
    for name in enabled:
        heads = assignments.get(name, ())
        if name == "coref":
            loss = coref_loss(bundle.layer_attn, aux_layers, heads,
                              supervision.token_entity)
        elif name == "binding_attn":
            loss = binding_attn_loss(bundle.layer_attn, aux_layers, heads,
                                     supervision.value_owner_pos)
        elif name == "temporal":
            loss = temporal_loss(modules.temporal_probe, bundle.aux_hidden,
                                 supervision.order_positions,
                                 supervision.order_mask)
        elif name == "negation":
            loss = negation_loss(modules.negation_probe, bundle.aux_hidden,
                                 supervision.token_negated)
        elif name == "salience":
            loss = salience_loss(modules.salience_probe, bundle.aux_hidden,
                                 supervision.token_salient)
        else:
            raise ValueError(f"unknown aux loss {name!r}")
        components[f"aux_{name}"] = loss.detach()
        term = aux_cfg.weight(name) * loss
        total = term if total is None else total + term
    assert total is not None
    return total, components
