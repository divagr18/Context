"""Aux-loss configuration + token-level supervision contract (PLAN 6.3).

Five toggleable auxiliary losses, each with its own weight (default 0.1):
    coref         -- entity coreference InfoNCE over attention patterns
    temporal      -- temporal/state-order probe over the final representation
    negation      -- negation cue binary classifier on the representation
    binding_attn  -- attention from value tokens to their owning entity
    salience      -- queried (important) vs decoy span classifier

Attention-pattern losses (coref, binding_attn) consume AuxBundle.layer_attn;
probe losses (temporal, negation, salience) consume AuxBundle.aux_hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor

LOSS_NAMES: tuple[str, ...] = (
    "coref", "temporal", "negation", "binding_attn", "salience",
)

ATTENTION_LOSSES: tuple[str, ...] = ("coref", "binding_attn")


@dataclass(frozen=True)
class AuxLossConfig:
    """Per-loss toggles + weights + head budget (all config-driven)."""

    coref: bool = False
    temporal: bool = False
    negation: bool = False
    binding_attn: bool = False
    salience: bool = False
    coref_weight: float = 0.1
    temporal_weight: float = 0.1
    negation_weight: float = 0.1
    binding_attn_weight: float = 0.1
    salience_weight: float = 0.1
    heads_per_loss: int = 2

    def __post_init__(self) -> None:
        if self.heads_per_loss < 1:
            raise ValueError("heads_per_loss must be >= 1")
        for name in LOSS_NAMES:
            w = getattr(self, f"{name}_weight")
            if w < 0:
                raise ValueError(f"{name}_weight must be >= 0")

    def enabled(self) -> tuple[str, ...]:
        return tuple(n for n in LOSS_NAMES if getattr(self, n))

    def weight(self, name: str) -> float:
        if name not in LOSS_NAMES:
            raise ValueError(f"unknown aux loss {name!r}")
        return getattr(self, f"{name}_weight")


@dataclass
class AuxSupervision:
    """Token-level ground truth for the aux losses (batch-aligned tensors).

    All tensors are shaped (B, T) unless noted; -1 marks "no supervision".
    token_entity[b, t]      entity id owning token t, -1 for background.
    token_negated[b, t]     1 negation-cue token, 0 non-cue token, -1 ignore.
    token_salient[b, t]     1 queried-fact token, 0 decoy token, -1 ignore.
    value_owner_pos[b, t]   owning-entity token index for value token t,
                            -1 for non-value tokens (owner must precede t).
    order_positions[b, k]   (B, K) token positions ascending in story time;
    order_mask[b, k]        (B, K) bool validity of each position slot.
    """

    token_entity: Optional[Tensor] = None
    token_negated: Optional[Tensor] = None
    token_salient: Optional[Tensor] = None
    value_owner_pos: Optional[Tensor] = None
    order_positions: Optional[Tensor] = None
    order_mask: Optional[Tensor] = None
