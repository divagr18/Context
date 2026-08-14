"""Per-layer attention entropy diagnostic (PLAN 6, entropy diagnostics).

Runs ONE eager forward pass with two 512-token id prefixes, computes per-layer
mean attention entropy in nats. Pure function, no state mutation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def layer_attention_entropy(
    model, prefix_a: Tensor, prefix_b: Tensor,
) -> dict[int, float]:
    """Per-layer mean attention entropy (nats) for two diagnostic prefixes.

    Args:
        model: Transformer instance (will be set to eval mode internally).
        prefix_a: 1D tensor of token IDs (length 512).
        prefix_b: 1D tensor of token IDs (length 512).

    Returns:
        {layer_idx: mean_entropy_nats} for all layers.
    """
    model.eval()
    # Stack prefixes into batch dimension.
    ids = torch.stack([prefix_a, prefix_b], dim=0).long()

    with torch.no_grad():
        _, all_probs = model.forward_with_attention_probs(ids)

    entropies: dict[int, float] = {}
    for layer_idx, probs in enumerate(all_probs):
        # probs shape: (B, n_heads, T, T)
        # Entropy per attention head per position: -sum(p * log(p))
        # Mean over batch, heads, and query positions.
        p = probs.clamp(min=1e-9)
        entropy = -(p * p.log()).sum(dim=-1)  # (B, n_heads, T)
        entropies[layer_idx] = entropy.mean().item()

    return entropies
