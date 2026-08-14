"""Transformer — pre-norm decoder-only with configurable ratio/GQA/RoPE/softcap.

Forward returns (logits, AuxBundle). AuxBundle.layer_attn holds attention
probability tensors for the LAST ceil(n_layers/3) layers ONLY when
aux_enabled=True. aux_enabled is for short diagnostic sequences (T <= 2048).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from track_a.model.config import AuxBundle, ModelConfig
from track_a.model.layers.block import TransformerBlock
from track_a.model.layers.norm import RMSNorm
from track_a.model.layers.rope import build_rope_freqs


class Transformer(nn.Module):
    """Pre-norm decoder-only transformer (PLAN 6)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.n_kv_heads,
                cfg.ffn_hidden, softcap=cfg.softcap,
            )
            for _ in range(cfg.n_layers)
        ])
        self.final_norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Tied embeddings: share weight between input and output projections.
        self.head.weight = self.embed.weight

        # RoPE frequency table (precomputed, registered as buffer).
        freqs = build_rope_freqs(cfg.head_dim, cfg.max_seq, cfg.rope_theta, torch.device("cpu"))
        self.register_buffer("rope_freqs", freqs, persistent=False)

        # Aux layer indices: last ceil(n_layers/3) layers.
        self._aux_indices = set(cfg.aux_layer_indices())

        self._init_weights()

    def _init_weights(self) -> None:
        """Normal(0, 0.02) default; norms initialized to 1."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def _move_rope(self, device: torch.device) -> Tensor:
        """Ensure rope freqs are on the right device."""
        if self.rope_freqs.device != device:
            freqs = build_rope_freqs(
                self.cfg.head_dim, self.cfg.max_seq, self.cfg.rope_theta, device,
            )
            self.rope_freqs = freqs
        return self.rope_freqs

    def forward(self, input_ids: Tensor) -> tuple[Tensor, AuxBundle]:
        """Standard forward. Returns (logits, AuxBundle)."""
        x = self.embed(input_ids)
        rope_freqs = self._move_rope(x.device)

        aux_probs: list[Tensor] = []
        for i, block in enumerate(self.blocks):
            want_probs = self.cfg.aux_enabled and (i in self._aux_indices)
            x, probs = block(x, rope_freqs, return_attn_probs=want_probs)
            if want_probs and probs is not None:
                aux_probs.append(probs)

        x = self.final_norm(x)
        logits = self.head(x)

        bundle = AuxBundle(
            layer_attn=tuple(aux_probs) if self.cfg.aux_enabled else None,
        )
        return logits, bundle

    def forward_with_attention_probs(self, input_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Diagnostic forward: returns (logits, all_layer_probs).

        Always materializes attention probabilities for EVERY layer,
        regardless of aux_enabled setting. Used by entropy diagnostics.
        """
        x = self.embed(input_ids)
        rope_freqs = self._move_rope(x.device)

        all_probs: list[Tensor] = []
        for block in self.blocks:
            x, probs = block(x, rope_freqs, return_attn_probs=True)
            if probs is not None:
                all_probs.append(probs)

        x = self.final_norm(x)
        logits = self.head(x)
        return logits, all_probs

    def param_groups(self) -> list[dict]:
        """Weight-decay param groups: norms + embeddings skip decay."""
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "embed" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
