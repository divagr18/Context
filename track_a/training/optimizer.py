"""Optimizer + LR schedule: AdamW with weight-decay groups and cosine+warmup.

Matches PLAN 8.1: AdamW betas, no decay on norms/embeddings (via
``Transformer.param_groups``), cosine decay to ``min_lr_frac`` of peak after a
linear warmup over ``warmup_frac`` of total steps.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


def build_optimizer(params: Iterable[nn.Parameter], cfg) -> torch.optim.AdamW:
    """AdamW with the run config's betas, weight decay, and peak LR.

    ``params`` may be an iterable of parameters or of param-group dicts; weight
    decay is applied to provided groups as-is (use ``Transformer.param_groups``
    to skip decay on norms/embeddings).
    """
    params = list(params)
    if params and isinstance(params[0], dict):
        groups = params
    else:
        groups = [{"params": params, "weight_decay": cfg.weight_decay}]
    for g in groups:
        g.setdefault("lr", cfg.lr)
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                             weight_decay=cfg.weight_decay)


def lr_at(step: int, cfg, total_steps: int) -> float:
    """LR for an optimiser step under linear warmup + cosine decay."""
    warmup_steps = max(1, int(cfg.warmup_frac * total_steps))
    min_lr = cfg.lr * cfg.min_lr_frac
    if step < warmup_steps:
        return cfg.lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return cfg.lr
    progress = (step - warmup_steps) / max(1, total_steps - 1 - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (cfg.lr - min_lr) * (1.0 + math.cos(math.pi * progress))
