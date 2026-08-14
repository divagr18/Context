"""RMSNorm — root-mean-square layer normalization."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Pre-norm RMSNorm (no bias, learnable scale only)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight
