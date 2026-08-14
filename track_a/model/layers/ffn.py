"""SwiGLU Feed-Forward Network: gate * up -> down."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLUFFN(nn.Module):
    """SwiGLU: gate(up(x)) * down(x) pattern, bias-free."""

    def __init__(self, d_model: int, ffn_hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_hidden, bias=False)
        self.up = nn.Linear(d_model, ffn_hidden, bias=False)
        self.down = nn.Linear(ffn_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
