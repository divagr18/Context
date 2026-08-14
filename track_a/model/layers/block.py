"""Transformer block: pre-norm attention + FFN with residual."""

from __future__ import annotations

from torch import Tensor, nn

from track_a.model.layers.attention import GroupedQueryAttention
from track_a.model.layers.ffn import SwiGLUFFN
from track_a.model.layers.norm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm: RMSNorm->Attn->residual->RMSNorm->FFN->residual."""

    def __init__(
        self, d_model: int, n_heads: int, head_dim: int, n_kv_heads: int,
        ffn_hidden: int, softcap: float | None = None,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model, n_heads, head_dim, n_kv_heads, softcap=softcap,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_hidden)

    def forward(
        self, x: Tensor, rope_freqs: Tensor, return_attn_probs: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        normed = self.attn_norm(x)
        attn_out, probs = self.attn(normed, rope_freqs, return_attn_probs)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, probs
