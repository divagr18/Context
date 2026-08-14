"""Grouped-Query Attention with SDPA backend selection and softcap.

Backend preference (PLAN 6.1): flash/cudnn -> mem_efficient -> math.
Backend is selected ONCE per process and printed to stderr (for logging).
Actual SDPA calls let PyTorch auto-select; softcap forces eager math path.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_BACKEND_CHOSEN: str | None = None


def _name_to_enum(name: str):
    return {
        "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        "cudnn": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
        "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        "math": torch.nn.attention.SDPBackend.MATH,
    }[name]


def _probe_backend(name: str) -> bool:
    try:
        sb = _name_to_enum(name)
        with torch.nn.attention.sdpa_kernel(sb):
            d = torch.randn(1, 1, 2, 4, device="cpu", dtype=torch.float32)
            F.scaled_dot_product_attention(d, d, d, is_causal=True)
        return True
    except (RuntimeError, Exception):
        return False


def _select_backend(softcap: float | None) -> str:
    global _BACKEND_CHOSEN
    if softcap is not None:
        name = "math (forced by softcap)"
        if _BACKEND_CHOSEN is None:
            print(f"[SDPA] backend={name}", file=sys.stderr, flush=True)
            _BACKEND_CHOSEN = name
        return name
    if _BACKEND_CHOSEN is not None:
        return _BACKEND_CHOSEN
    for candidate in ("flash", "cudnn", "efficient", "math"):
        if _probe_backend(candidate):
            _BACKEND_CHOSEN = candidate
            break
    else:
        _BACKEND_CHOSEN = "math"
    print(f"[SDPA] backend={_BACKEND_CHOSEN}", file=sys.stderr, flush=True)
    return _BACKEND_CHOSEN


class GroupedQueryAttention(nn.Module):
    """Multi-head / grouped-query attention with RoPE and optional softcap.

    Returns (projected_output, attn_probs_or_None).
    """

    def __init__(
        self, d_model: int, n_heads: int, head_dim: int, n_kv_heads: int,
        softcap: float | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.gqa_groups = n_heads // n_kv_heads
        self.scale = head_dim ** -0.5
        self.softcap = softcap

        attn_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        self.q_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(attn_dim, d_model, bias=False)

        self._backend_name = _select_backend(softcap)

    def _expand_kv(self, kv: Tensor) -> Tensor:
        """Repeat KV heads to match query heads for GQA."""
        if self.gqa_groups == 1:
            return kv
        return kv.repeat_interleave(self.gqa_groups, dim=1)

    def forward(
        self, x: Tensor, rope_freqs: Tensor, return_attn_probs: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        from track_a.model.layers.rope import apply_rope
        q, k = apply_rope(q, k, rope_freqs)

        k_exp = self._expand_kv(k)
        v_exp = self._expand_kv(v)

        needs_eager = return_attn_probs or self.softcap is not None
        if needs_eager:
            out, probs = self._eager_attention(q, k_exp, v_exp, return_attn_probs)
        else:
            out = self._sdpa_attention(q, k_exp, v_exp)
            probs = None

        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), probs

    def _sdpa_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        try:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        except RuntimeError:
            return self._eager_attention(q, k, v, False)[0]

    def _eager_attention(
        self, q: Tensor, k: Tensor, v: Tensor, return_probs: bool,
    ) -> tuple[Tensor, Tensor | None]:
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        causal = torch.ones(
            logits.shape[-2:], device=logits.device, dtype=torch.bool,
        ).triu(diagonal=1)
        logits.masked_fill_(causal, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        out = torch.matmul(probs, v)
        return out, probs if return_probs else None
