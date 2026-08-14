"""Rotary Position Embedding (RoPE) — configurable base theta."""

from __future__ import annotations

import torch
from torch import Tensor


def build_rope_freqs(
    head_dim: int, max_seq: int, theta: float, device: torch.device,
) -> Tensor:
    """Precompute complex frequency table (max_seq, head_dim//2)."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    positions = torch.arange(max_seq, device=device, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(q: Tensor, k: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate q/k halves by position-dependent complex angles."""
    T = q.shape[2]
    cos = freqs[:T].real.unsqueeze(0).unsqueeze(0)
    sin = freqs[:T].imag.unsqueeze(0).unsqueeze(0)
    half = q.shape[-1] // 2
    q_lo, q_hi = q[..., :half], q[..., half:]
    k_lo, k_hi = k[..., :half], k[..., half:]
    q_out = torch.empty_like(q)
    q_out[..., :half] = q_lo * cos - q_hi * sin
    q_out[..., half:] = q_lo * sin + q_hi * cos
    k_out = torch.empty_like(k)
    k_out[..., :half] = k_lo * cos - k_hi * sin
    k_out[..., half:] = k_lo * sin + k_hi * cos
    return q_out, k_out
