"""Parameter counting — single source of truth (PLAN 6.2).

The solver (configs/solve_ratio.py) imports these functions; no duplicate
formulas exist elsewhere. Every count assumes bias-free linear layers and
tied embeddings (output head shares the embedding weight).

Per-layer attention projections (no bias):
    Q:  d_model -> n_heads * head_dim        =>  d_model * (n_heads * head_dim)
    K:  d_model -> n_kv_heads * head_dim     =>  d_model * (n_kv_heads * head_dim)
    V:  d_model -> n_kv_heads * head_dim     =>  d_model * (n_kv_heads * head_dim)
    O:  n_heads * head_dim -> d_model        =>  (n_heads * head_dim) * d_model

Per-layer SwiGLU FFN (gate + up + down, no bias):
    3 * d_model * ffn_hidden

Embedding (tied, counted once):  vocab_size * d_model
Norms:  n_layers * 2 * d_model  +  d_model  (RMSNorm has one weight per dim)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from track_a.model.config import ModelConfig


def embedding_params(vocab_size: int, d_model: int) -> int:
    """Tied embedding counted ONCE (input + output share weight)."""
    return vocab_size * d_model


def attention_params_per_layer(
    d_model: int, n_heads: int, head_dim: int, n_kv_heads: int,
) -> int:
    """Q + K + V + O projections, all bias-free."""
    q = d_model * (n_heads * head_dim)
    k = d_model * (n_kv_heads * head_dim)
    v = d_model * (n_kv_heads * head_dim)
    o = (n_heads * head_dim) * d_model
    return q + k + v + o


def ffn_params_per_layer(d_model: int, ffn_hidden: int) -> int:
    """SwiGLU: gate + up + down, all bias-free."""
    return 3 * d_model * ffn_hidden


def norm_params(n_layers: int, d_model: int) -> int:
    """2 RMSNorm per layer (pre-attn, pre-ffn) + final RMSNorm."""
    return n_layers * 2 * d_model + d_model


def total_params(cfg: ModelConfig) -> int:
    """Full model parameter count (tied embeddings counted once)."""
    emb = embedding_params(cfg.vocab_size, cfg.d_model)
    attn = attention_params_per_layer(
        cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.n_kv_heads,
    )
    ffn = ffn_params_per_layer(cfg.d_model, cfg.ffn_hidden)
    norms = norm_params(cfg.n_layers, cfg.d_model)
    return emb + cfg.n_layers * (attn + ffn) + norms


def attn_ratio(cfg: ModelConfig) -> float:
    """attention_params / (attention_params + ffn_params) per layer."""
    attn = attention_params_per_layer(
        cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.n_kv_heads,
    )
    ffn = ffn_params_per_layer(cfg.d_model, cfg.ffn_hidden)
    return attn / (attn + ffn)
