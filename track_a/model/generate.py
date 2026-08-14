"""KV-cache greedy decoding for single-shot compaction samples.

Hooks the model's submodules directly (embed / blocks / norms / head) so the
frozen C7 ``Transformer.forward`` is untouched. Position-aware RoPE is applied
per decode step since ``apply_rope`` assumes positions start at 0. Returns the
generated token list excluding the stop token. Used by recall-shaping (T7) and
the eval battery (T8).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _rope_freqs_for(head_dim: int, positions: Tensor, theta: float,
                    device: torch.device) -> Tensor:
    """Complex RoPE frequencies for arbitrary absolute positions."""
    half = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(half, device=device,
                                        dtype=torch.float32) / half))
    angles = torch.outer(positions.to(device=device, dtype=torch.float32), inv)
    return torch.polar(torch.ones_like(angles), angles)


def _rotate(x: Tensor, freqs: Tensor) -> Tensor:
    """Rotate the two halves of ``x`` by ``freqs`` (same math as apply_rope)."""
    cos = freqs.real.unsqueeze(0).unsqueeze(0)
    sin = freqs.imag.unsqueeze(0).unsqueeze(0)
    half = x.shape[-1] // 2
    lo, hi = x[..., :half], x[..., half:]
    return torch.cat([lo * cos - hi * sin, lo * sin + hi * cos], dim=-1)


@torch.no_grad()
def greedy_generate(model, prompt_ids, max_new_tokens: int,
                    stop_token_ids) -> list[int]:
    """Greedy KV-cache decode of one sequence.

    Args:
        model: Transformer (frozen C7), weights read-only.
        prompt_ids: 1-D sequence of token ids.
        max_new_tokens: cap on generated tokens.
        stop_token_ids: iterable of token ids that stop generation (excluded).

    Returns:
        Generated token ids (stop token not included).
    """
    model.eval()
    cfg = model.cfg
    device = next(model.parameters()).device
    stop = {int(t) for t in stop_token_ids}
    hd = cfg.head_dim
    B = 1

    if torch.is_tensor(prompt_ids):
        ids = prompt_ids.detach().reshape(-1).tolist()
    else:
        ids = list(prompt_ids)

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    T = len(ids)
    x = model.embed(input_ids)
    freqs = _rope_freqs_for(hd, torch.arange(T, device=device),
                            cfg.rope_theta, device)

    k_cache: list[Tensor] = []
    v_cache: list[Tensor] = []
    for block in model.blocks:
        attn = block.attn
        normed = block.attn_norm(x)
        q = attn.q_proj(normed).view(B, T, attn.n_heads, hd).transpose(1, 2)
        k = attn.k_proj(normed).view(B, T, attn.n_kv_heads, hd).transpose(1, 2)
        v = attn.v_proj(normed).view(B, T, attn.n_kv_heads, hd).transpose(1, 2)
        q = _rotate(q, freqs)
        k = _rotate(k, freqs)
        k_cache.append(k)
        v_cache.append(v)
        ke = attn._expand_kv(k)
        ve = attn._expand_kv(v)
        ao = F.scaled_dot_product_attention(q, ke, ve, is_causal=True)
        ao = ao.transpose(1, 2).contiguous().view(B, T, attn.n_heads * hd)
        x = x + attn.o_proj(ao)
        x = x + block.ffn(block.ffn_norm(x))

    logits = model.head(model.final_norm(x)[:, -1, :])
    next_tok = int(logits.argmax(-1).item())
    generated: list[int] = []
    pos = T
    while next_tok not in stop and len(generated) < max_new_tokens:
        generated.append(next_tok)
        inp = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        x = model.embed(inp)
        step_freqs = _rope_freqs_for(hd, torch.tensor([pos], device=device),
                                     cfg.rope_theta, device)
        for i, block in enumerate(model.blocks):
            attn = block.attn
            normed = block.attn_norm(x)
            q = attn.q_proj(normed).view(B, 1, attn.n_heads, hd).transpose(1, 2)
            k = attn.k_proj(normed).view(B, 1, attn.n_kv_heads, hd).transpose(1, 2)
            v = attn.v_proj(normed).view(B, 1, attn.n_kv_heads, hd).transpose(1, 2)
            q = _rotate(q, step_freqs)
            k = _rotate(k, step_freqs)
            k_cache[i] = torch.cat([k_cache[i], k], dim=2)
            v_cache[i] = torch.cat([v_cache[i], v], dim=2)
            ke = attn._expand_kv(k_cache[i])
            ve = attn._expand_kv(v_cache[i])
            ao = F.scaled_dot_product_attention(q, ke, ve, is_causal=False)
            ao = ao.transpose(1, 2).contiguous().view(B, 1, attn.n_heads * hd)
            x = x + attn.o_proj(ao)
            x = x + block.ffn(block.ffn_norm(x))
        logits = model.head(model.final_norm(x)[:, -1, :])
        next_tok = int(logits.argmax(-1).item())
        pos += 1
    return generated
