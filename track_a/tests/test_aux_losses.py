"""T6B: aux losses — values finite, autograd healthy, heads disjoint."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from track_a.model.aux_losses import (
    AuxLossConfig, AuxModules, AuxSupervision, assign_heads,
    binding_attn_loss, compute_aux_losses, coref_loss, negation_loss,
    salience_loss, temporal_loss,
)
from track_a.model.config import ModelConfig
from track_a.model.core import Transformer

B, T = 2, 16


def _small_cfg(aux_enabled: bool = True, n_heads: int = 4) -> ModelConfig:
    return ModelConfig(
        d_model=32, n_layers=6, n_heads=n_heads, head_dim=8, n_kv_heads=2,
        ffn_hidden=64, rope_theta=10000.0, max_seq=64, aux_enabled=aux_enabled,
    )


def _wide_cfg() -> ModelConfig:
    # 3 aux layers x 8 heads = 24-slot pool: room for all 5 losses x 2 heads.
    return ModelConfig(
        d_model=32, n_layers=9, n_heads=8, head_dim=8, n_kv_heads=2,
        ffn_hidden=64, rope_theta=10000.0, max_seq=64, aux_enabled=True,
    )


@pytest.fixture(scope="module")
def small_setup():
    torch.manual_seed(0)
    cfg = _small_cfg(n_heads=6)  # 2 aux layers x 6 heads = 12 slots >= 10 needed
    model = Transformer(cfg)
    modules = AuxModules(cfg.d_model)
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    logits, bundle = model(ids)
    return cfg, model, modules, logits, bundle


@pytest.fixture(scope="module")
def supervision():
    g = torch.Generator().manual_seed(1)
    token_entity = torch.full((B, T), -1, dtype=torch.long)
    token_entity[0, 0:4] = 0
    token_entity[0, 4:8] = 1
    token_entity[0, 8:12] = 0
    token_entity[1, 0:8] = 2
    token_entity[1, 8:16] = 3
    token_negated = torch.full((B, T), -1, dtype=torch.long)
    token_negated[0, 2] = 1
    token_negated[0, 3] = 0
    token_negated[1, 5] = 0
    token_negated[1, 9] = 1
    token_salient = torch.full((B, T), -1, dtype=torch.long)
    token_salient[0, 0:8] = 1
    token_salient[0, 8:12] = 0
    token_salient[1, 4:12] = 1
    value_owner_pos = torch.full((B, T), -1, dtype=torch.long)
    value_owner_pos[0, 5] = 1
    value_owner_pos[0, 9] = 4
    value_owner_pos[1, 10] = 2
    order_positions = torch.tensor([[1, 4, 7, 10], [2, 5, 8, 11]])
    order_mask = torch.ones(B, 4, dtype=torch.bool)
    return AuxSupervision(
        token_entity=token_entity, token_negated=token_negated,
        token_salient=token_salient, value_owner_pos=value_owner_pos,
        order_positions=order_positions, order_mask=order_mask,
    )


def test_bundle_populated_when_aux_enabled(small_setup):
    cfg, _model, _modules, _logits, bundle = small_setup
    assert bundle.layer_attn is not None
    assert len(bundle.layer_attn) == len(cfg.aux_layer_indices()) == 2
    for probs in bundle.layer_attn:
        assert probs.shape == (B, cfg.n_heads, T, T)
    assert bundle.aux_hidden is not None
    assert bundle.aux_hidden.shape == (B, T, cfg.d_model)


def test_bundle_empty_when_aux_disabled():
    torch.manual_seed(0)
    cfg = _small_cfg(aux_enabled=False)
    model = Transformer(cfg)
    _logits, bundle = model(torch.randint(0, cfg.vocab_size, (B, T)))
    assert bundle.layer_attn is None
    assert bundle.aux_hidden is None


def test_assign_heads_disjoint_counts_and_layers():
    cfg = _wide_cfg()
    aux_cfg = AuxLossConfig(coref=True, temporal=True, negation=True,
                            binding_attn=True, salience=True, heads_per_loss=2)
    assignments = assign_heads(cfg, aux_cfg)
    assert set(assignments) == {"coref", "temporal", "negation",
                                "binding_attn", "salience"}
    seen = set()
    aux_layers = set(cfg.aux_layer_indices())
    for name, heads in assignments.items():
        assert len(heads) == 2
        for pair in heads:
            assert pair not in seen, f"{pair} assigned twice"
            seen.add(pair)
            assert pair[0] in aux_layers
            assert 0 <= pair[1] < cfg.n_heads


def test_assign_heads_raises_when_pool_too_small():
    cfg = _small_cfg()  # 2 aux layers x 4 heads = 8 slots < 5 losses x 2
    aux_cfg = AuxLossConfig(coref=True, temporal=True, negation=True,
                            binding_attn=True, salience=True, heads_per_loss=2)
    with pytest.raises(ValueError):
        assign_heads(cfg, aux_cfg)


def test_each_attention_loss_finite(small_setup, supervision):
    cfg, _model, _modules, _logits, bundle = small_setup
    assignments = assign_heads(cfg, AuxLossConfig(coref=True, binding_attn=True))
    aux_layers = cfg.aux_layer_indices()
    l_coref = coref_loss(bundle.layer_attn, aux_layers,
                         assignments["coref"], supervision.token_entity)
    l_bind = binding_attn_loss(bundle.layer_attn, aux_layers,
                               assignments["binding_attn"],
                               supervision.value_owner_pos)
    for loss in (l_coref, l_bind):
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() >= -1e-6


def test_each_probe_loss_finite(small_setup, supervision):
    cfg, _model, modules, _logits, bundle = small_setup
    hidden = bundle.aux_hidden
    l_temp = temporal_loss(modules.temporal_probe, hidden,
                           supervision.order_positions, supervision.order_mask)
    l_neg = negation_loss(modules.negation_probe, hidden,
                          supervision.token_negated)
    l_sal = salience_loss(modules.salience_probe, hidden,
                          supervision.token_salient)
    for loss in (l_temp, l_neg, l_sal):
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0


def test_compute_aux_losses_aggregation(small_setup, supervision):
    cfg, _model, modules, _logits, bundle = small_setup
    all_on = AuxLossConfig(coref=True, temporal=True, negation=True,
                           binding_attn=True, salience=True,
                           coref_weight=0.5, temporal_weight=0.1,
                           negation_weight=0.1, binding_attn_weight=0.1,
                           salience_weight=0.1)
    total, components = compute_aux_losses(modules, bundle, cfg, all_on,
                                           supervision)
    assert set(components) == {"aux_coref", "aux_temporal", "aux_negation",
                               "aux_binding_attn", "aux_salience"}
    for v in components.values():
        assert torch.isfinite(v) and not v.requires_grad
    assert total.requires_grad
    manual = (0.5 * components["aux_coref"] + 0.1 * components["aux_temporal"]
              + 0.1 * components["aux_negation"]
              + 0.1 * components["aux_binding_attn"]
              + 0.1 * components["aux_salience"])
    assert torch.allclose(total.detach(), manual, atol=1e-5)

    partial = AuxLossConfig(negation=True)
    total_p, comp_p = compute_aux_losses(modules, bundle, cfg, partial,
                                         supervision)
    assert set(comp_p) == {"aux_negation"}
    assert torch.isfinite(total_p)

    none_on = AuxLossConfig()
    total_n, comp_n = compute_aux_losses(modules, bundle, cfg, none_on,
                                         supervision)
    assert comp_n == {}
    assert float(total_n) == 0.0


def test_ce_plus_aux_backward_finite_grads(small_setup, supervision):
    cfg, model, modules, logits, bundle = small_setup
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    ce = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
    all_on = AuxLossConfig(coref=True, temporal=True, negation=True,
                           binding_attn=True, salience=True)
    aux_total, _components = compute_aux_losses(modules, bundle, cfg, all_on,
                                                supervision)
    loss = ce + aux_total
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for model param {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
    for name, p in modules.named_parameters():
        assert p.grad is not None, f"no grad for aux param {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_compute_aux_losses_requires_populated_bundle(supervision):
    torch.manual_seed(0)
    cfg = _small_cfg(aux_enabled=False)
    model = Transformer(cfg)
    modules = AuxModules(cfg.d_model)
    _logits, bundle = model(torch.randint(0, cfg.vocab_size, (B, T)))
    with pytest.raises(ValueError):
        compute_aux_losses(modules, bundle, cfg, AuxLossConfig(negation=True),
                           supervision)
