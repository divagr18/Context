"""S5: core architecture tests (PLAN 6) — tiny scale, all 5 variants."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
import torch

from track_a.configs.solve_ratio import PARAM_TOL, SCALE_SPECS
from track_a.model import params as P
from track_a.model.config import load_grid, model_config_from_grid_cell
from track_a.model.core import Transformer
from track_a.model.diag.entropy import layer_attention_entropy
from track_a.needle_gen.types import TOTAL_VOCAB

GRID_PATH = Path(__file__).resolve().parents[1] / "configs" / "variant_grid.yaml"
TINY_PARAMS = SCALE_SPECS["tiny"][2]
N_AUX_LAYERS = math.ceil(8 / 3)  # = 3 for tiny


@pytest.fixture(scope="module")
def tiny_configs() -> dict:
    grid = load_grid(GRID_PATH)
    return {
        v: model_config_from_grid_cell(grid, "tiny", v, "single_shot")
        for v in ("V0", "V1", "V2", "V3", "V4")
    }


def _seeded_model(cfg, seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(cfg).eval()


def test_param_counts_match_params_module_and_target(tiny_configs) -> None:
    for variant, cfg in tiny_configs.items():
        model = _seeded_model(cfg)
        n = sum(p.numel() for p in model.parameters())
        assert n == P.total_params(cfg), variant
        assert abs(n - TINY_PARAMS) <= PARAM_TOL * TINY_PARAMS, (variant, n)


def test_forward_shape_and_determinism(tiny_configs) -> None:
    for variant, cfg in tiny_configs.items():
        torch.manual_seed(123)
        ids = torch.randint(0, TOTAL_VOCAB, (2, 64))
        m1 = _seeded_model(cfg, seed=7)
        m2 = _seeded_model(cfg, seed=7)
        with torch.no_grad():
            logits1, bundle1 = m1(ids)
            logits2, _ = m2(ids)
        assert logits1.shape == (2, 64, TOTAL_VOCAB), variant
        assert torch.equal(logits1, logits2), variant
        assert torch.isfinite(logits1).all(), variant
        assert bundle1.layer_attn is None, variant  # aux disabled by default


def test_gqa_exercised_by_some_variant(tiny_configs) -> None:
    assert any(
        cfg.n_kv_heads < cfg.n_heads for cfg in tiny_configs.values()
    ), "at least one tiny variant must use grouped-query attention"


def test_aux_bundle_disabled_returns_none(tiny_configs) -> None:
    cfg = tiny_configs["V1"]
    assert cfg.aux_enabled is False
    model = _seeded_model(cfg)
    with torch.no_grad():
        _, bundle = model(torch.randint(0, TOTAL_VOCAB, (2, 32)))
    assert bundle.layer_attn is None


def test_aux_bundle_enabled_last_third_layers(tiny_configs) -> None:
    cfg = dataclasses.replace(tiny_configs["V1"], aux_enabled=True)
    model = _seeded_model(cfg)
    ids = torch.randint(0, TOTAL_VOCAB, (2, 32))
    with torch.no_grad():
        _, bundle = model(ids)
        _, all_probs = model.forward_with_attention_probs(ids)
    assert bundle.layer_attn is not None
    assert len(bundle.layer_attn) == N_AUX_LAYERS
    assert len(all_probs) == 8
    for probs in bundle.layer_attn:
        assert probs.shape == (2, cfg.n_heads, 32, 32)
        assert torch.isfinite(probs).all()
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    # bundle holds the LAST ceil(8/3) layers, matching the eager capture path
    for got, ref in zip(bundle.layer_attn, all_probs[-N_AUX_LAYERS:]):
        assert torch.allclose(got, ref, atol=1e-6)


def test_softcap_changes_logits(tiny_configs) -> None:
    base = tiny_configs["V3"]
    cfg_capped = dataclasses.replace(base, softcap=2.0)
    torch.manual_seed(11)
    plain = _seeded_model(base, seed=11)
    torch.manual_seed(11)
    capped = _seeded_model(cfg_capped, seed=11)
    ids = torch.randint(0, TOTAL_VOCAB, (2, 64))
    with torch.no_grad():
        l_plain, _ = plain(ids)
        l_capped, _ = capped(ids)
    assert not torch.allclose(l_plain, l_capped, atol=1e-4)
    assert torch.isfinite(l_capped).all()


def test_entropy_diagnostic_returns_per_layer_scalars(tiny_configs) -> None:
    cfg = tiny_configs["V1"]
    model = _seeded_model(cfg)
    torch.manual_seed(3)
    prefix_a = torch.randint(0, TOTAL_VOCAB, (512,))
    prefix_b = torch.randint(0, TOTAL_VOCAB, (512,))
    ent = layer_attention_entropy(model, prefix_a, prefix_b)
    assert set(ent) == set(range(8))
    for layer_idx, value in ent.items():
        assert math.isfinite(value), layer_idx
        assert value >= 0.0, layer_idx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_autocast_forward_is_finite(tiny_configs) -> None:
    cfg = tiny_configs["V4"]  # most attention-heavy variant
    model = _seeded_model(cfg).cuda()
    ids = torch.randint(0, TOTAL_VOCAB, (2, 128)).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(ids)
    assert logits.shape == (2, 128, TOTAL_VOCAB)
    assert torch.isfinite(logits.float()).all()
