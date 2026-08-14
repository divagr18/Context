"""S4: attention:FFN ratio solver acceptance, 5 variants x 3 scales (PLAN 6.2).

Asserts: params within +-2% of target; ratio within +-0.02 UNLESS the solver
flagged ratio relaxation (params tolerance is kept, ratio relaxed first, and
the relaxation must be recorded in variant_grid.yaml and logged by the report).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from track_a.configs.solve_ratio import (
    PARAM_TOL,
    RATIO_TOL,
    SCALE_SPECS,
    VARIANT_RATIOS,
    solve,
)
from track_a.model import params as P
from track_a.needle_gen.types import TOTAL_VOCAB

GRID_PATH = Path(__file__).resolve().parents[1] / "configs" / "variant_grid.yaml"
ALLOWED_HEAD_DIMS = (32, 64, 96, 128)


def _load_grid() -> dict:
    with GRID_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def grid() -> dict:
    return _load_grid()


def _all_cells():
    for scale in ("tiny", "small", "base"):
        d_model, n_layers, target_params = SCALE_SPECS[scale]
        for variant, target_ratio in VARIANT_RATIOS.items():
            yield scale, variant, d_model, n_layers, target_ratio, target_params


def test_grid_file_covers_all_15_cells(grid: dict) -> None:
    cells = [(s, v) for s, v, *_ in _all_cells()]
    for scale, variant in cells:
        cell = grid["scales"][scale]["variants"][variant]
        assert cell["ffn_hidden"] > 0


def test_solver_accepts_all_cells(grid: dict) -> None:
    """Every (variant, scale) cell meets tolerances, or is flagged relaxed."""
    n_relaxed = 0
    for scale, variant, d_model, n_layers, ratio, params in _all_cells():
        res = solve(d_model, n_layers, ratio, params)
        cell = grid["scales"][scale]["variants"][variant]

        # solver re-run agrees exactly with the committed grid data
        assert res.n_heads == cell["n_heads"], (scale, variant)
        assert res.head_dim == cell["head_dim"], (scale, variant)
        assert res.n_kv_heads == cell["n_kv_heads"], (scale, variant)
        assert res.ffn_hidden == cell["ffn_hidden"], (scale, variant)
        assert res.total_params == cell["total_params"], (scale, variant)
        assert res.ratio_relaxed == cell["ratio_relaxed"], (scale, variant)

        # search-space constraints
        assert res.head_dim in ALLOWED_HEAD_DIMS, (scale, variant)
        assert res.ffn_hidden % 128 == 0, (scale, variant)
        assert res.ffn_hidden >= 128, (scale, variant)
        assert res.n_heads % res.n_kv_heads == 0, (scale, variant)
        assert 1 <= res.n_kv_heads <= res.n_heads, (scale, variant)
        assert res.n_heads * res.head_dim >= d_model, (scale, variant)

        # params tolerance is absolute (never relaxed)
        assert abs(res.total_params - params) <= PARAM_TOL * params, (
            scale, variant, res.total_params)

        # ratio tolerance, with the documented relaxation escape hatch
        if res.ratio_relaxed:
            n_relaxed += 1
            print(f"[S4] ratio relaxation flagged: {scale}/{variant} "
                  f"achieved={res.ratio:.6f} target={ratio}")
        else:
            assert abs(res.ratio - ratio) <= RATIO_TOL, (scale, variant, res.ratio)


def test_at_most_few_relaxations(grid: dict) -> None:
    """Relaxation is the exception: most cells hit both tolerances exactly."""
    relaxed = [
        (s, v) for s, v, *_ in _all_cells()
        if grid["scales"][s]["variants"][v]["ratio_relaxed"]
    ]
    assert len(relaxed) <= 5, relaxed


def test_params_module_agrees_exactly(grid: dict) -> None:
    """params.py (single source of truth) reproduces the solver's numbers."""
    for scale, variant, d_model, n_layers, ratio, params in _all_cells():
        res = solve(d_model, n_layers, ratio, params)
        attn = P.attention_params_per_layer(
            d_model, res.n_heads, res.head_dim, res.n_kv_heads)
        ffn = P.ffn_params_per_layer(d_model, res.ffn_hidden)
        assert attn == res.attn_per_layer, (scale, variant)
        assert ffn == res.ffn_per_layer, (scale, variant)
        total = (P.embedding_params(TOTAL_VOCAB, d_model)
                 + n_layers * (attn + ffn)
                 + P.norm_params(n_layers, d_model))
        assert total == res.total_params, (scale, variant)
        expected_ratio = attn / (attn + ffn)
        assert math.isclose(res.ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-12)


def test_ad_hoc_solve_contract() -> None:
    """Ad-hoc solve honours arbitrary (d_model, n_layers, ratio, params)."""
    res = solve(512, 8, 0.40, 50_000_000)
    assert res.n_heads * res.head_dim >= 512
    assert abs(res.ratio - 0.40) <= RATIO_TOL or res.ratio_relaxed


def test_single_resolution_yaml_roundtrip(tmp_path: Path) -> None:
    from track_a.model.config import load_config

    from track_a.configs.solve_ratio import solve_to_yaml

    out = tmp_path / "tiny_v2_single_shot.yaml"
    solve_to_yaml(scale="tiny", variant="V2", framing="single_shot", out_path=out)
    cfg = load_config(out)
    assert cfg.vocab_size == TOTAL_VOCAB
    assert cfg.tied_embeddings is True
    assert cfg.max_seq == 13824
    assert cfg.rope_theta == 500000
    with out.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    assert doc["model"]["vocab_size"] == TOTAL_VOCAB
