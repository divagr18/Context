"""Attention:FFN ratio solver (PLAN 6.2, scenario S4).

Searches for (n_heads, head_dim, n_kv_heads, ffn_hidden) that achieve a target
attention-share ratio at a target parameter count. Param formulas are imported
from track_a.model.params (single source of truth).

CLI:
    --report                        Print 5 variants x 3 scales table.
    --scale <s> --variant <v>       Single-resolution mode (write YAML).
    --out <path.yaml>               Output file for single-resolution.
    --d-model --n-layers --ratio    Ad-hoc solve with explicit params.
      --params
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from track_a.needle_gen.types import TOTAL_VOCAB
from track_a.model.config import FRAMING_MAX_SEQ, ModelConfig, dump_config, rope_theta_for
from track_a.model.params import (
    attention_params_per_layer,
    embedding_params,
    ffn_params_per_layer,
    norm_params,
)

VARIANT_RATIOS: dict[str, float] = {
    "V0": 0.15, "V1": 0.25, "V2": 0.40, "V3": 0.55, "V4": 0.70,
}
SCALE_SPECS: dict[str, tuple[int, int, int]] = {
    "tiny": (512, 8, 50_000_000),
    "small": (768, 12, 150_000_000),
    "base": (1024, 16, 300_000_000),
}
PARAM_TOL = 0.02
RATIO_TOL = 0.02
HEAD_DIMS = (32, 64, 96, 128)


@dataclass(frozen=True)
class SolveResult:
    d_model: int
    n_layers: int
    n_heads: int
    head_dim: int
    n_kv_heads: int
    ffn_hidden: int
    attn_per_layer: int
    ffn_per_layer: int
    ratio: float
    total_params: int
    target_ratio: float
    target_params: int
    ratio_relaxed: bool


def _powers_of_two_leq(n: int) -> list[int]:
    result = []
    p = 1
    while p <= n:
        result.append(p)
        p *= 2
    return result


def solve(
    d_model: int, n_layers: int, target_ratio: float, target_params: int,
) -> SolveResult:
    """Find architecture dims that best match target ratio and param count.

    Priority: (1) prefer candidates within params tolerance, (2) among those
    minimize |ratio_err|, (3) if none pass params tol, minimize both errors.
    This ensures ratio is relaxed before params (per PLAN 6.2 spec).
    """
    emb = embedding_params(TOTAL_VOCAB, d_model)
    norms = norm_params(n_layers, d_model)
    fixed = emb + norms
    per_layer_budget = (target_params - fixed) / n_layers

    best: Optional[SolveResult] = None
    # Score: (params_violation_flag, ratio_err, param_err)
    # params_violation_flag=0 if within PARAM_TOL, else 1 → tiered selection.
    best_score = (2, float("inf"), float("inf"))

    for head_dim in HEAD_DIMS:
        min_heads = math.ceil(d_model / head_dim)
        max_heads = max(256, min_heads * 4)
        for n_heads in range(min_heads, max_heads + 1):
            for n_kv_heads in _powers_of_two_leq(n_heads):
                if n_heads % n_kv_heads != 0:
                    continue
                attn = attention_params_per_layer(
                    d_model, n_heads, head_dim, n_kv_heads,
                )
                ffn_remaining = per_layer_budget - attn
                if ffn_remaining < 3 * d_model * 128:
                    continue
                ffn_hidden_raw = ffn_remaining / (3 * d_model)
                ffn_hidden_lo = max(128, int(ffn_hidden_raw // 128) * 128)
                ffn_hidden_hi = ffn_hidden_lo + 128

                for ffn_hidden in (ffn_hidden_lo, ffn_hidden_hi):
                    ffn = ffn_params_per_layer(d_model, ffn_hidden)
                    total = fixed + n_layers * (attn + ffn)
                    ratio = attn / (attn + ffn)
                    ratio_err = abs(ratio - target_ratio)
                    param_err = abs(total - target_params) / target_params
                    params_ok_flag = 0 if param_err <= PARAM_TOL else 1
                    score = (params_ok_flag, ratio_err, param_err)
                    if score < best_score:
                        best_score = score
                        best = SolveResult(
                            d_model=d_model, n_layers=n_layers,
                            n_heads=n_heads, head_dim=head_dim,
                            n_kv_heads=n_kv_heads, ffn_hidden=ffn_hidden,
                            attn_per_layer=attn, ffn_per_layer=ffn,
                            ratio=ratio, total_params=total,
                            target_ratio=target_ratio,
                            target_params=target_params,
                            ratio_relaxed=False,
                        )

    if best is None:
        raise RuntimeError(
            f"No feasible config for d_model={d_model} n_layers={n_layers} "
            f"target_ratio={target_ratio} target_params={target_params}"
        )

    ratio_ok = abs(best.ratio - target_ratio) <= RATIO_TOL
    params_ok = abs(best.total_params - target_params) <= PARAM_TOL * target_params
    if not ratio_ok and params_ok:
        print(
            f"[SOLVER] RATIO RELAXATION: d_model={d_model} n_layers={n_layers} "
            f"target={target_ratio:.3f} achieved={best.ratio:.6f} "
            f"(params OK: {best.total_params} vs {target_params})",
            file=sys.stderr, flush=True,
        )
        object.__setattr__(best, "ratio_relaxed", True)

    return best


def build_grid() -> dict:
    """Generate the full 5x3 variant grid YAML structure."""
    doc: dict = {"scales": {}}
    for scale, (d_model, n_layers, target_params) in SCALE_SPECS.items():
        variants = {}
        for variant, target_ratio in VARIANT_RATIOS.items():
            res = solve(d_model, n_layers, target_ratio, target_params)
            p_ok = abs(res.total_params - target_params) <= PARAM_TOL * target_params
            variants[variant] = {
                "target_ratio": target_ratio,
                "n_heads": res.n_heads,
                "head_dim": res.head_dim,
                "n_kv_heads": res.n_kv_heads,
                "ffn_hidden": res.ffn_hidden,
                "attn_params_per_layer": res.attn_per_layer,
                "ffn_params_per_layer": res.ffn_per_layer,
                "achieved_ratio": res.ratio,
                "total_params": res.total_params,
                "params_within_tol": p_ok,
                "ratio_relaxed": res.ratio_relaxed,
                "single_shot": {
                    "max_seq": FRAMING_MAX_SEQ[scale]["single_shot"],
                    "rope_theta": rope_theta_for(
                        FRAMING_MAX_SEQ[scale]["single_shot"],
                    ),
                },
                "streaming": {
                    "max_seq": FRAMING_MAX_SEQ[scale]["streaming"],
                    "rope_theta": rope_theta_for(
                        FRAMING_MAX_SEQ[scale]["streaming"],
                    ),
                },
            }
        doc["scales"][scale] = {
            "d_model": d_model,
            "n_layers": n_layers,
            "target_params": target_params,
            "variants": variants,
        }
    return doc


def write_grid(path: Path) -> None:
    """Write the full variant grid YAML."""
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(build_grid(), fh, sort_keys=False)


def solve_to_yaml(
    scale: str, variant: str, framing: str, out_path: Path,
) -> None:
    """Solve one config and write it as a standalone ModelConfig YAML."""
    d_model, n_layers, target_params = SCALE_SPECS[scale]
    res = solve(d_model, n_layers, VARIANT_RATIOS[variant], target_params)
    ms = FRAMING_MAX_SEQ[scale][framing]
    cfg = ModelConfig(
        d_model=d_model, n_layers=n_layers, n_heads=res.n_heads,
        head_dim=res.head_dim, n_kv_heads=res.n_kv_heads,
        ffn_hidden=res.ffn_hidden, rope_theta=rope_theta_for(ms),
        max_seq=ms,
    )
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"model": cfg.to_dict()}, fh, sort_keys=False)


def _print_report() -> None:
    """Print the 5x3 solver report table."""
    header = (
        f"{'variant':<8} {'scale':<6} {'d_model':>7} {'n_layers':>8} "
        f"{'n_heads':>7} {'hdim':>5} {'kv_h':>5} {'ffn_h':>7} "
        f"{'attn_p':>12} {'ffn_p':>12} {'ratio':>8} {'total_params':>14} "
        f"{'target_p':>14} {'status':<10}"
    )
    print(header)
    print("-" * len(header))
    for scale, (d_model, n_layers, target_params) in SCALE_SPECS.items():
        for variant, target_ratio in VARIANT_RATIOS.items():
            res = solve(d_model, n_layers, target_ratio, target_params)
            p_ok = abs(res.total_params - target_params) <= PARAM_TOL * target_params
            r_ok = abs(res.ratio - target_ratio) <= RATIO_TOL
            status = "PASS" if (p_ok and r_ok) else (
                "PASS(relax)" if (p_ok and res.ratio_relaxed) else "FAIL"
            )
            print(
                f"{variant:<8} {scale:<6} {d_model:>7} {n_layers:>8} "
                f"{res.n_heads:>7} {res.head_dim:>5} {res.n_kv_heads:>5} "
                f"{res.ffn_hidden:>7} {res.attn_per_layer:>12} "
                f"{res.ffn_per_layer:>12} {res.ratio:>8.4f} "
                f"{res.total_params:>14d} {target_params:>14d} {status:<10}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention:FFN ratio solver")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--scale", choices=list(SCALE_SPECS))
    parser.add_argument("--variant", choices=list(VARIANT_RATIOS))
    parser.add_argument("--out", type=str)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--params", type=int)
    parser.add_argument("--grid", type=str)
    args = parser.parse_args()

    if args.report:
        _print_report()
        return
    if args.grid:
        write_grid(Path(args.grid))
        print(f"wrote {args.grid}")
        return
    if args.d_model and args.n_layers and args.ratio and args.params:
        res = solve(args.d_model, args.n_layers, args.ratio, args.params)
        print(f"Solved: n_heads={res.n_heads} head_dim={res.head_dim} "
              f"n_kv_heads={res.n_kv_heads} ffn_hidden={res.ffn_hidden} "
              f"ratio={res.ratio:.6f} total={res.total_params}")
        return
    if args.scale and args.variant and args.out:
        framing = "single_shot"
        solve_to_yaml(args.scale, args.variant, framing, Path(args.out))
        print(f"wrote {args.out}")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
