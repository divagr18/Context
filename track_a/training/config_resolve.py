"""Resolve a run YAML into a fully-resolved RunConfig (contract C8).

The model config is resolved from the committed variant_grid.yaml via
(scale, variant, framing); ``aux_enabled`` is derived from the aux-loss flags.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

from track_a.model.aux_losses import AuxLossConfig
from track_a.model.config import (
    ModelConfig, load_grid, model_config_from_grid_cell,
)

VARIANT_GRID_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "variant_grid.yaml"
)

_AUX_FIELDS = {f.name for f in fields(AuxLossConfig)}


@dataclass(frozen=True)
class RunConfig:
    run_tag: str
    seed: int
    framing: str
    model: ModelConfig
    aux: AuxLossConfig
    train_shards: tuple[str, ...]
    val_shards: tuple[str, ...]
    total_tokens: int
    effective_batch_tokens: int
    max_seq: int
    lr: float
    warmup_frac: float
    min_lr_frac: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    use_bf16: bool
    recall_every: int
    recall_subsample: int
    lambda_drop: float
    lambda_waste: float
    run_dir: str
    log_every: int
    save_every_steps: int
    eval_every_steps: int
    val_max_samples: int


def _build_aux(raw: dict) -> AuxLossConfig:
    aux_raw = raw.get("aux", {}) or {}
    kwargs = {k: v for k, v in aux_raw.items() if k in _AUX_FIELDS}
    return AuxLossConfig(**kwargs)


def load_run_config(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    m = raw["model"]
    framing = raw.get("framing", "single_shot")
    if "scale" in m and "variant" in m:
        grid = load_grid(VARIANT_GRID_PATH)
        model = model_config_from_grid_cell(grid, m["scale"], m["variant"], framing)
    else:
        model = ModelConfig(**m)

    aux = _build_aux(raw)
    if aux.enabled():
        model = replace(model, aux_enabled=True)

    tr = raw.get("training", {})
    rs = raw.get("recall_shaping", {}) or {}
    lg = raw.get("logging", {})
    ck = raw.get("checkpoint", {})
    ev = raw.get("eval", {})
    data = raw.get("data", {})

    return RunConfig(
        run_tag=raw["run_tag"],
        seed=int(raw.get("seed", 0)),
        framing=framing,
        model=model,
        aux=aux,
        train_shards=tuple(data.get("train_shards", [])),
        val_shards=tuple(data.get("val_shards", [])),
        total_tokens=int(tr["total_tokens"]),
        effective_batch_tokens=int(tr["effective_batch_tokens"]),
        max_seq=int(tr.get("max_seq", model.max_seq)),
        lr=float(tr["lr"]),
        warmup_frac=float(tr.get("warmup_frac", 0.02)),
        min_lr_frac=float(tr.get("min_lr_frac", 0.1)),
        weight_decay=float(tr.get("weight_decay", 0.1)),
        beta1=float(tr.get("beta1", 0.9)),
        beta2=float(tr.get("beta2", 0.95)),
        grad_clip=float(tr.get("grad_clip", 1.0)),
        use_bf16=bool(tr.get("use_bf16", True)),
        recall_every=int(rs.get("every", 0)),
        recall_subsample=int(rs.get("subsample", 8)),
        lambda_drop=float(rs.get("lambda_drop", 5.0)),
        lambda_waste=float(rs.get("lambda_waste", 1.0)),
        run_dir=str(lg.get("run_dir", "runs")),
        log_every=int(lg.get("log_every", 10)),
        save_every_steps=int(ck.get("save_every_steps", 0)),
        eval_every_steps=int(ev.get("every_steps", 0)),
        val_max_samples=int(ev.get("val_max_samples", 32)),
    )
