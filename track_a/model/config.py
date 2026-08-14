"""ModelConfig (frozen contract C7), AuxBundle, and YAML round-trip.

C7 keys, exactly: d_model, n_layers, n_heads, head_dim, n_kv_heads,
ffn_hidden, vocab_size=50278, tied_embeddings=True, rope_theta, max_seq,
softcap: float|None (default None), aux_enabled: bool (default False).

AuxBundle.layer_attn holds attention-probability tensors for the LAST
ceil(n_layers/3) layers ONLY when aux_enabled=True, else None.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from track_a.needle_gen.types import TOTAL_VOCAB

# Frozen usage-bucket sequence lengths (PLAN 1/Q3+, task spec).
FRAMING_MAX_SEQ: dict[str, dict[str, int]] = {
    "tiny": {"single_shot": 13824, "streaming": 20480},
    "small": {"single_shot": 13824, "streaming": 69632},
    "base": {"single_shot": 13824, "streaming": 69632},
}
FRAMINGS = ("single_shot", "streaming")


def rope_theta_for(max_seq: int) -> int:
    """RoPE base frequency, keyed to max_seq (PLAN 1/Q3+)."""
    return 10000 if max_seq <= 4096 else 500000


@dataclass(frozen=True)
class ModelConfig:
    """Fully-resolved architecture config (contract C7)."""

    d_model: int
    n_layers: int
    n_heads: int
    head_dim: int
    n_kv_heads: int
    ffn_hidden: int
    rope_theta: float
    max_seq: int
    vocab_size: int = TOTAL_VOCAB
    tied_embeddings: bool = True
    softcap: Optional[float] = None
    aux_enabled: bool = False

    def __post_init__(self) -> None:
        pos = dict(
            d_model=self.d_model, n_layers=self.n_layers, n_heads=self.n_heads,
            head_dim=self.head_dim, n_kv_heads=self.n_kv_heads,
            ffn_hidden=self.ffn_hidden, max_seq=self.max_seq,
            vocab_size=self.vocab_size,
        )
        for name, value in pos.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.n_kv_heads > self.n_heads:
            raise ValueError("n_kv_heads must be <= n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads (GQA)")
        if self.n_heads * self.head_dim < self.d_model:
            raise ValueError("n_heads * head_dim must be >= d_model")
        if not self.tied_embeddings:
            raise ValueError("only tied embeddings are supported (PLAN 6)")
        if self.vocab_size != TOTAL_VOCAB:
            raise ValueError(f"vocab_size is frozen at {TOTAL_VOCAB}")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.softcap is not None and self.softcap <= 0:
            raise ValueError("softcap must be positive when enabled")

    @property
    def head_width(self) -> int:
        """Total attention width per layer (n_heads * head_dim, >= d_model)."""
        return self.n_heads * self.head_dim

    @property
    def gqa_group_size(self) -> int:
        """Query heads sharing one KV head."""
        return self.n_heads // self.n_kv_heads

    def aux_layer_indices(self) -> tuple[int, ...]:
        """Last ceil(n_layers/3) layer indices (aux hook, PLAN 6)."""
        k = math.ceil(self.n_layers / 3)
        return tuple(range(self.n_layers - k, self.n_layers))

    def to_dict(self) -> dict[str, Any]:
        """Exactly the C7 keys, in C7 order."""
        raw = asdict(self)
        order = (
            "d_model", "n_layers", "n_heads", "head_dim", "n_kv_heads",
            "ffn_hidden", "vocab_size", "tied_embeddings", "rope_theta",
            "max_seq", "softcap", "aux_enabled",
        )
        return {key: raw[key] for key in order}


@dataclass
class AuxBundle:
    """Side-channel output of Transformer.forward (contract C7).

    WARNING: when populated, layer_attn holds (B, n_heads, T, T) tensors;
    aux_enabled is only for short diagnostic sequences (T <= 2048).
    """

    layer_attn: Optional[tuple[torch.Tensor, ...]] = field(default=None)


def _config_from_mapping(raw: dict[str, Any]) -> ModelConfig:
    data = raw["model"] if "model" in raw else raw
    known = {f.name for f in fields(ModelConfig)}
    keys = set(data)
    if keys != known:
        missing, extra = known - keys, keys - known
        raise ValueError(f"ModelConfig keys mismatch: missing={sorted(missing)} "
                         f"extra={sorted(extra)}")
    return ModelConfig(**data)


def dump_config(cfg: ModelConfig, path: Path | str) -> None:
    """Write the C7 mapping as YAML (top-level keys are exactly C7)."""
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)


def load_config(path: Path | str) -> ModelConfig:
    """Read a config YAML; accepts bare C7 keys or a nested 'model' mapping."""
    with Path(path).open(encoding="utf-8") as fh:
        return _config_from_mapping(yaml.safe_load(fh))


def load_grid(path: Path | str) -> dict:
    """Load variant_grid.yaml (solver-generated, committed data)."""
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def model_config_from_grid_cell(
    grid: dict, scale: str, variant: str, framing: str,
) -> ModelConfig:
    """Build a ModelConfig from one variant_grid.yaml cell + framing bucket."""
    if framing not in FRAMINGS:
        raise ValueError(f"unknown framing {framing!r}, expected one of {FRAMINGS}")
    scale_doc = grid["scales"][scale]
    cell = scale_doc["variants"][variant]
    bucket = cell[framing]
    return ModelConfig(
        d_model=scale_doc["d_model"],
        n_layers=scale_doc["n_layers"],
        n_heads=cell["n_heads"],
        head_dim=cell["head_dim"],
        n_kv_heads=cell["n_kv_heads"],
        ffn_hidden=cell["ffn_hidden"],
        rope_theta=bucket["rope_theta"],
        max_seq=bucket["max_seq"],
    )
