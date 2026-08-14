"""T3C: split config discipline (PLAN.md 5.5)."""
from __future__ import annotations

from pathlib import Path

import pytest

from track_a.needle_gen.corpus_writer import load_split_config, make_gen_config
from track_a.needle_gen.types import DOC_LEN_TARGETS, Domain, Split

SPLITS_DIR = Path("track_a/needle_gen/splits")
SPLIT_FILES = ("train.yaml", "val.yaml", "test-id.yaml", "test-ood.yaml")

REQUIRED_KEYS = ("split", "kind", "domain", "doc_len_name", "n_docs", "seed",
                 "paraphrase_idx_range")


@pytest.mark.parametrize("fname", SPLIT_FILES)
def test_split_file_exists_and_valid(fname):
    path = SPLITS_DIR / fname
    assert path.exists()
    cfg = load_split_config(path)
    for key in REQUIRED_KEYS:
        assert key in cfg, f"{fname}: missing {key}"
    Split(cfg["split"])  # raises if invalid
    Domain(cfg["domain"])
    assert cfg["doc_len_name"] in DOC_LEN_TARGETS
    assert cfg["n_docs"] > 0
    lo, hi = cfg["paraphrase_idx_range"]
    assert 0 <= lo <= hi <= 7


@pytest.mark.parametrize("fname,expected_split,expected_paras", [
    ("train.yaml", "train", [0, 5]),
    ("val.yaml", "val", [6, 6]),
    ("test-id.yaml", "test-id", [7, 7]),
])
def test_paraphrase_split_discipline(fname, expected_split, expected_paras):
    cfg = load_split_config(SPLITS_DIR / fname)
    assert cfg["split"] == expected_split
    assert list(cfg["paraphrase_idx_range"]) == expected_paras
    assert cfg["domain"] == "project_updates"


def test_ood_domain_is_lexically_disjoint_domain():
    cfg = load_split_config(SPLITS_DIR / "test-ood.yaml")
    assert cfg["domain"] == "logistics_ops"
    assert cfg["split"] == "test-ood"


@pytest.mark.parametrize("fname", SPLIT_FILES)
def test_make_gen_config(fname):
    cfg = load_split_config(SPLITS_DIR / fname)
    gen = make_gen_config(cfg)
    assert gen.seed == cfg["seed"]
    assert gen.split.value == cfg["split"]
    assert gen.domain.value == cfg["domain"]
    assert gen.paraphrase_idx_range == tuple(cfg["paraphrase_idx_range"])
    assert gen.window_tokens == cfg.get("window_tokens", 2048)
