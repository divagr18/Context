"""T7: training-unit tests (LR schedule, collation, config, CE grad flow,
recall-shaping). All run on CPU."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from track_a.data.pack import IGNORE_INDEX, PackSample, pack_single_shot
from track_a.training.batching import Batch, collate
from track_a.needle_gen.core.docgen import build_document
from track_a.needle_gen.generate import canonical_order_and_render
from track_a.needle_gen.types import Domain, GenConfig, Split
from track_a.model.config import ModelConfig
from track_a.model.core import Transformer
from track_a.tokenize import get_tokenizer
from track_a.training.config_resolve import load_run_config
from track_a.training.optimizer import lr_at
from track_a.training.recall_shaping import recall_shaping_loss

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


def _small_cfg_model():
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=4, head_dim=8,
                      n_kv_heads=2, ffn_hidden=64, rope_theta=10000.0,
                      max_seq=4096)
    return cfg, Transformer(cfg)


class _SchedCfg:
    lr = 1.0
    warmup_frac = 0.1
    min_lr_frac = 0.1


def test_lr_schedule_warmup_and_cosine():
    cfg = _SchedCfg()
    total = 100
    warm = 10
    assert math.isclose(lr_at(0, cfg, total), cfg.lr * 1 / warm)
    assert math.isclose(lr_at(warm - 1, cfg, total), cfg.lr)
    assert math.isclose(lr_at(warm, cfg, total), cfg.lr)
    assert math.isclose(lr_at(total - 1, cfg, total), cfg.lr * cfg.min_lr_frac,
                        rel_tol=1e-6)
    lrs = [lr_at(s, cfg, total) for s in range(total)]
    assert max(lrs) <= cfg.lr + 1e-9
    assert min(lrs) >= cfg.lr * cfg.min_lr_frac - 1e-9


def test_collate_padding_and_mask():
    a = PackSample(input_ids=(1, 2, 3), labels=(-100, 5, 6))
    b = PackSample(input_ids=(7, 8), labels=(-100, 9))
    batch = collate([a, b], pad_id=0)
    assert isinstance(batch, Batch)
    assert batch.input_ids.shape == (2, 3)
    assert batch.labels.shape == (2, 3)
    assert batch.attention_mask.shape == (2, 3)
    assert batch.input_ids[1, 2].item() == 0  # pad
    assert batch.labels[1, 2].item() == IGNORE_INDEX
    assert batch.attention_mask[1, 2].item() is False or \
        batch.attention_mask[1, 2].item() == False
    assert batch.n_tokens == 5


def test_config_resolve_smoke():
    cfg = load_run_config("configs/train_smoke.yaml")
    assert cfg.run_tag == "smoke"
    assert cfg.model.d_model == 512
    assert cfg.model.n_layers == 8
    assert cfg.aux.enabled() == ()
    assert cfg.model.aux_enabled is False
    assert cfg.use_bf16 is True
    assert cfg.recall_every == 0
    assert cfg.total_tokens == 20000000


def test_ce_grad_flow_cpu():
    model_cfg, model = _small_cfg_model()
    model.train()
    input_ids = torch.randint(0, model_cfg.vocab_size, (2, 16))
    labels = input_ids.clone()
    labels[:, :8] = IGNORE_INDEX
    logits, _ = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, model_cfg.vocab_size), labels.reshape(-1),
        ignore_index=IGNORE_INDEX)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_recall_shaping_loss_cpu(tok):
    torch.manual_seed(0)
    gen_cfg = GenConfig(seed=7, split=Split.TRAIN, domain=Domain.PROJECT_UPDATES,
                        doc_len_name="short", n_docs=1, family_ids=(),
                        paraphrase_idx_range=(0, 5))
    built = build_document(gen_cfg, 0, tok)
    db = built.fact_db
    budget = db.doc_len_tokens // 4
    text = canonical_order_and_render(db, budget, tok)
    target_ids = tuple(tok.encode(text, add_special_tokens=False))
    sample = pack_single_shot(built.doc_ids, target_ids, db.doc_len_tokens,
                              ratio=4, tok=tok)

    cfg = SimpleNamespace(lambda_drop=5.0, lambda_waste=1.0, use_bf16=False)
    model_cfg, model = _small_cfg_model()
    model.train()
    pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    c_close_id = tok.convert_tokens_to_ids("</C>")
    extra, r_bar, h_bar = recall_shaping_loss(
        model, [(sample, db)], tok, cfg, pad_id, c_close_id, CPU)
    assert torch.isfinite(extra)
    assert 0.0 <= r_bar <= 1.0
    assert 0.0 <= h_bar <= 1.0


def test_train_seed_override(tmp_path):
    """--seed must override the config seed and drive the run dir / resolved
    config (G1 grid runs variants x seeds from one config file each)."""
    import yaml
    import track_a.train as train_mod
    from track_a.tests._shard_fixtures import (
        make_single_shot_record, write_shards,
    )

    shard = write_shards([make_single_shot_record()],
                         tmp_path / "train.jsonl")
    cfg = {
        "run_tag": "seedtest", "seed": 11, "framing": "single_shot",
        "model": {"d_model": 32, "n_layers": 2, "n_heads": 4, "head_dim": 8,
                  "n_kv_heads": 2, "ffn_hidden": 64, "rope_theta": 10000.0,
                  "max_seq": 4096},
        "data": {"train_shards": [str(shard)], "val_shards": []},
        "training": {"total_tokens": 512, "effective_batch_tokens": 256,
                     "max_seq": 4096, "lr": 0.0003, "use_bf16": False},
        "recall_shaping": {"every": 0},
        "logging": {"run_dir": str(tmp_path / "runs"), "log_every": 1},
        "checkpoint": {"save_every_steps": 0},
        "eval": {"every_steps": 0},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run_root = train_mod.train(str(p), seed=22)
    assert run_root.name == "seedtest-22", \
        f"run dir must use the override seed, got {run_root.name}"
    assert "seed=22" in (run_root / "config.yaml").read_text(encoding="utf-8")


def test_g1_grid_configs_resolve():
    """All five G1 tiny variant configs must resolve with the G0-picked LR
    (6e-4, PLAN 7) and identical non-model recipe as V1."""
    base = load_run_config("configs/train_tiny.yaml")
    assert base.lr == 0.0006, "G0 LR pick (6e-4) must be applied to V1"
    for variant in ("V0", "V2", "V3", "V4"):
        cfg = load_run_config(f"configs/train_tiny_{variant}.yaml")
        assert cfg.run_tag == f"tiny-{variant}"
        assert cfg.lr == 0.0006, f"{variant}: G0 LR pick not applied"
        assert cfg.model.n_layers == base.model.n_layers
        assert cfg.model.d_model == base.model.d_model
        assert cfg.total_tokens == base.total_tokens
        assert cfg.effective_batch_tokens == base.effective_batch_tokens
        assert cfg.weight_decay == base.weight_decay
        assert cfg.use_bf16 == base.use_bf16
