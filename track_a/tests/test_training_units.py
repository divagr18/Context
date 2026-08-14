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
