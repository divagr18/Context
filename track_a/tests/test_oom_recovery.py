"""OOM-recovery tests for the training loop and evaluate() (S6 hardening).

Real failure mode observed on the 8GB 4060: the micro-batch auto-tuned on
short probe samples hits a longer real sample later and CUDA OOMs mid-step.
Contract:

* the main loop shrinks the micro-batch and re-queues the samples on OOM;
* if even a single sample OOMs, it is skipped (loud log line);
* if nothing fits (50 consecutive single-sample OOMs), abort loudly instead
  of looping forever;
* evaluate() skips OOM samples instead of crashing a whole training run.

All tests run on CPU with synthetic ``torch.cuda.OutOfMemoryError`` injection
(the class is identical to ``torch.OutOfMemoryError``). OOM is injected on
sequence LENGTH (batch.input_ids.shape[1] > threshold), which reproduces the
real incident: short auto-tune probes, then a longer padded batch later.
"""
from __future__ import annotations

import pytest
import torch
import yaml

import track_a.train as train_mod
from track_a.model.config import ModelConfig
from track_a.model.core import Transformer
from track_a.tests._shard_fixtures import make_single_shot_record, write_shards
from track_a.tokenize import get_tokenizer

OOM_SEQ_THRESHOLD = 20  # short fixture samples pack to ~15 tokens


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


def _tiny_model():
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=4, head_dim=8,
                      n_kv_heads=2, ffn_hidden=64, rope_theta=10000.0,
                      max_seq=4096)
    return Transformer(cfg)


def _write_cfg(tmp_path, train_shard):
    cfg = {
        "run_tag": "oomtest", "seed": 3, "framing": "single_shot",
        "model": {"d_model": 32, "n_layers": 2, "n_heads": 4, "head_dim": 8,
                  "n_kv_heads": 2, "ffn_hidden": 64, "rope_theta": 10000.0,
                  "max_seq": 4096},
        "data": {"train_shards": [str(train_shard)], "val_shards": []},
        "training": {"total_tokens": 512, "effective_batch_tokens": 256,
                     "max_seq": 4096, "lr": 0.0003, "use_bf16": False},
        "recall_shaping": {"every": 0},
        "logging": {"run_dir": str(tmp_path / "runs"), "log_every": 1},
        "checkpoint": {"save_every_steps": 0},
        "eval": {"every_steps": 0},
    }
    p = tmp_path / "oom_cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _oom_on_long_seq(real_ce):
    def ce(model, batch, use_bf16):
        if batch.input_ids.shape[1] > OOM_SEQ_THRESHOLD:
            raise torch.cuda.OutOfMemoryError(
                f"synthetic OOM: seq_len {batch.input_ids.shape[1]}")
        return real_ce(model, batch, use_bf16)
    return ce


def test_loop_recovers_from_oom_by_shrinking_micro_batch(tmp_path, monkeypatch,
                                                          capsys):
    records = [make_single_shot_record(doc_id=f"ss-{i}") for i in range(3)]
    # Long sample (40 doc tokens -> packed ~47 > threshold) after short ones:
    # auto-tune probes pass, then a padded batch containing it OOMs mid-step.
    records.append(make_single_shot_record(
        doc_id="ss-long", doc_ids=tuple(range(1, 40))))
    shard = write_shards(records, tmp_path / "train.jsonl")
    cfg_path = _write_cfg(tmp_path, shard)
    monkeypatch.setattr(train_mod, "ce_on_batch",
                        _oom_on_long_seq(train_mod.ce_on_batch))
    run_root = train_mod.train(str(cfg_path))
    assert (run_root / "final.pt").exists(), \
        "training must survive the OOM and save final.pt"
    out = capsys.readouterr().out
    assert "[oom]" in out, "expected OOM recovery log line(s) on stdout"


def test_loop_aborts_when_device_cannot_fit_any_sample(tmp_path, monkeypatch):
    shard = write_shards([make_single_shot_record()], tmp_path / "train.jsonl")
    cfg_path = _write_cfg(tmp_path, shard)

    def always_oom(model, batch, use_bf16):
        raise torch.cuda.OutOfMemoryError("synthetic OOM always")

    monkeypatch.setattr(train_mod, "ce_on_batch", always_oom)
    with pytest.raises(RuntimeError, match="cannot fit"):
        train_mod.train(str(cfg_path))


def test_evaluate_skips_oom_samples(tmp_path, tok):
    # Long record FIRST so the OOM sample is reached well before max_samples.
    records = [make_single_shot_record(
        doc_id="ss-long", doc_ids=tuple(range(1, 40)))]
    records += [make_single_shot_record(doc_id=f"ss-{i}") for i in range(2)]
    val = write_shards(records, tmp_path / "val.jsonl")
    model = _tiny_model()
    device = torch.device("cpu")
    original = train_mod.ce_on_batch
    train_mod.ce_on_batch = _oom_on_long_seq(original)
    try:
        val_ce = train_mod.evaluate(model, [str(val)], tok, "single_shot",
                                    device, False, 15, pad_id=0)
    finally:
        train_mod.ce_on_batch = original
    assert val_ce == val_ce, "val_ce must be a finite float, not NaN"
    assert val_ce >= 0.0, f"val_ce must be >= 0, got {val_ce}"


def test_ce_on_batch_characterization_chunk_equiv():
    """Characterization pin (pre-refactor GREEN, must survive the chunked-CE
    refactor): ce_on_batch == full fp32 CE over non-pad positions. The
    refactor exists because full-batch fp32 logits cost bs*T*50278*4 bytes
    (3.9 GiB at bs=8,T=2600) and cap micro-batch on the 8GB 4060."""
    from track_a.data.pack import IGNORE_INDEX
    from track_a.training.batching import collate
    from track_a.data.pack import PackSample

    torch.manual_seed(11)
    model = _tiny_model()
    model.train()
    a = PackSample(input_ids=tuple(torch.randint(0, 100, (21,)).tolist()),
                   labels=tuple([IGNORE_INDEX] * 5
                                + torch.randint(0, 100, (16,)).tolist()))
    b = PackSample(input_ids=tuple(torch.randint(0, 100, (17,)).tolist()),
                   labels=tuple([IGNORE_INDEX] * 3
                                + torch.randint(0, 100, (14,)).tolist()))
    batch = collate([a, b], pad_id=0)

    logits, _ = model(batch.input_ids)
    ref = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        batch.labels.reshape(-1), ignore_index=IGNORE_INDEX)
    got = train_mod.ce_on_batch(model, batch, use_bf16=False)
    assert torch.allclose(got.float(), ref.float(), atol=1e-5, rtol=1e-4), \
        f"ce_on_batch={got.item()} != reference CE {ref.item()}"
