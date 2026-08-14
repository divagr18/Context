"""W8 eval-battery tests (contract T8, PLAN.md section 9). All on CPU.

Contract pinned here:
* ``grade_generation(fact_db, text)`` is model-free: parses the generated C
  and scores it against the ground-truth FactDB (per-type exact/partial
  survival, chain current/prior values, hallucination, decoy emission,
  salience inversion, parse failures).
* ``build_single_shot_prompt`` reproduces the training source prefix exactly.
* ``evaluate_shards`` runs greedy Ĉ generation + grading + QA probes over a
  shard and returns {"meta","primary","diag"} with the PLAN 9 metric keys.
* ``load_eval_model`` rebuilds the model from run YAML + checkpoint.
"""
from __future__ import annotations

import pytest
import torch
import yaml

import track_a.eval as eval_mod
from track_a.model.config import ModelConfig
from track_a.model.core import Transformer
from track_a.needle_gen.core.docgen import build_document
from track_a.needle_gen.corpus_writer import build_record
from track_a.needle_gen.generate import canonical_order_and_render
from track_a.needle_gen.types import (
    Domain, Entity, Fact, FactDB, FactType, GenConfig, Split,
)
from track_a.shard_schema import to_json
from track_a.tokenize import get_tokenizer
from track_a.training.checkpoints import save_checkpoint

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


def _gen_cfg():
    return GenConfig(seed=7, split=Split.TEST_ID,
                     domain=Domain.PROJECT_UPDATES, doc_len_name="short",
                     n_docs=1, family_ids=(), paraphrase_idx_range=(0, 5))


@pytest.fixture(scope="module")
def built_doc(tok):
    return build_document(_gen_cfg(), 0, tok)


@pytest.fixture(scope="module")
def shard_line(tok):
    rec = build_record({}, _gen_cfg(), 0, tok, "single_shot")
    return to_json(rec)


def _tiny_model():
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=4, head_dim=8,
                      n_kv_heads=2, ffn_hidden=64, rope_theta=10000.0,
                      max_seq=4096)
    return Transformer(cfg)


# ---------------------------------------------------------------------------
# grade_generation semantics (model-free)
# ---------------------------------------------------------------------------

def test_grade_oracle_c_star_full_survival(tok, built_doc):
    db = built_doc.fact_db
    budget = db.doc_len_tokens // 4
    text = canonical_order_and_render(db, budget, tok)
    g = eval_mod.grade_generation(db, text, tok=tok, budget=budget)
    assert g["recall"] == 1.0, "oracle C* must recover every C*(budget) fact"
    assert g["hallucination_rate"] == 0.0
    assert g["parse_fail_rate"] == 0.0
    assert g["inversion"] is False
    for ftype, frac in g["survival_exact"].items():
        assert frac == 1.0, f"oracle C* must fully survive type {ftype}"


def test_grade_truncated_c_star_partial_recall(tok, built_doc):
    db = built_doc.fact_db
    text = canonical_order_and_render(db, db.doc_len_tokens // 4, tok)
    lines = text.split("\n")
    truncated = "\n".join(lines[: max(1, len(lines) // 3)])
    g = eval_mod.grade_generation(db, truncated)
    assert 0.0 <= g["recall"] < 1.0, \
        f"truncated C* must lose queried facts, recall={g['recall']}"
    assert g["hallucination_rate"] == 0.0
    assert g["inversion"] is False


def test_grade_garbage_is_parse_failure_not_halucination(tok, built_doc):
    g = eval_mod.grade_generation(built_doc.fact_db,
                                  "not grammar at all\nstill not grammar")
    assert g["recall"] == 0.0
    assert g["parse_fail_rate"] == 1.0
    assert g["hallucination_rate"] == 0.0, \
        "unparseable lines are parse failures, never hallucinations"


def _chain_db():
    ent = Entity(id="E0001", type="system", name="Alpha_Core", aliases=())
    chain = Fact(id="F0001", type=FactType.STATE_TRANSITION,
                 entity_ids=("E0001",), attribute="status",
                 values=("v1", "v2"), scene_positions=(0, 5),
                 is_queried=True)
    return FactDB(doc_id="chain-doc", entities=(ent,), facts=(chain,),
                  questions=())


def test_chain_current_and_prior_value_recovery(tok):
    db = _chain_db()
    text = canonical_order_and_render(db, 512, tok)
    assert "supersedes=v1" in text, "chain render must carry the prior value"
    g = eval_mod.grade_generation(db, text)
    assert g["state_current"] is True
    assert g["state_prior"] is True
    stripped = "\n".join(
        line.replace("supersedes=v1", "").replace("  ", " ")
        for line in text.split("\n"))
    g2 = eval_mod.grade_generation(db, stripped)
    assert g2["state_current"] is True
    assert g2["state_prior"] is False


def test_hallucination_and_decoy_emission_distances(tok, built_doc):
    db = built_doc.fact_db
    full = canonical_order_and_render(db, db.doc_len_tokens * 4, tok)
    g_full = eval_mod.grade_generation(db, full)
    assert g_full["recall"] == 1.0
    assert g_full["decoy_emission_rate"] > 0.0, \
        "full-budget C* includes unqueried doc facts (tolerated decoys)"
    fabricated = ("ENTITY E9999 type=system name=Ghost_System\n"
                  "FACT E9999.port = 9999 pos=99")
    g_hal = eval_mod.grade_generation(db, fabricated)
    assert g_hal["recall"] == 0.0
    assert g_hal["hallucination_rate"] > 0.0
    assert g_hal["inversion"] is False, \
        "fabricated facts are hallucinations, not decoys (PLAN 2.7/Q10)"


def test_inversion_requires_both_drop_and_decoy(tok, built_doc):
    db = built_doc.fact_db
    decoy = next(f for f in db.facts
                 if not f.is_queried and f.values
                 and f.type in (FactType.EXACT_VALUE, FactType.BINDING))
    ent = db.entity_by_id(decoy.entity_ids[0])
    decoy_only = (f"ENTITY {ent.id} type={ent.type} name={ent.name}\n"
                  f"FACT {ent.id}.{decoy.attribute} = {decoy.values[-1]} "
                  f"pos=0")
    g = eval_mod.grade_generation(db, decoy_only)
    assert g["inversion"] is True
    q_only = canonical_order_and_render(db, db.doc_len_tokens // 8, tok)
    g2 = eval_mod.grade_generation(db, q_only)
    assert g2["inversion"] is False


# ---------------------------------------------------------------------------
# prompt format + end-to-end report structure (tiny model, greedy)
# ---------------------------------------------------------------------------

def test_build_single_shot_prompt_matches_training_pack(tok, built_doc):
    from track_a.data.pack import pack_single_shot
    db = built_doc.fact_db
    render = canonical_order_and_render(db, db.doc_len_tokens // 4, tok)
    target_ids = tuple(tok.encode(render, add_special_tokens=False))
    sample = pack_single_shot(built_doc.doc_ids, target_ids,
                              db.doc_len_tokens, ratio=4, tok=tok)
    prompt = eval_mod.build_single_shot_prompt(built_doc.doc_ids,
                                               db.doc_len_tokens, 4, tok)
    assert list(prompt) == list(sample.input_ids[:len(prompt)]), \
        "eval prompt must be byte-identical to the training source prefix"


def test_evaluate_shards_report_structure(tmp_path, tok, shard_line):
    model = _tiny_model()
    shard_path = tmp_path / "eval.jsonl"
    shard_path.write_text(shard_line + "\n", encoding="utf-8")
    report = eval_mod.evaluate_shards(model, tok, [str(shard_path)],
                                      device=CPU, max_docs=1, qa_per_doc=2,
                                      ratios=(32,))
    assert {"meta", "primary", "diag"} <= set(report)
    p = report["primary"]
    for key in ("needle_survival_exact", "needle_survival_partial",
                "survival_state_current", "survival_state_prior",
                "hallucination_rate", "decoy_emission_rate",
                "salience_inversions", "parse_fail_rate", "info_pressure"):
        assert key in p, f"missing primary key {key}"
    assert isinstance(report["diag"]["qa_probe_acc"], dict)
    for key in ("hallucination_rate", "decoy_emission_rate"):
        v = p[key]
        assert 0.0 <= v <= 1.0, f"{key}={v} out of [0,1]"
    assert p["info_pressure"], "info_pressure must be reported per ratio"


def test_load_eval_model_restores_weights(tmp_path):
    cfg = {
        "run_tag": "evaltest", "seed": 5, "framing": "single_shot",
        "model": {"d_model": 32, "n_layers": 2, "n_heads": 4, "head_dim": 8,
                  "n_kv_heads": 2, "ffn_hidden": 64, "rope_theta": 10000.0,
                  "max_seq": 4096},
        "data": {"train_shards": [], "val_shards": []},
        "training": {"total_tokens": 131072, "effective_batch_tokens": 131072,
                     "max_seq": 4096, "lr": 0.0003},
        "logging": {"run_dir": str(tmp_path / "runs")},
    }
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    model = Transformer(ModelConfig(
        d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        ffn_hidden=64, rope_theta=10000.0, max_seq=4096))
    ckpt = tmp_path / "final.pt"
    save_checkpoint(ckpt, model, step=3, tokens_seen=42)
    loaded = eval_mod.load_eval_model(str(cfg_path), str(ckpt), CPU)
    ref = dict(model.named_parameters())
    got = dict(loaded.named_parameters())
    assert set(ref) == set(got)
    for name in ref:
        assert torch.equal(ref[name], got[name]), f"weight mismatch {name}"
    assert not loaded.training, "eval model must be in eval mode"
