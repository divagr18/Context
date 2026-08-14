"""Track A eval battery (contract T9, PLAN.md section 9).

Usage:
    python -m track_a.eval --config configs/train_smoke.yaml \
        --checkpoint runs/smoke-11/final.pt \
        --shards data/dev/smoke_ss.jsonl \
        [--out runs/smoke-11/eval_report.json] [--max-docs N] \
        [--qa-per-doc N] [--ratios 2,4,8,16,32]

For every shard record x compression ratio: greedy-generate C-hat from the
training-identical prompt, parse it, grade it against ground truth. Metrics
live under ``primary/``; QA probes are diagnostic-only under ``diag/``
(PLAN Q5+). Grading never raises on malformed output -- parse failures are
counted separately from factual errors (PLAN 4).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from track_a.data.budget_sampler import available_ratios
from track_a.data.pack import budget_token_id, special_id
from track_a.evalkit.grading import (
    decoy_triples, fact_triples_from_records, ground_truth_triples,
    recall_and_hallucination,
)
from track_a.model.core import Transformer
from track_a.model.generate import greedy_generate
from track_a.needle_gen.generate import canonical_order_and_render
from track_a.needle_gen.types import (
    FactDB, FactType, UncertaintyKind, canonicalize_value,
)
from track_a.parse_compact import (
    EntityRec, FactRec, NegRec, RelRec, UnresolvedRec, parse_target_lines,
)
from track_a.shard_schema import KIND_SINGLE_SHOT, from_json
from track_a.tokenize import get_tokenizer
from track_a.training.checkpoints import load_checkpoint
from track_a.training.config_resolve import load_run_config

_VALUE_TYPES = (FactType.EXACT_VALUE, FactType.BINDING,
                FactType.STATE_TRANSITION)
_QA_MAX_NEW = 32


def build_single_shot_prompt(doc_ids, doc_len_tokens: int, ratio: int,
                             tok) -> list[int]:
    """Source prefix of the single-shot training format (PLAN 4), verbatim."""
    budget = max(1, doc_len_tokens // ratio)
    return ([special_id(tok, "<doc>")] + list(doc_ids)
            + [special_id(tok, "</doc>"), budget_token_id(tok, budget)])


def normalize_answer(text: str) -> str:
    """Canonical answer comparison form (PLAN 4 value canonicalization)."""
    return canonicalize_value(text.strip())


def _fact_name(fact_db, fact):
    ent = fact_db.entity_by_id(fact.entity_ids[0]) if fact.entity_ids else None
    return canonicalize_value(ent.name) if ent is not None else None


def _exact_hit(fact_db, f, parsed_triples, parsed_rels, parsed_negs,
               parsed_unres, parsed_hedged) -> bool:
    if f.type in _VALUE_TYPES:
        name = _fact_name(fact_db, f)
        if name is None or not f.values:
            return False
        return (name, f.attribute or "",
                canonicalize_value(f.values[-1])) in parsed_triples
    if f.type is FactType.RELATIONAL:
        if len(f.entity_ids) < 2:
            return False
        ea = fact_db.entity_by_id(f.entity_ids[0])
        eb = fact_db.entity_by_id(f.entity_ids[1])
        if ea is None or eb is None:
            return False
        return (canonicalize_value(ea.name), f.relation,
                canonicalize_value(eb.name)) in parsed_rels
    if f.type is FactType.NEGATIVE:
        name = _fact_name(fact_db, f)
        return name is not None and (name, f.attribute or "") in parsed_negs
    if f.type is FactType.UNCERTAINTY:
        name = _fact_name(fact_db, f)
        if name is None:
            return False
        if f.uncertainty_kind is UncertaintyKind.CONFLICT:
            return (name, f.attribute or "") in parsed_unres
        if not f.values:
            return False
        return (name, f.attribute or "",
                canonicalize_value(f.values[-1])) in parsed_hedged
    return False


def _parsed_sets(records, names):
    """Membership sets over parsed records (used for both the generated C
    and the per-budget C* ground-truth scoping)."""
    triples = fact_triples_from_records(records)
    pairs = {(names.get(r.eid), r.attr) for r in records
             if isinstance(r, FactRec) and names.get(r.eid)}
    rels = {(names.get(r.eid_a), r.rel, names.get(r.eid_b))
            for r in records if isinstance(r, RelRec)
            and names.get(r.eid_a) and names.get(r.eid_b)}
    negs = {(names.get(r.eid), r.attr) for r in records
            if isinstance(r, NegRec) and names.get(r.eid)}
    unres = {(names.get(r.eid), r.attr) for r in records
             if isinstance(r, UnresolvedRec) and names.get(r.eid)}
    hedged = {(names.get(r.eid), r.attr, r.value) for r in records
              if isinstance(r, FactRec) and r.hedged and names.get(r.eid)}
    return triples, pairs, rels, negs, unres, hedged


def grade_generation(fact_db, text: str, tok=None, budget=None) -> dict:
    """Model-free grading of one generated C against ground truth.

    Ground truth is C*(budget) when ``budget`` (and ``tok``) are given
    (PLAN 2.6: per-budget ground truth -- the longest-prefix render may
    legitimately cut queried units); otherwise all queried facts. Survival
    denominators count only facts that C*(budget) itself contains.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    records, failures = parse_target_lines(lines)

    names: dict[str, str] = {r.eid: r.name for r in records
                             if isinstance(r, EntityRec)}
    (parsed_triples, parsed_pairs, parsed_rels, parsed_negs,
     parsed_unres, parsed_hedged) = _parsed_sets(records, names)

    scoped = budget is not None and tok is not None
    if scoped:
        c_star = canonical_order_and_render(fact_db, budget, tok)
        c_records, _ = parse_target_lines(
            [l for l in c_star.split("\n") if l.strip()])
        c_names = {r.eid: r.name for r in c_records
                   if isinstance(r, EntityRec)}
        (gt, _c_pairs, c_rels, c_negs, c_unres,
         c_hedged) = _parsed_sets(c_records, c_names)

        def in_c_star(f):
            return _exact_hit(fact_db, f, gt, c_rels, c_negs, c_unres,
                              c_hedged)
    else:
        gt = ground_truth_triples(fact_db)

        def in_c_star(f):
            return True

    decoys = decoy_triples(fact_db)
    recall, halluc = recall_and_hallucination(parsed_triples, gt, decoys)

    queried = [f for f in fact_db.facts if f.is_queried]
    survival_exact: dict[str, list] = defaultdict(lambda: [0, 0])
    survival_partial: dict[str, list] = defaultdict(lambda: [0, 0])
    fact_outcomes: list[tuple] = []
    for f in queried:
        if not in_c_star(f):
            continue
        exact = _exact_hit(fact_db, f, parsed_triples, parsed_rels,
                           parsed_negs, parsed_unres, parsed_hedged)
        if f.type in _VALUE_TYPES:
            name = _fact_name(fact_db, f)
            partial = (name is not None
                       and (name, f.attribute or "") in parsed_pairs)
        else:
            partial = exact
        tval = f.type.value
        survival_exact[tval][0] += int(exact)
        survival_exact[tval][1] += 1
        survival_partial[tval][0] += int(partial)
        survival_partial[tval][1] += 1
        dist = f.distance_bucket.value if f.distance_bucket else "all"
        fact_outcomes.append((tval, dist, exact, partial))

    chains = [f for f in queried if in_c_star(f)
              and f.type is FactType.STATE_TRANSITION and len(f.values) >= 2]
    state_current = state_prior = None
    if chains:
        cur_hits = prior_hits = 0
        for f in chains:
            name = _fact_name(fact_db, f)
            final = canonicalize_value(f.values[-1])
            prev = canonicalize_value(f.values[-2])
            cur = (name is not None
                   and (name, f.attribute or "", final) in parsed_triples)
            prior = any(
                isinstance(r, FactRec) and names.get(r.eid) == name
                and r.attr == (f.attribute or "")
                and canonicalize_value(r.value) == final
                and r.supersedes is not None
                and canonicalize_value(r.supersedes) == prev
                for r in records)
            cur_hits += int(cur)
            prior_hits += int(prior)
        state_current = cur_hits == len(chains)
        state_prior = prior_hits == len(chains)

    dropped_queried = len(gt - parsed_triples)
    emitted_decoy = len(parsed_triples & decoys)
    return {
        "recall": recall,
        "hallucination_rate": halluc,
        "n_parse_failures": len(failures),
        "n_lines": len(lines),
        "parse_fail_rate": len(failures) / max(1, len(lines)),
        "decoy_emission_rate": emitted_decoy / max(1, len(decoys)),
        "dropped_queried": dropped_queried,
        "emitted_decoy": emitted_decoy,
        "inversion": dropped_queried >= 1 and emitted_decoy >= 1,
        "state_current": state_current,
        "state_prior": state_prior,
        "survival_exact": {t: h / n for t, (h, n) in survival_exact.items()},
        "survival_partial": {t: h / n
                             for t, (h, n) in survival_partial.items()},
        "fact_outcomes": fact_outcomes,
    }


def run_qa_probe(model, tok, c_hat_ids, question) -> str:
    """Greedy answer for one question conditioned on the model's own C-hat."""
    prompt = ([special_id(tok, "<doc>")] + list(c_hat_ids)
              + [special_id(tok, "</doc>"), special_id(tok, "<Q>")]
              + tok.encode(question.text, add_special_tokens=False)
              + [special_id(tok, "</Q>"), special_id(tok, "<A>")])
    a_close = tok.convert_tokens_to_ids("</A>")
    gen = greedy_generate(model, prompt, _QA_MAX_NEW, [a_close])
    return tok.decode(gen).strip()


def evaluate_shards(model, tok, shard_paths, device, max_docs=None,
                    qa_per_doc: int = 8, ratios=None,
                    max_new_pad: int = 32) -> dict:
    """Full single-shot eval battery over shard files; returns the report."""
    model.eval()
    c_close = tok.convert_tokens_to_ids("</C>")
    ratio_filter = set(ratios) if ratios else None

    surv_exact: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: [0, 0])))
    surv_partial: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: [0, 0])))
    per_ratio_hall: dict = defaultdict(list)
    per_ratio_decoy: dict = defaultdict(list)
    per_ratio_inversion: dict = defaultdict(list)
    per_ratio_failures = defaultdict(int)
    per_ratio_lines = defaultdict(int)
    info_pressure: dict = defaultdict(list)
    state_cur_list: list = []
    state_prior_list: list = []
    qa_hits: dict = defaultdict(lambda: [0, 0])
    grade_recalls: list = []
    n_docs = 0
    stop = False

    for path in shard_paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = from_json(line)
                if rec.kind != KIND_SINGLE_SHOT:
                    continue
                if max_docs is not None and n_docs >= max_docs:
                    stop = True
                    break
                n_docs += 1
                fact_db = FactDB(doc_id=rec.doc_id, entities=rec.entities,
                                 facts=rec.facts, questions=rec.questions)
                rec_ratios = [r for r in available_ratios(rec.c_renders)
                              if ratio_filter is None or r in ratio_filter]
                for ratio in rec_ratios:
                    rkey = str(ratio)
                    prompt = build_single_shot_prompt(
                        rec.doc_ids, rec.doc_len_tokens, ratio, tok)
                    target_len = len(rec.c_renders[rkey].target_ids)
                    gen = greedy_generate(model, prompt,
                                          target_len + max_new_pad, [c_close])
                    text = tok.decode(gen)
                    g = grade_generation(
                        fact_db, text, tok=tok,
                        budget=max(1, rec.doc_len_tokens // ratio))
                    grade_recalls.append(g["recall"])
                    per_ratio_hall[rkey].append(g["hallucination_rate"])
                    per_ratio_decoy[rkey].append(g["decoy_emission_rate"])
                    per_ratio_inversion[rkey].append(float(g["inversion"]))
                    per_ratio_failures[rkey] += g["n_parse_failures"]
                    per_ratio_lines[rkey] += max(1, g["n_lines"])
                    info_pressure[rkey].append(
                        rec.c_renders[rkey].info_pressure_fact_count)
                    for tval, dist, exact, partial in g["fact_outcomes"]:
                        surv_exact[tval][dist][rkey][0] += int(exact)
                        surv_exact[tval][dist][rkey][1] += 1
                        surv_partial[tval][dist][rkey][0] += int(partial)
                        surv_partial[tval][dist][rkey][1] += 1
                    if g["state_current"] is not None:
                        state_cur_list.append(float(g["state_current"]))
                        state_prior_list.append(float(g["state_prior"]))
                    if qa_per_doc > 0 and rec.questions:
                        qa_ids = gen
                        fact_by_id = {f.id: f for f in rec.facts}
                        for q in rec.questions[:qa_per_doc]:
                            answer = run_qa_probe(model, tok, qa_ids, q)
                            qtype = "unknown"
                            if q.fact_ids and q.fact_ids[0] in fact_by_id:
                                qtype = fact_by_id[q.fact_ids[0]].type.value
                            hit = (normalize_answer(answer)
                                   == normalize_answer(q.answer))
                            qa_hits[qtype][0] += int(hit)
                            qa_hits[qtype][1] += 1
        if stop:
            break

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def _frac_tree(tree):
        return {
            t: {d: {r: vals[0] / vals[1] for r, vals in dist_map.items()}
                for d, dist_map in type_map.items()}
            for t, type_map in tree.items()}

    primary = {
        "needle_survival_exact": _frac_tree(surv_exact),
        "needle_survival_partial": _frac_tree(surv_partial),
        "survival_state_current": _mean(state_cur_list) if state_cur_list
        else None,
        "survival_state_prior": _mean(state_prior_list) if state_prior_list
        else None,
        "hallucination_rate": _mean([h for v in per_ratio_hall.values()
                                     for h in v]),
        "decoy_emission_rate": _mean([d for v in per_ratio_decoy.values()
                                      for d in v]),
        "salience_inversions": {r: _mean(v)
                                for r, v in per_ratio_inversion.items()},
        "parse_fail_rate": {
            "overall": (sum(per_ratio_failures.values())
                        / max(1, sum(per_ratio_lines.values()))),
            **{r: per_ratio_failures[r] / max(1, per_ratio_lines[r])
               for r in per_ratio_lines},
        },
        "info_pressure": {r: _mean(v) for r, v in info_pressure.items()},
        "recall_exact_value_facts": _mean(grade_recalls),
        "n_docs": n_docs,
    }
    diag = {
        "qa_probe_acc": {t: h / n for t, (h, n) in qa_hits.items()},
        "qa_probe_counts": {t: n for t, (_h, n) in qa_hits.items()},
    }
    meta = {
        "shards": [str(p) for p in shard_paths],
        "ratios": sorted({int(r) for r in info_pressure}),
        "device": str(device),
        "max_docs": max_docs,
        "qa_per_doc": qa_per_doc,
    }
    return {"meta": meta, "primary": primary, "diag": diag}


def load_eval_model(config_path: str, checkpoint_path: str, device):
    """Rebuild the run's model from its YAML and load checkpoint weights."""
    cfg = load_run_config(config_path)
    model = Transformer(cfg.model).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    return model


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Track A eval battery")
    parser.add_argument("--config", required=True,
                        help="training run YAML (model resolution)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--out", default=None, help="report JSON path")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--qa-per-doc", type=int, default=8)
    parser.add_argument("--ratios", default=None,
                        help="comma-separated subset, e.g. 2,4,8,16,32")
    args = parser.parse_args(argv)

    tok = get_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_eval_model(args.config, args.checkpoint, device)
    ratios = ([int(r) for r in args.ratios.split(",")]
              if args.ratios else None)
    report = evaluate_shards(model, tok, args.shards, device,
                             max_docs=args.max_docs,
                             qa_per_doc=args.qa_per_doc, ratios=ratios)
    report["meta"]["config"] = str(args.config)
    report["meta"]["checkpoint"] = str(args.checkpoint)

    p = report["primary"]
    print(f"[eval] docs={p['n_docs']} ratios={report['meta']['ratios']} "
          f"device={device}", flush=True)
    print(f"[eval] recall_exact={p['recall_exact_value_facts']:.4f} "
          f"halluc={p['hallucination_rate']:.4f} "
          f"decoy_emit={p['decoy_emission_rate']:.4f} "
          f"parse_fail={p['parse_fail_rate']['overall']:.4f}", flush=True)
    if report["diag"]["qa_probe_acc"]:
        print(f"[eval] qa_probe_acc={report['diag']['qa_probe_acc']}",
              flush=True)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=False),
                       encoding="utf-8")
        print(f"[eval] report -> {out}", flush=True)


if __name__ == "__main__":
    main()
