"""Isolation diagnostic: teacher-forced target recovery vs greedy decode.

Answers the "recall_exact=0.0" mystery by splitting it into three questions:
  1. Does the checkpoint reproduce C* under full forward (teacher-forced)?
     -> next-token argmax accuracy on the target span.
  2. Does greedy_generate's first token match the first target token?
  3. Does the full greedy output parse / grade at all?

If (1) is high but (3) is zero, the KV-cache decoder diverges from the
training forward pass. If (1) is zero, the checkpoint/prompt pair itself
is broken (load, tokenizer, or format mismatch).

Usage:
    python -m track_a.diag_generation --config configs/train_smoke.yaml \
        --checkpoint runs/smoke-11/best.pt \
        --shards data/dev/smoke_ss.jsonl [--n-docs 2] [--ratio 2]
"""
from __future__ import annotations

import argparse

import torch

from track_a.data.pack import IGNORE_INDEX, pack_single_shot
from track_a.eval import build_single_shot_prompt, grade_generation
from track_a.evalkit.grading import parse_c_text
from track_a.model.core import Transformer
from track_a.model.generate import greedy_generate
from track_a.shard_schema import KIND_SINGLE_SHOT, from_json
from track_a.tokenize import get_tokenizer
from track_a.training.checkpoints import load_checkpoint
from track_a.training.config_resolve import load_run_config


@torch.no_grad()
def teacher_forced_accuracy(model, sample, device) -> tuple[float, int]:
    """Argmax next-token accuracy over the target span (labels != IGNORE)."""
    ids = torch.tensor([sample.input_ids], dtype=torch.long, device=device)
    labels = torch.tensor([sample.labels], dtype=torch.long)
    logits, _ = model(ids)
    # labels[i] is the token to predict from position i (pack.py contract).
    preds = logits[0].argmax(-1).cpu()
    mask = labels[0] != IGNORE_INDEX
    if int(mask.sum()) == 0:
        return 0.0, 0
    correct = (preds[mask] == labels[0][mask]).sum().item()
    return correct / int(mask.sum()), int(mask.sum())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--n-docs", type=int, default=2)
    ap.add_argument("--ratio", type=int, default=2)
    args = ap.parse_args(argv)

    tok = get_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_run_config(args.config)
    model = Transformer(cfg.model).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    c_close = tok.convert_tokens_to_ids("</C>")

    n = 0
    for path in args.shards:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = from_json(line)
                if rec.kind != KIND_SINGLE_SHOT:
                    continue
                if n >= args.n_docs:
                    break
                n += 1
                ratio_key = str(args.ratio)
                if ratio_key not in rec.c_renders:
                    print(f"[doc {n}] ratio {args.ratio} not in renders, skip")
                    continue
                render = rec.c_renders[ratio_key]
                sample = pack_single_shot(rec.doc_ids, render.target_ids,
                                          rec.doc_len_tokens, args.ratio, tok)
                tf_acc, tf_n = teacher_forced_accuracy(model, sample, device)

                prompt = build_single_shot_prompt(rec.doc_ids,
                                                  rec.doc_len_tokens,
                                                  args.ratio, tok)
                prompt_match = tuple(prompt) == sample.input_ids[:len(prompt)]
                target_ids = sample.input_ids[len(prompt):]
                gen = greedy_generate(model, prompt, len(target_ids) + 32,
                                      [c_close])
                first_match = bool(gen) and int(gen[0]) == int(target_ids[0])
                text = tok.decode(gen)
                records, failures = parse_c_text(text)
                g = grade_generation(
                    __import__("track_a.needle_gen.types",
                               fromlist=["FactDB"]).FactDB(
                        doc_id=rec.doc_id, entities=rec.entities,
                        facts=rec.facts, questions=rec.questions),
                    text, tok=tok,
                    budget=max(1, rec.doc_len_tokens // args.ratio))

                target_text = tok.decode(list(render.target_ids))
                print(f"--- doc {n} (ratio {args.ratio}) ---")
                print(f"  prompt==training_prefix : {prompt_match}")
                print(f"  teacher_forced_acc      : {tf_acc:.4f} "
                      f"({tf_n} target tokens)")
                print(f"  greedy_first_token_match: {first_match} "
                      f"(gen[0]={gen[0] if gen else None} "
                      f"target[0]={int(target_ids[0])})")
                print(f"  greedy_len={len(gen)} target_len={len(target_ids)} "
                      f"parsed_lines={len(records)} parse_fails={len(failures)}")
                print(f"  grade: recall={g['recall']:.4f} "
                      f"halluc={g['hallucination_rate']:.4f} "
                      f"parse_fail={g['n_parse_failures']}")
                print(f"  target[:160]  : {target_text[:160]!r}")
                print(f"  generated[:160]: {text[:160]!r}")
        if n >= args.n_docs:
            break


if __name__ == "__main__":
    main()
