"""Corpus writer CLI (contract C8).

Generates a JSONL shard from a split config. Two record kinds:

* ``single_shot``: doc ids + per-compression-ratio C* renders (``c_renders``).
* ``streaming``: doc ids + per-window (state, window, edit-ops) samples.

Usage:
    python -m track_a.needle_gen.corpus_writer \
        --config track_a/needle_gen/splits/train.yaml \
        --out data/shards/train.jsonl [--limit-docs 4] [--kind streaming]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from track_a.needle_gen.core.docgen import build_document
from track_a.needle_gen.generate import canonical_order_and_render
from track_a.needle_gen.streaming import build_streaming_windows
from track_a.needle_gen.types import Domain, GenConfig, Split
from track_a.parse_compact import parse_target_lines
from track_a.parse_compact.records import EntityRec
from track_a.shard_schema import (
    CRender, KIND_SINGLE_SHOT, KIND_STREAMING, ShardRecord, to_json,
)
from track_a.tokenize import get_tokenizer

COMPRESSION_RATIOS = (2, 4, 8, 16, 32)


def load_split_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def make_gen_config(cfg: dict) -> GenConfig:
    return GenConfig(
        seed=cfg["seed"],
        split=Split(cfg["split"]),
        domain=Domain(cfg["domain"]),
        doc_len_name=cfg["doc_len_name"],
        n_docs=cfg["n_docs"],
        family_ids=tuple(cfg.get("family_ids", [])),
        paraphrase_idx_range=tuple(cfg["paraphrase_idx_range"]),
        decoys_per_queried=tuple(cfg.get("decoys_per_queried", (1, 3))),
        chain_fraction=cfg.get("chain_fraction", 0.3),
        chain_depth_max=cfg.get("chain_depth_max", 3),
        multihop_doc_fraction=cfg.get("multihop_doc_fraction", 0.15),
        queried_per_100_tokens=cfg.get("queried_per_100_tokens", 1.0),
        window_tokens=cfg.get("window_tokens", 2048),
        window_overlap_frac=cfg.get("window_overlap_frac", 0.10),
    )


def _fact_count(rendered_text: str) -> int:
    if not rendered_text:
        return 0
    records, _failures = parse_target_lines(rendered_text.split("\n"))
    return sum(1 for r in records if not isinstance(r, EntityRec))


def build_c_renders(fact_db, tokenizer) -> dict[str, CRender]:
    """Single-shot C* renders keyed by compression-ratio string."""
    doc_len = fact_db.doc_len_tokens
    renders: dict[str, CRender] = {}
    for ratio in COMPRESSION_RATIOS:
        budget = max(1, doc_len // ratio)
        text = canonical_order_and_render(fact_db, budget, tokenizer)
        ids = tuple(tokenizer.encode(text, add_special_tokens=False)) if text else ()
        renders[str(ratio)] = CRender(target_ids=ids,
                                      info_pressure_fact_count=_fact_count(text))
    return renders


def build_record(cfg: dict, gen_config: GenConfig, doc_idx: int, tokenizer,
                 kind: str):
    built = build_document(gen_config, doc_idx, tokenizer)
    fact_db = built.fact_db
    common = dict(
        doc_id=fact_db.doc_id, split=gen_config.split, domain=gen_config.domain,
        doc_len_name=gen_config.doc_len_name, doc_ids=built.doc_ids,
        entities=fact_db.entities, facts=fact_db.facts, questions=fact_db.questions,
        mention_spans=fact_db.mentions, value_spans=fact_db.value_spans,
        cue_spans=fact_db.cue_spans, scene_boundaries=fact_db.scene_boundaries,
        doc_len_tokens=fact_db.doc_len_tokens,
    )
    if kind == KIND_SINGLE_SHOT:
        return ShardRecord(kind=KIND_SINGLE_SHOT, **common,
                           c_renders=build_c_renders(fact_db, tokenizer), windows=None)
    entity_map = {e.id: e for e in fact_db.entities}
    budget = max(1, fact_db.doc_len_tokens // cfg.get("streaming_ratio", 8))
    windows = build_streaming_windows(
        built.doc_ids, fact_db.facts, fact_db.scene_boundaries, entity_map, budget,
        gen_config.window_tokens, gen_config.window_overlap_frac, tokenizer,
    )
    return ShardRecord(kind=KIND_STREAMING, **common, c_renders=None,
                       windows=tuple(windows))


def write_corpus(config_path: str | Path, out_path: str | Path,
                 limit_docs: int | None = None, kind: str | None = None,
                 tokenizer=None) -> int:
    """Generate the shard; returns the number of records written."""
    cfg = load_split_config(config_path)
    if kind is not None:
        cfg["kind"] = kind
    shard_kind = cfg.get("kind", KIND_SINGLE_SHOT)
    if shard_kind not in (KIND_SINGLE_SHOT, KIND_STREAMING):
        raise ValueError(f"unknown kind: {shard_kind}")
    gen_config = make_gen_config(cfg)
    tok = tokenizer if tokenizer is not None else get_tokenizer()
    n_docs = cfg["n_docs"] if limit_docs is None else min(limit_docs, cfg["n_docs"])

    lines: list[str] = []
    for doc_idx in range(n_docs):
        record = build_record(cfg, gen_config, doc_idx, tok, shard_kind)
        lines.append(to_json(record))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Track A corpus writer")
    parser.add_argument("--config", required=True, help="split YAML config")
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--kind", default=None,
                        choices=[KIND_SINGLE_SHOT, KIND_STREAMING])
    args = parser.parse_args(argv)
    n = write_corpus(args.config, args.out, limit_docs=args.limit_docs, kind=args.kind)
    print(f"wrote {n} records -> {args.out}")


if __name__ == "__main__":
    main()
