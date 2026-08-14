"""Bank-driven document + FactDB generator (T3B orchestrator).

build_document() plans a fact roster from the authored banks, renders scenes,
lays them out with distance padding, assembles tokens, and computes grounded
annotations + questions. Deterministic from (config.seed, doc_idx).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from track_a.needle_gen import assets_loader
from track_a.needle_gen.core import builders_chains as bc
from track_a.needle_gen.core import builders_relational as br
from track_a.needle_gen.core import builders_unary as bu
from track_a.needle_gen.core import distances, layout, questions, rendering
from track_a.needle_gen.core.entities import EntitySpec, make_entities
from track_a.needle_gen.core.factspec import FactSpec, SceneSpec
from track_a.needle_gen.core.layout import LaidScene, PreparedScene
from track_a.needle_gen.types import (
    DOC_LEN_TARGETS, CueKind, DistanceBucket, Entity, Fact, FactDB,
    FactType, GenConfig, MentionSpan, Question, UncertaintyKind, ValueSpan,
    CueSpan,
)


@dataclass(frozen=True)
class BuiltDocument:
    """A generated document: ground-truth FactDB + document token ids."""

    fact_db: FactDB
    doc_ids: tuple[int, ...]


_EXTRA_WEIGHTS = [
    ("exact", 3), ("relational", 2), ("chain_shallow", 2),
    ("negative", 1), ("hedge", 1), ("conflict", 1),
    ("binding", 1), ("chain_deep", 1),
]
_DECOY_KINDS = ["exact", "exact", "relational", "chain_shallow", "negative", "hedge"]


def _weighted_choice(rng: random.Random, weighted: list) -> str:
    total = sum(w for _, w in weighted)
    r = rng.uniform(0, total)
    acc = 0
    for kind, w in weighted:
        acc += w
        if r <= acc:
            return kind
    return weighted[-1][0]


_SCENE_EST = {
    "exact": 1, "negative": 1, "hedge": 1, "conflict": 2,
    "relational": 1, "chain_shallow": 2, "chain_deep": 3, "binding": 2,
}


def _bucket_caps(doc_len_name: str) -> dict:
    """Per-bucket count caps so cumulative distance padding fits the doc."""
    return {
        "short": {"mid": 1, "far": 0, "extreme": 0},
        "medium": {"mid": 99, "far": 1, "extreme": 0},
        "long": {"mid": 99, "far": 1, "extreme": 1},
        "xlong": {"mid": 99, "far": 99, "extreme": 1},
    }[doc_len_name]


def _binding_bucket(rng: random.Random, feasible: tuple, caps: dict,
                    state: dict) -> DistanceBucket:
    """Pick a binding distance bucket, biased near, capped by length budget."""
    options: list[DistanceBucket] = [DistanceBucket.NEAR] * 3
    for b in (DistanceBucket.MID, DistanceBucket.FAR, DistanceBucket.EXTREME):
        if b in feasible and state[b.value] < caps.get(b.value, 0):
            options.append(b)
    bucket = rng.choice(options)
    if bucket is not DistanceBucket.NEAR:
        state[bucket.value] += 1
    return bucket


def _chain_bucket(rng: random.Random, feasible: tuple, caps: dict,
                  state: dict) -> DistanceBucket | None:
    """Deep-chain spread: near-biased, at most mid, capped."""
    options: list[DistanceBucket] = [DistanceBucket.NEAR] * 3
    if DistanceBucket.MID in feasible and state["mid"] < caps.get("mid", 0):
        options.append(DistanceBucket.MID)
    bucket = rng.choice(options)
    if bucket is DistanceBucket.MID:
        state["mid"] += 1
    return bucket


def _plan_roster(rng, config, pools, target_tokens, multihop):
    """Build the list of FactSpecs (queried + decoys) with entity allocation.

    When *multihop* is set, appends a guaranteed compositional pair: a
    relational fact (owner -> target) plus an exact-value fact on the SAME
    target entity, so a 2-hop question has a grounded answer.
    """
    n_queried = max(8, int(round(target_tokens * config.queried_per_100_tokens / 100)))
    avg_scene_tokens = 25
    max_fact_scenes = int(target_tokens * 0.85 / avg_scene_tokens)
    feasible = distances.feasible_buckets(config)
    caps = _bucket_caps(config.doc_len_name)
    state = {"mid": 0, "far": 0, "extreme": 0}

    recipe: list[tuple[str, bool]] = []
    # Guaranteed coverage of every needle type (queried).
    for kind in ("exact", "relational", "chain_shallow", "chain_deep",
                 "negative", "hedge", "conflict", "binding"):
        recipe.append((kind, True))
    for _ in range(n_queried - len(recipe)):
        recipe.append((_weighted_choice(rng, _EXTRA_WEIGHTS), True))

    mult = rng.randint(config.decoys_per_queried[0], config.decoys_per_queried[1])
    for i in range(n_queried * mult):
        recipe.append((_DECOY_KINDS[i % len(_DECOY_KINDS)], False))

    # Cap the recipe to the scene budget before allocating entities.
    capped: list[tuple[str, bool]] = []
    scene_count = 0
    for kind, queried in recipe:
        est = _SCENE_EST[kind]
        if scene_count + est > max_fact_scenes:
            break
        capped.append((kind, queried))
        scene_count += est

    attr_kinds = {"exact", "chain_shallow", "chain_deep", "binding",
                  "negative", "hedge", "conflict"}
    attr_facts = sum(1 for k, _ in capped if k in attr_kinds)
    rel_facts = sum(1 for k, _ in capped if k == "relational")
    person_rel_pairs = rel_facts * 2 // 5
    obj_rel_slots = (rel_facts - person_rel_pairs) * 2
    n_objects = attr_facts + obj_rel_slots + (2 if multihop else 0)
    n_persons = person_rel_pairs * 2
    entities = make_entities(rng, pools, config.domain.value, n_objects, n_persons)
    objects = [e for e in entities if e.type != "person"]
    persons = [e for e in entities if e.type == "person"]
    obj_cursor = 0
    person_cursor = 0
    rel_person_left = person_rel_pairs

    def take_obj() -> EntitySpec:
        nonlocal obj_cursor
        e = objects[obj_cursor % len(objects)]
        obj_cursor += 1
        return e

    def take_rel_pair() -> tuple[EntitySpec, EntitySpec]:
        nonlocal person_cursor, rel_person_left
        if rel_person_left > 0 and len(persons) >= 2:
            rel_person_left -= 1
            a = persons[person_cursor % len(persons)]
            b = persons[(person_cursor + 1) % len(persons)]
            person_cursor += 2
            return a, b
        a = take_obj()
        b = take_obj()
        if b.eid == a.eid:
            b = take_obj()
        return a, b

    specs: list[FactSpec] = []
    for kind, queried in capped:
        if kind == "exact":
            specs.append(bu.build_exact(rng, pools, take_obj(), queried))
        elif kind == "negative":
            specs.append(bu.build_negative(rng, pools, take_obj(), queried))
        elif kind == "hedge":
            specs.append(bu.build_hedge(rng, pools, take_obj(), queried))
        elif kind == "conflict":
            specs.append(bu.build_conflict(rng, pools, take_obj(), queried))
        elif kind == "relational":
            a, b = take_rel_pair()
            specs.append(br.build_relational(rng, pools, a, b, queried=queried))
        elif kind == "chain_shallow":
            specs.append(bc.build_chain(rng, pools, take_obj(), queried, depth=2,
                                        deep_bucket=None))
        elif kind == "chain_deep":
            specs.append(bc.build_chain(rng, pools, take_obj(), queried, depth=3,
                                        deep_bucket=_chain_bucket(rng, feasible,
                                                                  caps, state)))
        elif kind == "binding":
            specs.append(bu.build_binding(rng, pools, take_obj(), queried,
                                          _binding_bucket(rng, feasible, caps, state)))
    multihop_indices = None
    if multihop:
        target = take_obj()
        owner = take_obj()
        base = len(specs)
        specs.append(br.build_relational(rng, pools, owner, target, queried=True,
                                         relation_hint="owns"))
        specs.append(bu.build_exact(rng, pools, target, True))
        multihop_indices = (base, base + 1)
    return entities, specs, multihop_indices


def _render_fact_scenes(rng, config, templates_by_id, keyed_specs):
    """Render every spec's scenes to PreparedScenes.

    Returns (facts_with_scenes for layout, prepared list keyed by fact key).
    """
    idx_range = config.paraphrase_idx_range
    facts_with_scenes: list[tuple[str, list, int | None]] = []
    prepared_by_key: dict[str, list] = {}
    for key, spec in keyed_specs:
        prepared: list[PreparedScene] = []
        for scene_spec in spec.scenes:
            family = templates_by_id[scene_spec.family_id]
            text, spans = rendering.render_fact_scene(scene_spec, family, rng, idx_range)
            prepared.append(PreparedScene(
                text=text, family_id=scene_spec.family_id, fact_key=key,
                is_filler=False,
                entity_slot_map=dict(scene_spec.entity_slot_map),
                value_slot_map=dict(scene_spec.value_slot_map),
                cue_kind=scene_spec.cue_kind, char_spans=spans,
            ))
        target_dist = (distances.pad_target(spec.distance_bucket)
                       if spec.distance_bucket is not None else None)
        facts_with_scenes.append((key, prepared, target_dist))
        prepared_by_key[key] = prepared
    return facts_with_scenes, prepared_by_key


def _filler_factory(rng, templates_by_id, pools, idx_range):
    """Cycle FILL families to produce fresh filler PreparedScenes."""
    fill_ids = sorted(fid for fid in templates_by_id if fid.startswith("FILL"))
    counter = [0]

    def make_filler() -> PreparedScene:
        fid = fill_ids[counter[0] % len(fill_ids)]
        counter[0] += 1
        fam = templates_by_id[fid]
        slot_values = rendering.filler_slot_values(rng, pools, fam["slots"])
        scene_spec = SceneSpec(family_id=fid, slot_values=slot_values, is_filler=True)
        text, spans = rendering.render_fact_scene(scene_spec, fam, rng, idx_range)
        return PreparedScene(text=text, family_id=fid, fact_key=None,
                             is_filler=True, char_spans=spans)

    return make_filler


def _encode_scene(tokenizer, text: str):
    """Per-scene token ids + offset mapping (offsets relative to scene text)."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(enc["input_ids"]), list(enc["offset_mapping"])


def _materialize_facts(keyed_specs, prepared_by_key, ps_index):
    """Build final Fact objects with global scene_positions."""
    facts: list[Fact] = []
    for key, spec in keyed_specs:
        prepared = prepared_by_key[key]
        global_idxs = [ps_index[id(p)] for p in prepared]
        if spec.values:
            positions: list[int | None] = [None] * len(spec.values)
            for value_idx, local_scene in spec.value_scene_idx.items():
                positions[value_idx] = global_idxs[local_scene]
            if any(p is None for p in positions):
                raise ValueError(f"fact {key}: unmapped value scene position")
            scene_positions = tuple(positions)
        else:
            scene_positions = (global_idxs[0],)
        facts.append(Fact(
            id=key, type=spec.type, entity_ids=spec.entity_ids,
            attribute=spec.attribute, values=spec.values,
            scene_positions=scene_positions, relation=spec.relation,
            uncertainty_kind=spec.uncertainty_kind, is_queried=spec.is_queried,
            distance_bucket=spec.distance_bucket,
        ))
    return facts


def build_document(config: GenConfig, doc_idx: int, tokenizer) -> BuiltDocument:
    """Generate one document + ground-truth FactDB, deterministic from seed."""
    rng = random.Random(config.seed * 1_000_003 + doc_idx)
    domain = config.domain.value
    pools = assets_loader.load_pools(domain)
    templates = assets_loader.load_templates(domain)
    qbank = assets_loader.load_questions(domain)["by_type"]
    templates_by_id = {f["family_id"]: f for f in templates["families"]}
    target_tokens = DOC_LEN_TARGETS[config.doc_len_name]

    multihop = rng.random() < config.multihop_doc_fraction
    entities, specs, multihop_indices = _plan_roster(rng, config, pools,
                                                     target_tokens, multihop)
    keyed_specs = [(f"F{i + 1:04d}", sp) for i, sp in enumerate(specs)]
    entity_by_id = {e.eid: e for e in entities}

    facts_with_scenes, prepared_by_key = _render_fact_scenes(
        rng, config, templates_by_id, keyed_specs)
    count_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    make_filler = _filler_factory(rng, templates_by_id, pools,
                                  config.paraphrase_idx_range)
    laid = layout.lay_out(rng, count_tokens, facts_with_scenes, make_filler,
                          target_tokens)

    doc_ids: list[int] = []
    scene_starts: list[int] = []
    scene_token_info: list[tuple[int, int, list]] = []
    ps_index: dict[int, int] = {}
    for i, ls in enumerate(laid):
        ids, offsets = _encode_scene(tokenizer, ls.scene.text)
        start = len(doc_ids)
        scene_starts.append(start)
        scene_token_info.append((start, start + len(ids), offsets))
        ps_index[id(ls.scene)] = i
        doc_ids.extend(ids)

    facts = _materialize_facts(keyed_specs, prepared_by_key, ps_index)

    used_eids = sorted({eid for f in facts for eid in f.entity_ids})
    entities_out = tuple(
        Entity(id=entity_by_id[e].eid, type=entity_by_id[e].type,
               name=entity_by_id[e].canonical,
               aliases=(entity_by_id[e].alias,) if entity_by_id[e].alias else ())
        for e in used_eids
    )

    mentions: list[MentionSpan] = []
    value_spans: list[ValueSpan] = []
    cue_spans: list[CueSpan] = []
    for i, ls in enumerate(laid):
        ps = ls.scene
        if ps.is_filler:
            continue
        start_tok, end_tok, offsets = scene_token_info[i]
        for slot, eid in ps.entity_slot_map.items():
            if slot not in ps.char_spans or eid not in entity_by_id:
                continue
            cs, ce = ps.char_spans[slot]
            t0, t1 = _char_span_to_tokens(offsets, cs, ce)
            slot_text = ps.text[cs:ce]
            mentions.append(MentionSpan(
                entity_id=eid, start_tok=start_tok + t0, end_tok=start_tok + t1,
                is_alias=(slot_text != entity_by_id[eid].surface)))
        for slot, value_idx in ps.value_slot_map.items():
            if slot not in ps.char_spans:
                continue
            cs, ce = ps.char_spans[slot]
            t0, t1 = _char_span_to_tokens(offsets, cs, ce)
            value_spans.append(ValueSpan(
                fact_id=ps.fact_key, value_idx=value_idx,
                start_tok=start_tok + t0, end_tok=start_tok + t1))
        if ps.cue_kind is not None:
            cue_spans.append(CueSpan(kind=ps.cue_kind, start_tok=start_tok,
                                     end_tok=end_tok))

    question_list: list[Question] = []
    qid = 0
    for fact in facts:
        if not fact.is_queried:
            continue
        for (key, text, answer, is_mh, probes_prior, ph_idx) in \
                questions.questions_for_fact(rng, qbank, fact, entity_by_id):
            question_list.append(Question(
                id=f"Q{qid + 1:04d}", fact_ids=(fact.id,), text=text,
                answer=answer, is_multihop=is_mh,
                probes_prior_value=probes_prior, phrasing_idx=ph_idx))
            qid += 1
    if multihop_indices is not None:
        rel_fact = facts[multihop_indices[0]]
        partner_fact = facts[multihop_indices[1]]
        for (key, text, answer, is_mh, probes_prior, ph_idx) in \
                questions.multihop_questions(rng, qbank, rel_fact, partner_fact,
                                             entity_by_id):
            question_list.append(Question(
                id=f"Q{qid + 1:04d}", fact_ids=(rel_fact.id, partner_fact.id),
                text=text, answer=answer, is_multihop=is_mh,
                probes_prior_value=probes_prior, phrasing_idx=ph_idx))
            qid += 1

    fact_db = FactDB(
        doc_id=f"{config.split.value}-{domain}-{config.doc_len_name}-{doc_idx}",
        entities=entities_out, facts=tuple(facts), questions=tuple(question_list),
        mentions=tuple(mentions), value_spans=tuple(value_spans),
        cue_spans=tuple(cue_spans), scene_boundaries=tuple(scene_starts),
        doc_len_tokens=len(doc_ids),
    )
    return BuiltDocument(fact_db=fact_db, doc_ids=tuple(doc_ids))


def _char_span_to_tokens(offsets: list, char_start: int, char_end: int) -> tuple[int, int]:
    """Map a scene-local char span to a token span using the scene offsets."""
    from track_a.needle_gen.core import annotations

    return annotations.char_span_to_token_span(offsets, char_start, char_end)
