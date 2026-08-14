"""Scenario S3: canonical C* ordering and rendering (PLAN.md section 2)."""

from __future__ import annotations

import pytest

from track_a.needle_gen import fixtures
from track_a.needle_gen.generate import (
    canonical_order_and_render,
    render_entity,
    render_unit,
    sort_key,
)
from track_a.needle_gen.types import Entity, Fact, FactDB, FactType, UncertaintyKind
from track_a.tokenize import get_tokenizer

SEEDS = tuple(range(200))
BIG_BUDGET = 10**6
BUDGETS = (40, 64, 96, 128, 192, 256, 384, 512, 1024)


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


def _db(seed: int) -> FactDB:
    return fixtures.make_fixture_factdb(
        seed=seed,
        n_queried=7 + (seed % 3),
        n_decoys=2 + (seed % 4),
        with_chain=seed % 2 == 0,
    )


def _line_to_fact(db: FactDB) -> dict[str, Fact]:
    mapping: dict[str, Fact] = {}
    for fact in db.facts:
        for line in render_unit(fact):
            mapping[line] = fact
    return mapping


def _included(rendered: str, db: FactDB) -> list[Fact]:
    """Included facts, in render order (record lines only, no ENTITY lines)."""
    if not rendered:
        return []
    mapping = _line_to_fact(db)
    return [
        mapping[line]
        for line in rendered.split("\n")
        if not line.startswith("ENTITY ")
    ]


@pytest.mark.parametrize("seed", SEEDS)
def test_render_within_budget(seed: int, tok) -> None:
    rendered = canonical_order_and_render(_db(seed), BUDGETS[seed % len(BUDGETS)], tok)
    assert len(tok.encode(rendered)) <= BUDGETS[seed % len(BUDGETS)]


@pytest.mark.parametrize("seed", SEEDS)
def test_budget_monotonicity(seed: int, tok) -> None:
    db = _db(seed)
    b1, b2 = BUDGETS[seed % 4], BUDGETS[4 + (seed % 4)]  # b1 < b2
    s1 = {f.id for f in _included(canonical_order_and_render(db, b1, tok), db)}
    s2 = {f.id for f in _included(canonical_order_and_render(db, b2, tok), db)}
    assert s1 <= s2


@pytest.mark.parametrize("seed", SEEDS)
def test_determinism(seed: int, tok) -> None:
    for budget in (64, 512):
        a = canonical_order_and_render(_db(seed), budget, tok)
        b = canonical_order_and_render(_db(seed), budget, tok)
        assert a == b


@pytest.mark.parametrize("seed", SEEDS)
def test_included_order_matches_sort_key(seed: int, tok) -> None:
    """Sorted order, position-ascending tie-break, decoys strictly last."""
    db = _db(seed)
    for budget in BUDGETS:
        included = _included(canonical_order_and_render(db, budget, tok), db)
        keys = [sort_key(f) for f in included]
        assert keys == sorted(keys)
        classes = [f.fact_class.value for f in included]
        assert classes == sorted(classes)


@pytest.mark.parametrize("seed", SEEDS)
def test_entity_lines_cover_included_only(seed: int, tok) -> None:
    db = _db(seed)
    rendered = canonical_order_and_render(db, BUDGETS[seed % len(BUDGETS)], tok)
    included = _included(rendered, db)
    expected = [
        render_entity(db.entity_by_id(eid))
        for eid in sorted({eid for f in included for eid in f.entity_ids})
    ]
    lines = rendered.split("\n") if rendered else []
    assert lines[: len(expected)] == expected
    assert all(not line.startswith("ENTITY ") for line in lines[len(expected):])


@pytest.mark.parametrize("seed", SEEDS)
def test_chains_never_split(seed: int, tok) -> None:
    db = fixtures.make_fixture_factdb(seed=seed, n_queried=8, n_decoys=4, with_chain=True)
    chains = [f for f in db.facts if f.is_chain]
    assert chains, "fixture must include a chain"
    for chain in chains:
        chain_lines = set(render_unit(chain))
        for budget in BUDGETS:
            lines = set(canonical_order_and_render(db, budget, tok).split("\n"))
            assert chain_lines <= lines or lines.isdisjoint(chain_lines)


def test_full_render_contains_all_facts_sorted(tok) -> None:
    db = fixtures.make_fixture_factdb(seed=7, n_queried=9, n_decoys=5, with_chain=True)
    rendered = canonical_order_and_render(db, BIG_BUDGET, tok)
    included = _included(rendered, db)
    assert {f.id for f in included} == {f.id for f in db.facts}
    assert {f.type for f in included} == set(FactType)
    keys = [sort_key(f) for f in included]
    assert keys == sorted(keys)


def test_no_skip_at_margin(tok) -> None:
    """Unit 2 too big -> STOP, even though unit 3 alone would still fit."""
    entity = Entity(id="E0001", type="project", name="alpha_rig")
    f1 = Fact(id="F0001", type=FactType.EXACT_VALUE, entity_ids=("E0001",),
              attribute="attr_a", values=("v_F1",), scene_positions=(1,), is_queried=True)
    f2 = Fact(id="F0002", type=FactType.EXACT_VALUE, entity_ids=("E0001",),
              attribute="attr_b", values=("wx" * 160,), scene_positions=(2,), is_queried=True)
    f3 = Fact(id="F0003", type=FactType.EXACT_VALUE, entity_ids=("E0001",),
              attribute="attr_c", values=("v_F3",), scene_positions=(3,), is_queried=True)
    db = FactDB(doc_id="no-skip", entities=(entity,), facts=(f1, f2, f3), questions=())
    header = render_entity(entity)
    block_f1 = "\n".join((header, *render_unit(f1)))
    block_f1_f2 = "\n".join((header, *render_unit(f1), *render_unit(f2)))
    block_f1_f3 = "\n".join((header, *render_unit(f1), *render_unit(f3)))
    budget = len(tok.encode(block_f1_f3))  # f1+f3 fits; f2 does not
    assert len(tok.encode(block_f1_f2)) > budget
    rendered = canonical_order_and_render(db, budget, tok)
    assert rendered == block_f1


def test_decoys_only_after_queried_and_cut_at_margin(tok) -> None:
    entity = Entity(id="E0001", type="person", name="aria_vale")
    queried = Fact(id="F0002", type=FactType.EXACT_VALUE, entity_ids=("E0001",),
                   attribute="attr_q", values=("v_Q",), scene_positions=(1,), is_queried=True)
    decoy = Fact(id="F0001", type=FactType.EXACT_VALUE, entity_ids=("E0001",),
                 attribute="attr_d", values=("v_D",), scene_positions=(2,), is_queried=False)
    db = FactDB(doc_id="decoy", entities=(entity,), facts=(decoy, queried), questions=())
    header = render_entity(entity)
    both = "\n".join((header, *render_unit(queried), *render_unit(decoy)))
    budget_both = len(tok.encode(both))
    # queried first even though decoy has earlier position AND lower fact id
    assert canonical_order_and_render(db, budget_both, tok) == both
    only_queried = "\n".join((header, *render_unit(queried)))
    budget_q = len(tok.encode(only_queried))
    assert canonical_order_and_render(db, budget_q, tok) == only_queried


def test_record_grammar_literal() -> None:
    """C2 grammar pinned verbatim (PLAN.md section 4)."""
    e = Entity(id="E0001", type="project", name="Alpha Rig")
    assert render_entity(e) == "ENTITY E0001 type=project name=Alpha_Rig"
    exact = Fact(id="F1", type=FactType.EXACT_VALUE, entity_ids=("E0001",),
                 attribute="owner", values=("new york",), scene_positions=(4,), is_queried=True)
    assert render_unit(exact) == ("FACT E0001.owner = new_york pos=4",)
    chain = Fact(id="F2", type=FactType.STATE_TRANSITION, entity_ids=("E0001",),
                 attribute="status", values=("v0", "v1", "v2"), scene_positions=(2, 7, 12),
                 is_queried=True)
    assert render_unit(chain) == (
        "FACT E0001.status = v2 supersedes=v0 pos=12",
        "FACT E0001.status = v1 supersedes=v0 pos=7",
    )
    hedge = Fact(id="F3", type=FactType.UNCERTAINTY, uncertainty_kind=UncertaintyKind.HEDGE,
                 entity_ids=("E0001",), attribute="eta", values=("soon",), scene_positions=(3,),
                 is_queried=True)
    assert render_unit(hedge) == ("FACT E0001.eta = soon [hedged] pos=3",)
    conflict = Fact(id="F4", type=FactType.UNCERTAINTY,
                    uncertainty_kind=UncertaintyKind.CONFLICT, entity_ids=("E0001",),
                    attribute="owner", values=("v1", "v2"), scene_positions=(4, 9),
                    is_queried=True)
    assert render_unit(conflict) == ("UNRESOLVED E0001.owner candidates=[v1@4, v2@9]",)
    neg = Fact(id="F5", type=FactType.NEGATIVE, entity_ids=("E0001",), attribute="recall",
               values=(), scene_positions=(5,), is_queried=True)
    assert render_unit(neg) == ("NEG E0001.recall denied pos=5",)
    rel = Fact(id="F6", type=FactType.RELATIONAL, entity_ids=("E0001", "E0002"),
               relation="blocks", values=(), scene_positions=(6,), is_queried=True)
    assert render_unit(rel) == ("REL E0001 blocks E0002 pos=6",)
