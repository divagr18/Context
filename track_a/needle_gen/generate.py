"""Canonical C* ordering + rendering (contract C3; PLAN.md section 2).

Single inspectable function ``canonical_order_and_render`` that produces
full-state C* records from a FactDB. Sort key, longest-prefix inclusion,
atomic chains, and decoy ordering are ALL frozen here and nowhere else.
"""

from __future__ import annotations

from track_a.needle_gen.types import (
    TYPE_RANK, Fact, FactClass, FactDB, FactType, UncertaintyKind,
    canonicalize_value, Entity,
)


def render_entity(entity: Entity) -> str:
    """ENTITY line per PLAN.md section 4 grammar."""
    return f"ENTITY {entity.id} type={entity.type} name={canonicalize_value(entity.name)}"


def render_unit(fact: Fact) -> tuple[str, ...]:
    """Record lines for one atomic unit (chain or single-value fact)."""
    eid = fact.entity_ids[0]
    attr = fact.attribute or ""
    pos0 = fact.scene_positions[0] if fact.scene_positions else 0

    if fact.type is FactType.RELATIONAL:
        eid_b = fact.entity_ids[1] if len(fact.entity_ids) > 1 else ""
        return (f"REL {eid} {fact.relation} {eid_b} pos={pos0}",)

    if fact.type is FactType.NEGATIVE:
        return (f"NEG {eid}.{attr} denied pos={pos0}",)

    if fact.type is FactType.UNCERTAINTY:
        if fact.uncertainty_kind is UncertaintyKind.CONFLICT:
            parts = ", ".join(
                f"{canonicalize_value(v)}@{p}"
                for v, p in zip(fact.values, fact.scene_positions)
            )
            return (f"UNRESOLVED {eid}.{attr} candidates=[{parts}]",)
        val = canonicalize_value(fact.values[0]) if fact.values else ""
        return (f"FACT {eid}.{attr} = {val} hedged pos={pos0}",)

    if fact.is_chain:
        vals = [canonicalize_value(v) for v in fact.values]
        poses = fact.scene_positions
        lines = [f"FACT {eid}.{attr} = {vals[-1]} supersedes={vals[0]} pos={poses[-1]}"]
        for i in range(len(vals) - 2, 0, -1):
            lines.append(
                f"FACT {eid}.{attr} = {vals[i]} supersedes={vals[i - 1]} pos={poses[i]}"
            )
        return tuple(lines)

    val = canonicalize_value(fact.values[-1]) if fact.values else ""
    return (f"FACT {eid}.{attr} = {val} pos={pos0}",)


def sort_key(fact: Fact) -> tuple:
    """Sort key per PLAN.md section 2: (class, type_rank, position, fact_id)."""
    return (
        fact.fact_class.value,
        TYPE_RANK[fact.type],
        fact.chain_start_position,
        fact.id,
    )


def _entity_header(facts: tuple[Fact, ...], db: FactDB) -> list[str]:
    """ENTITY lines for entities referenced by *facts*, ordered by eid."""
    eids = sorted({eid for f in facts for eid in f.entity_ids})
    return [render_entity(db.entity_by_id(eid)) for eid in eids if db.entity_by_id(eid)]


def canonical_order_and_render(fact_db: FactDB, budget_tokens: int, tokenizer) -> str:
    """Canonical C* render within budget (longest prefix, no skipping)."""
    ordered = sorted(fact_db.facts, key=sort_key)
    included: list[Fact] = []

    for fact in ordered:
        tentative = tuple(included) + (fact,)
        header_lines = _entity_header(tentative, fact_db)
        fact_lines = [ln for f in tentative for ln in render_unit(f)]
        all_lines = header_lines + fact_lines
        text = "\n".join(all_lines) if all_lines else ""
        if len(tokenizer.encode(text, add_special_tokens=False)) <= budget_tokens:
            included.append(fact)
        else:
            break

    if not included:
        return ""
    header = _entity_header(tuple(included), fact_db)
    body = [ln for f in included for ln in render_unit(f)]
    return "\n".join(header + body)
