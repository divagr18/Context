"""Per-document entity registry (fresh entity set per doc, PLAN T3B)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from track_a.needle_gen.types import canonicalize_value

# Entity-type vocabularies per domain (surface word used in BIND-01's
# {entity_type} slot and stored as Entity.type).
DOMAIN_ENTITY_TYPES: dict[str, tuple[tuple[str, str], ...]] = {
    # domain -> ((entity_type, pool_key), ...)
    "project_updates": (("person", "person_names"), ("project", "org_names"),
                        ("system", "org_names")),
    "logistics_ops": (("person", "person_names"), ("carrier", "org_names"),
                      ("depot", "org_names")),
}


@dataclass(frozen=True)
class EntitySpec:
    eid: str
    type: str
    surface: str  # prose form (may contain spaces)
    alias: str | None  # short prose alias, or None
    index: int = field(default=0)

    @property
    def canonical(self) -> str:
        return canonicalize_value(self.surface)


def make_entities(rng: random.Random, pools: dict, domain: str,
                  n_objects: int, n_persons: int) -> list[EntitySpec]:
    """Create fresh entities: *n_objects* system/project entities first, then
    *n_persons* person entities. Attribute facts consume the object block, so
    ports/versions attach to systems, not people.

    Persons get a first-name alias. Exhausted pools fall back to deterministic
    unique synthetic names (never raises, never duplicates).
    """
    type_entries = DOMAIN_ENTITY_TYPES[domain]
    object_entries = [(t, p) for (t, p) in type_entries if p != "person_names"]
    person_entry = next((t, p) for (t, p) in type_entries if p == "person_names")
    people = list(pools["person_names"])
    orgs = list(pools["org_names"])
    rng.shuffle(people)
    rng.shuffle(orgs)
    out: list[EntitySpec] = []
    person_i = 0
    org_i = 0

    def add(etype: str, pool_key: str, idx: int) -> None:
        nonlocal person_i, org_i
        if pool_key == "person_names":
            surface = people[person_i] if person_i < len(people) else f"person-{idx:03d}"
            person_i += 1
            alias = surface.split()[0] if " " in surface else None
        else:
            surface = orgs[org_i] if org_i < len(orgs) else f"{etype}-{idx:03d}"
            org_i += 1
            alias = None
        out.append(EntitySpec(eid=f"E{len(out) + 1:04d}", type=etype, surface=surface,
                              alias=alias, index=len(out)))

    for i in range(n_objects):
        etype, pool_key = object_entries[i % len(object_entries)]
        add(etype, pool_key, i)
    for i in range(n_persons):
        add(person_entry[0], person_entry[1], i)
    return out
