"""Load and validate hand-authored content assets (templates, pools, questions).

Raises ValueError with precise messages on any validation failure.
Every public function accepts a domain string matching types.Domain.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from track_a.needle_gen.types import VALUE_PATTERN

ASSETS_ROOT = Path(__file__).resolve().parent / "assets"
PARAPHRASE_COUNT = 8

# (family_id, fact_type, expected_slots, carries_fact)
EXPECTED_FAMILIES: list[tuple[str, str, list[str], bool]] = [
    ("FILL-01", "filler", ["team"], False),
    ("FILL-02", "filler", ["project"], False),
    ("FILL-03", "filler", ["person", "team"], False),
    ("FILL-04", "filler", ["project", "person"], False),
    ("EXACT-01", "exact_value", ["entity", "attribute", "value"], True),
    ("EXACT-02", "exact_value", ["entity", "attribute", "value"], True),
    ("REL-01", "relational", ["entity_a", "relation", "entity_b"], True),
    ("REL-02", "relational", ["entity_a", "relation", "entity_b"], True),
    ("STATE-01", "state_transition",
     ["entity", "attribute", "old_value", "new_value"], True),
    ("NEG-01", "negative", ["entity", "attribute"], True),
    ("UNRES-01", "uncertainty_hedge",
     ["source", "entity", "attribute", "value"], True),
    ("UNRES-02", "uncertainty_conflict",
     ["source", "entity", "attribute", "value"], True),
    ("BIND-01", "binding_intro",
     ["entity", "entity_type", "attribute", "value"], True),
    ("BIND-02", "binding_reference",
     ["entity", "attribute", "new_value"], True),
]
NUM_FAMILIES = len(EXPECTED_FAMILIES)

QUESTION_KEYS: tuple[str, ...] = (
    "exact_value", "relational",
    "state_transition_current", "state_transition_prior",
    "negative", "uncertainty_hedge", "uncertainty_conflict",
    "binding", "multihop_owns_attr", "multihop_superseded_owner",
)

QUESTION_ANSWERS: dict[str, str] = {
    "exact_value": "{value}",
    "relational": "{entity_a}",
    "state_transition_current": "{value}",
    "state_transition_prior": "{old_value}",
    "negative": "denied",
    "uncertainty_hedge": "unconfirmed:{value}",
    "uncertainty_conflict": "unresolved",
    "binding": "{value}",
    "multihop_owns_attr": "{entity_a}",
    "multihop_superseded_owner": "{entity_a}",
}

SLOT_RE = re.compile(r"\{(\w+)\}")


def _read_json(path: Path) -> dict:
    """Read and parse a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _domain_dir(domain: str) -> Path:
    return ASSETS_ROOT / domain


def _check_slots(tmpl: str, expected: list[str], fid: str, idx: int) -> None:
    found = sorted(SLOT_RE.findall(tmpl))
    want = sorted(expected)
    if found != want:
        raise ValueError(
            f"{fid}[{idx}]: slots {found} != expected {want}"
        )


def _check_value(v: str, ctx: str) -> None:
    if not VALUE_PATTERN.match(v):
        raise ValueError(f"{ctx}: value {v!r} fails VALUE_PATTERN")


def load_templates(domain: str) -> dict:
    """Load and validate templates.json for *domain*."""
    path = _domain_dir(domain) / "templates.json"
    data = _read_json(path)
    if data.get("domain") != domain:
        raise ValueError(
            f"templates domain {data.get('domain')!r} != {domain!r}"
        )
    families = data.get("families", [])
    if len(families) != NUM_FAMILIES:
        raise ValueError(
            f"{domain}: expected {NUM_FAMILIES} families, got {len(families)}"
        )
    by_id = {f["family_id"]: f for f in families}
    for fid, ftype, slots, carries in EXPECTED_FAMILIES:
        fam = by_id.get(fid)
        if fam is None:
            raise ValueError(f"{domain}: missing family {fid}")
        if fam["fact_type"] != ftype:
            raise ValueError(
                f"{domain}/{fid}: fact_type {fam['fact_type']!r} != {ftype!r}"
            )
        if sorted(fam["slots"]) != sorted(slots):
            raise ValueError(
                f"{domain}/{fid}: slots {fam['slots']} != {slots}"
            )
        if fam["carries_fact"] is not carries:
            raise ValueError(
                f"{domain}/{fid}: carries_fact {fam['carries_fact']} != {carries}"
            )
        paras = fam.get("paraphrases", [])
        if len(paras) != PARAPHRASE_COUNT:
            raise ValueError(
                f"{domain}/{fid}: expected {PARAPHRASE_COUNT} paraphrases, "
                f"got {len(paras)}"
            )
        if len(set(paras)) != len(paras):
            raise ValueError(f"{domain}/{fid}: duplicate paraphrases")
        for i, p in enumerate(paras):
            _check_slots(p, slots, fid, i)
    return data


def load_pools(domain: str) -> dict:
    """Load and validate pools.json for *domain*."""
    path = _domain_dir(domain) / "pools.json"
    data = _read_json(path)
    size_checks = [
        ("person_names", 40), ("org_names", 40),
        ("sources", 8), ("filler_teams", 12),
    ]
    for key, minimum in size_checks:
        n = len(data.get(key, []))
        if n < minimum:
            raise ValueError(
                f"{domain} pools: {key} has {n}, need >= {minimum}"
            )
    abf = data.get("attributes_by_family", {})
    for fam, min_attrs in [("EXACT-01", 4), ("EXACT-02", 2)]:
        attrs = abf.get(fam, {})
        if len(attrs) < min_attrs:
            raise ValueError(
                f"{domain} pools: {fam} needs >= {min_attrs} attr types, "
                f"got {len(attrs)}"
            )
        for aname, adata in attrs.items():
            vals = adata.get("values", [])
            if len(vals) < 12:
                raise ValueError(
                    f"{domain} pools: {fam}/{aname}: {len(vals)} values, "
                    f"need >= 12"
                )
            for v in vals:
                _check_value(v, f"{domain}/{fam}/{aname}")
    rels = data.get("relations", {})
    for rk in ("REL-01", "REL-02"):
        n = len(rels.get(rk, []))
        if n < 6:
            raise ValueError(
                f"{domain} pools: {rk} has {n} relations, need >= 6"
            )
    return data


def load_questions(domain: str) -> dict:
    """Load and validate questions.json for *domain*."""
    path = _domain_dir(domain) / "questions.json"
    data = _read_json(path)
    by_type = data.get("by_type", {})
    for key in QUESTION_KEYS:
        n = len(by_type.get(key, []))
        if n < 2:
            raise ValueError(
                f"{domain} questions: {key} has {n} templates, need >= 2"
            )
    answers = data.get("answers", {})
    for key, expected in QUESTION_ANSWERS.items():
        actual = answers.get(key)
        if actual != expected:
            raise ValueError(
                f"{domain} questions: answers[{key}] = {actual!r}, "
                f"expected {expected!r}"
            )
    return data
