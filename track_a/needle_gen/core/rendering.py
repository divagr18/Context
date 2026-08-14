"""Scene rendering: fill authored template paraphrases for fact/filler scenes."""
from __future__ import annotations

import random

from track_a.needle_gen.core.factspec import SceneSpec
from track_a.needle_gen.core.render_scene import fill_template


def paraphrase_index(rng: random.Random, idx_range: tuple[int, int]) -> int:
    """Pick a paraphrase index within the split's inclusive range."""
    lo, hi = idx_range
    return rng.randint(lo, hi)


def render_fact_scene(
    spec: SceneSpec,
    family: dict,
    rng: random.Random,
    idx_range: tuple[int, int],
) -> tuple[str, dict[str, tuple[int, int]]]:
    """Render one fact scene; returns (text, char_spans within scene text)."""
    para = family["paraphrases"][paraphrase_index(rng, idx_range)]
    return fill_template(para, spec.slot_values)


def filler_slot_values(rng: random.Random, pools: dict, slots: list[str]) -> dict[str, str]:
    """Fill a filler scene's slots with pool content (team/project/person)."""
    out: dict[str, str] = {}
    for slot in slots:
        if slot == "team":
            out[slot] = rng.choice(pools["filler_teams"])
        elif slot == "project":
            out[slot] = rng.choice(pools["org_names"])
        elif slot == "person":
            out[slot] = rng.choice(pools["person_names"])
        else:
            out[slot] = rng.choice(pools["org_names"])
    return out
