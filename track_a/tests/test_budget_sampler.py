"""T5: budget/ratio sampler invariants."""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

from track_a.data.budget_sampler import available_ratios, sample_ratio
from track_a.shard_schema import CRender
from track_a.tests._shard_fixtures import make_single_shot_record


def test_available_ratios_all_present():
    rec = make_single_shot_record(ratios=("2", "4", "8", "16", "32"))
    assert available_ratios(rec.c_renders) == (2, 4, 8, 16, 32)


def test_available_ratios_subset():
    rec = make_single_shot_record(ratios=("4", "16"))
    assert available_ratios(rec.c_renders) == (4, 16)


def test_available_ratios_excludes_empty_renders():
    rec = make_single_shot_record(ratios=("2", "4"))
    renders = dict(rec.c_renders)
    renders["4"] = CRender(target_ids=(), info_pressure_fact_count=0)
    rec2 = replace(rec, c_renders=renders)
    assert available_ratios(rec2.c_renders) == (2,)


def test_available_ratios_canonical_order():
    rec = make_single_shot_record(ratios=("32", "2", "8"))
    assert available_ratios(rec.c_renders) == (2, 8, 32)


def test_sample_ratio_within_available():
    rec = make_single_shot_record(ratios=("2", "8", "32"))
    rng = random.Random(0)
    for _ in range(50):
        assert sample_ratio(rng, rec.c_renders) in (2, 8, 32)


def test_sample_ratio_covers_options_eventually():
    rec = make_single_shot_record(ratios=("2", "4"))
    rng = random.Random(1)
    seen = {sample_ratio(rng, rec.c_renders) for _ in range(100)}
    assert seen == {2, 4}


def test_sample_ratio_deterministic_same_seed():
    rec = make_single_shot_record()
    a = [sample_ratio(random.Random(7), rec.c_renders) for _ in range(5)]
    b = [sample_ratio(random.Random(7), rec.c_renders) for _ in range(5)]
    assert a == b


def test_sample_ratio_raises_when_none_available():
    rec = make_single_shot_record(ratios=("2",))
    renders = {"2": CRender(target_ids=(), info_pressure_fact_count=0)}
    rec2 = replace(rec, c_renders=renders)
    with pytest.raises(ValueError):
        sample_ratio(random.Random(0), rec2.c_renders)
