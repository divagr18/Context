"""Structural validation of hand-authored assets for both domains.

Imports the loader (which asserts structural correctness) and adds
checks the loader intentionally omits (pool uniqueness, paraphrase
non-triviality, answer-template slot alignment).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from track_a.needle_gen.assets_loader import (
    ASSETS_ROOT,
    EXPECTED_FAMILIES,
    NUM_FAMILIES,
    PARAPHRASE_COUNT,
    load_pools,
    load_questions,
    load_templates,
)

DOMAINS = ("project_updates", "logistics_ops")

SLOT_RE = re.compile(r"\{(\w+)\}")


# ---------------------------------------------------------------------------
# templates.json
# ---------------------------------------------------------------------------


class TestTemplates:
    @pytest.mark.parametrize("domain", DOMAINS)
    def test_loads_without_error(self, domain: str) -> None:
        load_templates(domain)

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_family_count(self, domain: str) -> None:
        data = load_templates(domain)
        assert len(data["families"]) == NUM_FAMILIES

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_paraphrases_non_empty_strings(self, domain: str) -> None:
        data = load_templates(domain)
        for fam in data["families"]:
            for p in fam["paraphrases"]:
                assert isinstance(p, str) and len(p.strip()) > 20, (
                    f"{fam['family_id']}: paraphrase too short: {p!r}"
                )

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_paraphrases_varied_structure(self, domain: str) -> None:
        """At least 4 distinct leading words across 8 paraphrases."""
        data = load_templates(domain)
        for fam in data["families"]:
            leads = {p.split()[0].lower() for p in fam["paraphrases"]}
            assert len(leads) >= 4, (
                f"{domain}/{fam['family_id']}: only {len(leads)} distinct "
                f"leading words, need >= 4"
            )


# ---------------------------------------------------------------------------
# pools.json
# ---------------------------------------------------------------------------


class TestPools:
    @pytest.mark.parametrize("domain", DOMAINS)
    def test_loads_without_error(self, domain: str) -> None:
        load_pools(domain)

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_person_names_unique_casefold(self, domain: str) -> None:
        data = load_pools(domain)
        names = [n.casefold() for n in data["person_names"]]
        assert len(set(names)) == len(names), "duplicate person_names"

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_org_names_unique_casefold(self, domain: str) -> None:
        data = load_pools(domain)
        names = [n.casefold() for n in data["org_names"]]
        assert len(set(names)) == len(names), "duplicate org_names"

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_attribute_values_no_spaces(self, domain: str) -> None:
        data = load_pools(domain)
        abf = data["attributes_by_family"]
        for fam_id, attrs in abf.items():
            for atr, adata in attrs.items():
                for v in adata["values"]:
                    assert " " not in v, (
                        f"{fam_id}/{atr}: space in value {v!r}"
                    )

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_relations_no_overlap_between_keys(self, domain: str) -> None:
        data = load_pools(domain)
        r1 = set(data["relations"]["REL-01"])
        r2 = set(data["relations"]["REL-02"])
        assert r1.isdisjoint(r2), f"REL-01 and REL-02 overlap: {r1 & r2}"


# ---------------------------------------------------------------------------
# questions.json
# ---------------------------------------------------------------------------


class TestQuestions:
    @pytest.mark.parametrize("domain", DOMAINS)
    def test_loads_without_error(self, domain: str) -> None:
        load_questions(domain)

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_question_templates_are_strings(self, domain: str) -> None:
        data = load_questions(domain)
        for key, templates in data["by_type"].items():
            for t in templates:
                assert isinstance(t, str) and len(t) > 5, (
                    f"{key}: bad template {t!r}"
                )

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_question_templates_have_slots(self, domain: str) -> None:
        """Every question template contains at least one {slot}."""
        data = load_questions(domain)
        for key, templates in data["by_type"].items():
            for t in templates:
                assert SLOT_RE.search(t), (
                    f"{key}: no {{slot}} found in {t!r}"
                )
