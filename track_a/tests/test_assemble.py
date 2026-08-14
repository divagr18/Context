"""Token-accurate scene packing (PLAN.md 5.2): +/-5% of target, exact boundaries."""

from __future__ import annotations

import pytest

from track_a.needle_gen import fixtures
from track_a.needle_gen.core.assemble import assemble_doc
from track_a.needle_gen.types import DOC_LEN_TARGETS
from track_a.tokenize import get_tokenizer


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


def test_short_doc_within_five_percent(tok) -> None:
    target = DOC_LEN_TARGETS["short"]
    doc = assemble_doc(target, fixtures.make_scene_text_maker(seed=11), tok)
    assert doc.doc_len_tokens >= target
    assert doc.doc_len_tokens - target <= 0.05 * target
    assert doc.n_scenes == len(doc.scene_boundaries) > 1


def test_boundaries_consistent_with_token_ids(tok) -> None:
    seen: list[str] = []
    maker = fixtures.make_scene_text_maker(seed=22)

    def recorder(scene_idx: int) -> str:
        text = maker(scene_idx)
        seen.append(text)
        return text

    doc = assemble_doc(DOC_LEN_TARGETS["short"], recorder, tok)
    boundaries = doc.scene_boundaries
    assert boundaries[0] == 0
    assert all(lo < hi for lo, hi in zip(boundaries, boundaries[1:]))
    assert boundaries[-1] < len(doc.token_ids)
    assert len(seen) == doc.n_scenes
    ids = doc.token_ids
    for i, scene in enumerate(seen):
        lo = boundaries[i]
        hi = boundaries[i + 1] if i + 1 < len(boundaries) else len(ids)
        assert tok.decode(list(ids[lo:hi])).strip() == scene.strip()


def test_deterministic_from_seed(tok) -> None:
    target = DOC_LEN_TARGETS["short"]
    doc_a = assemble_doc(target, fixtures.make_scene_text_maker(seed=33), tok)
    doc_b = assemble_doc(target, fixtures.make_scene_text_maker(seed=33), tok)
    assert doc_a.token_ids == doc_b.token_ids
    assert doc_a.scene_boundaries == doc_b.scene_boundaries
    doc_c = assemble_doc(target, fixtures.make_scene_text_maker(seed=34), tok)
    assert doc_c.token_ids != doc_a.token_ids
