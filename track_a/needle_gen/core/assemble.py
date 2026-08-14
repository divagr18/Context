"""Token-accurate scene packing (PLAN.md 5.2).

Greedy scene-by-scene packing that hits ±5% of the target token length.
Takes a callable that produces scene text strings — this module only
packs and counts; it has no knowledge of templates, facts, or families.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class AssembledDoc:
    """Packed document: token ids + token index of each scene start."""

    token_ids: tuple[int, ...]
    scene_boundaries: tuple[int, ...]

    @property
    def doc_len_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def n_scenes(self) -> int:
        return len(self.scene_boundaries)


def assemble_doc(
    target_tokens: int,
    scene_text: Callable[[int], str],
    tokenizer,
) -> AssembledDoc:
    """Pack scenes greedily until token count reaches *target_tokens*.

    Args:
        target_tokens: minimum token goal (PLAN.md ``DOC_LEN_TARGETS``).
        scene_text: deterministic callable; ``scene_text(i)`` returns the
            text for scene ``i``. Called in order starting from 0.
        tokenizer: any object with ``.encode(text, add_special_tokens=False)``.

    Returns:
        ``AssembledDoc`` with total tokens >= *target* and within ±5%.
    """
    all_ids: list[int] = []
    boundaries: list[int] = []
    scene_idx = 0

    while len(all_ids) < target_tokens:
        text = scene_text(scene_idx)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            break
        boundaries.append(len(all_ids))
        all_ids.extend(ids)
        scene_idx += 1

    return AssembledDoc(token_ids=tuple(all_ids), scene_boundaries=tuple(boundaries))
