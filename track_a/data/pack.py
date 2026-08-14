"""Sample packing with loss masking (PLAN.md section 4 input formats).

Builds training samples for the three framings. Loss is computed ONLY over
the C* / edit-ops / answer span (PLAN.md: "Loss is computed only over the
C*/ops/answer span"); every source-side token is masked with IGNORE_INDEX.

Formats:
    single-shot: <doc> {doc} </doc> <budget=N> {C*} </C>
    streaming:   <state> {state} </state> <window> {win} </window>
                 <budget=N> {edit-ops} </OPS>
    qa-probe:    <doc> {C} </doc> <Q> {q} </Q> <A> {answer} </A>

C*/ops/answer ids are read verbatim from the shard (c_renders / windows /
questions); this module never re-renders or reorders them.
"""
from __future__ import annotations

from dataclasses import dataclass

IGNORE_INDEX = -100


@dataclass(frozen=True)
class PackSample:
    """One packed training sample.

    ``input_ids`` is the full sequence (source prefix + target + closing
    token). ``labels`` is aligned to ``input_ids``: labels[i] = input_ids[i+1]
    for positions that predict a target token, else IGNORE_INDEX. The training
    loop feeds input_ids and scores against labels directly (no extra shift).
    """

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.labels):
            raise ValueError("input_ids and labels length mismatch")


def _mask_labels(input_ids: tuple[int, ...], target_start: int) -> tuple[int, ...]:
    """Mask all source positions; predict input_ids[target_start:] only."""
    labels = [IGNORE_INDEX] * len(input_ids)
    for i in range(target_start - 1, len(input_ids) - 1):
        labels[i] = input_ids[i + 1]
    return tuple(labels)


def special_id(tok, token: str) -> int:
    """Single-token id for an added special token."""
    ids = tok.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"special token {token!r} encoded to {len(ids)} ids")
    return ids[0]


def budget_token_id(tok, budget: int) -> int:
    from track_a.needle_gen.types import budget_token_for

    return special_id(tok, budget_token_for(budget))


def pack_single_shot(doc_ids, c_render_ids, doc_len_tokens: int, ratio: int,
                     tok) -> PackSample:
    """Pack a single-shot sample: predict C* (+ </C>) from doc + budget."""
    budget = max(1, doc_len_tokens // ratio)
    source = [special_id(tok, "<doc>")] + list(doc_ids) + [
        special_id(tok, "</doc>"), budget_token_id(tok, budget)]
    target = list(c_render_ids) + [special_id(tok, "</C>")]
    input_ids = tuple(source + target)
    return PackSample(input_ids, _mask_labels(input_ids, len(source)))


def pack_streaming_window(window, tok) -> PackSample:
    """Pack a streaming sample: predict edit-ops (+ </OPS>) from state+window."""
    budget_tok = budget_token_id(tok, window.budget)
    ops_ids = (list(tok.encode(window.ops_text, add_special_tokens=False))
               if window.ops_text else [])
    source = ([special_id(tok, "<state>")] + list(window.state_ids)
              + [special_id(tok, "</state>")]
              + [special_id(tok, "<window>")] + list(window.window_ids)
              + [special_id(tok, "</window>"), budget_tok])
    target = ops_ids + [special_id(tok, "</OPS>")]
    input_ids = tuple(source + target)
    return PackSample(input_ids, _mask_labels(input_ids, len(source)))


def pack_qa(c_ids, question_text: str, answer_text: str, tok) -> PackSample:
    """Pack a QA-probe sample: predict answer (+ </A>) from C + question."""
    q_ids = tok.encode(question_text, add_special_tokens=False)
    a_ids = tok.encode(answer_text, add_special_tokens=False)
    source = ([special_id(tok, "<doc>")] + list(c_ids)
              + [special_id(tok, "</doc>")]
              + [special_id(tok, "<Q>")] + list(q_ids)
              + [special_id(tok, "</Q>"), special_id(tok, "<A>")])
    target = list(a_ids) + [special_id(tok, "</A>")]
    input_ids = tuple(source + target)
    return PackSample(input_ids, _mask_labels(input_ids, len(source)))
