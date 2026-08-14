"""Character-span -> token-span mapping for annotations (T3B).

After the full document text is tokenized with offset mapping, every recorded
character span (entity mention, value, cue) is converted to a token range so
aux losses get grounded spans.
"""
from __future__ import annotations


def char_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    """Map a document-level char span [char_start, char_end) to token [s, e).

    ``offsets`` is the tokenizer's per-token (start, end) char mapping.
    Returns (tok_start, tok_end) with tok_end exclusive. Returns (0, 0) if the
    span maps to no token (degenerate).
    """
    tok_start = -1
    tok_end = -1
    for i, (s, e) in enumerate(offsets):
        if e <= char_start or s >= char_end:
            continue  # token entirely outside the char span
        if tok_start == -1:
            tok_start = i
        tok_end = i + 1
    if tok_start == -1:
        return (0, 0)
    return (tok_start, tok_end)


def map_char_spans(
    offsets: list[tuple[int, int]], spans: dict[str, tuple[int, int]]
) -> dict[str, tuple[int, int]]:
    """Map a dict of name -> char span to name -> token span."""
    return {
        name: char_span_to_token_span(offsets, cs, ce)
        for name, (cs, ce) in spans.items()
    }
