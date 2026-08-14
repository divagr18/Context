"""Streaming window split + scene-cutoff mapping (PLAN.md section 3).

Windows tile the document token stream at ``window_tokens`` with
``window_overlap_frac`` overlap. Each window maps to a scene cutoff: the
number of scenes that have started by the window's end token. Events with
scene_idx below the cutoff are "seen" after reading that window.
"""
from __future__ import annotations

import bisect


def split_windows(doc_len_tokens: int, window_tokens: int,
                  overlap_frac: float) -> list[tuple[int, int]]:
    """Return (start, end) token ranges tiling [0, doc_len_tokens)."""
    if window_tokens <= 0:
        raise ValueError("window_tokens must be positive")
    stride = max(1, int(window_tokens * (1.0 - overlap_frac)))
    windows: list[tuple[int, int]] = []
    start = 0
    while start < doc_len_tokens:
        end = min(start + window_tokens, doc_len_tokens)
        windows.append((start, end))
        if end >= doc_len_tokens:
            break
        start += stride
    return windows


def scene_cutoff(scene_boundaries: tuple[int, ...], window_end: int) -> int:
    """Number of scenes that have started by ``window_end`` tokens."""
    return bisect.bisect_right(scene_boundaries, window_end)
