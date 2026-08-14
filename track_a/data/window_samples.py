"""Streaming window sample pass-through (scene-aligned upstream).

Window splitting + scene-alignment already happened inside the generator
(streaming/windows), so this module only maps each ``StreamingWindow`` of a
streaming shard record to a packed sample, skipping degenerate windows whose
edit-op text is empty (nothing to learn from a no-op update).
"""
from __future__ import annotations

from collections.abc import Iterator

from track_a.data.pack import PackSample, pack_streaming_window


def window_samples(record, tok) -> Iterator[PackSample]:
    """Yield one packed sample per non-empty-ops window of a record."""
    if not record.windows:
        return
    for window in record.windows:
        if not window.ops_text:
            continue
        yield pack_streaming_window(window, tok)
