"""Streaming ground-truth construction (PLAN.md section 3).

``windows`` tiles the document; ``events`` folds scene-level mutations into a
belief state; ``state_diff`` renders/truncates/diffs that state into the
edit-op grammar (consistent with the T4 parser / OpLog fold).
"""

from .events import Event, StreamState, apply_event, events_for_fact, extract_events
from .state_diff import (
    build_streaming_windows,
    diff_states,
    render_state_lines,
    truncate_state,
)
from .windows import scene_cutoff, split_windows

__all__ = [
    "Event",
    "StreamState",
    "apply_event",
    "build_streaming_windows",
    "diff_states",
    "events_for_fact",
    "extract_events",
    "render_state_lines",
    "scene_cutoff",
    "split_windows",
    "truncate_state",
]
