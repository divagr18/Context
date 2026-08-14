"""Template slot filling with character-span tracking (T3B rendering base).

Fills ``{slot}`` placeholders and records each filled value's character
span so annotations can later be mapped to token spans.
"""
from __future__ import annotations

from track_a.needle_gen.assets_loader import SLOT_RE


def fill_template(template: str, slot_values: dict[str, str]) -> tuple[str, dict[str, tuple[int, int]]]:
    """Return (text, char_spans) with each slot value's [start, end) range."""
    spans: dict[str, tuple[int, int]] = {}
    parts: list[str] = []
    cursor = 0
    for m in SLOT_RE.finditer(template):
        slot = m.group(1)
        if slot not in slot_values:
            raise KeyError(f"template slot {slot!r} has no value")
        parts.append(template[cursor:m.start()])
        value = slot_values[slot]
        start = sum(len(p) for p in parts)
        parts.append(value)
        spans[slot] = (start, start + len(value))
        cursor = m.end()
    parts.append(template[cursor:])
    return "".join(parts), spans
