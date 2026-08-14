"""JSONL shard line schema (contract C6).

Dataclasses + ``to_json`` / ``from_json`` for one JSONL line. Handles the
two record kinds (single_shot, streaming) with kind-specific fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, asdict
from enum import Enum
from typing import Optional

from track_a.needle_gen.types import (
    CueKind, CueSpan, DistanceBucket, Domain, Entity, Fact, FactType,
    MentionSpan, Question, Split, UncertaintyKind, ValueSpan,
)

SCHEMA_VERSION = 1
KIND_SINGLE_SHOT = "single_shot"
KIND_STREAMING = "streaming"
_VALID_KINDS = frozenset({KIND_SINGLE_SHOT, KIND_STREAMING})

# Lookup tables for enum reconstruction.
_ENUM_MAP: dict[str, type[Enum]] = {
    "type": FactType, "uncertainty_kind": UncertaintyKind,
    "distance_bucket": DistanceBucket, "fact_class": type(None),
    "split": Split, "domain": Domain, "kind": CueKind,
}


@dataclass(frozen=True)
class CRender:
    """Single-shot C* render at one budget bucket."""
    target_ids: tuple[int, ...]
    info_pressure_fact_count: int


@dataclass(frozen=True)
class StreamingWindow:
    """One streaming sample: current state + window + next-state edit ops."""
    budget: int
    state_ids: tuple[int, ...]
    window_ids: tuple[int, ...]
    ops_text: str


@dataclass(frozen=True)
class ShardRecord:
    """One JSONL line: doc ids + annotations + per-kind renders."""
    kind: str
    doc_id: str
    split: Split
    domain: object
    doc_len_name: str
    doc_ids: tuple[int, ...]
    entities: tuple[Entity, ...]
    facts: tuple[Fact, ...]
    questions: tuple[Question, ...]
    mention_spans: tuple[MentionSpan, ...]
    value_spans: tuple[ValueSpan, ...]
    cue_spans: tuple[CueSpan, ...]
    scene_boundaries: tuple[int, ...]
    doc_len_tokens: int
    c_renders: dict[str, CRender] | None = None
    windows: tuple[StreamingWindow, ...] | None = None
    schema_version: int = SCHEMA_VERSION


# -- Serialisation helpers ---------------------------------------------------

def _to_json_obj(obj: object) -> object:
    """Recursively convert to JSON-serialisable plain objects."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, tuple):
        return [_to_json_obj(x) for x in obj]
    if isinstance(obj, list):
        return [_to_json_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json_obj(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_json_obj(getattr(obj, k)) for k in obj.__dataclass_fields__}
    raise TypeError(f"Cannot serialize {type(obj)}")


def _restore_enum(name: str, val: object, dc_cls: type | None = None) -> object:
    if val is None:
        return val
    _DC_ENUM: dict[tuple[type, str], type[Enum]] = {
        (Fact, "type"): FactType,
        (Fact, "uncertainty_kind"): UncertaintyKind,
        (Fact, "distance_bucket"): DistanceBucket,
        (CueSpan, "kind"): CueKind,
    }
    if dc_cls is not None:
        cls = _DC_ENUM.get((dc_cls, name))
        if cls is None:
            return val
        return cls(val)
    cls = _ENUM_MAP.get(name)
    if cls is type(None):
        return val
    if cls is not None:
        return cls(val)
    return val


_DC_BY_TYPE: dict[str, type] = {
    "Entity": Entity, "Fact": Fact, "Question": Question,
    "MentionSpan": MentionSpan, "ValueSpan": ValueSpan,
    "CueSpan": CueSpan, "CRender": CRender, "StreamingWindow": StreamingWindow,
}


def _restore_dc(type_name: str, data: dict) -> object:
    cls = _DC_BY_TYPE[type_name]
    out: dict[str, object] = {}
    for f in fields(cls):
        raw = data.get(f.name)
        if raw is None:
            out[f.name] = None
            continue
        out[f.name] = _restore_field(f.name, raw, cls)
    return cls(**out)


def _restore_field(name: str, raw: object, dc_cls: type) -> object:
    if isinstance(raw, dict) and "id" in raw and "type" in raw:
        return _restore_dc("Entity" if "name" in raw else "Fact", raw)
    if isinstance(raw, dict) and "text" in raw and "answer" in raw:
        return _restore_dc("Question", raw)
    if isinstance(raw, list):
        return _restore_tuple(name, raw)
    return _restore_enum(name, raw, dc_cls)


def _restore_tuple(name: str, raw: list) -> tuple:
    if not raw:
        return ()
    first = raw[0]
    if isinstance(first, int):
        return tuple(raw)
    if isinstance(first, str):
        return tuple(raw)
    if isinstance(first, dict):
        tname = _guess_dc_type(name, first)
        return tuple(_restore_dc(tname, d) for d in raw)
    return tuple(raw)


def _guess_dc_type(field_name: str, _d: dict) -> str:
    return {
        "entities": "Entity", "facts": "Fact", "questions": "Question",
        "mention_spans": "MentionSpan", "value_spans": "ValueSpan",
        "cue_spans": "CueSpan", "windows": "StreamingWindow",
    }[field_name]


# -- Public API --------------------------------------------------------------

def _validate(record: ShardRecord) -> None:
    if record.kind not in _VALID_KINDS:
        raise ValueError(f"unknown kind: {record.kind}")
    if record.kind == KIND_SINGLE_SHOT:
        if record.c_renders is None:
            raise ValueError("single_shot requires c_renders")
        if record.windows is not None:
            raise ValueError("single_shot must not have windows")
    else:
        if record.windows is None:
            raise ValueError("streaming requires windows")
        if record.c_renders is not None:
            raise ValueError("streaming must not have c_renders")


def to_json(record: ShardRecord) -> str:
    """Serialise one shard record to a single-line JSON string."""
    _validate(record)
    return json.dumps(_to_json_obj(record), separators=(",", ":"))


def from_json(line: str) -> ShardRecord:
    """Deserialise one JSONL line into a ``ShardRecord``."""
    raw = json.loads(line)
    ver = raw.get("schema_version", 0)
    if ver != SCHEMA_VERSION:
        raise ValueError(f"unknown schema_version: {ver} (expected {SCHEMA_VERSION})")
    kind = raw.get("kind", "")

    def _entities(data: list) -> tuple[Entity, ...]:
        return tuple(_restore_dc("Entity", d) for d in data)

    def _facts(data: list) -> tuple[Fact, ...]:
        return tuple(_restore_dc("Fact", d) for d in data)

    def _questions(data: list) -> tuple[Question, ...]:
        return tuple(_restore_dc("Question", d) for d in data)

    def _spans(name: str, data: list) -> tuple:
        return tuple(_restore_dc(_guess_dc_type(name, {}), d) for d in data) if data else ()

    def _c_renders(data: dict | None) -> dict[str, CRender] | None:
        if data is None:
            return None
        return {k: CRender(tuple(v["target_ids"]), v["info_pressure_fact_count"])
                for k, v in data.items()}

    def _windows(data: list | None) -> tuple[StreamingWindow, ...] | None:
        if data is None:
            return None
        return tuple(
            StreamingWindow(d["budget"], tuple(d["state_ids"]),
                            tuple(d["window_ids"]), d["ops_text"])
            for d in data
        )

    domain_val = raw.get("domain")
    try:
        domain_val = Domain(domain_val)
    except (ValueError, KeyError):
        pass

    return ShardRecord(
        kind=kind,
        doc_id=raw["doc_id"],
        split=Split(raw["split"]),
        domain=domain_val,
        doc_len_name=raw["doc_len_name"],
        doc_ids=tuple(raw["doc_ids"]),
        entities=_entities(raw.get("entities", [])),
        facts=_facts(raw.get("facts", [])),
        questions=_questions(raw.get("questions", [])),
        mention_spans=_spans("mention_spans", raw.get("mention_spans", [])),
        value_spans=_spans("value_spans", raw.get("value_spans", [])),
        cue_spans=_spans("cue_spans", raw.get("cue_spans", [])),
        scene_boundaries=tuple(raw.get("scene_boundaries", [])),
        doc_len_tokens=raw.get("doc_len_tokens", 0),
        c_renders=_c_renders(raw.get("c_renders")),
        windows=_windows(raw.get("windows")),
        schema_version=ver,
    )
