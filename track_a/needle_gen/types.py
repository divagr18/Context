"""Frozen contract types for Track A (see PLAN.md sections 2-4).

Keystone module: needle_gen core, parser, dataset layer, model code, and eval
all import from here. Changing an existing field is a contract change and must
be recorded in PLAN.md before touching this file.

Pure standard library only -- no torch/numpy imports allowed here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FactType(str, Enum):
    EXACT_VALUE = "exact_value"
    RELATIONAL = "relational"
    STATE_TRANSITION = "state_transition"
    NEGATIVE = "negative"
    UNCERTAINTY = "uncertainty"
    BINDING = "binding"


class UncertaintyKind(str, Enum):
    HEDGE = "hedge"        # single-source hedged claim -> FACT ... [hedged]
    CONFLICT = "conflict"  # two unresolved claims     -> UNRESOLVED record


class DistanceBucket(str, Enum):
    NEAR = "near"          # < 500 tokens between bind endpoints
    MID = "mid"            # 500 - 4000
    FAR = "far"            # 4000 - 20000
    EXTREME = "extreme"    # > 20000 (long/xlong docs only)


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST_ID = "test-id"
    TEST_OOD = "test-ood"


class Domain(str, Enum):
    PROJECT_UPDATES = "project_updates"  # training / in-distribution domain
    LOGISTICS_OPS = "logistics_ops"      # lexically disjoint OOD domain


class FactClass(int, Enum):
    QUERIED = 0
    DECOY = 1


class CueKind(str, Enum):
    NEGATION = "negation"
    HEDGE = "hedge"


# ---------------------------------------------------------------------------
# Canonical-order constants (PLAN.md section 2). Frozen across ALL variants,
# splits, scale points, and budget buckets. Do not parameterize per-run.
# ---------------------------------------------------------------------------

TYPE_RANK: dict[FactType, int] = {
    FactType.STATE_TRANSITION: 0,
    FactType.BINDING: 1,
    FactType.EXACT_VALUE: 2,
    FactType.RELATIONAL: 3,
    FactType.NEGATIVE: 4,
    FactType.UNCERTAINTY: 5,
}

# Distance-bucket boundaries (tokens between bind endpoints).
DISTANCE_BUCKET_MAX = {
    DistanceBucket.NEAR: 500,
    DistanceBucket.MID: 4000,
    DistanceBucket.FAR: 20000,
}


def distance_bucket_for(n_tokens: int) -> DistanceBucket:
    """Bucket for a measured binding distance (PLAN.md 5.2.6)."""
    if n_tokens < DISTANCE_BUCKET_MAX[DistanceBucket.NEAR]:
        return DistanceBucket.NEAR
    if n_tokens < DISTANCE_BUCKET_MAX[DistanceBucket.MID]:
        return DistanceBucket.MID
    if n_tokens < DISTANCE_BUCKET_MAX[DistanceBucket.FAR]:
        return DistanceBucket.FAR
    return DistanceBucket.EXTREME


# ---------------------------------------------------------------------------
# Grammar / tokenizer contract (PLAN.md sections 1 & 4)
# ---------------------------------------------------------------------------

DOC_LEN_TARGETS: dict[str, int] = {
    "short": 2048,
    "medium": 8192,
    "long": 32768,
    "xlong": 131072,
}

BUDGET_VALUES: tuple[int, ...] = (
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
)

STRUCT_TOKENS: tuple[str, ...] = (
    "<doc>", "</doc>", "</C>",
    "<state>", "</state>", "<window>", "</window>", "</OPS>",
    "<Q>", "</Q>", "<A>", "</A>",
)

BUDGET_TOKENS: tuple[str, ...] = tuple(f"<budget={v}>" for v in BUDGET_VALUES)

SPECIAL_TOKENS: tuple[str, ...] = STRUCT_TOKENS + BUDGET_TOKENS
NUM_ADDED_TOKENS: int = len(SPECIAL_TOKENS)  # 21

GPT2_BASE_VOCAB: int = 50257
TOTAL_VOCAB: int = GPT2_BASE_VOCAB + NUM_ADDED_TOKENS  # 50278

# Fact values come from pools and must match this (no whitespace), so that
# C records are trivially parseable. See PLAN.md section 4 canonicalization.
VALUE_PATTERN = re.compile(r"^[A-Za-z0-9_$./:%-]+$")

# Input formats (single-shot / streaming / QA-probe), PLAN.md section 4:
#   single-shot: <doc> {doc} </doc> <budget=N> {C* lines} </C>
#   streaming:   <state> {state render} </state> <window> {win} </window>
#                <budget=N> {edit-ops} </OPS>
#   qa-probe:    <doc> {C} </doc> <Q> {question} </Q> <A> {answer} </A>
# Loss is computed only over the C*/ops/answer span.


def budget_token_for(target_tokens: int) -> str:
    """Nearest budget token for a target budget, nearest in log space."""
    target = max(1, target_tokens)
    best = min(
        BUDGET_VALUES,
        key=lambda v: abs(math.log(v) - math.log(target)),
    )
    return f"<budget={best}>"


def canonicalize_value(raw: str) -> str:
    """Canonical form used for all value matching (PLAN.md section 4).

    whitespace runs -> '_', digit-grouping commas stripped, case preserved.
    """
    out = "_".join(raw.split())
    out = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", out)
    return out


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    id: str  # generator-assigned stable id, e.g. "E0007"
    type: str  # "person" | "project" | "system" | "depot" | "route" | ...
    name: str  # canonical surface name
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Fact:
    """One grounded fact. A state-transition chain (>=2 values) is ONE unit.

    chains: `values` in chronological order (current = values[-1]),
    `scene_positions` aligned element-wise. Atomic under canonical order.
    """

    id: str
    type: FactType
    entity_ids: tuple[str, ...]  # 1 for unary facts, 2 for relational
    attribute: Optional[str] = None  # None for pure relational facts
    values: tuple[str, ...] = ()  # chronological; () for NEGATIVE facts
    scene_positions: tuple[int, ...] = ()
    relation: Optional[str] = None  # relational facts only
    uncertainty_kind: Optional[UncertaintyKind] = None
    is_queried: bool = False
    distance_bucket: Optional[DistanceBucket] = None  # binding needles only

    @property
    def fact_class(self) -> FactClass:
        return FactClass.QUERIED if self.is_queried else FactClass.DECOY

    @property
    def is_chain(self) -> bool:
        return self.type is FactType.STATE_TRANSITION and len(self.values) >= 2

    @property
    def chain_start_position(self) -> int:
        """Tie-break position for canonical ordering (PLAN.md 2.2)."""
        return self.scene_positions[0] if self.scene_positions else 0


@dataclass(frozen=True)
class Question:
    id: str
    fact_ids: tuple[str, ...]
    text: str
    answer: str  # canonical answer string
    is_multihop: bool = False
    probes_prior_value: bool = False  # chain "what was it before" probe
    phrasing_idx: int = 0


@dataclass(frozen=True)
class MentionSpan:
    entity_id: str
    start_tok: int
    end_tok: int  # exclusive
    is_alias: bool = False


@dataclass(frozen=True)
class ValueSpan:
    fact_id: str
    value_idx: int  # index into Fact.values
    start_tok: int
    end_tok: int  # exclusive


@dataclass(frozen=True)
class CueSpan:
    kind: CueKind
    start_tok: int
    end_tok: int  # exclusive


@dataclass(frozen=True)
class FactDB:
    """Ground truth for one generated document."""

    doc_id: str
    entities: tuple[Entity, ...]
    facts: tuple[Fact, ...]
    questions: tuple[Question, ...]
    mentions: tuple[MentionSpan, ...] = ()
    value_spans: tuple[ValueSpan, ...] = ()
    cue_spans: tuple[CueSpan, ...] = ()
    scene_boundaries: tuple[int, ...] = ()  # token idx of each scene start
    doc_len_tokens: int = 0

    def queried_facts(self) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if f.is_queried)

    def decoy_facts(self) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if not f.is_queried)

    def entity_by_id(self, eid: str) -> Optional[Entity]:
        for e in self.entities:
            if e.id == eid:
                return e
        return None


@dataclass(frozen=True)
class GenConfig:
    """Fully determines a generated corpus (committed under splits/, PLAN 5.2)."""

    seed: int
    split: Split
    domain: Domain
    doc_len_name: str  # key into DOC_LEN_TARGETS
    n_docs: int
    family_ids: tuple[str, ...]
    paraphrase_idx_range: tuple[int, int]  # inclusive
    decoys_per_queried: tuple[int, int] = (1, 3)
    chain_fraction: float = 0.3  # fraction of state facts that are chains
    chain_depth_max: int = 3
    multihop_doc_fraction: float = 0.15
    queried_per_100_tokens: float = 1.0
    # Streaming (PLAN.md section 3): window split params.
    window_tokens: int = 2048
    window_overlap_frac: float = 0.10
