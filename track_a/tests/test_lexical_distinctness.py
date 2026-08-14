"""Lexical-disjointness gate between Domain A and Domain B.

Computes character-3-gram Jaccard over template + pool content vocabularies
(unique words, function words and scaffolding verbs stripped).  Failing this
test is the objective trigger for the "revisit with LLM pass" gate
(PLAN.md §5.1, [A7]).
"""
from __future__ import annotations

import json
import re

from track_a.needle_gen.assets_loader import ASSETS_ROOT

LEXICAL_JACCARD_MAX = 0.15
DOMAINS = ("project_updates", "logistics_ops")
SLOT_RE = re.compile(r"\{[^}]+\}")
WORD_RE = re.compile(r"[a-zA-Z0-9_./:%$-]+")

# Excluded from comparison: function words and domain-neutral scaffolding
# verbs/adjectives that any English factual prose shares.  Only domain-specific
# content words should drive the Jaccard score.
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "for", "yet", "so",
    "at", "by", "in", "of", "on", "to", "up", "as", "is", "it",
    "am", "be", "do", "go", "he", "if", "me", "my", "no", "we",
    "was", "are", "had", "has", "her", "his", "its", "our", "out",
    "own", "say", "she", "too", "who", "all", "any", "can", "did",
    "get", "got", "him", "how", "let", "may", "new", "not", "now",
    "old", "see", "way", "you", "also", "been", "both", "each",
    "from", "have", "into", "just", "like", "more", "most", "only",
    "over", "such", "than", "that", "them", "then", "they", "this",
    "very", "were", "what", "when", "with", "will", "your", "about",
    "after", "being", "could", "every", "other", "their", "there",
    "these", "those", "which", "while", "would", "per", "via",
    "shall", "should", "through", "upon", "under", "during",
    "according", "ahead", "around", "assigned", "between",
    "carries", "carrying", "change", "changed", "check", "claim",
    "confirmed", "completed", "coordinated", "current", "currently",
    "entered", "equals", "field", "gathered", "gives", "holds",
    "initial", "last", "latest", "listed", "lists", "logged",
    "members", "might", "moved", "moving", "proceeded", "reads",
    "recorded", "records", "rejected", "replaced", "review",
    "reviewed", "revised", "set", "shifted", "show", "shows",
    "stands", "states", "suggests", "time", "today", "took",
    "updated", "valid", "verified", "week", "weekly", "without",
    "detail", "direction", "schedule", "noted", "final", "open",
    "brief", "recent", "routine", "usual", "steady", "open",
    "afternoon", "morning", "night", "main", "several",
    "board", "crew", "delay", "finalized", "floor", "hedged",
    "inspection", "logs", "notes", "opened", "record",
    "registration", "squad", "system", "target", "team", "update",
    "gate", "log", "manifest", "routes", "stand",
})


def _read_json(rel: str) -> dict:
    path = ASSETS_ROOT / rel
    return json.loads(path.read_text(encoding="utf-8"))


def _strip_trailing_punct(token: str) -> str:
    return token.rstrip(".,;:!?")


def _content_words(domain: str) -> set[str]:
    parts: list[str] = []
    tpl = _read_json(f"{domain}/templates.json")
    for fam in tpl["families"]:
        for p in fam["paraphrases"]:
            parts.append(SLOT_RE.sub("", p))
    pools = _read_json(f"{domain}/pools.json")
    for key in ("person_names", "org_names", "sources", "filler_teams"):
        parts.extend(pools.get(key, []))
    for fam_attrs in pools.get("attributes_by_family", {}).values():
        for adata in fam_attrs.values():
            parts.extend(adata.get("values", []))
    for rel_list in pools.get("relations", {}).values():
        parts.extend(rel_list)
    raw = " ".join(parts).lower()
    tokens = WORD_RE.findall(raw)
    cleaned = {_strip_trailing_punct(t) for t in tokens}
    return {w for w in cleaned if w and w not in _STOP_WORDS and len(w) > 2}


def _word_char_ngrams(words: set[str], n: int = 4) -> set[str]:
    result: set[str] = set()
    for w in words:
        for i in range(len(w) - n + 1):
            result.add(w[i : i + n])
    return result


def test_lexical_distinctness() -> None:
    words_a = _content_words(DOMAINS[0])
    words_b = _content_words(DOMAINS[1])
    grams_a = _word_char_ngrams(words_a)
    grams_b = _word_char_ngrams(words_b)
    union = grams_a | grams_b
    jaccard = len(grams_a & grams_b) / len(union) if union else 0.0
    assert jaccard < LEXICAL_JACCARD_MAX, (
        f"n-gram Jaccard={jaccard:.4f} >= {LEXICAL_JACCARD_MAX}; "
        f"domain content vocabularies too similar"
    )
