"""Track A eval package (T8); shared fact grading lives in grading.py."""

from track_a.evalkit.grading import (
    decoy_triples, eid_name_map, fact_triples_from_records,
    ground_truth_triples, parse_c_text, recall_and_hallucination,
)

__all__ = [
    "decoy_triples",
    "eid_name_map",
    "fact_triples_from_records",
    "ground_truth_triples",
    "parse_c_text",
    "recall_and_hallucination",
]
