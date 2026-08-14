"""Rate-distortion curve tests for the eval battery (PLAN 9 deliverable)."""
from __future__ import annotations

import pytest

from track_a.plot_curves import plot_rate_distortion


def _report():
    ratios = ["2", "8", "32"]
    return {
        "meta": {"ratios": [2, 8, 32]},
        "primary": {
            "needle_survival_exact": {
                "exact_value": {"all": {"2": 0.9, "8": 0.7, "32": 0.4}},
                "binding": {"near": {"2": 1.0, "8": 0.8, "32": 0.5}},
            },
            "needle_survival_partial": {},
            "hallucination_rate": 0.05,
            "hallucination_rate_by_ratio": {"2": 0.02, "8": 0.05, "32": 0.09},
            "decoy_emission_rate": 0.3,
            "decoy_emission_rate_by_ratio": {"2": 0.6, "8": 0.3, "32": 0.0},
            "salience_inversions": {"2": 0.0, "8": 0.1, "32": 0.4},
            "parse_fail_rate": {"overall": 0.01, "2": 0.0, "8": 0.01,
                                "32": 0.03},
            "info_pressure": {"2": 30.0, "8": 10.0, "32": 3.0},
        },
        "diag": {"qa_probe_acc": {}},
    }


def test_plot_rate_distortion_writes_png(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    out = tmp_path / "curves.png"
    plot_rate_distortion(_report(), str(out))
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "must write a real PNG"
    assert len(data) > 1000, "curve grid must contain drawn content"


def test_plot_rate_distortion_missing_optionals(tmp_path):
    """Report without by_ratio optionals still plots (smoke/minimal runs)."""
    pytest.importorskip("matplotlib")
    rep = _report()
    del rep["primary"]["hallucination_rate_by_ratio"]
    del rep["primary"]["decoy_emission_rate_by_ratio"]
    out = tmp_path / "curves_min.png"
    plot_rate_distortion(rep, str(out))
    assert out.exists() and out.stat().st_size > 1000
