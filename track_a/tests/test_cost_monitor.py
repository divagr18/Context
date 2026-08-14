"""W9 cost-monitor tests (PLAN 11: ping at 25/50/75% of budget).

Pure functions only under test -- the CLI merely wires wall-clock + optional
RunPod polling into these. Stdlib-only design (urllib for RunPod queries).
"""
from __future__ import annotations

import pytest

from track_a.runbook.cost_monitor import (
    crossing_events, estimate_spend, format_alert, monitor_step,
    parse_runpod_spend,
)


def test_estimate_spend_wall_clock():
    assert estimate_spend(2.0, 0.79) == pytest.approx(1.58)
    assert estimate_spend(0.0, 0.79) == 0.0


def test_crossing_events_partial_jump():
    got = crossing_events(0.20, 0.55, (0.25, 0.50, 0.75))
    assert got == [0.25, 0.50], "both crossed thresholds, in order"


def test_crossing_events_boundaries_and_no_repeat():
    assert crossing_events(0.0, 0.25, (0.25, 0.5, 0.75)) == [0.25]
    assert crossing_events(0.25, 0.30, (0.25, 0.5, 0.75)) == []
    assert crossing_events(0.75, 1.2, (0.25, 0.5, 0.75)) == []
    assert crossing_events(0.3, 0.29, (0.25, 0.5, 0.75)) == []


def test_crossing_events_full_budget():
    got = crossing_events(0.7, 1.05, (0.25, 0.5, 0.75, 1.0))
    assert got == [0.75, 1.0]


def test_format_alert_contains_facts():
    msg = format_alert(0.5, 37.5, 75.0)
    assert "50%" in msg
    assert "37.5" in msg and "75.0" in msg


def test_monitor_step_accumulates_and_fires_once():
    state = {"last_frac": 0.0, "alerts": []}
    alerts, state = monitor_step(state, 20.0, budget_usd=100.0)
    assert alerts == [] and state["last_frac"] == 0.20
    alerts, state = monitor_step(state, 60.0, budget_usd=100.0)
    assert [a["threshold"] for a in alerts] == [0.25, 0.5]
    assert len(state["alerts"]) == 2
    alerts, state = monitor_step(state, 61.0, budget_usd=100.0)
    assert alerts == [], "no re-fire inside the same band"


def test_parse_runpod_spend_graphql_shape():
    payload = {"data": {"userSelfServe": {"userSpent": 1234.5,
                                           "userPromoter": 0.0}}}
    assert parse_runpod_spend(payload) == pytest.approx(12.345)
    alt = {"data": {"userSelfServe": None}}
    assert parse_runpod_spend(alt) is None
    assert parse_runpod_spend({"errors": [{"message": "denied"}]}) is None
