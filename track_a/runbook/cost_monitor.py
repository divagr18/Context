"""RunPod cost monitor (W9, PLAN 11): ping at 25/50/75% of credit budget.

Spend source: wall-clock estimate (elapsed hours x $/hr); when a RunPod API
key is supplied, live spend from the GraphQL userSelfServe endpoint is
blended in (max of both). Stdlib only -- no third-party HTTP client.

CLI:
    python -m track_a.runbook.cost_monitor --budget-usd 100 --rate 0.79 \
        [--interval 300] [--once] [--runpod-api-key KEY]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

DEFAULT_THRESHOLDS = (0.25, 0.50, 0.75, 1.00)
_RUNPOD_GQL = "https://api.runpod.io/graphql"


def estimate_spend(hours_elapsed: float, rate_per_hour: float) -> float:
    """Wall-clock spend estimate; never negative."""
    return max(0.0, hours_elapsed) * rate_per_hour


def crossing_events(prev_frac: float, cur_frac: float,
                    thresholds=DEFAULT_THRESHOLDS) -> list[float]:
    """Thresholds t with prev < t <= cur, ascending (fires once per band)."""
    return [t for t in thresholds if prev_frac < t <= cur_frac]


def format_alert(threshold: float, spend: float, budget: float) -> str:
    pct = int(round(threshold * 100))
    return (f"[cost] ALERT >= {pct}% of budget: spent ${spend:.2f} "
            f"of ${budget:.2f}")


def monitor_step(state: dict, spend_usd: float, budget_usd: float):
    """Advance monitor state with current spend.

    Returns (new_alerts, new_state); each threshold band fires at most once.
    """
    frac = spend_usd / max(1e-9, budget_usd)
    alerts = [{"threshold": t, "spend_usd": spend_usd,
               "message": format_alert(t, spend_usd, budget_usd)}
              for t in crossing_events(state["last_frac"], frac)]
    return alerts, {"last_frac": frac, "alerts": state["alerts"] + alerts}


def parse_runpod_spend(payload: dict):
    """USD total from a RunPod GraphQL userSelfServe payload.

    RunPod reports ``userSpent`` in cents. Returns None when the payload
    carries errors or no self-serve data.
    """
    if not isinstance(payload, dict) or payload.get("errors"):
        return None
    self_serve = (payload.get("data") or {}).get("userSelfServe")
    if not isinstance(self_serve, dict):
        return None
    cents = self_serve.get("userSpent")
    if cents is None:
        return None
    return float(cents) / 100.0


def fetch_runpod_spend(api_key: str, timeout: float = 15.0):
    """Live spend in USD via RunPod GraphQL (stdlib urllib)."""
    body = json.dumps(
        {"query": "query { userSelfServe { userSpent } }"}).encode("utf-8")
    req = urllib.request.Request(
        _RUNPOD_GQL, data=body,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return parse_runpod_spend(json.loads(resp.read().decode("utf-8")))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Track A RunPod cost monitor")
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--rate", type=float, required=True,
                    help="USD per GPU-hour (PLAN 8: ~0.75-1.0 for 5090)")
    ap.add_argument("--interval", type=float, default=300.0,
                    help="seconds between checks")
    ap.add_argument("--once", action="store_true",
                    help="single check then exit (for tests/cron)")
    ap.add_argument("--runpod-api-key", default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    state = {"last_frac": 0.0, "alerts": []}
    while True:
        spend = estimate_spend((time.time() - t0) / 3600.0, args.rate)
        if args.runpod_api_key:
            try:
                live = fetch_runpod_spend(args.runpod_api_key)
            except Exception:
                live = None
            if live is not None:
                spend = max(spend, live)
        alerts, state = monitor_step(state, spend, args.budget_usd)
        for alert in alerts:
            print(alert["message"], flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
