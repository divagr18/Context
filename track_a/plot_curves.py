"""Rate-distortion curves from eval report JSON (PLAN 9 deliverable).

``plot_rate_distortion(report, out_png)`` renders a fixed panel grid:
per-type exact survival vs ratio, hallucination vs ratio, decoy emission
vs ratio, salience inversions vs ratio, parse failures vs ratio, and
info_pressure context vs ratio. Optional keys missing from minimal reports
(the ``*_by_ratio`` additions) are skipped gracefully.
"""
from __future__ import annotations


def _ratios_from(report) -> list[str]:
    ratios = report.get("meta", {}).get("ratios")
    if ratios:
        return [str(r) for r in sorted(int(r) for r in ratios)]
    return [str(r) for r in sorted(int(k)
            for k in report["primary"].get("info_pressure", {}))]


def _type_curves(surv_tree) -> dict[str, dict[str, float]]:
    """Per-type survival vs ratio: mean over distance buckets."""
    out: dict[str, dict[str, float]] = {}
    for tval, dist_map in (surv_tree or {}).items():
        per_ratio: dict[str, list[float]] = {}
        for _dist, ratio_map in dist_map.items():
            for rkey, frac in ratio_map.items():
                per_ratio.setdefault(rkey, []).append(frac)
        out[tval] = {r: sum(v) / len(v) for r, v in per_ratio.items()}
    return out


def plot_rate_distortion(report: dict, out_png: str) -> None:
    """Write the rate-distortion panel grid as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = report["primary"]
    ratios = _ratios_from(report)
    xs = [int(r) for r in ratios]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Track A rate-distortion (single-shot)")

    ax = axes[0][0]
    for tval, curve in _type_curves(p.get("needle_survival_exact")).items():
        ys = [curve.get(r) for r in ratios]
        ax.plot(xs, ys, marker="o", label=tval)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("compression ratio")
    ax.set_ylabel("exact survival")
    ax.set_title("needle_survival_exact by type")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    def _panel(ax, key, title, ylabel, use_ratios=ratios):
        data = p.get(key) or {}
        xs_ = [int(r) for r in use_ratios]
        ys = [data.get(r) for r in use_ratios]
        ax.plot(xs_, ys, marker="o", color="tab:red")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("compression ratio")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    _panel(axes[0][1], "hallucination_rate_by_ratio",
           "hallucination_rate", "rate")
    _panel(axes[0][2], "decoy_emission_rate_by_ratio",
           "decoy_emission_rate", "rate")
    _panel(axes[1][0], "salience_inversions",
           "salience_inversions", "frac docs")
    pf = {k: v for k, v in (p.get("parse_fail_rate") or {}).items()
          if k != "overall"}
    ax = axes[1][1]
    xs_ = [int(r) for r in pf]
    ax.plot(xs_, [pf[r] for r in pf], marker="o", color="tab:purple")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("compression ratio")
    ax.set_ylabel("parse failure rate")
    ax.set_title("parse_fail_rate")
    ax.grid(True, alpha=0.3)
    _panel(axes[1][2], "info_pressure",
           "info_pressure (C* fact count)", "facts in C*")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
