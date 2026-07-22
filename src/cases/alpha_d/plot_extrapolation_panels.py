"""Combined 4-panel extrapolation-crossover figure (one panel per axis).

Aggregates the per-axis sweep artifacts produced by the extrapolation study
(``data/cases/extrap/<tag>/`` + ``data/models/case_pressure_drop__ext_<tag>/``)
into mean |relative error| vs shell level, for coupled / RF / MLP / analytic
baseline, on a shared log scale (MLP overshoots span several decades).

Run from repo root: ``PYTHONPATH=src python -m cases.alpha_d.plot_extrapolation_panels``
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cases.alpha_d.plot_extrapolation_sweep import collect

# (tag, axis, panel title)
PANELS = [
    ("Dr_low", "Dr", "Dr ↓  (small throat)"),
    ("Re_high", "Re", "Re ↑  (high Reynolds)"),
    ("Lr_high", "Lr", "Lr ↑  (long throat)"),
    ("Lr_low", "Lr", "Lr ↓  (short throat)"),
]
# (record key for rel-error, label, colour)
SERIES = [
    ("coupled_relerr", "coupled (α_D→integral)", "#2ca02c"),
    ("random_forest_relerr", "regressor: RF (CV-best)", "#d62728"),
    ("mlp_relerr", "regressor: MLP", "#9467bd"),
    ("baseline_relerr", "analytic baseline", "#7f7f7f"),
]
# In-distribution anchor (target held out of both models), §7.8.1.
INDIST_COUPLED_PCT = 10.2


def _amean_pct(xs) -> float:
    return 100.0 * sum(abs(x) for x in xs) / len(xs)


def _panel_data(tag: str, axis: str, data_root: Path):
    recs = collect(
        axis=axis,
        eval_metrics_path=str(
            data_root / f"models/case_pressure_drop__ext_{tag}/eval_metrics.json"
        ),
        coupled_dir=str(data_root / f"cases/extrap/{tag}/coupled"),
        shell_names=[
            s
            for s in (data_root / f"cases/extrap/{tag}/shell_names.txt")
            .read_text()
            .split()
            if s
        ],
    )
    for r in recs:  # add baseline rel-error alongside the model rel-errors
        r["baseline_relerr"] = (r["baseline"] - r["truth"]) / r["truth"]
    by = defaultdict(list)
    for r in recs:
        by[r["axis_value"]].append(r)
    levels = sorted(by)
    return levels, by


def build_figure(data_root: Path, out_png: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    for ax, (tag, axis, title) in zip(axes.flat, PANELS):
        levels, by = _panel_data(tag, axis, data_root)
        x = np.arange(len(levels))
        width = 0.8 / len(SERIES)
        for i, (key, label, color) in enumerate(SERIES):
            vals = [_amean_pct([r[key] for r in by[lvl]]) for lvl in levels]
            bars = ax.bar(
                x + (i - (len(SERIES) - 1) / 2) * width,
                vals,
                width,
                color=color,
                label=label,
                edgecolor="black",
                linewidth=0.4,
            )
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v * 1.05,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
        ax.axhline(INDIST_COUPLED_PCT, color="#2ca02c", ls=":", lw=1.0, alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 1e5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{axis}={lvl:.4g}" for lvl in levels])
        ax.set_title(title, fontsize=11, fontweight="600")
        ax.set_ylabel("mean |rel. error| vs CFD truth (%)")
        ax.grid(True, axis="y", which="both", alpha=0.2)
        n = sum(len(v) for v in by.values())
        ax.text(
            0.98,
            0.96,
            f"n={n}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#555",
        )
    axes.flat[0].legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.suptitle(
        "When does the coupled approach beat direct ΔP regression? "
        "(dotted line = coupled in-distribution +10.2%)",
        fontsize=13,
        fontweight="600",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    return out_png


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--out-png",
        type=Path,
        default=Path("docs/_static/alpha_d_extrapolation_crossover.png"),
    )
    ns = p.parse_args(argv)
    out = build_figure(ns.data_root, ns.out_png)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
