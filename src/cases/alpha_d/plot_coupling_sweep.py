"""Aggregate visualisation of the alpha_D coupling sweep.

Reads ``data/cases/train_conv1d/coupling_sweep.json`` (one entry per case
with truth / surrogate / MOOSE pressures and signed relative errors) and
draws a two-panel figure:

  • parity plot — truth vs predicted, log-log, one marker per case;
    surrogate-alone and coupled-MOOSE drawn with distinct shapes so the
    reader can see whether MOOSE moves any case off the diagonal that
    the surrogate would have nailed.
  • CDF — cumulative distribution of |relative error| across the sweep,
    one line per predictor, with median / p90 callouts.

Outputs:
  • data/cases/train_conv1d/coupling_sweep.png  (canonical artifact)
  • docs/_static/alpha_d_coupling_sweep.png     (tracked, embeddable)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
SWEEP_JSON = REPO / "data/cases/train_conv1d/coupling_sweep.json"
CONST_F_JSON = REPO / "data/cases/train_conv1d/constF_sweep.json"
OUT_DATA = REPO / "data/cases/train_conv1d/coupling_sweep.png"
OUT_DOCS = REPO / "docs/_static/alpha_d_coupling_sweep.png"

results = json.loads(SWEEP_JSON.read_text())
truth = np.array([r["delta_p_truth"] for r in results])
surro = np.array([r["delta_p_surrogate"] for r in results])
moose = np.array([r["delta_p_moose"] for r in results])
err_surro = np.array([abs(r["rel_err_surrogate"]) for r in results])
err_moose = np.array([abs(r["rel_err_moose"]) for r in results])

# Optional constant-F sweep (no ML, single F applied to all cases).
constF = None
if CONST_F_JSON.exists():
    constF_results = json.loads(CONST_F_JSON.read_text())
    # Order by the same case list as the coupling sweep
    case_to_const = {r["case"]: r for r in constF_results}
    constF = np.array([case_to_const[r["case"]]["delta_p_constF"] for r in results])
    err_constF = np.array([abs(case_to_const[r["case"]]["rel_err"]) for r in results])
    F_value = constF_results[0]["F_constant"]

# ─── Figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12.5, 5.5), constrained_layout=False)
gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[1.05, 1.0],
    wspace=0.28,
    left=0.07,
    right=0.97,
    top=0.90,
    bottom=0.13,
)

# ─── Parity plot ────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0])

all_pts = [truth, surro, moose]
if constF is not None:
    all_pts.append(constF)
lo = float(min(p.min() for p in all_pts) * 0.5)
hi = float(max(p.max() for p in all_pts) * 1.8)
ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", zorder=0, label="y = x")
ax.plot(
    [lo, hi],
    [1.1 * lo, 1.1 * hi],
    color="#888",
    linewidth=0.8,
    linestyle=":",
    zorder=0,
    label="±10%",
)
ax.plot([lo, hi], [0.9 * lo, 0.9 * hi], color="#888", linewidth=0.8, linestyle=":", zorder=0)

if constF is not None:
    ax.scatter(
        truth,
        constF,
        s=85,
        marker="x",
        color="#d62728",
        linewidths=1.5,
        alpha=0.85,
        label=f"Vanilla MOOSE, const F={F_value}",
        zorder=2,
    )
ax.scatter(
    truth,
    surro,
    s=85,
    marker="o",
    facecolors="#1f77b4",
    edgecolors="white",
    linewidths=0.7,
    alpha=0.85,
    label="Surrogate",
    zorder=3,
)
ax.scatter(
    truth,
    moose,
    s=85,
    marker="^",
    facecolors="#2ca02c",
    edgecolors="white",
    linewidths=0.7,
    alpha=0.85,
    label="Coupled MOOSE",
    zorder=3,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("CFD truth ΔP  (Pa)", fontsize=11)
ax.set_ylabel("Predicted ΔP  (Pa)", fontsize=11)
ax.set_title(f"Parity over {len(results)} test cases", fontsize=12, pad=8)
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.grid(True, which="both", alpha=0.25)
ax.set_aspect("equal", adjustable="box")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# ─── CDF of |relative error| ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])


def cdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    # Start the line at the leftmost point on the y=0 axis so it reads cleanly.
    return np.concatenate(([0.0], xs * 100)), np.concatenate(([0.0], ys))


xs, ys = cdf(err_surro)
ax2.step(
    xs,
    ys,
    where="post",
    color="#1f77b4",
    linewidth=2.2,
    label=f"Surrogate (median {np.median(err_surro) * 100:.1f}%, "
    f"p90 {np.percentile(err_surro, 90) * 100:.1f}%)",
)
xs, ys = cdf(err_moose)
ax2.step(
    xs,
    ys,
    where="post",
    color="#2ca02c",
    linewidth=2.2,
    label=f"Coupled MOOSE (median {np.median(err_moose) * 100:.1f}%, "
    f"p90 {np.percentile(err_moose, 90) * 100:.1f}%)",
)
if constF is not None:
    xs, ys = cdf(err_constF)
    ax2.step(
        xs,
        ys,
        where="post",
        color="#d62728",
        linewidth=2.2,
        linestyle="--",
        label=f"Vanilla MOOSE, const F={F_value} "
        f"(median {np.median(err_constF) * 100:.0f}%, "
        f"max {err_constF.max() * 100:.0f}%)",
    )

ax2.axvline(10.0, color="#888", linestyle=":", linewidth=0.9, label="10% threshold")
ax2.axhline(0.5, color="#bbb", linestyle=":", linewidth=0.7, zorder=0)
ax2.axhline(0.9, color="#bbb", linestyle=":", linewidth=0.7, zorder=0)
ax2.text(0.4, 0.52, "median", color="#888", fontsize=8.5)
ax2.text(0.4, 0.92, "p90", color="#888", fontsize=8.5)

err_max_all = err_moose.max()
if constF is not None:
    err_max_all = max(err_max_all, err_constF.max())
xmax = float(err_max_all * 100 * 1.10)
ax2.set_xscale("symlog", linthresh=20)
ax2.set_xlim(0, xmax)
ax2.set_ylim(0, 1.02)
ax2.set_xlabel("|Relative error| vs CFD truth  (%, symlog above 20%)", fontsize=11)
ax2.set_ylabel("Fraction of cases with error ≤ X", fontsize=11)
ax2.set_title(f"Error CDF across {len(results)} cases", fontsize=12, pad=8)
ax2.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax2.grid(True, alpha=0.25)
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)

fig.suptitle(
    "α_D surrogate vs coupled MOOSE — sweep across the test set",
    fontsize=13,
    fontweight="600",
    y=0.985,
)

for out in (OUT_DATA, OUT_DOCS):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"Saved: {out}", file=sys.stderr)
