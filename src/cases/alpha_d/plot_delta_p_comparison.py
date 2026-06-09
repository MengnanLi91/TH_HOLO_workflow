"""Generate a comparison plot for the alpha_D -> MOOSE coupling end-to-end results.

Target case: Re_43938__Dr_0p522__Lr_0p073
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Resolve repo root from this file's location: src/cases/alpha_d/<this>.py
# → parents[3] is the repo root regardless of where the worktree lives.
REPO = Path(__file__).resolve().parents[3]
CASE = "Re_43938__Dr_0p522__Lr_0p073"
SIDECAR = REPO / f"data/cases/train_conv1d/{CASE}/forchheimer_profile.meta.json"
CSV = REPO / "src/cases/alpha_d/moose/forchheimer_profile.csv"
# Two output locations:
#   1. Next to the sidecar (data/, gitignored) — the canonical produced artifact.
#   2. docs/_static/ (tracked) — embedded in docs/dev/alpha_d_coupling_physics.md.
OUT_PNG_DATA = REPO / f"data/cases/train_conv1d/{CASE}/delta_p_comparison.png"
OUT_PNG_DOCS = REPO / "docs/_static/alpha_d_coupling_delta_p.png"

sidecar = json.loads(SIDECAR.read_text())
csv = np.loadtxt(CSV, delimiter=",", skiprows=1)
z_csv, F_csv = csv[:, 0], csv[:, 1]

alpha_D = np.array(sidecar["alpha_D_bulk_roi"])
z_phys = np.array(sidecar["z_phys_roi"])
D_big = sidecar["D_big"]
Dr = sidecar["Dr"]
end_len = sidecar["buffer_diams"] * D_big
throat_end = end_len + sidecar["throat_length_m"]
in_throat = (z_phys >= end_len - 1e-9) & (z_phys <= throat_end + 1e-9)
D_h = np.where(in_throat, Dr * D_big, D_big)
integrand_surrogate = alpha_D / (2.0 * D_h)  # rho=V_bulk=1

# End-to-end ΔP comparison (Pa). delta_p_moose is read from the verifier
# and hardcoded here so the figure regenerates without rerunning MOOSE.
delta_p_moose = 11.282
labels = ["Surrogate\nintegral", "Coupled\nMOOSE", "CFD\ntruth"]
values = [sidecar["delta_p_surrogate"], delta_p_moose, sidecar["delta_p_truth"]]
colors = ["#1f77b4", "#2ca02c", "#444444"]
rel_err_pct = [(v - sidecar["delta_p_truth"]) / sidecar["delta_p_truth"] * 100 for v in values]

# Three-panel layout: comparison spans the top row; α_D(z) and F(z) sit below.
fig = plt.figure(figsize=(13, 9.5), constrained_layout=False)
gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[0.85, 1.0],
    hspace=0.42,
    wspace=0.22,
    left=0.07,
    right=0.97,
    top=0.93,
    bottom=0.07,
)

# ── Top (full width): ΔP comparison bar chart ──
ax1 = fig.add_subplot(gs[0, :])
bar_w = 0.55
x_pos = np.arange(len(labels))
bars = ax1.bar(x_pos, values, width=bar_w, color=colors, edgecolor="black", linewidth=1.1)
ax1.axhline(
    sidecar["delta_p_truth"],
    color="#444444",
    linestyle="--",
    linewidth=1.2,
    alpha=0.55,
    zorder=0,
    label=f"CFD truth = {sidecar['delta_p_truth']:.2f} Pa",
)
for bar, val, err in zip(bars, values, rel_err_pct):
    txt = f"{val:.2f} Pa" if abs(err) < 0.5 else f"{val:.2f} Pa\n({err:+.1f}%)"
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.18,
        txt,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="600",
    )
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel("ΔP across ROI (Pa)", fontsize=11)
ax1.set_title(
    "End-to-end ΔP comparison — Re_43938__Dr_0p522__Lr_0p073",
    fontsize=12,
    pad=10,
)
ax1.set_ylim(0, max(values) * 1.18)
ax1.set_xlim(-0.6, len(labels) - 0.4)
ax1.legend(loc="upper left", fontsize=9, framealpha=0.95)
ax1.grid(True, axis="y", alpha=0.25)
for spine in ("top", "right"):
    ax1.spines[spine].set_visible(False)

# ── Bottom-left: α_D(z) and the porosity step ──
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(
    z_phys, alpha_D, "o-", color="#1f77b4", markersize=4, linewidth=1.5, label="α_D(z) (surrogate)"
)
ax3.axvspan(
    end_len, throat_end, color="orange", alpha=0.15, label=f"throat (block 2, ε={Dr**2:.3f})"
)
ax3.axvline(end_len, color="orange", linestyle="--", linewidth=1, alpha=0.7)
ax3.axvline(throat_end, color="orange", linestyle="--", linewidth=1, alpha=0.7)
peak_idx = int(np.argmax(alpha_D))
peak_y = alpha_D[peak_idx]
# Headroom above the peak so the legend (upper-left) and the annotation
# (just right of the peak) don't fight with the data.
ax3.set_ylim(top=peak_y * 1.30, bottom=min(alpha_D) - 10)
ax3.annotate(
    f"vena-contracta\npeak α_D={peak_y:.1f}",
    xy=(z_phys[peak_idx], peak_y),
    xytext=(z_phys[peak_idx] + 0.04, peak_y * 1.05),
    fontsize=9,
    ha="left",
    arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
)
ax3.set_xlabel("z (m), MOOSE mesh coord (x=0 at inlet)", fontsize=10)
ax3.set_ylabel("α_D (dimensionless)", fontsize=10)
ax3.set_title("Surrogate-predicted Darcy-Weisbach α_D(z)", fontsize=11)
ax3.legend(loc="upper left", fontsize=9, framealpha=0.95)
ax3.grid(True, alpha=0.3)
for spine in ("top", "right"):
    ax3.spines[spine].set_visible(False)

# ── Bottom-right: F(z) split by block, log scale ──
ax4 = fig.add_subplot(gs[1, 1])
in_throat_csv = (z_csv >= end_len - 1e-9) & (z_csv <= throat_end + 1e-9)
F_abs = np.abs(F_csv) + 1e-3  # log-scale floor for visibility
ax4.bar(
    z_csv[~in_throat_csv],
    F_abs[~in_throat_csv],
    width=0.008,
    color="#1f77b4",
    alpha=0.7,
    label=f"buffer F (mult = {sidecar['forchheimer_multiplier_buffer']:.2f})",
)
ax4.bar(
    z_csv[in_throat_csv],
    F_abs[in_throat_csv],
    width=0.008,
    color="#d62728",
    alpha=0.7,
    label=f"throat F (mult = {sidecar['forchheimer_multiplier_throat']:.3f})",
)
ax4.axvspan(end_len, throat_end, color="orange", alpha=0.10)
ax4.set_yscale("log")
ax4.set_xlabel("z (m), MOOSE mesh coord", fontsize=10)
ax4.set_ylabel("|F(z)| (1/m)", fontsize=10)
ax4.set_title(
    f"Forchheimer profile fed to MOOSE   (peak F = {np.abs(F_csv).max():.1f})", fontsize=11
)
ax4.legend(loc="upper left", fontsize=9)
ax4.grid(True, alpha=0.3, which="both")
for spine in ("top", "right"):
    ax4.spines[spine].set_visible(False)

# Overall title + tagline
fig.suptitle("α_D → MOOSE coupling pipeline", fontsize=13.5, y=0.985, fontweight="600")
fig.text(
    0.5,
    0.012,
    "Coupled MOOSE reproduces the surrogate ΔP integral to within 0.7% — "
    "the coupling preserves the surrogate's prediction without adding distortion.",
    ha="center",
    fontsize=9,
    style="italic",
    color="#444",
)

for out in (OUT_PNG_DATA, OUT_PNG_DOCS):
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140)
    print(f"Saved: {out}", file=sys.stderr)
