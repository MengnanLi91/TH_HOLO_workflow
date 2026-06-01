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

REPO = Path("/data/lim2/projects/multifid-th/worktrees/integration")
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

# End-to-end ΔP comparison (Pa)
labels = ["Constant\nF=1 MOOSE", "Surrogate\nintegral", "Coupled\nMOOSE", "CFD\ntruth"]
values = [0.484, sidecar["delta_p_surrogate"], 12.77, sidecar["delta_p_truth"]]
colors = ["#888888", "#1f77b4", "#2ca02c", "#000000"]
rel_err_pct = [(v - sidecar["delta_p_truth"]) / sidecar["delta_p_truth"] * 100 for v in values]

fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.32, wspace=0.28)

# ── Top-left: ΔP comparison bar chart ──
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)
ax1.axhline(
    sidecar["delta_p_truth"],
    color="black",
    linestyle=":",
    linewidth=1,
    alpha=0.5,
    label="truth = 11.83 Pa",
)
for bar, val, err in zip(bars, values, rel_err_pct):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.3,
        f"{val:.2f} Pa\n({err:+.1f}%)",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
ax1.set_ylabel("ΔP across ROI (Pa)", fontsize=11)
ax1.set_title("End-to-end ΔP comparison — target Re_43938__Dr_0p522__Lr_0p073", fontsize=11)
ax1.set_ylim(0, 16)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, axis="y", alpha=0.3)

# ── Top-right: relative error vs truth (signed bar chart) ──
ax2 = fig.add_subplot(gs[0, 1])
err_bars = ax2.bar(labels, rel_err_pct, color=colors, edgecolor="black", linewidth=1.2)
ax2.axhline(0, color="black", linewidth=1)
for bar, err in zip(err_bars, rel_err_pct):
    y_offset = 4 if err >= 0 else -8
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        err + y_offset,
        f"{err:+.1f}%",
        ha="center",
        va="bottom" if err >= 0 else "top",
        fontsize=10,
        fontweight="bold",
    )
ax2.set_ylabel("Relative error vs CFD truth (%)", fontsize=11)
ax2.set_title("Fidelity vs CFD truth (closer to 0 is better)", fontsize=11)
ax2.set_ylim(-110, 35)
ax2.grid(True, axis="y", alpha=0.3)

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

# ── Bottom-right: F(z) split by block, log scale ──
ax4 = fig.add_subplot(gs[1, 1])
in_throat_csv = (z_csv >= end_len - 1e-9) & (z_csv <= throat_end + 1e-9)
sign_F = np.sign(F_csv)
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

# Overall title
fig.suptitle(f"α_D → MOOSE coupling pipeline — {CASE}", fontsize=13, y=0.995)
fig.text(
    0.5,
    0.005,
    "Coupled MOOSE (+7.9% from truth) is closer than the surrogate integral alone (-30.8%) "
    "or vanilla MOOSE with constant F=1 (-95.9%).",
    ha="center",
    fontsize=9,
    style="italic",
)

for out in (OUT_PNG_DATA, OUT_PNG_DOCS):
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}", file=sys.stderr)
