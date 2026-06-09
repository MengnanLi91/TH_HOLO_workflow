"""Single-case ΔP comparison: independent case-level regressor vs coupled MOOSE.

Target case: Re_43938__Dr_0p522__Lr_0p073

Bar 1 — case_pressure_drop regressor, HELD-OUT prediction (target forced out
        of training via data.split.force_test).
Bar 2 — coupled MOOSE PINSFV pressure (hardcoded; no MOOSE rerun here).
Bar 3 — resolved-CFD truth (case_metadata.delta_p_case).

This replaces the earlier "surrogate integral vs coupled MOOSE" chart, which
was near-circular: MOOSE was fed the Conv1D's own α_D profile, so it merely
reproduced bar 1's integral. The case-level regressor predicts ΔP directly
from (Re, Dr, Lr) and never sees the α_D profile or MOOSE — a genuinely
independent estimate.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
CASE = "Re_43938__Dr_0p522__Lr_0p073"

# Independent case-level regressor artifacts (held-out target).
RUN_META = REPO / "data/models/case_pressure_drop/run_meta.json"
EVAL = REPO / "data/models/case_pressure_drop/eval_metrics.json"
# alpha_d sidecar — used only to cross-check the CFD truth baseline.
SIDECAR = REPO / f"data/cases/train_conv1d/{CASE}/forchheimer_profile.meta.json"

OUT_PNG_DATA = REPO / f"data/cases/train_conv1d/{CASE}/delta_p_comparison.png"
OUT_PNG_DOCS = REPO / "docs/_static/alpha_d_coupling_delta_p.png"

# Coupled MOOSE PINSFV pressure for the target case (postprocessor / inlet
# area). Hardcoded so the figure regenerates without rerunning MOOSE.
delta_p_moose = 11.282

if not EVAL.exists() or not RUN_META.exists():
    raise SystemExit(
        f"Missing case_pressure_drop artifacts under {EVAL.parent}.\n"
        f"Train + evaluate first, forcing the target held out:\n"
        f"  python cases/case_pressure_drop/run_case_pressure_drop.py "
        f"data.split.force_test='[{CASE}]'\n"
        f"  python cases/case_pressure_drop/evaluate_case_pressure_drop.py"
    )

run_meta = json.loads(RUN_META.read_text())
best_model = run_meta["best_model"]["name"]
if CASE not in run_meta["split"].get("force_test", []):
    raise SystemExit(
        f"{CASE} was not forced held-out (run_meta.split.force_test="
        f"{run_meta['split'].get('force_test')}). Retrain with "
        f"data.split.force_test='[{CASE}]' so the prediction is out-of-sample."
    )

evald = json.loads(EVAL.read_text())
rows = {r["case"]: r for r in evald["per_case_predictions"]}
if CASE not in rows:
    raise SystemExit(f"{CASE} not in case_pressure_drop test predictions.")
row = rows[CASE]
dp_case_reg = float(row[f"{best_model}_pred"])
dp_truth = float(row["delta_p_true"])

# Cross-check the truth baseline against the alpha_d sidecar — both should be
# case_metadata.delta_p_case. Bail if the two halves of the chart disagree.
dp_truth_sidecar = float(json.loads(SIDECAR.read_text())["delta_p_truth"])
if abs(dp_truth - dp_truth_sidecar) / dp_truth_sidecar > 0.01:
    raise SystemExit(
        f"Truth baseline mismatch: eval delta_p_true={dp_truth:.4f} vs "
        f"sidecar delta_p_truth={dp_truth_sidecar:.4f} (>1%)."
    )

labels = ["Case ΔP\nregressor\n(held-out)", "Coupled\nMOOSE", "CFD\ntruth"]
values = [dp_case_reg, delta_p_moose, dp_truth]
colors = ["#1f77b4", "#2ca02c", "#444444"]
rel_err_pct = [(v - dp_truth) / dp_truth * 100 for v in values]

# Ready-to-paste line for docs/dev/alpha_d_coupling_physics.md §7.4.
print(
    f"[numbers] case_regressor({best_model})={dp_case_reg:.2f} Pa "
    f"({rel_err_pct[0]:+.1f}%) | coupled_moose={delta_p_moose:.2f} Pa "
    f"({rel_err_pct[1]:+.1f}%) | cfd_truth={dp_truth:.2f} Pa",
    file=sys.stderr,
)

fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
x_pos = np.arange(len(labels))
bars = ax.bar(x_pos, values, width=0.55, color=colors, edgecolor="black", linewidth=1.1)
ax.axhline(
    dp_truth,
    color="#444444",
    linestyle="--",
    linewidth=1.2,
    alpha=0.55,
    zorder=0,
    label=f"CFD truth = {dp_truth:.2f} Pa",
)
for bar, val, err in zip(bars, values, rel_err_pct):
    txt = f"{val:.2f} Pa" if abs(err) < 0.5 else f"{val:.2f} Pa\n({err:+.1f}%)"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + max(values) * 0.015,
        txt,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="600",
    )
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("ΔP across ROI (Pa)", fontsize=11)
ax.set_title(f"Independent ΔP estimates vs CFD truth — {CASE}", fontsize=12, pad=10)
ax.set_ylim(0, max(values) * 1.18)
ax.set_xlim(-0.6, len(labels) - 0.4)
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
ax.grid(True, axis="y", alpha=0.25)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.suptitle(
    "Case-level ΔP regressor vs coupled Conv1D→MOOSE pipeline",
    fontsize=13.5,
    fontweight="600",
)
fig.text(
    0.5,
    -0.02,
    f"Bar 1: case_pressure_drop ({best_model}) predicts ΔP directly from "
    "(Re, Dr, Lr) — single HELD-OUT case, independent of the α_D profile and "
    "MOOSE. Bar 2: coupled MOOSE, still driven by the Conv1D α_D profile. "
    "Both compared against the same CFD truth.",
    ha="center",
    fontsize=8.5,
    style="italic",
    color="#444",
    wrap=True,
)

for out in (OUT_PNG_DATA, OUT_PNG_DOCS):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}", file=sys.stderr)
