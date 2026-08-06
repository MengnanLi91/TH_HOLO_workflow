"""Aggregate regressor vs coupled ΔP over an extrapolation shell and plot the
crossover.

Reads only existing artifacts:
  - regressor: data/models/<reg_case>/eval_metrics.json (per_case_predictions:
    delta_p_true + {linear_regression,random_forest,mlp}_pred)
  - coupled:   data/cases/<conv_case>/<case>/forchheimer_profile.meta.json
               (delta_p_surrogate, delta_p_truth, Re/Dr/Lr + geometry)
  - baseline:  computed via integrated_baseline_delta_p() from sidecar geometry

Headline comparison uses the extrapolation-capable regressor families
(linear_regression, mlp); random_forest is plotted as a documented artifact
because it cannot extrapolate (piecewise-constant).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cases.alpha_d.extrapolation import parse_case_params
from cases.alpha_d.physics.baseline import BaselineGeometry, integrated_baseline_delta_p

REG_MODELS = ("linear_regression", "random_forest", "mlp")
HEADLINE_MODELS = ("linear_regression", "mlp")


def relerr(pred: float, truth: float) -> float:
    return (pred - truth) / truth


def find_crossover(axis_vals, regr_err, coupled_err):
    """First axis value (in given order) where |coupled| < |regressor|, else None."""
    for x, r, c in zip(axis_vals, regr_err, coupled_err):
        if abs(c) < abs(r):
            return x
    return None


def baseline_delta_p_from_sidecar(sc: dict) -> float:
    geom = BaselineGeometry(
        Re=sc["Re"],
        Dr=sc["Dr"],
        Lr=sc["Lr"],
        D_big=sc.get("D_big", 0.2),
        outer_height_m=sc.get("outer_height_m", 1.0),
        buffer_diams=sc.get("buffer_diams", 1.0),
        rho=sc.get("rho", 1.0),
        V_bulk=sc.get("V_bulk", 1.0),
    )
    return integrated_baseline_delta_p(geom)


def collect(*, axis, eval_metrics_path, coupled_dir, shell_names):
    em = json.loads(Path(eval_metrics_path).read_text())
    rows = {r["case"]: r for r in em["per_case_predictions"]}
    out = []
    for name in shell_names:
        sc_path = Path(coupled_dir) / name / "forchheimer_profile.meta.json"
        if name not in rows or not sc_path.exists():
            continue  # logged by caller; skip cases missing either artifact
        row = rows[name]
        sc = json.loads(sc_path.read_text())
        truth = float(row["delta_p_true"])
        rec = {
            "case": name,
            "axis_value": getattr(parse_case_params(name), axis),
            "truth": truth,
            "coupled": float(sc["delta_p_surrogate"]),
            "baseline": baseline_delta_p_from_sidecar(sc),
            "coupled_relerr": relerr(float(sc["delta_p_surrogate"]), truth),
        }
        for m in REG_MODELS:
            rec[m] = float(row[f"{m}_pred"])
            rec[f"{m}_relerr"] = relerr(float(row[f"{m}_pred"]), truth)
        out.append(rec)
    out.sort(key=lambda r: r["axis_value"])
    return out


def plot(records, axis, out_png):
    xs = [r["axis_value"] for r in records]
    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    ax.axhline(0.0, color="#444", lw=1.0, ls="--", alpha=0.5)
    for m, color in (
        ("linear_regression", "#1f77b4"),
        ("mlp", "#9467bd"),
        ("random_forest", "#d62728"),
    ):
        ax.plot(
            xs,
            [abs(r[f"{m}_relerr"]) * 100 for r in records],
            "o-",
            color=color,
            label=f"regressor: {m}" + (" (cannot extrapolate)" if m == "random_forest" else ""),
        )
    ax.plot(
        xs,
        [abs(r["coupled_relerr"]) * 100 for r in records],
        "s-",
        color="#2ca02c",
        label="coupled (α_D→integral)",
    )
    ax.set_xlabel(axis)
    ax.set_ylabel("|relative error vs CFD truth| (%)")
    ax.set_title(f"Extrapolation along {axis}: regressor vs coupled")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    return out_png


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--axis", required=True)
    p.add_argument("--eval-metrics", required=True)
    p.add_argument("--coupled-dir", required=True)
    p.add_argument(
        "--shell-names-file",
        required=True,
        help="Newline-delimited shell case names (extrapolation.py --emit shell-names).",
    )
    p.add_argument("--out-png", required=True)
    ns = p.parse_args(argv)

    shell_names = [s for s in Path(ns.shell_names_file).read_text().split() if s]
    records = collect(
        axis=ns.axis,
        eval_metrics_path=ns.eval_metrics,
        coupled_dir=ns.coupled_dir,
        shell_names=shell_names,
    )
    if not records:
        raise SystemExit("No records collected — check eval_metrics / coupled sidecars exist.")
    for m in HEADLINE_MODELS:
        xo = find_crossover(
            [r["axis_value"] for r in records],
            [r[f"{m}_relerr"] for r in records],
            [r["coupled_relerr"] for r in records],
        )
        print(f"[crossover] headline model {m}: {xo}")
    out = plot(records, ns.axis, ns.out_png)
    print(f"Saved {out} over {len(records)} shell cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
