"""Compare HPO training versions across the project's Optuna databases.

Usage (inside the container):
    python -m evaluation.compare_hpo_versions                    # all versions
    python -m evaluation.compare_hpo_versions --versions v6 v7   # specific versions
    python -m evaluation.compare_hpo_versions --save report.png  # save plots

Reads HPO databases from ``data/hpo/`` and eval_metrics from ``data/models*/``,
prints a comparison table and optionally generates comparison plots.
"""

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HPO_DIR = PROJECT_ROOT / "data" / "hpo"

# Known database filenames in chronological order.
KNOWN_DBS: dict[str, Path] = {}

# Discover databases dynamically.
def _discover_dbs() -> dict[str, Path]:
    dbs: dict[str, Path] = {}
    # Legacy v1
    v1_path = PROJECT_ROOT / "data" / "hpo_v1" / "alpha_d_mlp_hpo.db"
    if v1_path.exists():
        dbs["v1"] = v1_path
    # Standard naming: alpha_d_mlp_hpo_vN.db
    for db_file in sorted(HPO_DIR.glob("alpha_d_mlp_hpo_v*.db")):
        version = db_file.stem.split("_hpo_")[1]  # e.g. "v3", "v5b"
        dbs[version] = db_file
    return dbs


# Known eval_metrics.json paths.
KNOWN_EVAL_METRICS: dict[str, Path] = {
    "v1": PROJECT_ROOT / "data" / "models_v1" / "eval_metrics.json",
    "latest": PROJECT_ROOT / "data" / "models" / "eval_metrics.json",
}


@dataclass
class TrialSummary:
    number: int
    value: float
    params: dict[str, float] = field(default_factory=dict)
    datetime_start: str = ""
    datetime_complete: str = ""


@dataclass
class VersionSummary:
    version: str
    db_path: Path
    study_name: str = ""
    n_complete: int = 0
    n_pruned: int = 0
    best_trial: TrialSummary | None = None
    all_completed_values: list[float] = field(default_factory=list)
    eval_metrics: dict[str, Any] | None = None


def load_version(version: str, db_path: Path) -> VersionSummary:
    """Load summary from an Optuna SQLite database."""
    summary = VersionSummary(version=version, db_path=db_path)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("SELECT study_name FROM studies LIMIT 1")
    row = cur.fetchone()
    summary.study_name = row[0] if row else ""

    cur.execute("SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE'")
    summary.n_complete = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM trials WHERE state = 'PRUNED'")
    summary.n_pruned = cur.fetchone()[0]

    # All completed trial values
    cur.execute("""
        SELECT tv.value
        FROM trials t
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.state = 'COMPLETE'
        ORDER BY t.number
    """)
    summary.all_completed_values = [row[0] for row in cur.fetchall()]

    # Best trial
    cur.execute("""
        SELECT t.number, tv.value, t.datetime_start, t.datetime_complete
        FROM trials t
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.state = 'COMPLETE'
        ORDER BY tv.value ASC
        LIMIT 1
    """)
    best_row = cur.fetchone()
    if best_row:
        trial_num, val, dt_start, dt_complete = best_row
        # Get params
        cur.execute("""
            SELECT tp.param_name, tp.param_value
            FROM trial_params tp
            JOIN trials t ON tp.trial_id = t.trial_id
            WHERE t.number = ?
        """, (trial_num,))
        params = {row[0]: row[1] for row in cur.fetchall()}
        summary.best_trial = TrialSummary(
            number=trial_num,
            value=val,
            params=params,
            datetime_start=dt_start or "",
            datetime_complete=dt_complete or "",
        )

    conn.close()
    return summary


def load_eval_metrics(path: Path) -> dict[str, Any] | None:
    """Load evaluation metrics JSON if it exists."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(val: float, fmt: str = ".4f") -> str:
    return f"{val:{fmt}}"


def _pct_change(old: float, new: float) -> str:
    if old == 0:
        return "N/A"
    change = (new - old) / abs(old) * 100
    sign = "+" if change > 0 else ""
    return f"{sign}{change:.1f}%"


def print_hpo_comparison(summaries: list[VersionSummary]) -> None:
    """Print a formatted comparison table of HPO versions."""
    print("\n" + "=" * 80)
    print("  HPO VERSION COMPARISON")
    print("=" * 80)

    # Header
    header = f"{'Version':<8} {'Study':<28} {'Complete':>8} {'Pruned':>7} {'Best Val Loss':>14}"
    print(header)
    print("-" * 80)

    for s in summaries:
        best_val = _format_float(s.best_trial.value, ".6f") if s.best_trial else "N/A"
        print(f"{s.version:<8} {s.study_name:<28} {s.n_complete:>8} {s.n_pruned:>7} {best_val:>14}")

    # Progression
    if len(summaries) >= 2:
        prev = summaries[-2]
        curr = summaries[-1]
        if prev.best_trial and curr.best_trial:
            change = _pct_change(prev.best_trial.value, curr.best_trial.value)
            print(f"\n  Latest vs previous: {change} "
                  f"({prev.version} {prev.best_trial.value:.6f} -> "
                  f"{curr.version} {curr.best_trial.value:.6f})")

    if summaries[0].best_trial and summaries[-1].best_trial:
        change = _pct_change(summaries[0].best_trial.value, summaries[-1].best_trial.value)
        print(f"  Latest vs first:    {change} "
              f"({summaries[0].version} {summaries[0].best_trial.value:.6f} -> "
              f"{summaries[-1].version} {summaries[-1].best_trial.value:.6f})")
    print()


def print_best_params_comparison(summaries: list[VersionSummary]) -> None:
    """Print best hyperparameters side by side."""
    print("=" * 80)
    print("  BEST HYPERPARAMETERS")
    print("=" * 80)

    # Collect all param names across versions
    all_params: set[str] = set()
    for s in summaries:
        if s.best_trial:
            all_params.update(s.best_trial.params.keys())

    sorted_params = sorted(all_params)

    # Print header
    versions_str = "".join(f"{s.version:>14}" for s in summaries)
    print(f"{'Parameter':<32}{versions_str}")
    print("-" * (32 + 14 * len(summaries)))

    for param in sorted_params:
        row = f"{param:<32}"
        for s in summaries:
            if s.best_trial and param in s.best_trial.params:
                val = s.best_trial.params[param]
                if isinstance(val, float):
                    if val < 0.01 and val != 0:
                        row += f"{val:>14.6e}"
                    else:
                        row += f"{val:>14.4f}"
                else:
                    row += f"{str(val):>14}"
            else:
                row += f"{'---':>14}"
        print(row)
    print()


def print_eval_comparison(
    eval_a: dict[str, Any] | None,
    eval_b: dict[str, Any] | None,
    label_a: str = "previous",
    label_b: str = "latest",
) -> None:
    """Print side-by-side comparison of two evaluation results."""
    if not eval_a and not eval_b:
        print("  No evaluation metrics available.\n")
        return

    print("=" * 80)
    print(f"  EVALUATION METRICS: {label_a} vs {label_b}")
    print("=" * 80)

    def _get(d: dict | None, *keys: str, default: Any = None) -> Any:
        if d is None:
            return default
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d

    metrics_rows: list[tuple[str, Any, Any]] = [
        ("Overall MSE", _get(eval_a, "overall", "mse"), _get(eval_b, "overall", "mse")),
        ("Overall RMSE", _get(eval_a, "overall", "rmse"), _get(eval_b, "overall", "rmse")),
        ("Test cases", _get(eval_a, "test_cases"), _get(eval_b, "test_cases")),
        ("Test samples", _get(eval_a, "num_samples"), _get(eval_b, "num_samples")),
    ]

    # Extended metrics
    ext_a = _get(eval_a, "extended")
    ext_b = _get(eval_b, "extended")
    if ext_a or ext_b:
        pf_a = (_get(ext_a, "per_field") or [{}])[0] if ext_a else {}
        pf_b = (_get(ext_b, "per_field") or [{}])[0] if ext_b else {}
        metrics_rows.extend([
            ("R-squared", pf_a.get("r2"), pf_b.get("r2")),
            ("MAE", pf_a.get("mae"), pf_b.get("mae")),
            ("Phys median rel error", pf_a.get("physical_median_relative_error"),
             pf_b.get("physical_median_relative_error")),
            ("Phys mean rel error", pf_a.get("physical_mean_relative_error"),
             pf_b.get("physical_mean_relative_error")),
            ("Phys p90 rel error", pf_a.get("physical_p90_relative_error"),
             pf_b.get("physical_p90_relative_error")),
        ])

    header = f"{'Metric':<30} {label_a:>14} {label_b:>14} {'Change':>10}"
    print(header)
    print("-" * 70)

    for name, va, vb in metrics_rows:
        sa = _format_val(va)
        sb = _format_val(vb)
        sc = _pct_change(va, vb) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else ""
        print(f"{name:<30} {sa:>14} {sb:>14} {sc:>10}")

    # Per-region comparison
    if ext_a or ext_b:
        region_a = _get(ext_a, "per_region") or {} if ext_a else {}
        region_b = _get(ext_b, "per_region") or {} if ext_b else {}
        all_regions = sorted(set(list(region_a.keys()) + list(region_b.keys())))

        if all_regions:
            print(f"\n{'Per-Region Breakdown':}")
            print(f"{'Region':<20} {'Metric':<12} {label_a:>14} {label_b:>14} {'Change':>10}")
            print("-" * 70)
            for region in all_regions:
                ra = region_a.get(region, {})
                rb = region_b.get(region, {})
                # Find field metrics (skip n_samples)
                fields = set(k for k in list(ra.keys()) + list(rb.keys()) if k != "n_samples")
                for fld in sorted(fields):
                    fa = ra.get(fld, {})
                    fb = rb.get(fld, {})
                    for metric in ["r2", "rmse", "median_relative_error"]:
                        va = fa.get(metric)
                        vb = fb.get(metric)
                        if va is not None or vb is not None:
                            sa = _format_val(va)
                            sb = _format_val(vb)
                            sc = _pct_change(va, vb) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else ""
                            m_label = {"r2": "R2", "rmse": "RMSE", "median_relative_error": "MedRelErr"}.get(metric, metric)
                            print(f"{region:<20} {m_label:<12} {sa:>14} {sb:>14} {sc:>10}")

    # Worst/best cases comparison
    for label, key in [("Worst", "worst_cases"), ("Best", "best_cases")]:
        cases_a = _get(ext_a, key) if ext_a else None
        cases_b = _get(ext_b, key) if ext_b else None
        if cases_b:
            print(f"\n{label} 5 cases ({label_b}):")
            for c in (cases_b or [])[:5]:
                line = f"  {c['case']}: RMSE={c['rmse']:.4e}"
                if "median_relative_error" in c:
                    line += f", MedRelErr={c['median_relative_error']:.1%}"
                print(line)

    # Delta-p comparison
    dp_a = _get(ext_a, "delta_p") if ext_a else None
    dp_b = _get(ext_b, "delta_p") if ext_b else None
    if dp_a or dp_b:
        print(f"\n{'Delta-p Prediction':}")
        dp_rows = [
            ("Cases evaluated",
             _get(dp_a, "n_cases") if dp_a else None,
             _get(dp_b, "n_cases") if dp_b else None),
            ("Median rel error",
             _get(dp_a, "relative_error_median") if dp_a else None,
             _get(dp_b, "relative_error_median") if dp_b else None),
            ("Mean rel error",
             _get(dp_a, "relative_error_mean") if dp_a else None,
             _get(dp_b, "relative_error_mean") if dp_b else None),
            ("P90 rel error",
             _get(dp_a, "relative_error_p90") if dp_a else None,
             _get(dp_b, "relative_error_p90") if dp_b else None),
            ("Max rel error",
             _get(dp_a, "relative_error_max") if dp_a else None,
             _get(dp_b, "relative_error_max") if dp_b else None),
        ]
        print(f"{'Metric':<30} {label_a:>14} {label_b:>14} {'Change':>10}")
        print("-" * 70)
        for name, va, vb in dp_rows:
            sa = _format_val(va)
            sb = _format_val(vb)
            sc = _pct_change(va, vb) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else ""
            print(f"{name:<30} {sa:>14} {sb:>14} {sc:>10}")

        # Show worst delta_p cases for latest
        dp_worst = _get(dp_b, "worst_cases") if dp_b else None
        if dp_worst:
            print(f"\n  Worst 5 delta-p cases ({label_b}):")
            for c in dp_worst[:5]:
                print(
                    f"    {c['case']}: "
                    f"gt={c['delta_p_gt']:.2f} Pa, "
                    f"pred={c['delta_p_pred']:.2f} Pa, "
                    f"rel_err={c['relative_error']:.1%}"
                )

    print()


def _format_val(val: Any) -> str:
    if val is None:
        return "---"
    if isinstance(val, float):
        if abs(val) < 0.01 and val != 0:
            return f"{val:.4e}"
        if abs(val) > 100:
            return f"{val:.1f}"
        return f"{val:.4f}"
    return str(val)


def generate_comparison_plots(
    summaries: list[VersionSummary],
    save_path: str | None = None,
) -> None:
    """Generate comparison plots across HPO versions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("HPO Version Comparison", fontsize=14, fontweight="bold")

    # 1. Best validation loss across versions
    ax = axes[0, 0]
    versions = [s.version for s in summaries if s.best_trial]
    best_vals = [s.best_trial.value for s in summaries if s.best_trial]
    bars = ax.bar(versions, best_vals, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Best Validation Loss by Version")
    for bar, val in zip(bars, best_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(best_vals) * 1.15)

    # 2. Trial distribution (box plot of completed trial values)
    ax = axes[0, 1]
    data_for_box = [s.all_completed_values for s in summaries if s.all_completed_values]
    labels_for_box = [s.version for s in summaries if s.all_completed_values]
    if data_for_box:
        bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightsteelblue")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Trial Value Distribution")

    # 3. Trials completed vs pruned
    ax = axes[1, 0]
    x_pos = np.arange(len(summaries))
    width = 0.35
    ax.bar(x_pos - width/2, [s.n_complete for s in summaries], width,
           label="Complete", color="steelblue", alpha=0.8)
    ax.bar(x_pos + width/2, [s.n_pruned for s in summaries], width,
           label="Pruned", color="coral", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.version for s in summaries])
    ax.set_ylabel("Count")
    ax.set_title("Trial Outcomes")
    ax.legend()

    # 4. Convergence: running best across versions that have multiple trials
    ax = axes[1, 1]
    for s in summaries:
        if len(s.all_completed_values) > 1:
            running_best = []
            current_best = float("inf")
            for v in s.all_completed_values:
                current_best = min(current_best, v)
                running_best.append(current_best)
            ax.plot(range(1, len(running_best) + 1), running_best, marker=".", label=s.version)
    ax.set_xlabel("Completed Trial #")
    ax.set_ylabel("Running Best Loss")
    ax.set_title("Convergence per Version")
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")
    else:
        default_path = HPO_DIR / "version_comparison.png"
        fig.savefig(str(default_path), dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to {default_path}")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare HPO training versions")
    parser.add_argument(
        "--versions", nargs="*", default=None,
        help="Specific versions to compare (e.g. v6 v7). Default: all found.",
    )
    parser.add_argument(
        "--save", default=None,
        help="Path to save comparison plot (default: data/hpo/version_comparison.png).",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation (useful without matplotlib).",
    )
    parser.add_argument(
        "--eval-a", default=None,
        help="Path to eval_metrics.json for baseline (default: data/models_v1/eval_metrics.json).",
    )
    parser.add_argument(
        "--eval-b", default=None,
        help="Path to eval_metrics.json for latest (default: data/models/eval_metrics.json).",
    )
    args = parser.parse_args(argv)

    all_dbs = _discover_dbs()
    if not all_dbs:
        print("No HPO databases found.", file=sys.stderr)
        sys.exit(1)

    if args.versions:
        selected = {v: all_dbs[v] for v in args.versions if v in all_dbs}
        missing = [v for v in args.versions if v not in all_dbs]
        if missing:
            print(f"Warning: versions not found: {missing}", file=sys.stderr)
    else:
        selected = all_dbs

    # Load summaries
    summaries = []
    for version, db_path in selected.items():
        s = load_version(version, db_path)
        summaries.append(s)

    if not summaries:
        print("No valid versions loaded.", file=sys.stderr)
        sys.exit(1)

    # Print HPO comparison
    print_hpo_comparison(summaries)
    print_best_params_comparison(summaries)

    # Load and compare evaluation metrics
    eval_a_path = Path(args.eval_a) if args.eval_a else KNOWN_EVAL_METRICS.get("v1")
    eval_b_path = Path(args.eval_b) if args.eval_b else KNOWN_EVAL_METRICS.get("latest")

    eval_a = load_eval_metrics(eval_a_path) if eval_a_path else None
    eval_b = load_eval_metrics(eval_b_path) if eval_b_path else None

    label_a = args.eval_a or "v1"
    label_b = args.eval_b or "latest (v7)"
    print_eval_comparison(eval_a, eval_b, label_a, label_b)

    # Generate plots
    if not args.no_plot:
        try:
            generate_comparison_plots(summaries, save_path=args.save)
        except ImportError:
            print("matplotlib not available; skipping plots. Use --no-plot to suppress.", file=sys.stderr)


if __name__ == "__main__":
    main()
