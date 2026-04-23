"""Feature analysis + selection entry point for the alpha_D surrogate.

Runs six sklearn-based feature-selection methods with leak-safe
GroupKFold CV (grouped by case), aggregates them via Borda count,
respects atomic grouped-feature blocks, and retrains ridge / GBR
baselines on the selected feature subset.

Writes artifacts to ``output.dir``:
  - report.json      per-method rankings, consensus, baseline metrics
  - manifest.json    config, data hash, lib versions, git SHA (reproducibility)
  - selected_features.txt  one feature name per line (for copy into training configs)
  - *.png            optional plots when matplotlib is available

Usage (from src/ directory):
    python run_feature_analysis.py
    python run_feature_analysis.py top_k=8
    python run_feature_analysis.py methods='[f_regression,lasso,gbr_permutation]'
"""

import json
import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(__file__))

from feature_analysis import build_manifest, load_feature_matrix, write_manifest
from feature_analysis.methods import (
    borda_consensus,
    build_report,
    collapse_blocks_to_selection,
    run_baseline,
    run_methods,
)
from feature_analysis.plotting import save_feature_analysis_plots

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="feature_analysis/config",
    config_name="feature_analysis",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    data_cfg = cfg_dict["data"]
    output_cfg = cfg_dict["output"]
    cv_cfg = cfg_dict["cv"]
    grouped_cfg = cfg_dict.get("grouped_features", {}) or {}
    methods = list(cfg_dict.get("methods", []))
    top_k = int(cfg_dict.get("top_k", 6))
    mi_cfg = cfg_dict.get("mi", {}) or {}
    consensus_cfg = cfg_dict.get("consensus", {}) or {}
    stability_min = float(consensus_cfg.get("stability_min", 0.6))
    baseline_cfg = cfg_dict.get("baseline", {}) or {}

    if str(consensus_cfg.get("method", "borda")) != "borda":
        raise ValueError(
            f"consensus.method={consensus_cfg.get('method')!r} not supported. "
            "Only 'borda' is implemented."
        )

    # --- Load ---
    data = load_feature_matrix(
        zarr_dir=data_cfg["zarr_dir"],
        target=data_cfg.get("target", "log_alpha_D"),
        selected_from_allowlist=data_cfg.get("selected_from_allowlist"),
        local_velocity_normalization=bool(data_cfg.get("local_velocity_normalization", True)),
        min_Dr=data_cfg.get("min_Dr"),
        exclude_cases=data_cfg.get("exclude_cases") or [],
    )
    logger.info(
        "Loaded %d rows across %d cases; %d features: %s",
        data.X.shape[0], data.n_cases, data.X.shape[1], data.feature_names,
    )

    # --- Per-method CV ---
    method_results = run_methods(
        data,
        methods=methods,
        cv_cfg=cv_cfg,
        top_k=top_k,
        mi_cfg=mi_cfg,
    )

    # --- Consensus ---
    borda = borda_consensus(method_results, data.feature_names)
    consensus = collapse_blocks_to_selection(
        feature_names=data.feature_names,
        borda=borda,
        method_results=method_results,
        grouped_cfg=grouped_cfg,
        top_k=top_k,
        stability_min=stability_min,
    )
    logger.info("Selected features: %s", consensus["selected"])

    # --- Baseline retrain on selected subset ---
    baseline: dict = {}
    if baseline_cfg.get("enabled", False):
        baseline = run_baseline(
            data,
            selected_features=consensus["selected"],
            models=list(baseline_cfg.get("models", ["ridge", "gbr"])),
            n_splits=int(baseline_cfg.get("n_splits", cv_cfg.get("n_splits", 5))),
            seed=int(cv_cfg.get("seed", 42)),
        )
        for m, stats in baseline.get("per_model", {}).items():
            if "r2_mean" in stats:
                logger.info(
                    "baseline %s: R²=%.4f±%.4f, RMSE=%.4e±%.4e",
                    m, stats["r2_mean"], stats["r2_std"],
                    stats["rmse_mean"], stats["rmse_std"],
                )

    # --- Write artifacts ---
    out_dir = Path(output_cfg["dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(
        data=data,
        method_results=method_results,
        consensus=consensus,
        baseline=baseline,
        top_k=top_k,
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=False))

    manifest = build_manifest(
        config=cfg_dict,
        zarr_dir=data_cfg["zarr_dir"],
        feature_names=data.feature_names,
        target_name=data.target_name,
        n_rows=int(data.X.shape[0]),
        n_cases=data.n_cases,
        seeds={"cv": int(cv_cfg.get("seed", 42))},
    )
    write_manifest(manifest, out_dir)

    (out_dir / "selected_features.txt").write_text(
        "\n".join(consensus["selected"]) + "\n"
    )

    plot_cfg = dict(output_cfg.get("plots") or {})
    if bool(plot_cfg.get("enabled", True)):
        plot_files = save_feature_analysis_plots(
            report,
            out_dir,
            data=data,
            top_n=int(plot_cfg.get("top_n", 12)),
            relationship_top_n=int(plot_cfg.get("relationship_top_n", 6)),
            sample_max=int(plot_cfg.get("sample_max", 5000)),
            dpi=int(plot_cfg.get("dpi", 150)),
        )
        if plot_files:
            logger.info("Saved %d plot(s) to %s", len(plot_files), out_dir)

    print(f"\nWrote feature-analysis artifacts to {out_dir}")


if __name__ == "__main__":
    main()
