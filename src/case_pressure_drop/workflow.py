"""Train and evaluate case-level pressure-drop regressors."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from case_pressure_drop.data import CANDIDATE_FEATURES, CasePressureDropDataset
from case_pressure_drop.feature_selection import run_feature_selection
from case_pressure_drop.modeling import (
    compute_metrics,
    cross_validate_models,
    fit_and_save_models,
    inverse_transform_target,
    load_saved_model,
)
from case_pressure_drop.plotting import save_prediction_plots


def to_plain_dict(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            plain = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(plain, dict):
                return plain
    except ModuleNotFoundError:
        pass
    raise TypeError(f"Expected dict-like config, got {type(cfg)}")


def _resolve_path(raw_path: str | Path) -> Path:
    return Path(raw_path).expanduser().resolve()


def _normalize_split_cfg(split_cfg: dict[str, Any], default_seed: int) -> dict[str, Any]:
    normalized = dict(split_cfg)
    normalized.setdefault("strategy", "stratified")
    if normalized["strategy"] in {"sequential", "random", "stratified"}:
        normalized.setdefault("train_ratio", 0.8)
    if normalized["strategy"] in {"random", "stratified"}:
        normalized.setdefault("seed", default_seed)
    return normalized


def _stratified_split(
    sim_names: list[str],
    train_ratio: float,
    seed: int,
    n_bins: int = 3,
) -> tuple[list[int], list[int]]:
    import random
    import re
    from collections import defaultdict

    def _parse_case_params(sim_name: str) -> dict[str, float]:
        params: dict[str, float] = {}
        for key in ("Re", "Dr", "Lr"):
            match = re.search(rf"{key}_([0-9p]+)", sim_name)
            if not match:
                params[key] = 0.0
                continue
            try:
                params[key] = float(match.group(1).replace("p", "."))
            except ValueError:
                params[key] = 0.0
        return params

    def _quantile_bin(values: list[float], n: int) -> list[int]:
        sorted_unique = sorted(set(values))
        if len(sorted_unique) <= n:
            mapping = {value: idx for idx, value in enumerate(sorted_unique)}
            return [mapping[value] for value in values]
        edges = [sorted_unique[int(len(sorted_unique) * i / n)] for i in range(n)]
        bins: list[int] = []
        for value in values:
            bucket = n - 1
            for idx in range(1, n):
                if value < edges[idx]:
                    bucket = idx - 1
                    break
            bins.append(bucket)
        return bins

    parsed = [_parse_case_params(name) for name in sim_names]
    re_vals = [row["Re"] for row in parsed]
    dr_vals = [row["Dr"] for row in parsed]
    lr_vals = [row["Lr"] for row in parsed]

    re_bins = _quantile_bin(re_vals, n_bins)
    dr_bins = _quantile_bin(dr_vals, n_bins)
    lr_bins = _quantile_bin(lr_vals, n_bins)

    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx in range(len(sim_names)):
        grouped[(re_bins[idx], dr_bins[idx], lr_bins[idx])].append(idx)

    rng = random.Random(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for _, indices in sorted(grouped.items()):
        rng.shuffle(indices)
        n_train = max(1, min(len(indices) - 1, round(len(indices) * train_ratio)))
        if len(indices) == 1:
            train_idx.extend(indices)
        else:
            train_idx.extend(indices[:n_train])
            test_idx.extend(indices[n_train:])

    if not test_idx:
        rng.shuffle(train_idx)
        n_train = max(1, round(len(train_idx) * train_ratio))
        test_idx = train_idx[n_train:]
        train_idx = train_idx[:n_train]

    return sorted(train_idx), sorted(test_idx)


def split_case_indices(
    sim_names: list[str],
    split_cfg: dict[str, Any],
) -> tuple[list[int], list[int], list[str], list[str]]:
    import random

    if len(sim_names) < 2:
        raise ValueError("Need at least 2 cases to create a train/test split.")

    strategy = str(split_cfg.get("strategy", "stratified"))
    if strategy in {"sequential", "random"}:
        train_ratio = float(split_cfg.get("train_ratio", 0.8))
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1.")
        n_train = max(1, min(len(sim_names) - 1, int(len(sim_names) * train_ratio)))
        indices = list(range(len(sim_names)))
        if strategy == "random":
            rng = random.Random(int(split_cfg.get("seed", 42)))
            rng.shuffle(indices)
            train_idx = sorted(indices[:n_train])
            test_idx = sorted(indices[n_train:])
        else:
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
    elif strategy == "stratified":
        train_idx, test_idx = _stratified_split(
            sim_names,
            train_ratio=float(split_cfg.get("train_ratio", 0.8)),
            seed=int(split_cfg.get("seed", 42)),
            n_bins=int(split_cfg.get("n_bins", 3)),
        )
    else:
        raise ValueError(
            "split.strategy must be one of {'sequential', 'random', 'stratified'}."
        )

    train_sims = [sim_names[idx] for idx in train_idx]
    test_sims = [sim_names[idx] for idx in test_idx]
    return train_idx, test_idx, train_sims, test_sims


def _comparison_table(metrics: dict[str, dict[str, float]], best_model_name: str) -> str:
    header = (
        "| Model | RMSE [Pa] | MAE [Pa] | R² | MAPE | "
        "Median Abs Rel Err | Max Abs Rel Err |"
    )
    divider = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, divider]
    for model_name, row in sorted(
        metrics.items(),
        key=lambda item: item[1]["rmse_pa"],
    ):
        label = f"**{model_name}**" if model_name == best_model_name else model_name
        lines.append(
            "| "
            f"{label} | {row['rmse_pa']:.6f} | {row['mae_pa']:.6f} | "
            f"{row['r2_pa']:.6f} | {row['mape']:.6f} | "
            f"{row.get('abs_relative_error_median', float('nan')):.6f} | "
            f"{row.get('abs_relative_error_max', float('nan')):.6f} |"
        )
    return "\n".join(lines) + "\n"


def _compute_relative_error_summary(
    y_true_pa,
    predictions_for_plots: dict[str, Any],
) -> dict[str, dict[str, float]]:
    import numpy as np

    y_true = np.asarray(y_true_pa, dtype=np.float64)
    summary: dict[str, dict[str, float]] = {}
    denom = np.clip(np.abs(y_true), 1e-8, None)

    for model_name, pred_values in predictions_for_plots.items():
        pred = np.asarray(pred_values, dtype=np.float64)
        signed = (pred - y_true) / denom
        abs_rel = np.abs(signed)
        summary[model_name] = {
            "relative_error_signed_min": float(np.min(signed)),
            "relative_error_signed_max": float(np.max(signed)),
            "abs_relative_error_median": float(np.median(abs_rel)),
            "abs_relative_error_p90": float(np.quantile(abs_rel, 0.9)),
            "abs_relative_error_max": float(np.max(abs_rel)),
        }
    return summary


def _best_worst_cases_by_model(
    per_case_rows: list[dict[str, Any]],
    *,
    model_names: list[str],
    top_n: int = 5,
) -> dict[str, dict[str, list[dict[str, float | str]]]]:
    out: dict[str, dict[str, list[dict[str, float | str]]]] = {}
    for model_name in model_names:
        entries: list[dict[str, float | str]] = []
        pred_key = f"{model_name}_pred"
        for row in per_case_rows:
            true = float(row["delta_p_true"])
            pred = float(row[pred_key])
            signed_rel = (pred - true) / max(abs(true), 1e-8)
            entries.append(
                {
                    "case": str(row["case"]),
                    "delta_p_true": true,
                    "delta_p_pred": pred,
                    "signed_relative_error": float(signed_rel),
                    "abs_relative_error": float(abs(signed_rel)),
                    "abs_error_pa": float(abs(pred - true)),
                }
            )
        entries.sort(key=lambda item: float(item["abs_relative_error"]))
        out[model_name] = {
            "best_cases": entries[:top_n],
            "worst_cases": list(reversed(entries[-top_n:])),
        }
    return out


def _print_evaluation_summary(
    *,
    run_meta: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    best_model_name: str,
    n_test_cases: int,
    case_rankings: dict[str, dict[str, list[dict[str, float | str]]]],
) -> None:
    """Render the evaluation summary, using ``rich`` when available."""
    try:
        _print_evaluation_summary_rich(
            run_meta=run_meta,
            metrics=metrics,
            best_model_name=best_model_name,
            n_test_cases=n_test_cases,
            case_rankings=case_rankings,
        )
    except ImportError:
        _print_evaluation_summary_plain(
            run_meta=run_meta,
            metrics=metrics,
            best_model_name=best_model_name,
            n_test_cases=n_test_cases,
            case_rankings=case_rankings,
        )


def _fmt_pa(value: float) -> str:
    """Compact human-friendly formatter for pressures in Pa."""
    a = abs(value)
    if a >= 1_000_000:
        return f"{value / 1_000_000:.2f} MPa"
    if a >= 1_000:
        return f"{value / 1_000:.2f} kPa"
    if a >= 1:
        return f"{value:.2f} Pa"
    return f"{value:.3f} Pa"


def _print_evaluation_summary_rich(
    *,
    run_meta: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    best_model_name: str,
    n_test_cases: int,
    case_rankings: dict[str, dict[str, list[dict[str, float | str]]]],
) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console(width=120)

    selected = run_meta["data"].get("selected_features", [])
    header_lines = [
        f"[bold]Test cases:[/bold] {n_test_cases}",
        f"[bold]Best model on test:[/bold] [green]{best_model_name}[/green] ★",
        f"[bold]Selected features:[/bold] "
        + (", ".join(f"[cyan]{name}[/cyan]" for name in selected) or "[dim]n/a[/dim]"),
    ]
    console.print(
        Panel(
            "\n".join(header_lines),
            title="[bold bright_blue]Case Pressure-Drop Evaluation[/bold bright_blue]",
            border_style="bright_blue",
            expand=False,
        )
    )

    # Model comparison table
    table = Table(
        title="[bold]Model comparison[/bold]  (sorted by RMSE)",
        show_lines=False,
        header_style="bold magenta",
        padding=(0, 1),
    )
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("RMSE", justify="right", no_wrap=True)
    table.add_column("MAE", justify="right", no_wrap=True)
    table.add_column("R²", justify="right", no_wrap=True)
    table.add_column("MAPE", justify="right", no_wrap=True)
    table.add_column("Median\n|rel|", justify="right", no_wrap=True)
    table.add_column("P90\n|rel|", justify="right", no_wrap=True)
    table.add_column("Max\n|rel|", justify="right", no_wrap=True)

    sorted_items = sorted(metrics.items(), key=lambda item: item[1]["rmse_pa"])
    for model_name, row in sorted_items:
        is_best = model_name == best_model_name
        label = Text(model_name + (" ★" if is_best else ""))
        if is_best:
            label.stylize("bold green")
        r2 = row["r2_pa"]
        r2_text = Text(f"{r2:.4f}")
        if r2 >= 0.9:
            r2_text.stylize("green")
        elif r2 >= 0.5:
            r2_text.stylize("yellow")
        else:
            r2_text.stylize("red")
        table.add_row(
            label,
            _fmt_pa(row["rmse_pa"]),
            _fmt_pa(row["mae_pa"]),
            r2_text,
            f"{100.0 * row['mape']:.2f}%",
            f"{100.0 * row.get('abs_relative_error_median', float('nan')):.2f}%",
            f"{100.0 * row.get('abs_relative_error_p90', float('nan')):.2f}%",
            f"{100.0 * row.get('abs_relative_error_max', float('nan')):.2f}%",
        )
    console.print(table)

    # Per-model best/worst case tables
    for model_name in sorted(case_rankings):
        ranking = case_rankings[model_name]
        marker = " ★" if model_name == best_model_name else ""
        model_style = "bold green" if model_name == best_model_name else "bold cyan"

        best_table = Table(
            title=f"[{model_style}]{model_name}{marker}[/{model_style}]  —  Best 3 cases",
            header_style="bold magenta",
            show_lines=False,
            padding=(0, 1),
        )
        best_table.add_column("Case", style="cyan", no_wrap=True)
        best_table.add_column("True", justify="right", no_wrap=True)
        best_table.add_column("Pred", justify="right", no_wrap=True)
        best_table.add_column("Abs Rel Err", justify="right", style="green", no_wrap=True)
        for row in ranking["best_cases"][:3]:
            best_table.add_row(
                str(row["case"]),
                _fmt_pa(float(row["delta_p_true"])),
                _fmt_pa(float(row["delta_p_pred"])),
                f"{100.0 * float(row['abs_relative_error']):.2f}%",
            )

        worst_table = Table(
            title=f"[{model_style}]{model_name}{marker}[/{model_style}]  —  Worst 3 cases",
            header_style="bold magenta",
            show_lines=False,
            padding=(0, 1),
        )
        worst_table.add_column("Case", style="cyan", no_wrap=True)
        worst_table.add_column("True", justify="right", no_wrap=True)
        worst_table.add_column("Pred", justify="right", no_wrap=True)
        worst_table.add_column("Signed Rel Err", justify="right", no_wrap=True)
        worst_table.add_column("Abs Rel Err", justify="right", style="red", no_wrap=True)
        for row in ranking["worst_cases"][:3]:
            signed = 100.0 * float(row["signed_relative_error"])
            signed_text = Text(f"{signed:+.2f}%")
            signed_text.stylize("red" if abs(signed) > 50 else "yellow")
            worst_table.add_row(
                str(row["case"]),
                _fmt_pa(float(row["delta_p_true"])),
                _fmt_pa(float(row["delta_p_pred"])),
                signed_text,
                f"{100.0 * float(row['abs_relative_error']):.2f}%",
            )

        console.print(best_table)
        console.print(worst_table)


def _print_evaluation_summary_plain(
    *,
    run_meta: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    best_model_name: str,
    n_test_cases: int,
    case_rankings: dict[str, dict[str, list[dict[str, float | str]]]],
) -> None:
    selected = ", ".join(run_meta["data"].get("selected_features", [])) or "n/a"
    print(f"Selected features: {selected}")
    print(f"Test cases: {n_test_cases}")
    print("Model comparison (sorted by RMSE):")
    for model_name, row in sorted(metrics.items(), key=lambda item: item[1]["rmse_pa"]):
        marker = "*" if model_name == best_model_name else " "
        print(
            f"{marker} {model_name}: "
            f"RMSE={row['rmse_pa']:.3f} Pa, "
            f"MAE={row['mae_pa']:.3f} Pa, "
            f"R2={row['r2_pa']:.4f}, "
            f"MAPE={100.0 * row['mape']:.2f}%, "
            f"median_abs_rel={100.0 * row.get('abs_relative_error_median', float('nan')):.2f}%, "
            f"p90_abs_rel={100.0 * row.get('abs_relative_error_p90', float('nan')):.2f}%, "
            f"max_abs_rel={100.0 * row.get('abs_relative_error_max', float('nan')):.2f}%"
        )
    print(f"Best model on test: {best_model_name}")
    for model_name, ranking in case_rankings.items():
        print(f"Best 3 cases for {model_name}:")
        for row in ranking["best_cases"][:3]:
            print(
                f"  {row['case']}: "
                f"true={row['delta_p_true']:.3f} Pa, "
                f"pred={row['delta_p_pred']:.3f} Pa, "
                f"abs_rel={100.0 * row['abs_relative_error']:.2f}%"
            )
        print(f"Worst 3 cases for {model_name}:")
        for row in ranking["worst_cases"][:3]:
            print(
                f"  {row['case']}: "
                f"true={row['delta_p_true']:.3f} Pa, "
                f"pred={row['delta_p_pred']:.3f} Pa, "
                f"signed_rel={100.0 * row['signed_relative_error']:.2f}%, "
                f"abs_rel={100.0 * row['abs_relative_error']:.2f}%"
            )


def _evaluation_summary_markdown(
    *,
    run_meta: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    best_model_name: str,
    n_test_cases: int,
    case_rankings: dict[str, dict[str, list[dict[str, float | str]]]],
) -> str:
    selected = ", ".join(run_meta["data"].get("selected_features", [])) or "n/a"
    lines = [
        "# Case Pressure-Drop Evaluation",
        "",
        f"- Test cases: {n_test_cases}",
        f"- Selected features: `{selected}`",
        f"- Best model on test: `{best_model_name}`",
        "- Relative error columns are fractions, not percentages.",
        "",
        _comparison_table(metrics, best_model_name).rstrip(),
        "",
    ]
    for model_name in sorted(case_rankings):
        lines.extend(
            [
                f"## {model_name}",
                "",
                "Best cases:",
                "",
                "| Case | True [Pa] | Pred [Pa] | Abs Rel Err |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in case_rankings[model_name]["best_cases"]:
            lines.append(
                f"| {row['case']} | {row['delta_p_true']:.6f} | "
                f"{row['delta_p_pred']:.6f} | {row['abs_relative_error']:.6f} |"
            )
        lines.extend(
            [
                "",
                "Worst cases:",
                "",
                "| Case | True [Pa] | Pred [Pa] | Signed Rel Err | Abs Rel Err |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in case_rankings[model_name]["worst_cases"]:
            lines.append(
                f"| {row['case']} | {row['delta_p_true']:.6f} | "
                f"{row['delta_p_pred']:.6f} | {row['signed_relative_error']:.6f} | "
                f"{row['abs_relative_error']:.6f} |"
            )
        lines.extend(["", ""])
    return "\n".join(lines) + "\n"


def train_case_pressure_drop(cfg: dict | Any) -> dict[str, Any]:
    cfg_dict = to_plain_dict(cfg)
    data_cfg = dict(cfg_dict.get("data") or {})
    feature_cfg = dict(cfg_dict.get("feature_selection") or {})
    models_cfg = dict(cfg_dict.get("models") or {})
    output_cfg = dict(cfg_dict.get("output") or {})

    if not data_cfg.get("zarr_dir"):
        raise ValueError("data.zarr_dir is required.")

    min_Dr = data_cfg.get("min_Dr")
    dataset = CasePressureDropDataset.from_zarr_dir(
        data_cfg["zarr_dir"],
        exclude_cases=[str(name) for name in data_cfg.get("exclude_cases", [])],
        min_Dr=float(min_Dr) if min_Dr is not None else None,
    )

    split_seed = int(
        (data_cfg.get("split") or {}).get(
            "seed",
            feature_cfg.get("seed", (models_cfg.get("cv") or {}).get("seed", 42)),
        )
    )
    split_cfg = _normalize_split_cfg(dict(data_cfg.get("split") or {}), split_seed)
    train_idx, test_idx, train_sims, test_sims = split_case_indices(
        dataset.sim_names,
        split_cfg,
    )
    train_dataset = dataset.subset_by_case_indices(train_idx)

    case_dir = _resolve_path(str(output_cfg.get("case_dir") or "./case_pressure_drop"))
    feature_dir = _resolve_path(
        str(output_cfg.get("feature_selection_dir") or (case_dir / "feature_selection"))
    )
    model_dir = _resolve_path(str(output_cfg.get("model_dir") or (case_dir / "models")))
    run_meta_path = _resolve_path(str(output_cfg.get("run_meta") or (case_dir / "run_meta.json")))

    case_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    candidate_features = list(feature_cfg.get("candidate_features") or CANDIDATE_FEATURES)
    if bool(feature_cfg.get("enabled", True)):
        _rt = feature_cfg.get("redundancy_threshold", 0.95)
        fs_result = run_feature_selection(
            train_dataset,
            feature_names=candidate_features,
            methods=list(feature_cfg.get("methods") or []),
            top_k=int(feature_cfg.get("top_k", 3)),
            n_splits=int(feature_cfg.get("n_splits", 5)),
            seed=int(feature_cfg.get("seed", 42)),
            stability_min=float(feature_cfg.get("stability_min", 0.5)),
            mutual_info_n_seeds=int(feature_cfg.get("mutual_info_n_seeds", 10)),
            output_dir=feature_dir,
            config=cfg_dict,
            redundancy_threshold=float(_rt) if _rt is not None else None,
        )
        selected_features = list(fs_result.selected_features)
    else:
        selected_features = list(candidate_features)
        selected_path = feature_dir / "selected_features.txt"
        selected_path.write_text("\n".join(selected_features) + "\n", encoding="utf-8")
        fs_result = None

    cv_results = cross_validate_models(
        train_dataset,
        feature_names=selected_features,
        model_cfg=models_cfg,
        cv_cfg=dict(models_cfg.get("cv") or {}),
    )
    best_model_name = min(
        cv_results,
        key=lambda name: cv_results[name]["cv"]["rmse_pa_mean"],
    )

    saved_models = fit_and_save_models(
        train_dataset,
        feature_names=selected_features,
        model_cfg=models_cfg,
        seed=int((models_cfg.get("cv") or {}).get("seed", 42)),
        model_dir=model_dir,
    )

    run_meta = {
        "task": "case_pressure_drop_regression",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(case_dir),
        "data": {
            "zarr_dir": str(dataset.zarr_dir),
            "exclude_cases": [str(name) for name in data_cfg.get("exclude_cases", [])],
            "min_Dr": float(min_Dr) if min_Dr is not None else None,
            "target_name": "delta_p_case",
            "target_transform": "log1p",
            "candidate_features": candidate_features,
            "selected_features": selected_features,
            "n_total_cases": len(dataset),
            "n_train_cases": len(train_sims),
            "n_test_cases": len(test_sims),
        },
        "split": {
            "strategy": split_cfg["strategy"],
            "train_ratio": float(split_cfg.get("train_ratio", 0.8)),
            "seed": int(split_cfg.get("seed", split_seed)),
            "n_bins": int(split_cfg.get("n_bins", 3)),
            "train_sims": train_sims,
            "test_sims": test_sims,
        },
        "feature_selection": {
            "enabled": bool(feature_cfg.get("enabled", True)),
            "methods": list(feature_cfg.get("methods") or []),
            "top_k": int(feature_cfg.get("top_k", 3)),
            "n_splits": int(feature_cfg.get("n_splits", 5)),
            "seed": int(feature_cfg.get("seed", 42)),
            "stability_min": float(feature_cfg.get("stability_min", 0.5)),
            "candidate_features": candidate_features,
            "selected_features": selected_features,
            "report_path": str(fs_result.report_path) if fs_result else None,
            "manifest_path": str(fs_result.manifest_path) if fs_result else None,
            "selected_features_path": (
                str(fs_result.selected_features_path)
                if fs_result
                else str(feature_dir / "selected_features.txt")
            ),
            "case_ids_used": list(train_sims),
        },
        "models": {
            model_name: {
                **cv_results[model_name],
                **saved_models[model_name],
            }
            for model_name in ("linear_regression", "random_forest", "mlp")
        },
        "best_model": {
            "name": best_model_name,
            "selection_metric": "cv.rmse_pa_mean",
            "selection_value": float(cv_results[best_model_name]["cv"]["rmse_pa_mean"]),
        },
    }
    run_meta_path.parent.mkdir(parents=True, exist_ok=True)
    run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(
        f"Trained case-level pressure-drop models on {len(train_sims)} train case(s); "
        f"selected features: {selected_features}. Best CV model: {best_model_name}."
    )
    print(f"Saved run metadata to {run_meta_path}")

    return {
        "run_meta": str(run_meta_path),
        "output_dir": str(case_dir),
        "best_model": best_model_name,
        "selected_features": selected_features,
        "train_cases": len(train_sims),
        "test_cases": len(test_sims),
    }


def evaluate_case_pressure_drop(cfg: dict | Any) -> dict[str, Any]:
    cfg_dict = to_plain_dict(cfg)
    eval_cfg = dict(cfg_dict.get("eval") or {})
    output_cfg = dict(cfg_dict.get("output") or {})

    run_meta_value = eval_cfg.get("run_meta")
    if run_meta_value:
        run_meta_path = _resolve_path(str(run_meta_value))
    elif eval_cfg.get("run_dir"):
        run_meta_path = _resolve_path(str(eval_cfg["run_dir"])) / "run_meta.json"
    else:
        default_case_dir = output_cfg.get("case_dir")
        if not default_case_dir:
            raise ValueError("Set eval.run_meta or eval.run_dir for evaluation.")
        run_meta_path = _resolve_path(str(default_case_dir)) / "run_meta.json"

    if not run_meta_path.exists():
        raise FileNotFoundError(f"run_meta.json not found: {run_meta_path}")

    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    run_dir = run_meta_path.parent

    rm_min_Dr = run_meta["data"].get("min_Dr")
    dataset = CasePressureDropDataset.from_zarr_dir(
        run_meta["data"]["zarr_dir"],
        exclude_cases=[str(name) for name in run_meta["data"].get("exclude_cases", [])],
        min_Dr=float(rm_min_Dr) if rm_min_Dr is not None else None,
    )
    test_dataset = dataset.subset_by_case_names(list(run_meta["split"]["test_sims"]))

    y_true_pa = test_dataset.delta_p_case
    y_true_log = test_dataset.target_log1p()

    per_model: dict[str, Any] = {}
    predictions_for_plots: dict[str, Any] = {}
    per_case_rows: list[dict[str, Any]] = [
        {
            "case": case_name,
            "delta_p_true": float(y_true_pa[idx]),
        }
        for idx, case_name in enumerate(test_dataset.sim_names)
    ]

    for model_name, model_meta in run_meta["models"].items():
        artifact = load_saved_model(model_meta["artifact"])
        estimator = artifact["estimator"]
        feature_names = list(artifact.get("feature_names") or run_meta["data"]["selected_features"])
        X_test = test_dataset.build_feature_matrix(feature_names)
        pred_log = estimator.predict(X_test)
        pred_pa = inverse_transform_target(pred_log)
        predictions_for_plots[model_name] = pred_pa
        per_model[model_name] = {
            **compute_metrics(
                y_true_pa,
                pred_pa,
                y_true_log=y_true_log,
                y_pred_log=pred_log,
            ),
            "artifact": model_meta["artifact"],
            "feature_names": feature_names,
        }
        for idx, row in enumerate(per_case_rows):
            row[f"{model_name}_pred"] = float(pred_pa[idx])

    best_model_name = min(per_model, key=lambda name: per_model[name]["rmse_pa"])
    relative_summary = _compute_relative_error_summary(y_true_pa, predictions_for_plots)
    for model_name, rel_stats in relative_summary.items():
        per_model[model_name].update(rel_stats)
    case_rankings = _best_worst_cases_by_model(
        per_case_rows,
        model_names=list(per_model.keys()),
        top_n=5,
    )

    metrics_payload = {
        "task": "case_pressure_drop_regression",
        "run_meta": str(run_meta_path),
        "output_dir": str(run_dir),
        "n_test_cases": len(test_dataset),
        "test_sims": list(test_dataset.sim_names),
        "best_model_on_test": best_model_name,
        "selected_features": list(run_meta["data"]["selected_features"]),
        "per_model": per_model,
        "per_case_predictions": per_case_rows,
        "case_rankings": case_rankings,
    }

    metrics_out_value = output_cfg.get("metrics_out", "auto")
    if str(metrics_out_value).strip().lower() == "auto":
        metrics_out_path = run_dir / "eval_metrics.json"
    else:
        metrics_out_path = _resolve_path(str(metrics_out_value))
    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_out_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    table_out_path = None
    if bool(eval_cfg.get("save_table", True)):
        table_out_value = output_cfg.get("table_out", "auto")
        if str(table_out_value).strip().lower() == "auto":
            table_out_path = run_dir / "model_comparison.md"
        else:
            table_out_path = _resolve_path(str(table_out_value))
        comparison = _evaluation_summary_markdown(
            run_meta=run_meta,
            metrics=per_model,
            best_model_name=best_model_name,
            n_test_cases=len(test_dataset),
            case_rankings=case_rankings,
        )
        table_out_path.parent.mkdir(parents=True, exist_ok=True)
        table_out_path.write_text(comparison, encoding="utf-8")

    plot_dir_value = eval_cfg.get("plot_dir")
    plot_dir = _resolve_path(str(plot_dir_value)) if plot_dir_value else (run_dir / "plots")
    plot_files = []
    if bool(eval_cfg.get("save_plots", True)):
        try:
            plot_files = save_prediction_plots(
                y_true=y_true_pa,
                predictions=predictions_for_plots,
                output_dir=plot_dir,
                feature_names=list(run_meta["data"].get("selected_features", [])),
            )
        except OSError as exc:
            print(f"Skipping plot generation: {exc}")
            plot_files = []

    metrics_payload["artifacts"] = {
        "metrics_json": str(metrics_out_path),
        "comparison_table": str(table_out_path) if table_out_path is not None else None,
        "plots": plot_files,
    }
    metrics_out_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(
        f"Evaluated {len(per_model)} case-level models on {len(test_dataset)} test case(s). "
        f"Best test RMSE: {best_model_name}."
    )
    _print_evaluation_summary(
        run_meta=run_meta,
        metrics=per_model,
        best_model_name=best_model_name,
        n_test_cases=len(test_dataset),
        case_rankings=case_rankings,
    )
    print(f"Saved metrics to {metrics_out_path}")

    return metrics_payload
