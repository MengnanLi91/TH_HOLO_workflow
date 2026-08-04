"""Alpha-D-specific extended evaluation metrics.

The generic runner calls
:meth:`training.experiment.Experiment.compute_extended_metrics`, which
``AlphaDExperiment`` (in ``cases/alpha_d/experiment.py``) overrides to invoke
the helpers in this module. Base experiments inherit a no-op and skip this
module entirely, so the runner stays alpha-D-agnostic.

Three functions live here:

- :func:`compute_pointwise_extended_metrics` — R², physical-space, per-region,
  per-case error breakdown for the TabularPairDataset adapter.
- :func:`compute_delta_p_metrics` — trapezoidal integration of the predicted
  α_D profile per case and comparison vs the ground-truth ``delta_p_case``.
- :func:`print_extended_metrics` — human-readable summary used by ``evaluate()``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from cases.alpha_d.physics.targets import (
    alpha_d_values_to_bulk,
    field_values_to_physical,
    is_alpha_d_target,
)


def compute_delta_p_metrics(
    model: Any,
    eval_dataset,
    device: torch.device,
    *,
    alpha_d_target_name: str = "log_alpha_D",
    local_velocity_normalization: bool = False,
) -> dict[str, Any]:
    """Per-case Δp prediction error statistics.

    Integrates the predicted α_D profile via the trapezoidal rule to obtain
    ``delta_p_pred``, then compares with ``delta_p_case`` stored in the zarr
    metadata. Per-case geometry constants (``D_big``, ``outer_height_m``,
    ``buffer_diams``, ``rho``, ``V_bulk``) are read from each case's metadata
    and fall back to the historical AlphaD-ETL defaults when missing.
    """
    if not is_alpha_d_target(alpha_d_target_name):
        return {}

    case_meta = getattr(eval_dataset, "_case_meta", None)
    case_names = getattr(eval_dataset, "_case_ids_unique", None)
    row_case_idx = getattr(eval_dataset, "_row_case_idx", None)
    raw_z_hat = getattr(eval_dataset, "_raw_z_hat", None)
    raw_d_over_D = getattr(eval_dataset, "_raw_d_local_over_D", None)

    if case_meta is None or case_names is None or row_case_idx is None:
        return {}
    if raw_z_hat is None or raw_d_over_D is None:
        return {}

    per_case: list[dict[str, Any]] = []
    model.eval()

    # Profile-dataset wrappers expose _case_slices (per-case row indices
    # into the inner TabularPair, sorted by z_hat). When present, feed the
    # Conv1D model its expected [1, C, S] layout instead of the pointwise
    # [S, C] forward used by the MLP path.
    case_slices = getattr(eval_dataset, "_case_slices", None)
    profile_mode = case_slices is not None

    with torch.no_grad():
        for ci, case_name in enumerate(case_names):
            cm = case_meta[ci]
            delta_p_gt = float(cm.get("delta_p_case", 0.0))
            if delta_p_gt <= 0:
                continue

            D_big = float(cm.get("D_big", 0.2))
            outer_height_m = float(cm.get("outer_height_m", 1.0))
            buffer_diams = float(cm.get("buffer_diams", 1.0))
            rho = float(cm.get("rho", 1.0))
            V_bulk = float(cm.get("V_bulk", 1.0))

            if profile_mode:
                rows = torch.as_tensor(case_slices[ci], dtype=torch.long)
                x_case = eval_dataset._x[rows].T.unsqueeze(0).to(device)  # [1, C, S]
                z_hat = raw_z_hat[rows].to(device)
                d_over_D = raw_d_over_D[rows].to(device)
                pred_values = model(x_case)[0, 0]  # [n_stations]
                pred_values = eval_dataset.add_baseline_to_encoded(
                    pred_values,
                    row_mask=rows,
                    field_idx=0,
                )
            else:
                mask = row_case_idx == ci
                x_full = eval_dataset._x[mask].to(device)
                z_hat = raw_z_hat[mask].to(device)
                d_over_D = raw_d_over_D[mask].to(device)
                pred_values = model(x_full).squeeze(-1)  # [n_stations]
                pred_values = eval_dataset.add_baseline_to_encoded(
                    pred_values,
                    row_mask=mask,
                    field_idx=0,
                )
            alpha_D_bulk = alpha_d_values_to_bulk(
                pred_values,
                target_name=alpha_d_target_name,
                d_over_D=d_over_D,
                local_velocity_normalization=local_velocity_normalization,
            )

            # dp/dz = alpha_D_bulk * rho * V_bulk^2 / (2 * D_h)
            D_h = d_over_D * D_big
            dp_dz = alpha_D_bulk * rho * V_bulk**2 / (2.0 * D_h)

            # Trapezoidal integration over physical z
            L_roi = cm["Lr"] * outer_height_m + 2.0 * buffer_diams * D_big
            z_physical = z_hat * L_roi
            delta_p_pred = float(torch.trapezoid(dp_dz, z_physical).cpu())

            rel_err = abs(delta_p_pred - delta_p_gt) / abs(delta_p_gt)
            signed_error = delta_p_pred - delta_p_gt
            signed_rel_err = (delta_p_pred - delta_p_gt) / abs(delta_p_gt)
            log_err = abs(
                math.log(max(delta_p_pred, 1e-8)) - math.log(max(delta_p_gt, 1e-8))
            )

            per_case.append(
                {
                    "case": case_name,
                    "delta_p_gt": delta_p_gt,
                    "delta_p_pred": delta_p_pred,
                    "relative_error": rel_err,
                    "signed_error": signed_error,
                    "signed_relative_error": signed_rel_err,
                    "log_abs_error": log_err,
                    "Dr": float(cm.get("Dr", 0.0)),
                    "Lr": float(cm.get("Lr", 0.0)),
                    "Re": float(cm.get("Re", 0.0)),
                }
            )

    if not per_case:
        return {}

    rel_errors = [c["relative_error"] for c in per_case]
    log_errors = [c["log_abs_error"] for c in per_case]
    signed_errors = [c["signed_relative_error"] for c in per_case]
    lr_level_biases = {
        str(level): float(
            np.mean(
                [
                    case["signed_relative_error"]
                    for case in per_case
                    if case["Lr"] == level
                ]
            )
        )
        for level in sorted({case["Lr"] for case in per_case})
    }

    per_case.sort(key=lambda x: x["relative_error"], reverse=True)

    return {
        "n_cases": len(per_case),
        "relative_error_median": float(np.median(rel_errors)),
        "relative_error_mean": float(np.mean(rel_errors)),
        "relative_error_p90": float(np.quantile(rel_errors, 0.9)),
        "relative_error_max": float(max(rel_errors)),
        "log_abs_error_mean": float(np.mean(log_errors)),
        "log_abs_error_median": float(np.median(log_errors)),
        "signed_bias": float(np.mean(signed_errors)),
        "lr_level_signed_bias": lr_level_biases,
        "max_abs_lr_level_bias": float(
            max(abs(value) for value in lr_level_biases.values())
        ),
        "worst_cases": per_case[:10],
        "best_cases": list(reversed(per_case[-10:])),
        "per_case": per_case,
    }


def compute_pointwise_extended_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    dataset,
    output_fields: list[str],
) -> dict[str, Any]:
    """R², physical-space, per-region, and per-case metrics.

    Only meaningful for the pointwise adapter with TabularPairDataset.
    """
    metrics: dict[str, Any] = {}
    raw_d_over_D = getattr(dataset, "_raw_d_local_over_D", None)
    local_vel_norm = bool(getattr(dataset, "local_velocity_normalization", False))

    def _has_physical_inverse(name: str) -> bool:
        return (
            is_alpha_d_target(name)
            or name.startswith("log_")
            or name.startswith("log10_")
        )

    per_field_extended = []
    for i, name in enumerate(output_fields):
        p, t = preds[:, i], targets[:, i]
        ss_res = float(((p - t) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = float((p - t).abs().mean())
        entry: dict[str, Any] = {"name": name, "r2": r2, "mae": mae}

        if _has_physical_inverse(name):
            d_over_D = raw_d_over_D if is_alpha_d_target(name) else None
            p_full = dataset.add_baseline_to_encoded(p, field_idx=i)
            t_full = dataset.add_baseline_to_encoded(t, field_idx=i)
            p_phys = field_values_to_physical(
                p_full,
                field_name=name,
                d_over_D=d_over_D,
                local_velocity_normalization=local_vel_norm,
            )
            t_phys = field_values_to_physical(
                t_full,
                field_name=name,
                d_over_D=d_over_D,
                local_velocity_normalization=local_vel_norm,
            )
            rel_err = ((p_phys - t_phys) / t_phys.abs().clamp(min=1e-8)).abs()
            entry["physical_median_relative_error"] = float(rel_err.median())
            entry["physical_p90_relative_error"] = float(rel_err.quantile(0.9))
            log_abs_err = (
                p_phys.abs().clamp(min=1e-8).log() - t_phys.abs().clamp(min=1e-8).log()
            ).abs()
            entry["physical_log_abs_error_median"] = float(log_abs_err.median())
            entry["physical_log_abs_error_mean"] = float(log_abs_err.mean())
            entry["physical_log_abs_error_p90"] = float(log_abs_err.quantile(0.9))

        per_field_extended.append(entry)
    metrics["per_field"] = per_field_extended

    input_columns = getattr(dataset, "input_columns", [])
    region_col_indices: dict[str, int] = {}
    for col_name in ("is_upstream", "is_throat", "is_downstream"):
        if col_name in input_columns:
            region_col_indices[col_name] = input_columns.index(col_name)

    if region_col_indices:
        raw_x = dataset._x.clone().cpu()
        norm_stats = getattr(dataset, "norm_stats", None)
        if getattr(dataset, "normalize", False) and norm_stats is not None:
            raw_x = raw_x * norm_stats["x_std"].cpu() + norm_stats["x_mean"].cpu()

        per_region: dict[str, Any] = {}
        for region_name, col_idx in region_col_indices.items():
            mask = raw_x[:, col_idx] > 0.5
            n_region = int(mask.sum())
            if n_region == 0:
                continue
            region_entry: dict[str, Any] = {"n_samples": n_region}
            for i, field_name in enumerate(output_fields):
                p, t = preds[mask, i], targets[mask, i]
                ss_res = float(((p - t) ** 2).sum())
                ss_tot = float(((t - t.mean()) ** 2).sum())
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                rmse = float(((p - t) ** 2).mean().sqrt())
                field_metrics: dict[str, Any] = {"r2": r2, "rmse": rmse}
                if _has_physical_inverse(field_name):
                    d_over_D = (
                        raw_d_over_D[mask]
                        if (raw_d_over_D is not None and is_alpha_d_target(field_name))
                        else None
                    )
                    p_full = dataset.add_baseline_to_encoded(
                        p, row_mask=mask, field_idx=i
                    )
                    t_full = dataset.add_baseline_to_encoded(
                        t, row_mask=mask, field_idx=i
                    )
                    p_phys = field_values_to_physical(
                        p_full,
                        field_name=field_name,
                        d_over_D=d_over_D,
                        local_velocity_normalization=local_vel_norm,
                    )
                    t_phys = field_values_to_physical(
                        t_full,
                        field_name=field_name,
                        d_over_D=d_over_D,
                        local_velocity_normalization=local_vel_norm,
                    )
                    rel_err = ((p_phys - t_phys) / t_phys.abs().clamp(min=1e-8)).abs()
                    field_metrics["median_relative_error"] = float(rel_err.median())
                region_entry[field_name] = field_metrics
            per_region[region_name] = region_entry
        metrics["per_region"] = per_region

    case_idx_arr = getattr(dataset, "_row_case_idx", None)
    case_names = getattr(dataset, "_case_ids_unique", None)
    if case_idx_arr is not None and case_names is not None:
        per_case: list[dict[str, Any]] = []
        case_idx_t = torch.from_numpy(case_idx_arr)
        for ci, case_name in enumerate(case_names):
            mask = case_idx_t == ci
            if mask.sum() == 0:
                continue
            for i, field_name in enumerate(output_fields):
                p, t = preds[mask, i], targets[mask, i]
                case_rmse = float(((p - t) ** 2).mean().sqrt())
                entry = {"case": case_name, "field": field_name, "rmse": case_rmse}
                if _has_physical_inverse(field_name):
                    d_over_D = (
                        raw_d_over_D[mask]
                        if (raw_d_over_D is not None and is_alpha_d_target(field_name))
                        else None
                    )
                    p_full = dataset.add_baseline_to_encoded(
                        p, row_mask=mask, field_idx=i
                    )
                    t_full = dataset.add_baseline_to_encoded(
                        t, row_mask=mask, field_idx=i
                    )
                    p_phys = field_values_to_physical(
                        p_full,
                        field_name=field_name,
                        d_over_D=d_over_D,
                        local_velocity_normalization=local_vel_norm,
                    )
                    t_phys = field_values_to_physical(
                        t_full,
                        field_name=field_name,
                        d_over_D=d_over_D,
                        local_velocity_normalization=local_vel_norm,
                    )
                    rel_err = ((p_phys - t_phys) / t_phys.abs().clamp(min=1e-8)).abs()
                    entry["median_relative_error"] = float(rel_err.median())
                per_case.append(entry)
        per_case.sort(key=lambda x: x["rmse"], reverse=True)
        metrics["worst_cases"] = per_case[:10]
        metrics["best_cases"] = list(reversed(per_case[-10:]))

    return metrics


def print_extended_metrics(metrics: dict[str, Any]) -> None:
    """Human-readable summary of α_D extended metrics."""
    for entry in metrics.get("per_field", []):
        parts = [f"  {entry['name']}: R²={entry['r2']:.4f}, MAE={entry['mae']:.4e}"]
        if "physical_log_abs_error_median" in entry:
            line = (
                f"  {entry['name']} log_abs_err: "
                f"median={entry['physical_log_abs_error_median']:.3f}, "
                f"mean={entry['physical_log_abs_error_mean']:.3f}, "
                f"p90={entry['physical_log_abs_error_p90']:.3f}"
            )
            print(line)
        if "physical_median_relative_error" in entry:
            parts.append(
                f"    physical relative error: "
                f"median={entry['physical_median_relative_error']:.1%}, "
                f"p90={entry['physical_p90_relative_error']:.1%}"
            )
        print("\n".join(parts))

    per_region = metrics.get("per_region", {})
    if per_region:
        print("Per-region breakdown:")
        for region_name, region_data in per_region.items():
            n = region_data.get("n_samples", "?")
            for field_name, fm in region_data.items():
                if field_name == "n_samples":
                    continue
                line = f"  {region_name} ({n} pts): R²={fm['r2']:.4f}, RMSE={fm['rmse']:.4e}"
                if "median_relative_error" in fm:
                    line += f", median_rel_err={fm['median_relative_error']:.1%}"
                print(line)

    worst = metrics.get("worst_cases", [])
    best = metrics.get("best_cases", [])
    if worst:
        print("Worst 5 cases:")
        for c in worst[:5]:
            line = f"  {c['case']}: RMSE={c['rmse']:.4e}"
            if "median_relative_error" in c:
                line += f", median_rel_err={c['median_relative_error']:.1%}"
            print(line)
    if best:
        print("Best 5 cases:")
        for c in best[:5]:
            line = f"  {c['case']}: RMSE={c['rmse']:.4e}"
            if "median_relative_error" in c:
                line += f", median_rel_err={c['median_relative_error']:.1%}"
            print(line)

    dp = metrics.get("delta_p", {})
    if dp:
        print(
            f"Delta-p prediction ({dp['n_cases']} cases): "
            f"median_rel_err={dp['relative_error_median']:.1%}, "
            f"mean_rel_err={dp['relative_error_mean']:.1%}, "
            f"p90_rel_err={dp['relative_error_p90']:.1%}, "
            f"max_rel_err={dp['relative_error_max']:.1%}"
        )
        dp_worst = dp.get("worst_cases", [])
        if dp_worst:
            print("  Worst 5 delta-p cases:")
            for c in dp_worst[:5]:
                print(
                    f"    {c['case']}: "
                    f"gt={c['delta_p_gt']:.2f} Pa, "
                    f"pred={c['delta_p_pred']:.2f} Pa, "
                    f"rel_err={c['relative_error']:.1%}"
                )
