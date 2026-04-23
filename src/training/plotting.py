"""Plotting helpers for evaluation outputs."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from training.datasets import GridPairDataset
from training.datasets_tabular import TabularPairDataset
from training.alpha_d_targets import field_values_to_physical, is_alpha_d_target


def parse_index_list(raw: str | list[int] | None) -> list[int] | None:
    if raw is None:
        return None

    if isinstance(raw, list):
        values = [int(item) for item in raw]
    else:
        values: list[int] = []
        for item in str(raw).split(","):
            stripped = item.strip()
            if not stripped:
                continue
            values.append(int(stripped))

    if not values:
        raise ValueError("plot_case_indices is empty after parsing.")
    if any(value < 0 for value in values):
        raise ValueError(f"Plot indices must be non-negative, got {values}.")
    return values


def resolve_plot_indices(
    num_cases: int,
    raw_indices: str | list[int] | None,
    max_cases: int,
) -> list[int]:
    if max_cases < 0:
        raise ValueError("plot_max_cases must be >= 0.")
    if max_cases == 0 or num_cases == 0:
        return []

    explicit = parse_index_list(raw_indices)
    if explicit is None:
        return list(range(min(num_cases, max_cases)))

    deduped: list[int] = []
    for idx in explicit:
        if idx >= num_cases:
            raise ValueError(
                f"plot_case_indices contains {idx}, but dataset size is {num_cases}."
            )
        if idx not in deduped:
            deduped.append(idx)
    return deduped[:max_cases]


def _add_imshow(ax, arr: np.ndarray, title: str, cmap: str):
    im = ax.imshow(arr, origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _resolve_case_name(dataset, idx: int) -> str:
    if isinstance(dataset, Subset):
        parent_idx = int(dataset.indices[idx])
        return _resolve_case_name(dataset.dataset, parent_idx)

    if isinstance(dataset, GridPairDataset):
        return dataset.sim_names[idx]
    if isinstance(dataset, TabularPairDataset):
        return dataset.sim_names[idx]

    return f"case_{idx:03d}"


def select_best_worst_pointwise_cases(
    extended_metrics: dict[str, Any],
    output_fields: list[str],
) -> list[dict[str, Any]]:
    """Choose one best and one worst pointwise case for profile plotting."""
    if not output_fields:
        return []

    field_name = output_fields[0]
    selected: list[dict[str, Any]] = []
    used_cases: set[str] = set()

    for label, key in (("best", "best_cases"), ("worst", "worst_cases")):
        candidates = [
            entry
            for entry in extended_metrics.get(key, [])
            if entry.get("field") == field_name
        ]
        if not candidates:
            continue

        chosen = None
        for entry in candidates:
            case_name = str(entry.get("case"))
            if case_name not in used_cases:
                chosen = entry
                break
        if chosen is None:
            chosen = candidates[0]

        selected.append(
            {
                "label": label,
                "case": str(chosen["case"]),
                "field": field_name,
                "rmse": float(chosen["rmse"]),
                "median_relative_error": chosen.get("median_relative_error"),
            }
        )
        used_cases.add(str(chosen["case"]))

    return selected


def _to_physical_alpha_profile(
    values: torch.Tensor,
    *,
    field_name: str,
    d_over_D: torch.Tensor | None,
    local_velocity_normalization: bool,
) -> np.ndarray:
    """Convert model-space output to physical alpha_D for plotting."""
    profile = field_values_to_physical(
        values.detach().cpu().clone(),
        field_name=field_name,
        d_over_D=d_over_D.detach().cpu() if d_over_D is not None else None,
        local_velocity_normalization=local_velocity_normalization,
    )
    return profile.detach().cpu().numpy()


def save_pointwise_profile_plots(
    model,
    dataset,
    output_fields: list[str],
    device: torch.device,
    plot_dir: str | Path,
    case_entries: list[dict[str, Any]],
    *,
    plot_dpi: int = 150,
) -> list[str]:
    """Save best/worst alpha_D profile plots for pointwise/tabular models."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as import_error:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install it or omit output.plot_dir."
        ) from import_error

    if plot_dpi < 1:
        raise ValueError("plot_dpi must be >= 1.")
    if not output_fields:
        raise ValueError("No output fields are available for plotting.")
    if not hasattr(dataset, "_case_ids_unique"):
        raise ValueError("Pointwise profile plotting requires a case-indexed tabular dataset.")

    plot_dir_path = Path(plot_dir)
    plot_dir_path.mkdir(parents=True, exist_ok=True)

    field_name = output_fields[0]
    case_to_idx = {name: idx for idx, name in enumerate(dataset._case_ids_unique)}
    output_files: list[str] = []

    with torch.no_grad():
        for entry in case_entries:
            case_name = str(entry["case"])
            case_idx = case_to_idx.get(case_name)
            if case_idx is None:
                continue

            mask = dataset._row_case_idx == case_idx
            if not np.any(mask):
                continue

            x_case = dataset._x[mask].to(device)
            pred_case = model(x_case).detach().cpu()[:, 0]
            target_case = dataset._y[mask].detach().cpu()[:, 0]

            z_hat = (
                dataset._raw_z_hat[mask].detach().cpu()
                if dataset._raw_z_hat is not None
                else torch.arange(len(pred_case), dtype=torch.float32)
            )
            d_over_D = (
                dataset._raw_d_local_over_D[mask].detach().cpu()
                if dataset._raw_d_local_over_D is not None
                else None
            )

            order = torch.argsort(z_hat)
            z_axis = z_hat[order].numpy()
            pred_phys = _to_physical_alpha_profile(
                pred_case[order],
                field_name=field_name,
                d_over_D=d_over_D[order] if d_over_D is not None else None,
                local_velocity_normalization=bool(
                    getattr(dataset, "local_velocity_normalization", False)
                ),
            )
            target_phys = _to_physical_alpha_profile(
                target_case[order],
                field_name=field_name,
                d_over_D=d_over_D[order] if d_over_D is not None else None,
                local_velocity_normalization=bool(
                    getattr(dataset, "local_velocity_normalization", False)
                ),
            )

            fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
            ax.plot(z_axis, target_phys, label="Ground Truth", linewidth=2.5, marker="o", ms=3)
            ax.plot(z_axis, pred_phys, label="Predicted", linewidth=2.0, linestyle="--")
            if np.all(target_phys > 0.0) and np.all(pred_phys > 0.0):
                ax.set_yscale("log")
            ax.set_xlabel("z_hat")
            profile_label = "alpha_D" if is_alpha_d_target(field_name) else field_name
            ax.set_ylabel(profile_label)
            ax.grid(True, alpha=0.3)
            ax.legend()

            title = f"{entry['label'].title()} {profile_label} Profile | {case_name}"
            rmse = entry.get("rmse")
            rel = entry.get("median_relative_error")
            subtitle_parts: list[str] = []
            if rmse is not None:
                subtitle_parts.append(f"RMSE={float(rmse):.3e}")
            if rel is not None:
                subtitle_parts.append(f"median_rel_err={float(rel):.1%}")
            if subtitle_parts:
                title += "\n" + ", ".join(subtitle_parts)
            ax.set_title(title)

            safe_label = "alpha_D" if is_alpha_d_target(field_name) else field_name
            out_path = plot_dir_path / f"{entry['label']}_{case_name}_{safe_label}_profile.png"
            fig.savefig(out_path, dpi=plot_dpi)
            plt.close(fig)
            output_files.append(str(out_path))

    return output_files


def save_grid_prediction_plots(
    model,
    dataset,
    output_fields: list[str],
    device: torch.device,
    plot_dir: str | Path,
    plot_indices: list[int],
    plot_cmap: str = "viridis",
    plot_dpi: int = 150,
    quiver_step: int = 4,
    vel_x_field: str = "vel_x",
    vel_y_field: str = "vel_y",
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as import_error:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install it or omit output.plot_dir."
        ) from import_error

    if quiver_step < 1:
        raise ValueError("plot_quiver_step must be >= 1.")
    if plot_dpi < 1:
        raise ValueError("plot_dpi must be >= 1.")
    if not output_fields:
        raise ValueError("No output fields are available for plotting.")

    plot_dir_path = Path(plot_dir)
    plot_dir_path.mkdir(parents=True, exist_ok=True)

    has_vel_x = vel_x_field in output_fields
    has_vel_y = vel_y_field in output_fields
    can_plot_velocity = has_vel_x and has_vel_y
    vel_x_idx = output_fields.index(vel_x_field) if has_vel_x else -1
    vel_y_idx = output_fields.index(vel_y_field) if has_vel_y else -1

    output_files: list[str] = []

    with torch.no_grad():
        for idx in plot_indices:
            x, y = dataset[idx]
            pred = model(x.unsqueeze(0).to(device)).squeeze(0).detach().cpu()
            target = y.detach().cpu()
            case_name = _resolve_case_name(dataset, idx)

            if can_plot_velocity:
                target_u = target[vel_x_idx].numpy()
                target_v = target[vel_y_idx].numpy()
                pred_u = pred[vel_x_idx].numpy()
                pred_v = pred[vel_y_idx].numpy()

                speed_true = np.sqrt(target_u**2 + target_v**2)
                speed_pred = np.sqrt(pred_u**2 + pred_v**2)
                speed_err = np.abs(speed_pred - speed_true)
                vector_err = np.sqrt((pred_u - target_u) ** 2 + (pred_v - target_v) ** 2)

                yy, xx = np.mgrid[0 : target_u.shape[0], 0 : target_u.shape[1]]
                step = max(1, quiver_step)

                fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
                im = _add_imshow(
                    axes[0, 0], speed_true, f"Ground Truth |v| ({case_name})", plot_cmap
                )
                axes[0, 0].quiver(
                    xx[::step, ::step],
                    yy[::step, ::step],
                    target_u[::step, ::step],
                    target_v[::step, ::step],
                    color="white",
                    width=0.002,
                )
                fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

                im = _add_imshow(axes[0, 1], speed_pred, "Predicted |v|", plot_cmap)
                axes[0, 1].quiver(
                    xx[::step, ::step],
                    yy[::step, ::step],
                    pred_u[::step, ::step],
                    pred_v[::step, ::step],
                    color="white",
                    width=0.002,
                )
                fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

                im = _add_imshow(axes[1, 0], speed_err, "Absolute Error |v|", "magma")
                fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

                im = _add_imshow(axes[1, 1], vector_err, "Vector Error Magnitude", "magma")
                fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

                out_path = plot_dir_path / f"{idx:03d}_{case_name}_velocity.png"
                fig.savefig(out_path, dpi=plot_dpi)
                plt.close(fig)
            else:
                field_name = output_fields[0]
                target_scalar = target[0].numpy()
                pred_scalar = pred[0].numpy()
                abs_err = np.abs(pred_scalar - target_scalar)

                fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
                im = _add_imshow(
                    axes[0],
                    target_scalar,
                    f"Ground Truth {field_name} ({case_name})",
                    plot_cmap,
                )
                fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

                im = _add_imshow(axes[1], pred_scalar, f"Predicted {field_name}", plot_cmap)
                fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

                im = _add_imshow(axes[2], abs_err, f"Absolute Error {field_name}", "magma")
                fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

                out_path = plot_dir_path / f"{idx:03d}_{case_name}_{field_name}.png"
                fig.savefig(out_path, dpi=plot_dpi)
                plt.close(fig)

            output_files.append(str(out_path))

    return output_files
