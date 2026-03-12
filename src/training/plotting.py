"""Plotting helpers for grid-model evaluation."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from training.datasets import GridPairDataset


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

    return f"case_{idx:03d}"


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
