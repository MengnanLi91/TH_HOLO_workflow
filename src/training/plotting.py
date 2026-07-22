"""Plotting helpers for evaluation outputs."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from training.datasets import GridPairDataset
from training.datasets_tabular import TabularPairDataset


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
            raise ValueError(f"plot_case_indices contains {idx}, but dataset size is {num_cases}.")
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
            entry for entry in extended_metrics.get(key, []) if entry.get("field") == field_name
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


def save_pointwise_profile_plots(
    model,
    dataset,
    output_fields: list[str],
    device: torch.device,
    plot_dir: str | Path,
    case_entries: list[dict[str, Any]],
    *,
    plot_dpi: int = 150,
    decode_fn: Callable[..., tuple[np.ndarray, str] | None] | None = None,
    baseline_fn: Callable[..., tuple[np.ndarray, str] | None] | None = None,
) -> list[str]:
    """Save best/worst profile plots for pointwise/tabular models.

    When ``decode_fn`` is supplied (typically
    ``experiment.decode_for_plotting``), the plotter applies it to both
    predicted and target tensors before plotting; the function returns the
    decoded values and the y-axis label. When ``decode_fn`` is ``None`` or
    returns ``None``, the plotter shows raw encoded values with ``field_name``
    as the label.

    When ``baseline_fn`` is supplied (typically
    ``experiment.baseline_for_plotting``) and returns a non-None decoded
    array, it is overlaid as a third "Baseline" curve so model-versus-baseline
    improvement is visible. Has no effect when ``baseline_fn`` is ``None`` or
    the callback returns ``None``.
    """
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

            order = torch.argsort(z_hat)
            z_axis = z_hat[order].numpy()

            profile_label = field_name
            pred_phys: np.ndarray | None = None
            target_phys: np.ndarray | None = None
            decoded = False
            if decode_fn is not None:
                pred_dec = decode_fn(pred_case, dataset, field_name, mask)
                target_dec = decode_fn(target_case, dataset, field_name, mask)
                if pred_dec is not None and target_dec is not None:
                    pred_unordered, label = pred_dec
                    target_unordered, _ = target_dec
                    order_np = order.numpy()
                    pred_phys = pred_unordered[order_np]
                    target_phys = target_unordered[order_np]
                    profile_label = label
                    decoded = True
            if not decoded:
                pred_phys = pred_case[order].numpy()
                target_phys = target_case[order].numpy()

            baseline_phys: np.ndarray | None = None
            if decoded and baseline_fn is not None:
                baseline_dec = baseline_fn(dataset, field_name, mask)
                if baseline_dec is not None:
                    baseline_unordered, _ = baseline_dec
                    baseline_phys = baseline_unordered[order.numpy()]

            fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
            ax.plot(
                z_axis,
                target_phys,
                label="Ground Truth",
                linewidth=2.5,
                marker="o",
                ms=3,
            )
            ax.plot(z_axis, pred_phys, label="Predicted", linewidth=2.0, linestyle="--")
            if baseline_phys is not None:
                ax.plot(
                    z_axis,
                    baseline_phys,
                    label="Baseline",
                    linewidth=1.5,
                    linestyle=":",
                    color="gray",
                )
            positives = [target_phys > 0.0, pred_phys > 0.0]
            if baseline_phys is not None:
                positives.append(baseline_phys > 0.0)
            if all(np.all(p) for p in positives):
                ax.set_yscale("log")
            ax.set_xlabel("z_hat")
            ax.set_ylabel(profile_label)
            ax.grid(True, alpha=0.3)
            ax.legend()

            ax.set_title(f"{entry['label'].title()} {profile_label} Profile | {case_name}")

            out_path = plot_dir_path / f"{entry['label']}_{case_name}_{profile_label}_profile.png"
            fig.savefig(out_path, dpi=plot_dpi)
            plt.close(fig)
            output_files.append(str(out_path))

    return output_files


def save_profile_prediction_plots(
    model,
    dataset,
    output_fields: list[str],
    device: torch.device,
    plot_dir: str | Path,
    case_entries: list[dict[str, Any]],
    *,
    plot_dpi: int = 150,
    decode_fn: Callable[..., tuple[np.ndarray, str] | None] | None = None,
    baseline_fn: Callable[..., tuple[np.ndarray, str] | None] | None = None,
) -> list[str]:
    """Save best/worst per-case prediction plots for profile (1D-conv) models.

    The profile model consumes per-case ``[1, C, S]`` tensors and emits
    ``[1, O, S]`` predictions. This plotter mirrors
    :func:`save_pointwise_profile_plots` (same z_hat axis, same
    ``decode_fn`` / ``baseline_fn`` contract, same RMSE subtitle) but runs
    the model once per case rather than once per row.
    """
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
    if not hasattr(dataset, "_case_ids_unique") or not hasattr(dataset, "_row_case_idx"):
        raise ValueError("Profile prediction plotting requires a case-indexed dataset.")

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

            n_stations = int(mask.sum())
            z_hat_local = (
                dataset._raw_z_hat[mask].detach().cpu()
                if dataset._raw_z_hat is not None
                else torch.arange(n_stations, dtype=torch.float32)
            )
            order = torch.argsort(z_hat_local)
            z_axis = z_hat_local[order].numpy()

            x_sorted = dataset._x[mask][order]  # [S, C]
            x_in = x_sorted.T.unsqueeze(0).to(device)  # [1, C, S]
            pred_sorted = model(x_in).detach().cpu()[0, 0]  # [S]
            target_sorted = dataset._y[mask][order][:, 0]  # [S]

            # decode_fn / baseline_fn index baseline_encoded[mask] in flat
            # _row_case_idx order, so undo the z_hat sort before calling them,
            # then re-sort the decoded array for plotting.
            inverse_order = torch.empty_like(order)
            inverse_order[order] = torch.arange(order.numel())
            pred_roword = pred_sorted[inverse_order]
            target_roword = target_sorted[inverse_order]

            profile_label = field_name
            pred_phys: np.ndarray | None = None
            target_phys: np.ndarray | None = None
            decoded = False
            if decode_fn is not None:
                pred_dec = decode_fn(pred_roword, dataset, field_name, mask)
                target_dec = decode_fn(target_roword, dataset, field_name, mask)
                if pred_dec is not None and target_dec is not None:
                    pred_unordered, label = pred_dec
                    target_unordered, _ = target_dec
                    order_np = order.numpy()
                    pred_phys = pred_unordered[order_np]
                    target_phys = target_unordered[order_np]
                    profile_label = label
                    decoded = True
            if not decoded:
                pred_phys = pred_sorted.numpy()
                target_phys = target_sorted.numpy()

            baseline_phys: np.ndarray | None = None
            if decoded and baseline_fn is not None:
                baseline_dec = baseline_fn(dataset, field_name, mask)
                if baseline_dec is not None:
                    baseline_unordered, _ = baseline_dec
                    baseline_phys = baseline_unordered[order.numpy()]

            fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
            ax.plot(
                z_axis,
                target_phys,
                label="Ground Truth",
                linewidth=2.5,
                marker="o",
                ms=3,
            )
            ax.plot(z_axis, pred_phys, label="Predicted", linewidth=2.0, linestyle="--")
            if baseline_phys is not None:
                ax.plot(
                    z_axis,
                    baseline_phys,
                    label="Baseline",
                    linewidth=1.5,
                    linestyle=":",
                    color="gray",
                )
            positives = [target_phys > 0.0, pred_phys > 0.0]
            if baseline_phys is not None:
                positives.append(baseline_phys > 0.0)
            if all(np.all(p) for p in positives):
                ax.set_yscale("log")
            ax.set_xlabel("z_hat")
            ax.set_ylabel(profile_label)
            ax.grid(True, alpha=0.3)
            ax.legend()

            ax.set_title(f"{entry['label'].title()} {profile_label} Profile | {case_name}")

            out_path = plot_dir_path / f"{entry['label']}_{case_name}_{profile_label}_profile.png"
            fig.savefig(out_path, dpi=plot_dpi)
            plt.close(fig)
            output_files.append(str(out_path))

    return output_files


def save_parity_plot(
    *,
    cat_preds: torch.Tensor,
    cat_targets: torch.Tensor,
    dataset,
    output_fields: list[str],
    plot_dir: str | Path,
    decode_fn: Callable[..., tuple[np.ndarray, str] | None] | None = None,
    plot_dpi: int = 150,
) -> list[str]:
    """Save one parity plot per output field across the whole evaluation set.

    Each scatter point is one (case, station) pair: ground-truth value on X,
    predicted value on Y, with a y=x reference line. When ``decode_fn`` is
    supplied, predictions and targets are mapped to physical space first
    (case-by-case via the same hook the per-case plots use). When the
    dataset's ``input_columns`` include ``is_upstream`` / ``is_throat`` /
    ``is_downstream`` flags, points are colored by region.

    Accepts both pointwise (``[N_rows, O]``) and profile (``[N_cases, O, S]``)
    prediction tensors; profile tensors are scattered to flat row layout via
    ``dataset._case_slices`` so the same plotting path works for both.
    """
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
    if not hasattr(dataset, "_row_case_idx"):
        raise ValueError("Parity plotting requires a case-indexed dataset.")

    plot_dir_path = Path(plot_dir)
    plot_dir_path.mkdir(parents=True, exist_ok=True)

    if cat_preds.shape != cat_targets.shape:
        raise ValueError(
            f"cat_preds shape {tuple(cat_preds.shape)} does not match "
            f"cat_targets shape {tuple(cat_targets.shape)}."
        )

    # Profile adapter emits [N_cases, O, S]; scatter back to [N_rows, O]
    # so the case-by-case decode hook (which expects flat rows aligned with
    # _row_case_idx) works uniformly.
    if cat_preds.dim() == 3:
        case_slices = getattr(dataset, "_case_slices", None)
        if case_slices is None:
            raise ValueError("Profile-shaped predictions require dataset._case_slices.")
        n_rows = sum(len(s) for s in case_slices)
        n_fields = cat_preds.shape[1]
        flat_preds = torch.empty(n_rows, n_fields, dtype=cat_preds.dtype)
        flat_targets = torch.empty(n_rows, n_fields, dtype=cat_targets.dtype)
        for ci, rows in enumerate(case_slices):
            rows_t = torch.as_tensor(rows, dtype=torch.long)
            flat_preds[rows_t] = cat_preds[ci].T
            flat_targets[rows_t] = cat_targets[ci].T
        cat_preds, cat_targets = flat_preds, flat_targets
    elif cat_preds.dim() != 2:
        raise ValueError(
            f"Expected predictions of shape [N_rows, O] or [N_cases, O, S]; "
            f"got {tuple(cat_preds.shape)}."
        )

    region_cols = ("is_upstream", "is_throat", "is_downstream")
    input_columns = list(getattr(dataset, "input_columns", []) or [])
    region_idx_lookup = {
        name: input_columns.index(name) for name in region_cols if name in input_columns
    }

    output_files: list[str] = []
    for field_idx, field_name in enumerate(output_fields):
        pred_pieces: list[np.ndarray] = []
        target_pieces: list[np.ndarray] = []
        region_pieces: list[np.ndarray] = []
        axis_label = field_name

        for ci in range(len(dataset._case_ids_unique)):
            mask = dataset._row_case_idx == ci
            if not np.any(mask):
                continue
            pred_rows = cat_preds[mask, field_idx]
            target_rows = cat_targets[mask, field_idx]

            decoded = False
            if decode_fn is not None:
                pred_dec = decode_fn(pred_rows, dataset, field_name, mask)
                target_dec = decode_fn(target_rows, dataset, field_name, mask)
                if pred_dec is not None and target_dec is not None:
                    pred_phys, label = pred_dec
                    target_phys, _ = target_dec
                    axis_label = label
                    decoded = True
            if not decoded:
                pred_phys = pred_rows.numpy()
                target_phys = target_rows.numpy()

            pred_pieces.append(np.asarray(pred_phys))
            target_pieces.append(np.asarray(target_phys))

            if region_idx_lookup:
                x_rows = dataset._x[mask].numpy()
                labels_for_case = np.full(x_rows.shape[0], "other", dtype=object)
                for name in region_cols:
                    if name in region_idx_lookup:
                        col = region_idx_lookup[name]
                        is_in = x_rows[:, col] > 0.5
                        labels_for_case[is_in] = name
                region_pieces.append(labels_for_case)

        if not pred_pieces:
            continue

        pred_all = np.concatenate(pred_pieces)
        target_all = np.concatenate(target_pieces)
        region_all = np.concatenate(region_pieces) if region_pieces else None

        fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)

        if region_all is not None:
            colors = {
                "is_upstream": "#1f77b4",
                "is_throat": "#d62728",
                "is_downstream": "#2ca02c",
                "other": "#7f7f7f",
            }
            for name, color in colors.items():
                sel = region_all == name
                if not np.any(sel):
                    continue
                ax.scatter(
                    target_all[sel],
                    pred_all[sel],
                    s=10,
                    alpha=0.5,
                    color=color,
                    label=(name.replace("is_", "").capitalize() if name != "other" else "Other"),
                )
        else:
            ax.scatter(target_all, pred_all, s=10, alpha=0.5, color="#1f77b4")

        lo = float(np.nanmin([target_all.min(), pred_all.min()]))
        hi = float(np.nanmax([target_all.max(), pred_all.max()]))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        ax.plot(
            [lo, hi],
            [lo, hi],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="y = x",
        )
        ax.plot(
            [lo, hi],
            [1.1 * lo, 1.1 * hi],
            color="black",
            linestyle=":",
            linewidth=0.8,
            label="±10%",
        )
        ax.plot(
            [lo, hi],
            [0.9 * lo, 0.9 * hi],
            color="black",
            linestyle=":",
            linewidth=0.8,
            label="_nolegend_",
        )

        if np.all(target_all > 0.0) and np.all(pred_all > 0.0):
            ax.set_xscale("log")
            ax.set_yscale("log")

        rmse = float(np.sqrt(np.mean((pred_all - target_all) ** 2)))
        ax.set_xlabel(f"Ground Truth {axis_label}")
        ax.set_ylabel(f"Predicted {axis_label}")
        ax.set_title(f"Parity Plot — {axis_label}\nN={pred_all.size}, RMSE={rmse:.3e}")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", fontsize=9)

        out_path = plot_dir_path / f"parity_{axis_label}.png"
        fig.savefig(out_path, dpi=plot_dpi)
        plt.close(fig)
        output_files.append(str(out_path))

    return output_files


def save_delta_p_parity_plot(
    *,
    per_case: list[dict[str, Any]],
    plot_dir: str | Path,
    plot_dpi: int = 150,
) -> str | None:
    """Save a Δp parity plot: one marker per test case, log-log.

    ``per_case`` is the full sorted list produced by
    ``compute_delta_p_metrics`` (``extended.delta_p.per_case``). Each entry
    must carry ``delta_p_gt`` / ``delta_p_pred``; when ``Dr`` is present,
    points are colored on a discrete diameter-ratio colormap so the
    low-Dr cluster (the typical alpha-D failure mode) stands out.
    """
    if not per_case:
        return None

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as import_error:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install it or omit output.plot_dir."
        ) from import_error

    if plot_dpi < 1:
        raise ValueError("plot_dpi must be >= 1.")

    plot_dir_path = Path(plot_dir)
    plot_dir_path.mkdir(parents=True, exist_ok=True)

    gt = np.array([float(e["delta_p_gt"]) for e in per_case], dtype=np.float64)
    pred = np.array([float(e["delta_p_pred"]) for e in per_case], dtype=np.float64)
    drs = np.array([float(e.get("Dr", np.nan)) for e in per_case], dtype=np.float64)
    rel_err = np.array([float(e.get("relative_error", np.nan)) for e in per_case], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.8, 6.4), constrained_layout=True)

    have_dr = np.isfinite(drs).all() and drs.size > 0
    if have_dr:
        cmap = plt.get_cmap("viridis")
        scatter = ax.scatter(
            gt,
            pred,
            s=30,
            c=drs,
            cmap=cmap,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Dr (D_throat / D_big)")
    else:
        ax.scatter(gt, pred, s=30, alpha=0.7, color="#1f77b4")

    lo = float(np.nanmin([gt.min(), pred.min()]))
    hi = float(np.nanmax([gt.max(), pred.max()]))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = max(lo, 1e-3), max(hi, 1e-3) * 10.0
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0, label="y = x")
    ax.plot(
        [lo, hi],
        [1.1 * lo, 1.1 * hi],
        color="black",
        linestyle=":",
        linewidth=0.8,
        label="±10%",
    )
    ax.plot(
        [lo, hi],
        [0.9 * lo, 0.9 * hi],
        color="black",
        linestyle=":",
        linewidth=0.8,
        label="_nolegend_",
    )

    if (gt > 0).all() and (pred > 0).all():
        ax.set_xscale("log")
        ax.set_yscale("log")

    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    rel_med = float(np.median(rel_err)) if np.isfinite(rel_err).all() else float("nan")
    rel_max = float(np.max(rel_err)) if np.isfinite(rel_err).all() else float("nan")
    ax.set_xlabel("Ground Truth Δp [Pa]")
    ax.set_ylabel("Predicted Δp [Pa]")
    ax.set_title(
        f"Δp Parity Plot — N={len(per_case)}\n"
        f"RMSE={rmse:.3e} Pa, median rel. err={rel_med:.1%}, max={rel_max:.1%}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=9)

    out_path = plot_dir_path / "delta_p_parity.png"
    fig.savefig(out_path, dpi=plot_dpi)
    plt.close(fig)
    return str(out_path)


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
