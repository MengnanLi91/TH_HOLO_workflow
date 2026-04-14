"""Generic training/evaluation runners for supervised one-step models."""

import importlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any

import copy

import numpy as np
try:
    import torch
    from torch.utils.data import DataLoader, Subset
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyTorch is required for training/evaluation but is not installed in this "
        "environment. Use the `etl` or `etl-ngc` Docker service, or install `torch` "
        "in your active environment."
    ) from exc

from training import import_physicsnemo_attr
from training.adapters import get_adapter
from training.datasets import split_indices
from training.experiment import Experiment
from training.losses import get_loss_fn
from training.models import get_build_fn_and_adapter, model_entrypoint_string
from training.plotting import resolve_plot_indices, save_grid_prediction_plots

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


def to_plain_dict(cfg: Any) -> dict[str, Any]:
    """Convert OmegaConf DictConfig or dict to a plain dict."""
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


_to_plain_dict = to_plain_dict


def set_seed(seed: int) -> None:
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_set_seed = set_seed


def resolve_device(device_arg: str) -> torch.device:
    """Parse a device string ('auto', 'cpu', 'cuda') into a torch.device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but no CUDA device is available.")
    return torch.device(device_arg)


_resolve_device = resolve_device


def _resolve_path(raw_path: str | Path) -> Path:
    return Path(raw_path).expanduser().resolve()


def _load_object(entrypoint: str):
    if ":" not in entrypoint:
        raise ValueError(
            f"Invalid entrypoint '{entrypoint}'. Expected format 'module.path:object'."
        )
    module_path, object_name = entrypoint.rsplit(":", 1)
    module = importlib.import_module(module_path)
    if not hasattr(module, object_name):
        raise AttributeError(
            f"Entrypoint object '{object_name}' not found in module '{module_path}'."
        )
    return getattr(module, object_name)


def build_experiment(
    experiment_entrypoint: str | None,
    model,
    optimizer,
    loss_fn,
    adapter,
    device: torch.device,
    **kwargs,
) -> Experiment:
    """Instantiate an Experiment from an optional entrypoint string.

    Extra *kwargs* are forwarded to the experiment constructor, allowing
    custom experiments to accept domain-specific arguments such as
    ``consistency_weight`` or ``norm_stats``.
    """
    if experiment_entrypoint:
        experiment_cls = _load_object(experiment_entrypoint)
    else:
        experiment_cls = Experiment

    experiment = experiment_cls(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        adapter=adapter,
        device=device,
        **kwargs,
    )

    if not hasattr(experiment, "training_step") or not hasattr(experiment, "eval_step"):
        raise TypeError(
            "Experiment class must define training_step() and eval_step() methods."
        )

    return experiment


_build_experiment = build_experiment


def _git_code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _collect_resolved_model_params(
    model,
    model_params: dict,
    dataset_info: dict,
) -> dict[str, Any]:
    resolved_from_model = getattr(model, "_resolved_model_params", None)
    if isinstance(resolved_from_model, dict):
        return dict(resolved_from_model)

    resolved = dict(model_params)
    for key in ("in_channels", "out_channels", "edge_dim", "spatial_shape"):
        if key in dataset_info and key not in resolved:
            resolved[key] = dataset_info[key]
    return resolved


def _compute_pointwise_extended_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    dataset,
    output_fields: list[str],
) -> dict[str, Any]:
    """Compute R², physical-space, per-region, and per-case metrics.

    Only meaningful for the pointwise adapter with ``TabularPairDataset``.
    """
    metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Overall R², MAE per output field
    # ------------------------------------------------------------------
    per_field_extended = []
    for i, name in enumerate(output_fields):
        p, t = preds[:, i], targets[:, i]
        ss_res = float(((p - t) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = float((p - t).abs().mean())
        entry: dict[str, Any] = {"name": name, "r2": r2, "mae": mae}

        # Physical-space relative error for log-transformed outputs
        if name.startswith("log_") or name.startswith("log10_"):
            base_exp = torch.exp if name.startswith("log_") else lambda x: 10.0 ** x
            p_phys = base_exp(p)
            t_phys = base_exp(t)
            rel_err = ((p_phys - t_phys) / t_phys.clamp(min=1e-8)).abs()
            entry["physical_median_relative_error"] = float(rel_err.median())
            entry["physical_mean_relative_error"] = float(rel_err.mean())
            entry["physical_p90_relative_error"] = float(rel_err.quantile(0.9))

        per_field_extended.append(entry)
    metrics["per_field"] = per_field_extended

    # ------------------------------------------------------------------
    # Per-region metrics (upstream / throat / downstream)
    # ------------------------------------------------------------------
    input_columns = getattr(dataset, "input_columns", [])
    region_col_indices: dict[str, int] = {}
    for col_name in ("is_upstream", "is_throat", "is_downstream"):
        if col_name in input_columns:
            region_col_indices[col_name] = input_columns.index(col_name)

    if region_col_indices:
        # Recover raw binary indicators from normalized inputs.
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
                if field_name.startswith("log_") or field_name.startswith("log10_"):
                    base_exp = torch.exp if field_name.startswith("log_") else lambda x: 10.0 ** x
                    rel_err = (
                        (base_exp(p) - base_exp(t)) / base_exp(t).clamp(min=1e-8)
                    ).abs()
                    field_metrics["median_relative_error"] = float(rel_err.median())
                region_entry[field_name] = field_metrics
            per_region[region_name] = region_entry
        metrics["per_region"] = per_region

    # ------------------------------------------------------------------
    # Per-case metrics (sorted worst-to-best by RMSE)
    # ------------------------------------------------------------------
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
                if field_name.startswith("log_") or field_name.startswith("log10_"):
                    base_exp = torch.exp if field_name.startswith("log_") else lambda x: 10.0 ** x
                    rel_err = (
                        (base_exp(p) - base_exp(t)) / base_exp(t).clamp(min=1e-8)
                    ).abs()
                    entry["median_relative_error"] = float(rel_err.median())
                per_case.append(entry)
        per_case.sort(key=lambda x: x["rmse"], reverse=True)
        metrics["worst_cases"] = per_case[:10]
        metrics["best_cases"] = list(reversed(per_case[-10:]))

    return metrics


def _print_extended_metrics(metrics: dict[str, Any]) -> None:
    """Print a human-readable summary of extended metrics."""
    for entry in metrics.get("per_field", []):
        parts = [f"  {entry['name']}: R²={entry['r2']:.4f}, MAE={entry['mae']:.4e}"]
        if "physical_median_relative_error" in entry:
            parts.append(
                f"    alpha_D relative error: "
                f"median={entry['physical_median_relative_error']:.1%}, "
                f"mean={entry['physical_mean_relative_error']:.1%}, "
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


def _serialize_norm_stats(norm_stats: dict[str, torch.Tensor] | None) -> dict[str, list[float]] | None:
    if not norm_stats:
        return None
    return {
        "x_mean": norm_stats["x_mean"].detach().cpu().tolist(),
        "x_std": norm_stats["x_std"].detach().cpu().tolist(),
    }


def normalize_split_cfg(split_cfg: dict, default_seed: int) -> dict[str, Any]:
    """Fill in default values for a split config dict."""
    normalized = dict(split_cfg)
    normalized.setdefault("strategy", "sequential")
    if normalized["strategy"] in {"sequential", "random", "stratified"}:
        normalized.setdefault("train_ratio", 0.8)
    if normalized["strategy"] in {"random", "stratified"}:
        normalized.setdefault("seed", default_seed)
    return normalized


_normalize_split_cfg = normalize_split_cfg


def prepare_training(cfg_dict: dict) -> dict[str, Any]:
    """Build adapter, dataset, dataset_info, and build_fn from a config dict.

    Returns a dict with keys: model_cfg, data_cfg, training_cfg, output_cfg,
    adapter_name, adapter, dataset, dataset_info, build_fn, device, seed.
    This is the shared setup that both ``train()`` and HPO use.
    """
    model_cfg = dict(cfg_dict.get("model") or {})
    data_cfg = dict(cfg_dict.get("data") or {})
    training_cfg = dict(cfg_dict.get("training") or {})
    output_cfg = dict(cfg_dict.get("output") or {})

    if not data_cfg.get("zarr_dir"):
        raise ValueError("data.zarr_dir is required.")

    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

    device = resolve_device(str(training_cfg.get("device", "auto")))
    build_fn, adapter_name = get_build_fn_and_adapter(model_cfg)
    adapter = get_adapter(adapter_name)

    dataset = adapter.build_dataset(data_cfg)
    dataset_info = adapter.dataset_info(dataset)

    return {
        "model_cfg": model_cfg,
        "data_cfg": data_cfg,
        "training_cfg": training_cfg,
        "output_cfg": output_cfg,
        "adapter_name": adapter_name,
        "adapter": adapter,
        "dataset": dataset,
        "dataset_info": dataset_info,
        "build_fn": build_fn,
        "device": device,
        "seed": seed,
    }


def train_one_epoch(experiment: Experiment, dataloader: DataLoader) -> float:
    """Run one training epoch, return average loss.

    Raises ``RuntimeError`` if the dataloader produces zero batches.
    """
    running_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        running_loss += experiment.training_step(batch)
        num_batches += 1
    if num_batches == 0:
        raise RuntimeError("No training batches were produced.")
    return running_loss / num_batches


def compute_val_loss(experiment: Experiment, val_loader: DataLoader) -> float:
    """Evaluate on a validation loader, return average loss.

    Uses ``experiment.validation_step()`` which defaults to
    ``eval_step() + loss_fn``.  Custom experiments can override it.

    Raises ``RuntimeError`` if the loader produces zero batches.
    """
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            total += experiment.validation_step(batch)
            n += 1
    if n == 0:
        raise RuntimeError("Validation loader produced zero batches.")
    return total / n


def train(cfg: dict | Any) -> dict[str, Any]:
    """Train a supervised model and save checkpoint + run_meta.json."""
    cfg_dict = to_plain_dict(cfg)

    prep = prepare_training(cfg_dict)
    model_cfg = prep["model_cfg"]
    data_cfg = prep["data_cfg"]
    training_cfg = prep["training_cfg"]
    output_cfg = prep["output_cfg"]
    adapter_name = prep["adapter_name"]
    adapter = prep["adapter"]
    dataset = prep["dataset"]
    dataset_info = prep["dataset_info"]
    build_fn = prep["build_fn"]
    device = prep["device"]
    seed = prep["seed"]

    model_params = dict(model_cfg.get("params") or {})
    model = build_fn(model_params, dataset_info).to(device)

    loss_name = str(training_cfg.get("loss", "mse"))
    loss_fn = get_loss_fn(loss_name)
    lr = float(training_cfg.get("lr", 1.0e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    if weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    experiment_entrypoint = training_cfg.get("experiment")
    experiment_kwargs: dict[str, Any] = {}
    consistency_weight = float(training_cfg.get("consistency_weight", 0.0))
    if consistency_weight > 0:
        experiment_kwargs["consistency_weight"] = consistency_weight
    experiment = _build_experiment(
        experiment_entrypoint=experiment_entrypoint,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        adapter=adapter,
        device=device,
        **experiment_kwargs,
    )

    split_cfg = _normalize_split_cfg(dict(data_cfg.get("split") or {}), default_seed=seed)
    num_cases = len(dataset.sim_names) if hasattr(dataset, "sim_names") else len(dataset)
    train_idx, test_idx, train_sims, test_sims = split_indices(
        num_cases=num_cases,
        split_cfg=split_cfg,
        sim_names=dataset.sim_names,
    )

    # Early stopping: carve a validation split from the training cases.
    early_stop_cfg = dict(training_cfg.get("early_stopping") or {})
    patience = int(early_stop_cfg.get("patience", 0))
    use_early_stopping = patience > 0
    train_case_idx = list(train_idx)
    val_case_idx: list = []
    if use_early_stopping:
        val_ratio = float(early_stop_cfg.get("val_ratio", 0.15))
        rng_es = random.Random(seed + 1)
        shuffled_train = list(train_idx)
        rng_es.shuffle(shuffled_train)
        n_val_cases = max(1, round(len(shuffled_train) * val_ratio))
        val_case_idx = shuffled_train[:n_val_cases]
        train_case_idx = shuffled_train[n_val_cases:]
    if not train_case_idx:
        raise ValueError("Training split is empty after validation split. Reduce val_ratio.")

    # Pointwise normalization must be fit on training cases only.
    if adapter_name == "pointwise" and bool(data_cfg.get("normalize", False)):
        normalized_data_cfg = dict(data_cfg)
        normalized_data_cfg["norm_from_case_indices"] = train_case_idx
        dataset = adapter.build_dataset(normalized_data_cfg)

    # Inject norm_stats into the experiment for denormalisation (e.g. consistency loss)
    if consistency_weight > 0 and hasattr(dataset, "norm_stats") and dataset.norm_stats:
        norm_stats_for_exp = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in dataset.norm_stats.items()
        }
        experiment.y_mean = norm_stats_for_exp.get("y_mean")
        experiment.y_std = norm_stats_for_exp.get("y_std")

    if hasattr(dataset, "subset_by_case_indices"):
        train_dataset = dataset.subset_by_case_indices(train_case_idx)
        val_dataset = dataset.subset_by_case_indices(val_case_idx) if use_early_stopping else None
    else:
        train_dataset = Subset(dataset, train_case_idx)
        val_dataset = Subset(dataset, val_case_idx) if use_early_stopping else None

    epochs = int(training_cfg.get("epochs", 20))
    batch_size = int(training_cfg.get("batch_size", 4))
    num_workers = int(training_cfg.get("num_workers", 0))
    if epochs < 1:
        raise ValueError("training.epochs must be >= 1.")
    if batch_size < 1:
        raise ValueError("training.batch_size must be >= 1.")
    if num_workers < 0:
        raise ValueError("training.num_workers must be >= 0.")

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=adapter.collate_fn(),
    )

    val_loader = None
    if use_early_stopping and val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            collate_fn=adapter.collate_fn(),
        )

    scheduler_name = str(training_cfg.get("lr_scheduler") or "")
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-7
        )

    model_name = str(model_cfg.get("name", "custom"))
    if adapter_name == "pointwise":
        print(
            f"Training model='{model_name}' adapter='{adapter_name}' on "
            f"{len(train_case_idx)} train case(s) ({len(train_dataset)} samples), "
            f"{len(test_idx)} test case(s), device={device}."
        )
    else:
        print(
            f"Training model='{model_name}' adapter='{adapter_name}' on {len(train_dataset)} "
            f"train case(s), {len(test_idx)} test case(s), device={device}."
        )
    if use_early_stopping:
        print(f"Early stopping enabled (patience={patience}, val_cases={len(val_case_idx)}).")

    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0
    last_avg_loss = float("nan")
    epoch_iter = range(1, epochs + 1)
    epoch_progress = None
    if tqdm is not None:
        epoch_progress = tqdm(epoch_iter, total=epochs, desc="training", unit="epoch")
        epoch_iter = epoch_progress

    for epoch in epoch_iter:
        running_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            loss_value = experiment.training_step(batch)
            running_loss += loss_value
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("No training batches were produced.")

        avg_loss = running_loss / num_batches
        last_avg_loss = avg_loss
        experiment.on_epoch_end(int(epoch), avg_loss)

        if scheduler is not None:
            scheduler.step()

        if use_early_stopping and val_loader is not None:
            val_loss = compute_val_loss(experiment, val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if epoch_progress is not None:
                epoch_progress.set_postfix(
                    loss=f"{avg_loss:.3e}", val=f"{val_loss:.3e}", patience=patience_counter
                )
            else:
                print(
                    f"epoch {epoch}/{epochs}: loss={avg_loss:.6e} "
                    f"val_loss={val_loss:.6e} patience={patience_counter}/{patience}"
                )
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.6e}).")
                break
        else:
            if epoch_progress is not None:
                epoch_progress.set_postfix(loss=f"{avg_loss:.3e}")
            else:
                print(f"epoch {epoch}/{epochs}: loss={avg_loss:.6e}")

    if use_early_stopping and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Restored best weights (val_loss={best_val_loss:.6e}).")

    checkpoint_value = output_cfg.get("checkpoint")
    if not checkpoint_value:
        checkpoint_value = f"../data/models/{model_name}_model.mdlus"
    checkpoint_path = _resolve_path(str(checkpoint_value))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_path))

    model_params_resolved = _collect_resolved_model_params(model, model_params, dataset_info)
    split_meta: dict[str, Any] = {
        "strategy": split_cfg["strategy"],
        "train_sims": train_sims,
        "test_sims": test_sims,
    }
    if "train_ratio" in split_cfg:
        split_meta["train_ratio"] = float(split_cfg["train_ratio"])
    if "seed" in split_cfg:
        split_meta["seed"] = int(split_cfg["seed"])
    if split_cfg.get("strategy") == "file":
        split_meta["train_file"] = str(_resolve_path(str(split_cfg["train_file"])))
        split_meta["test_file"] = str(_resolve_path(str(split_cfg["test_file"])))

    if adapter_name == "pointwise":
        data_meta = {
            "zarr_dir": str(_resolve_path(str(data_cfg["zarr_dir"]))),
            "input_columns": list(dataset.input_columns),
            "output_columns": list(dataset.output_columns),
            "normalize": bool(getattr(dataset, "normalize", False)),
            "norm_stats": _serialize_norm_stats(getattr(dataset, "norm_stats", None)),
            "norm_fit_train_sims": [dataset.sim_names[i] for i in train_case_idx],
            "adapter": adapter_name,
        }
    else:
        data_meta = {
            "zarr_dir": str(_resolve_path(str(data_cfg["zarr_dir"]))),
            "input_fields": list(dataset.input_fields),
            "output_fields": list(dataset.output_fields),
            "input_time_idx": int(dataset.input_time_idx),
            "target_time_idx": int(dataset.target_time_idx),
        }

    run_meta = {
        "code_version": _git_code_version(),
        "model_name": model_name,
        "entrypoint": model_entrypoint_string(model_cfg, build_fn),
        "adapter": adapter_name,
        "model_params": model_params,
        "model_params_resolved": model_params_resolved,
        "data": data_meta,
        "split": split_meta,
        "training": {
            "epochs": epochs,
            "loss": loss_name,
            "lr": lr,
            "lr_scheduler": scheduler_name or None,
            "seed": seed,
            "final_train_loss": float(last_avg_loss),
            "best_val_loss": float(best_val_loss) if use_early_stopping else None,
            "experiment": experiment_entrypoint,
        },
        "checkpoint": str(checkpoint_path),
    }

    run_meta_path = checkpoint_path.with_name("run_meta.json")
    run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"Saved model checkpoint to {checkpoint_path}")
    print(f"Saved run metadata to {run_meta_path}")

    return {
        "checkpoint": str(checkpoint_path),
        "run_meta": str(run_meta_path),
        "final_train_loss": float(last_avg_loss),
        "train_cases": len(train_case_idx),
        "train_samples": len(train_dataset),
        "test_cases": len(test_idx),
    }


def _indices_for_test_split(
    sim_names: list[str],
    split_meta: dict[str, Any],
) -> tuple[list[int], list[str], list[str]]:
    train_sims = [str(name) for name in split_meta.get("train_sims", [])]
    test_sims = [str(name) for name in split_meta.get("test_sims", [])]

    if test_sims:
        sim_to_idx = {name: idx for idx, name in enumerate(sim_names)}
        unknown_test = [name for name in test_sims if name not in sim_to_idx]
        if unknown_test:
            raise ValueError(
                f"run_meta split contains unknown test sim name(s): {unknown_test}"
            )
        test_idx = [sim_to_idx[name] for name in test_sims]
        return test_idx, train_sims, test_sims

    reconstructed_split = _normalize_split_cfg(dict(split_meta), default_seed=42)
    train_idx, test_idx, train_sims, test_sims = split_indices(
        num_cases=len(sim_names),
        split_cfg=reconstructed_split,
        sim_names=sim_names,
    )
    _ = train_idx
    return test_idx, train_sims, test_sims


def evaluate(cfg: dict | Any) -> dict[str, Any]:
    """Evaluate checkpoint using run_meta.json to reconstruct dataset and split."""
    cfg_dict = _to_plain_dict(cfg)

    eval_cfg = dict(cfg_dict.get("eval") or {})
    output_cfg = dict(cfg_dict.get("output") or {})

    checkpoint_value = eval_cfg.get("checkpoint")
    if not checkpoint_value:
        raise ValueError("eval.checkpoint is required.")

    checkpoint_path = _resolve_path(str(checkpoint_value))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_meta_value = eval_cfg.get("run_meta")
    run_meta_path = (
        _resolve_path(str(run_meta_value))
        if run_meta_value
        else checkpoint_path.with_name("run_meta.json")
    )
    if not run_meta_path.exists():
        raise FileNotFoundError(
            f"run_meta.json not found: {run_meta_path}. "
            "Set eval.run_meta explicitly or train with the new runner first."
        )

    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    adapter_name = str(run_meta["adapter"])
    adapter = get_adapter(adapter_name)

    data_meta = dict(run_meta.get("data") or {})
    if not data_meta:
        raise ValueError(f"run_meta at {run_meta_path} is missing 'data' section.")

    if adapter_name == "pointwise":
        data_cfg = {
            "zarr_dir": data_meta["zarr_dir"],
            "input_columns": data_meta.get("input_columns"),
            "output_columns": data_meta.get("output_columns"),
            "normalize": bool(data_meta.get("normalize", False)),
            "norm_stats": data_meta.get("norm_stats"),
        }
    else:
        data_cfg = {
            "zarr_dir": data_meta["zarr_dir"],
            "input_fields": data_meta.get("input_fields"),
            "output_fields": data_meta.get("output_fields"),
            "input_time_idx": int(data_meta.get("input_time_idx", 0)),
            "target_time_idx": int(data_meta.get("target_time_idx", -1)),
        }

    dataset = adapter.build_dataset(data_cfg)
    split_meta = dict(run_meta.get("split") or {})
    test_idx, train_sims, test_sims = _indices_for_test_split(dataset.sim_names, split_meta)
    if hasattr(dataset, "subset_by_case_indices"):
        eval_dataset = dataset.subset_by_case_indices(test_idx)
    else:
        eval_dataset = Subset(dataset, test_idx)

    device = _resolve_device(str(eval_cfg.get("device", "auto")))
    batch_size = int(eval_cfg.get("batch_size", 4))
    num_workers = int(eval_cfg.get("num_workers", 0))
    if batch_size < 1:
        raise ValueError("eval.batch_size must be >= 1.")
    if num_workers < 0:
        raise ValueError("eval.num_workers must be >= 0.")

    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=adapter.collate_fn(),
    )

    module_cls = import_physicsnemo_attr("physicsnemo.core.module", "Module")
    model = module_cls.from_checkpoint(str(checkpoint_path)).to(device)

    loss_name = str(run_meta.get("training", {}).get("loss", "mse"))
    loss_fn = get_loss_fn(loss_name if loss_name in {"mse", "l1", "relative_l2"} else "mse")

    experiment_entrypoint = eval_cfg.get("experiment") or run_meta.get("training", {}).get("experiment")
    experiment = _build_experiment(
        experiment_entrypoint=experiment_entrypoint,
        model=model,
        optimizer=None,
        loss_fn=loss_fn,
        adapter=adapter,
        device=device,
    )

    if hasattr(dataset, "output_columns"):
        output_fields = list(dataset.output_columns)
    else:
        output_fields = list(dataset.output_fields)
    total_se_per_field = torch.zeros(len(output_fields), dtype=torch.float64)
    total_samples = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            pred, target = experiment.eval_step(batch)
            field_se, sample_count = adapter.accumulate_metrics(batch, pred, target)
            total_se_per_field += field_se.detach().to(torch.float64).cpu()
            total_samples += int(sample_count)
            all_preds.append(pred.detach().cpu())
            all_targets.append(target.detach().cpu())

    if total_samples == 0:
        raise RuntimeError("No evaluation samples were processed.")

    per_field_mse = total_se_per_field / float(total_samples)
    per_field_rmse = torch.sqrt(per_field_mse)
    overall_mse = float(per_field_mse.mean().item())
    overall_rmse = math.sqrt(overall_mse)

    # Extended metrics for pointwise adapter.
    extended_metrics: dict[str, Any] = {}
    if adapter_name == "pointwise" and hasattr(eval_dataset, "_row_case_idx"):
        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        extended_metrics = _compute_pointwise_extended_metrics(
            cat_preds, cat_targets, eval_dataset, output_fields,
        )

    plot_files: list[str] = []
    plot_dir_value = output_cfg.get("plot_dir")
    if plot_dir_value is not None:
        if adapter.family != "grid":
            raise ValueError(
                "Plotting currently supports only grid adapters. "
                f"Received adapter='{adapter.family}'."
            )

        plot_indices = resolve_plot_indices(
            num_cases=len(eval_dataset),
            raw_indices=output_cfg.get("plot_case_indices"),
            max_cases=int(output_cfg.get("plot_max_cases", 3)),
        )
        plot_files = save_grid_prediction_plots(
            model=model,
            dataset=eval_dataset,
            output_fields=output_fields,
            device=device,
            plot_dir=plot_dir_value,
            plot_indices=plot_indices,
            plot_cmap=str(output_cfg.get("plot_cmap", "viridis")),
            plot_dpi=int(output_cfg.get("plot_dpi", 150)),
            quiver_step=int(output_cfg.get("plot_quiver_step", 4)),
            vel_x_field=str(output_cfg.get("plot_velocity_x_field", "vel_x")),
            vel_y_field=str(output_cfg.get("plot_velocity_y_field", "vel_y")),
        )

    payload = {
        "zarr_dir": str(_resolve_path(str(data_cfg["zarr_dir"]))),
        "checkpoint": str(checkpoint_path),
        "run_meta": str(run_meta_path),
        "adapter": adapter.family,
        "num_cases": len(test_idx),
        "num_samples": total_samples,
        "train_cases": len(train_sims),
        "test_cases": len(test_idx),
        "overall": {
            "mse": overall_mse,
            "rmse": overall_rmse,
        },
        "per_field": [
            {
                "name": field_name,
                "mse": float(mse_value),
                "rmse": float(rmse_value),
            }
            for field_name, mse_value, rmse_value in zip(
                output_fields,
                per_field_mse.tolist(),
                per_field_rmse.tolist(),
            )
        ],
        "plots": {
            "plot_dir": str(_resolve_path(str(plot_dir_value)))
            if plot_dir_value is not None
            else None,
            "num_saved": len(plot_files),
            "files": plot_files,
        },
        "split": {
            "train_sims": train_sims,
            "test_sims": test_sims,
        },
    }
    if extended_metrics:
        payload["extended"] = extended_metrics

    print(
        f"Evaluated adapter='{adapter.family}' on {len(test_idx)} test case(s) "
        f"({total_samples} sample(s)), "
        f"overall mse={overall_mse:.6e}, rmse={overall_rmse:.6e}."
    )
    for row in payload["per_field"]:
        print(f"{row['name']}: mse={row['mse']:.6e}, rmse={row['rmse']:.6e}")
    if extended_metrics:
        _print_extended_metrics(extended_metrics)

    metrics_out_value = output_cfg.get("metrics_out")
    if metrics_out_value is not None:
        metrics_out_path = _resolve_path(str(metrics_out_value))
        metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved metrics JSON to {metrics_out_path}")

    if plot_files:
        print(f"Saved {len(plot_files)} plot(s) to {plot_dir_value}")

    return payload
