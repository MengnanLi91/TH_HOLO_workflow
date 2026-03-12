"""Generic training/evaluation runners for supervised one-step models."""

import importlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

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


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but no CUDA device is available.")
    return torch.device(device_arg)


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


def _build_experiment(
    experiment_entrypoint: str | None,
    model,
    optimizer,
    loss_fn,
    adapter,
    device: torch.device,
) -> Experiment:
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
    )

    if not hasattr(experiment, "training_step") or not hasattr(experiment, "eval_step"):
        raise TypeError(
            "Experiment class must define training_step() and eval_step() methods."
        )

    return experiment


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


def _normalize_split_cfg(split_cfg: dict, default_seed: int) -> dict[str, Any]:
    normalized = dict(split_cfg)
    normalized.setdefault("strategy", "sequential")
    if normalized["strategy"] in {"sequential", "random"}:
        normalized.setdefault("train_ratio", 0.8)
    if normalized["strategy"] == "random":
        normalized.setdefault("seed", default_seed)
    return normalized


def train(cfg: dict | Any) -> dict[str, Any]:
    """Train a supervised model and save checkpoint + run_meta.json."""
    cfg_dict = _to_plain_dict(cfg)

    model_cfg = dict(cfg_dict.get("model") or {})
    data_cfg = dict(cfg_dict.get("data") or {})
    training_cfg = dict(cfg_dict.get("training") or {})
    output_cfg = dict(cfg_dict.get("output") or {})

    if not data_cfg.get("zarr_dir"):
        raise ValueError("data.zarr_dir is required.")

    seed = int(training_cfg.get("seed", 42))
    _set_seed(seed)

    device = _resolve_device(str(training_cfg.get("device", "auto")))
    build_fn, adapter_name = get_build_fn_and_adapter(model_cfg)
    adapter = get_adapter(adapter_name)

    dataset = adapter.build_dataset(data_cfg)
    dataset_info = adapter.dataset_info(dataset)

    model_params = dict(model_cfg.get("params") or {})
    model = build_fn(model_params, dataset_info).to(device)

    loss_name = str(training_cfg.get("loss", "mse"))
    loss_fn = get_loss_fn(loss_name)
    lr = float(training_cfg.get("lr", 1.0e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    experiment_entrypoint = training_cfg.get("experiment")
    experiment = _build_experiment(
        experiment_entrypoint=experiment_entrypoint,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        adapter=adapter,
        device=device,
    )

    split_cfg = _normalize_split_cfg(dict(data_cfg.get("split") or {}), default_seed=seed)
    train_idx, test_idx, train_sims, test_sims = split_indices(
        num_cases=len(dataset),
        split_cfg=split_cfg,
        sim_names=dataset.sim_names,
    )

    train_dataset = Subset(dataset, train_idx)

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

    model_name = str(model_cfg.get("name", "custom"))
    print(
        f"Training model='{model_name}' adapter='{adapter_name}' on {len(train_dataset)} "
        f"train case(s), {len(test_idx)} test case(s), device={device}."
    )

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

        if epoch_progress is not None:
            epoch_progress.set_postfix(loss=f"{avg_loss:.3e}")
        else:
            print(f"epoch {epoch}/{epochs}: loss={avg_loss:.6e}")

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
            "seed": seed,
            "final_train_loss": float(last_avg_loss),
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
        "train_cases": len(train_dataset),
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

    output_fields = list(dataset.output_fields)
    total_se_per_field = torch.zeros(len(output_fields), dtype=torch.float64)
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            pred, target = experiment.eval_step(batch)
            field_se, sample_count = adapter.accumulate_metrics(batch, pred, target)
            total_se_per_field += field_se.detach().to(torch.float64).cpu()
            total_samples += int(sample_count)

    if total_samples == 0:
        raise RuntimeError("No evaluation samples were processed.")

    per_field_mse = total_se_per_field / float(total_samples)
    per_field_rmse = torch.sqrt(per_field_mse)
    overall_mse = float(per_field_mse.mean().item())
    overall_rmse = math.sqrt(overall_mse)

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
        "num_cases": len(eval_dataset),
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

    print(
        f"Evaluated adapter='{adapter.family}' on {len(eval_dataset)} test case(s), "
        f"overall mse={overall_mse:.6e}, rmse={overall_rmse:.6e}."
    )
    for row in payload["per_field"]:
        print(f"{row['name']}: mse={row['mse']:.6e}, rmse={row['rmse']:.6e}")

    metrics_out_value = output_cfg.get("metrics_out")
    if metrics_out_value is not None:
        metrics_out_path = _resolve_path(str(metrics_out_value))
        metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved metrics JSON to {metrics_out_path}")

    if plot_files:
        print(f"Saved {len(plot_files)} plot(s) to {plot_dir_value}")

    return payload
