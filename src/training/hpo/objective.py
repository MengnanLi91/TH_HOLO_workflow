"""Best-checkpoint HPO training and composite objective evaluation."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import Any

import optuna
from torch.utils.data import Subset

from training.hpo.search_space import apply_overrides, sample_from_search_space
from training.losses import get_loss_fn
from training.models import get_build_fn_and_adapter
from training.runner import (
    build_experiment,
    compute_val_loss,
    set_seed,
    train_one_epoch,
)
from training.runtime import build_dataloader, build_optimizer, build_scheduler


def composite_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    """Return a finite weighted objective, rejecting missing or invalid inputs."""
    missing = sorted(set(weights) - set(metrics))
    if missing:
        raise ValueError(f"Experiment did not provide required HPO metric(s): {missing}")
    invalid = sorted(
        name
        for name in weights
        if not math.isfinite(float(metrics[name])) or not math.isfinite(float(weights[name]))
    )
    if invalid:
        raise ValueError(f"HPO objective metric(s) or weight(s) are non-finite: {invalid}")
    score = sum(float(weights[name]) * float(metrics[name]) for name in weights)
    if not math.isfinite(score):
        raise ValueError("Composite HPO score is non-finite.")
    return float(score)


def _subsets(dataset, train_indices: list[int], val_indices: list[int]):
    if hasattr(dataset, "subset_by_case_indices"):
        return (
            dataset.subset_by_case_indices(train_indices),
            dataset.subset_by_case_indices(val_indices),
        )
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def train_and_score(
    *,
    base_cfg: dict,
    params: dict[str, Any],
    phase_cfg: dict,
    objective_weights: dict[str, float],
    prepared: dict[str, Any],
    train_indices: list[int],
    val_indices: list[int],
    seed: int,
    trial: optuna.Trial | None = None,
) -> dict[str, Any]:
    """Train one candidate on one fold and score its restored best checkpoint."""
    trial_cfg = apply_overrides(base_cfg, params)
    model_cfg = dict(trial_cfg.get("model") or {})
    training_cfg = dict(trial_cfg.get("training") or {})
    training_cfg["seed"] = int(seed)
    set_seed(seed)

    adapter = prepared["adapter"]
    device = prepared["device"]
    candidate_prepared = prepared
    if any(path.startswith("data.") for path in params):
        data_cfg = dict(trial_cfg.get("data") or {})
        if bool(data_cfg.get("normalize", False)):
            if not adapter.supports_fold_normalization:
                raise ValueError(
                    f"Adapter '{prepared['adapter_name']}' cannot normalize an HPO fold."
                )
            data_cfg["norm_from_case_indices"] = list(train_indices)
        dataset = adapter.build_dataset(data_cfg)
        candidate_prepared = dict(prepared)
        candidate_prepared["dataset"] = dataset
        candidate_prepared["dataset_info"] = adapter.dataset_info(dataset)
    dataset = candidate_prepared["dataset"]
    train_dataset, val_dataset = _subsets(dataset, train_indices, val_indices)

    build_fn, adapter_name = get_build_fn_and_adapter(model_cfg)
    if adapter_name != prepared["adapter_name"]:
        raise ValueError("An HPO candidate cannot change the configured model adapter.")
    model = build_fn(dict(model_cfg.get("params") or {}), candidate_prepared["dataset_info"]).to(
        device
    )

    optimizer = build_optimizer(model.parameters(), training_cfg)
    loss_fn = get_loss_fn(str(training_cfg.get("loss", "mse")))
    experiment = build_experiment(
        experiment_entrypoint=training_cfg.get("experiment"),
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        adapter=adapter,
        device=device,
    )
    experiment.prepare_for_training(train_dataset, val_dataset, device)

    epochs = int(phase_cfg["max_epochs"])
    batch_size = int(training_cfg.get("batch_size", 4))
    num_workers = int(training_cfg.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": device,
        "collate_fn": adapter.collate_fn(),
        "config_prefix": "HPO",
    }
    train_loader = build_dataloader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = build_dataloader(val_dataset, shuffle=False, **loader_kwargs)
    scheduler = build_scheduler(optimizer, training_cfg, epochs)

    early_cfg = dict(phase_cfg["early_stopping"])
    patience = int(early_cfg["patience"])
    min_delta = float(early_cfg["min_delta"])
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    epochs_trained = 0

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(experiment, train_loader)
        experiment.on_epoch_end(epoch, avg_loss)
        experiment.on_epoch_end_extra_step()
        if scheduler is not None:
            scheduler.step()
        val_loss = float(compute_val_loss(experiment, val_loader))
        if not math.isfinite(val_loss):
            raise ValueError(f"Profile validation loss is non-finite at epoch {epoch}.")
        epochs_trained = epoch

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("HPO training did not produce a best validation checkpoint.")
    model.load_state_dict(best_state)

    restored_profile_loss = float(compute_val_loss(experiment, val_loader))
    case_metrics = experiment.compute_hpo_metrics(val_dataset)
    if not isinstance(case_metrics, dict):
        raise TypeError("Experiment.compute_hpo_metrics() must return a dict.")
    if "profile_val_loss" in case_metrics:
        raise ValueError("Experiments must not override the generic profile_val_loss metric.")
    metrics = {name: float(value) for name, value in case_metrics.items()}
    metrics["profile_val_loss"] = restored_profile_loss
    score = composite_score(metrics, objective_weights)
    return {
        "score": score,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
    }


def make_objective(
    base_cfg: dict,
    search_space: dict[str, dict],
    phase_cfg: dict,
    objective_weights: dict[str, float],
    prepared: dict[str, Any],
    train_indices: list[int],
    val_indices: list[int],
    seed: int,
) -> Callable[[optuna.Trial], float]:
    """Create the screening objective for one configured fold."""

    def objective(trial: optuna.Trial) -> float:
        params = sample_from_search_space(trial, search_space)
        result = train_and_score(
            base_cfg=base_cfg,
            params=params,
            phase_cfg=phase_cfg,
            objective_weights=objective_weights,
            prepared=prepared,
            train_indices=train_indices,
            val_indices=val_indices,
            seed=seed,
            trial=trial,
        )
        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("epochs_trained", result["epochs_trained"])
        trial.set_user_attr("composite_score", result["score"])
        for name, value in result["metrics"].items():
            trial.set_user_attr(f"metric:{name}", value)
        return float(result["score"])

    return objective
