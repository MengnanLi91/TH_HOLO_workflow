"""Optuna objective function for hyperparameter optimization."""

import random
from collections.abc import Callable
from typing import Any

import optuna
import torch
from torch.utils.data import DataLoader

from training.hpo.search_space import apply_overrides, sample_from_search_space
from training.losses import get_loss_fn
from training.models import get_build_fn_and_adapter
from training.runner import (
    _build_case_geometry,
    build_experiment,
    compute_val_loss,
    set_seed,
    train_one_epoch,
)


def make_objective(
    base_cfg: dict,
    search_space: dict[str, dict],
    hpo_cfg: dict,
    prepared: dict[str, Any],
    train_inner_idx: list[int],
    val_idx: list[int],
) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective function.

    The returned closure rebuilds only the model, optimizer, and loss per
    trial.  The dataset, adapter, and dataset_info are cached from
    *prepared* (the output of ``runner.prepare_training``).

    Parameters
    ----------
    base_cfg : dict
        Full training config **without** the ``hpo`` section.
    search_space : dict
        YAML search-space definition (dot-path -> spec).
    hpo_cfg : dict
        The ``hpo`` section of the config.
    prepared : dict
        Output of ``runner.prepare_training(base_cfg)``.
    train_inner_idx : list[int]
        Case indices (into ``dataset.sim_names``) for inner training.
    val_idx : list[int]
        Case indices for validation.
    """
    dataset = prepared["dataset"]
    adapter = prepared["adapter"]
    dataset_info = prepared["dataset_info"]
    device = prepared["device"]

    # Build train/val subsets once (shared across trials)
    if hasattr(dataset, "subset_by_case_indices"):
        train_ds = dataset.subset_by_case_indices(train_inner_idx)
        val_ds = dataset.subset_by_case_indices(val_idx)
    else:
        from torch.utils.data import Subset

        train_ds = Subset(dataset, train_inner_idx)
        val_ds = Subset(dataset, val_idx)

    def objective(trial: optuna.Trial) -> float:
        # 1. Sample hyperparameters and apply to config
        overrides = sample_from_search_space(trial, search_space)
        trial_cfg = apply_overrides(base_cfg, overrides)

        model_cfg = dict(trial_cfg.get("model") or {})
        training_cfg = dict(trial_cfg.get("training") or {})

        seed = int(training_cfg.get("seed", 42))
        set_seed(seed)

        # 2. Build model (cheap per trial)
        build_fn, _ = get_build_fn_and_adapter(model_cfg)
        model_params = dict(model_cfg.get("params") or {})
        model = build_fn(model_params, dataset_info).to(device)

        # 3. Build optimizer and loss
        lr = float(training_cfg.get("lr", 1e-3))
        weight_decay = float(training_cfg.get("weight_decay", 0.0))
        if weight_decay > 0:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = get_loss_fn(str(training_cfg.get("loss", "mse")))

        # 4. Build experiment (respects training.experiment entrypoint)
        experiment_kwargs: dict[str, Any] = {}
        consistency_weight = float(training_cfg.get("consistency_weight", 0.0))
        if consistency_weight > 0:
            experiment_kwargs["consistency_weight"] = consistency_weight
        delta_p_weight = float(training_cfg.get("delta_p_weight", 0.0))
        if delta_p_weight > 0:
            experiment_kwargs["delta_p_weight"] = delta_p_weight
        experiment = build_experiment(
            experiment_entrypoint=training_cfg.get("experiment"),
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            adapter=adapter,
            device=device,
            **experiment_kwargs,
        )
        if hasattr(train_ds, "output_columns") and getattr(train_ds, "output_columns", None):
            experiment.alpha_d_target_name = str(train_ds.output_columns[0])

        # Inject case geometry for delta_p integral loss
        if delta_p_weight > 0 and hasattr(train_ds, "_case_meta"):
            experiment.case_geometry = _build_case_geometry(train_ds, device)
            experiment.local_velocity_normalization = getattr(
                train_ds, "local_velocity_normalization", False
            )
            if hasattr(val_ds, "_case_meta"):
                experiment.val_case_geometry = _build_case_geometry(val_ds, device)

        # 5. Build DataLoaders
        epochs = int(training_cfg.get("epochs", 20))
        batch_size = int(training_cfg.get("batch_size", 4))
        num_workers = int(training_cfg.get("num_workers", 0))

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=adapter.collate_fn(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=adapter.collate_fn(),
        )

        # 6. LR scheduler (match retrain behavior)
        scheduler_name = str(training_cfg.get("lr_scheduler") or "")
        scheduler = None
        if scheduler_name == "cosine":
            warmup_epochs = int(training_cfg.get("lr_warmup_epochs", 0))
            if warmup_epochs > 0 and warmup_epochs < epochs:
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
                )
                cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=1e-7
                )

        # 7. Training loop with pruning
        val_loss = float("nan")
        for epoch in range(1, epochs + 1):
            train_one_epoch(experiment, train_loader)
            if hasattr(experiment, "compute_delta_p_loss_step"):
                experiment.compute_delta_p_loss_step()
            if scheduler is not None:
                scheduler.step()
            val_loss = compute_val_loss(experiment, val_loader)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_loss

    return objective
