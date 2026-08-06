"""Shared construction helpers for training runtime objects."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.utils.data import DataLoader


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter], training_cfg: dict[str, Any]
) -> torch.optim.Optimizer:
    """Build the configured Adam-family optimizer."""
    lr = float(training_cfg.get("lr", 1.0e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    if weight_decay > 0.0:
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(parameters, lr=lr)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: dict[str, Any],
    epochs: int,
):
    """Build the optional cosine scheduler, including configured warmup."""
    if str(training_cfg.get("lr_scheduler") or "") != "cosine":
        return None

    warmup_epochs = int(training_cfg.get("lr_warmup_epochs", 0))
    if 0 < warmup_epochs < epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0e-3,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=1.0e-7,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1.0e-7,
    )


def build_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    collate_fn=None,
    config_prefix: str | None = None,
) -> DataLoader:
    """Build a DataLoader with the project's device and worker defaults."""
    prefix = f"{config_prefix}." if config_prefix else ""
    if batch_size < 1:
        raise ValueError(f"{prefix}batch_size must be >= 1.")
    if num_workers < 0:
        raise ValueError(f"{prefix}num_workers must be >= 0.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
