"""Loss registry for supervised one-step training."""

from collections.abc import Callable

import torch


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.norm(pred - target)
    denominator = torch.linalg.norm(target).clamp_min(1e-12)
    return numerator / denominator


LOSS_REGISTRY: dict[str, LossFn] = {
    "mse": mse_loss,
    "l1": l1_loss,
    "relative_l2": relative_l2_loss,
}


def get_loss_fn(name: str) -> LossFn:
    if name not in LOSS_REGISTRY:
        available = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(f"Unknown loss '{name}'. Available losses: {available}")
    return LOSS_REGISTRY[name]
