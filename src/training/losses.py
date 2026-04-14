"""Loss registry for supervised one-step training."""

from collections.abc import Callable

import torch


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    se = (pred - target) ** 2
    if weight is not None:
        se = se * weight
    return se.mean()


def l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    ae = torch.abs(pred - target)
    if weight is not None:
        ae = ae * weight
    return ae.mean()


def relative_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = pred - target
    if weight is not None:
        diff = diff * weight.sqrt()
    numerator = torch.linalg.norm(diff)
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
