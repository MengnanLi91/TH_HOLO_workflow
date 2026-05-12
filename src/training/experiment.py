"""Experiment abstraction for optional custom training/eval steps."""

from collections.abc import Callable
from typing import Any

import torch


class Experiment:
    """Default experiment for supervised one-step training."""

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer | None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
        adapter,
        device: torch.device,
        **kwargs,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.adapter = adapter
        self.device = device

    def training_step(self, batch) -> float:
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("Experiment.training_step requires optimizer and loss_fn.")

        self.model.train()
        prepared = self.adapter.build_batch(batch, self.device)
        result = self.adapter.forward_train(self.model, prepared)
        if len(result) >= 3:
            pred, target, weight = result[0], result[1], result[2]
            loss = self.loss_fn(pred, target, weight)
        else:
            pred, target = result
            loss = self.loss_fn(pred, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def eval_step(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        prepared = self.adapter.build_batch(batch, self.device)
        return self.adapter.forward_eval(self.model, prepared)

    def validation_step(self, batch) -> float:
        """Compute validation loss for one batch.

        Override in subclasses for custom validation logic (e.g., weighted
        metrics, physics-informed losses).  The default uses ``eval_step``
        followed by ``loss_fn``.
        """
        if self.loss_fn is None:
            raise RuntimeError("Experiment.validation_step requires loss_fn.")
        self.model.eval()
        prepared = self.adapter.build_batch(batch, self.device)
        result = self.adapter.forward_train(self.model, prepared)
        if len(result) >= 3:
            pred, target, weight = result[0], result[1], result[2]
            return float(self.loss_fn(pred, target, weight).detach().cpu())
        pred, target = result
        return float(self.loss_fn(pred, target).detach().cpu())

    def validation_epoch_loss(self, val_loader) -> float:
        """Optional validation loss term computed once per epoch.

        Subclasses can override this to add case-level or physics-based
        penalties that require access to the full validation set rather than
        individual batches.
        """
        _ = val_loader
        return 0.0

    def on_epoch_end(self, epoch: int, avg_loss: float) -> None:
        _ = (epoch, avg_loss)

    def compute_val_delta_p_metric(self) -> float:
        """HPO-side Δp metric (mean squared log-Δp error on val).

        Default no-op for experiments without a Δp integral.  Override in
        Δp-aware subclasses (e.g. ``AlphaDExperiment``).
        """
        return 0.0

    def compute_extended_metrics(
        self,
        eval_dataset,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Per-experiment extended evaluation metrics.

        Receives per-batch CPU tensor lists (uncatted because graph adapters
        have variable shapes) and the eval dataset. Subclasses concatenate
        as needed and produce a metrics dict the runner merges into the
        ``extended`` block of the metrics JSON. Default: empty.
        """
        _ = (eval_dataset, all_preds, all_targets)
        return {}

    def print_extended_metrics(self, metrics: dict[str, Any]) -> None:
        """Pretty-print the dict returned by ``compute_extended_metrics``.

        Default: no-op (the generic runner already prints overall + per-field
        MSE/RMSE). Subclasses override to format case-specific blocks.
        """
        _ = metrics

    def decode_for_plotting(
        self,
        values: torch.Tensor,
        dataset,
        field_name: str,
        mask,
    ):
        """Decode encoded model output to physical/labelled space for plotting.

        Returns ``(decoded_ndarray, label)`` or ``None``. When ``None``, the
        plotter shows raw encoded values with ``field_name`` as the y-axis
        label and skips the physical-RMSE subtitle. Case experiments override
        to apply baseline re-add, unit conversion, etc.
        """
        _ = (values, dataset, field_name, mask)
        return None

    def prepare_for_training(
        self,
        train_dataset,
        val_dataset,
        device: torch.device,
    ) -> None:
        """Bind dataset-derived state onto the experiment before training.

        Called once after datasets are subsetted but before the training loop.
        Case experiments override to capture per-case geometry, target names,
        normalisation statistics, etc. that the generic core would otherwise
        have to read by name. Default: no-op.
        """
        _ = (train_dataset, val_dataset, device)

    def on_epoch_end_extra_step(self) -> None:
        """Optional extra gradient step run once per epoch.

        Case experiments override to apply secondary objectives that aren't
        well-expressed as per-batch losses (e.g. integral-of-profile losses
        that need a full case at a time). Default: no-op.
        """
