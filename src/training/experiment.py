"""Experiment abstraction for optional custom training/eval steps."""

from collections.abc import Callable

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

    def on_epoch_end(self, epoch: int, avg_loss: float) -> None:
        _ = (epoch, avg_loss)
