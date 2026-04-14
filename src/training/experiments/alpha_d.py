"""Alpha-D experiment with case-level pressure-drop consistency loss.

The consistency loss encourages the model to match the *integrated*
alpha_D profile per case, not just pointwise values.  For each unique
case present in a training batch the mean of ``exp(log_alpha_D)`` is
computed for both predictions and targets (after denormalisation) and
the squared difference is penalised.

Usage -- set in the YAML config::

    training:
      experiment: training.experiments.alpha_d:AlphaDExperiment
      consistency_weight: 0.1
"""

import torch

from training.experiment import Experiment


class AlphaDExperiment(Experiment):
    """Extends the default experiment with a case-level consistency loss.

    Parameters
    ----------
    consistency_weight : float
        Scaling factor for the consistency term.  Set to 0 to disable.
    norm_stats : dict or None
        ``{"y_mean": Tensor, "y_std": Tensor}`` used to denormalise
        predictions before integrating.  If *None* the consistency loss
        operates on normalised values (less physically meaningful but
        still regularises).
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        adapter,
        device,
        *,
        consistency_weight: float = 0.0,
        norm_stats: dict | None = None,
        **kwargs,
    ):
        super().__init__(model, optimizer, loss_fn, adapter, device, **kwargs)
        self.consistency_weight = consistency_weight
        self.y_mean = None
        self.y_std = None
        if norm_stats and "y_mean" in norm_stats and "y_std" in norm_stats:
            self.y_mean = torch.as_tensor(norm_stats["y_mean"], dtype=torch.float32).to(device)
            self.y_std = torch.as_tensor(norm_stats["y_std"], dtype=torch.float32).to(device)

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def training_step(self, batch) -> float:
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("AlphaDExperiment.training_step requires optimizer and loss_fn.")

        self.model.train()
        prepared = self.adapter.build_batch(batch, self.device)
        result = self.adapter.forward_train(self.model, prepared)

        # Unpack -- 4-tuple when include_case_idx is True
        case_idx = None
        if len(result) == 4:
            pred, target, weight, case_idx = result
            pointwise_loss = self.loss_fn(pred, target, weight)
        elif len(result) == 3:
            pred, target, weight = result
            pointwise_loss = self.loss_fn(pred, target, weight)
        else:
            pred, target = result
            pointwise_loss = self.loss_fn(pred, target)

        # Consistency loss (only when case indices are available)
        loss = pointwise_loss
        if self.consistency_weight > 0 and case_idx is not None:
            c_loss = self._consistency_loss(pred, target, case_idx)
            loss = loss + self.consistency_weight * c_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    # ------------------------------------------------------------------
    # Validation step (includes consistency for comparable metrics)
    # ------------------------------------------------------------------

    def validation_step(self, batch) -> float:
        if self.loss_fn is None:
            raise RuntimeError("AlphaDExperiment.validation_step requires loss_fn.")
        self.model.eval()
        prepared = self.adapter.build_batch(batch, self.device)
        result = self.adapter.forward_train(self.model, prepared)

        case_idx = None
        if len(result) == 4:
            pred, target, weight, case_idx = result
            pointwise_loss = self.loss_fn(pred, target, weight)
        elif len(result) == 3:
            pred, target, weight = result
            pointwise_loss = self.loss_fn(pred, target, weight)
        else:
            pred, target = result
            pointwise_loss = self.loss_fn(pred, target)

        loss = pointwise_loss
        if self.consistency_weight > 0 and case_idx is not None:
            c_loss = self._consistency_loss(pred, target, case_idx)
            loss = loss + self.consistency_weight * c_loss

        return float(loss.detach().cpu())

    # ------------------------------------------------------------------
    # Consistency loss implementation
    # ------------------------------------------------------------------

    def _consistency_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        case_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Per-case mean-alpha_D consistency loss.

        Denormalises predictions and targets back to ``log(alpha_D)``
        space, exponentiates to get ``alpha_D``, computes the case-level
        mean, and penalises squared differences.
        """
        # Denormalise if stats are available
        if self.y_mean is not None and self.y_std is not None:
            pred_log = pred * self.y_std + self.y_mean
            tgt_log = target * self.y_std + self.y_mean
        else:
            pred_log = pred
            tgt_log = target

        # exp to get alpha_D (clamp for numerical stability)
        pred_alpha = torch.exp(pred_log.clamp(max=20.0))
        tgt_alpha = torch.exp(tgt_log.clamp(max=20.0))

        # Scatter-mean by case index
        unique_cases = case_idx.unique()
        n_cases = len(unique_cases)
        if n_cases == 0:
            return pred.new_tensor(0.0)

        total_se = pred.new_tensor(0.0)
        for c in unique_cases:
            mask = case_idx == c
            pred_mean = pred_alpha[mask].mean()
            tgt_mean = tgt_alpha[mask].mean()
            total_se = total_se + (pred_mean - tgt_mean) ** 2

        return total_se / n_cases
