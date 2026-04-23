"""Alpha-D experiment with case-level consistency and pressure-drop losses.

The consistency loss encourages the model to match the *integrated*
alpha_D profile per case, not just pointwise values.

The pressure-drop loss integrates the predicted alpha_D profile per case
via the trapezoidal rule and penalises mismatch vs the ground-truth
``delta_p_case`` stored in the zarr metadata.

Usage -- set in the YAML config::

    training:
      experiment: training.experiments.alpha_d:AlphaDExperiment
      consistency_weight: 0.0
      delta_p_weight: 0.1
"""

import math

import torch

from training.experiment import Experiment
from training.alpha_d_targets import alpha_d_values_to_bulk, decode_alpha_d_target


class AlphaDExperiment(Experiment):
    """Extends the default experiment with case-level physics losses.

    Parameters
    ----------
    consistency_weight : float
        Scaling factor for the consistency term.  Set to 0 to disable.
    delta_p_weight : float
        Scaling factor for the pressure-drop integral loss.  Set to 0 to
        disable.
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
        delta_p_weight: float = 0.0,
        norm_stats: dict | None = None,
        **kwargs,
    ):
        super().__init__(model, optimizer, loss_fn, adapter, device, **kwargs)
        self.consistency_weight = consistency_weight
        self.delta_p_weight = delta_p_weight
        self.y_mean = None
        self.y_std = None
        if norm_stats and "y_mean" in norm_stats and "y_std" in norm_stats:
            self.y_mean = torch.as_tensor(norm_stats["y_mean"], dtype=torch.float32).to(device)
            self.y_std = torch.as_tensor(norm_stats["y_std"], dtype=torch.float32).to(device)

        # Populated by runner after construction
        self.case_geometry: dict[int, dict] = {}
        self.val_case_geometry: dict[int, dict] = {}
        self.local_velocity_normalization: bool = False
        self.alpha_d_target_name: str = "log_alpha_D"

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

        Decodes predictions and targets back to their alpha_D-space
        representation, computes the case-level mean, and penalises
        squared differences.
        """
        # Denormalise if stats are available
        if self.y_mean is not None and self.y_std is not None:
            pred_values = pred * self.y_std + self.y_mean
            tgt_values = target * self.y_std + self.y_mean
        else:
            pred_values = pred
            tgt_values = target

        pred_alpha = decode_alpha_d_target(
            pred_values,
            target_name=self.alpha_d_target_name,
        )
        tgt_alpha = decode_alpha_d_target(
            tgt_values,
            target_name=self.alpha_d_target_name,
        )

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

    # ------------------------------------------------------------------
    # Pressure-drop integral loss (per-epoch step)
    # ------------------------------------------------------------------

    def _mean_delta_p_loss(
        self,
        case_geometry: dict[int, dict],
    ) -> torch.Tensor | None:
        """Average unweighted log-space delta_p loss across a case set."""
        if not case_geometry:
            return None

        case_ids = list(case_geometry.keys())
        chunk_size = 50  # cases per forward-pass chunk

        total_loss = torch.tensor(0.0, device=self.device)
        n_valid = 0

        for chunk_start in range(0, len(case_ids), chunk_size):
            chunk = case_ids[chunk_start : chunk_start + chunk_size]
            x_list = [case_geometry[ci]["x_full"] for ci in chunk]
            x_batch = torch.cat(x_list, dim=0)

            pred_batch = self.model(x_batch)

            offset = 0
            for ci in chunk:
                geo = case_geometry[ci]
                n = geo["n_stations"]
                pred_case = pred_batch[offset : offset + n]
                case_loss = self._single_case_dp_loss(pred_case, geo)
                if case_loss is not None:
                    total_loss = total_loss + case_loss
                    n_valid += 1
                offset += n

        if n_valid == 0:
            return None
        return total_loss / n_valid

    def compute_delta_p_loss_step(self) -> float:
        """Run a separate gradient step enforcing pressure-drop consistency.

        For each training case, forward-pass all stations through the model,
        integrate the predicted alpha_D profile to obtain ``delta_p_pred``,
        and penalise the log-space squared error vs. ground truth.

        Returns the scalar loss value (0.0 if disabled or no geometry).
        """
        if self.delta_p_weight <= 0 or not self.case_geometry:
            return 0.0
        if self.optimizer is None:
            return 0.0

        self.model.train()
        mean_loss = self._mean_delta_p_loss(self.case_geometry)
        if mean_loss is None:
            return 0.0

        loss = self.delta_p_weight * mean_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return float(loss.detach().cpu())

    def validation_epoch_loss(self, val_loader) -> float:
        """Add validation delta_p loss so model selection matches training."""
        _ = val_loader
        if self.delta_p_weight <= 0 or not self.val_case_geometry:
            return 0.0

        self.model.eval()
        with torch.no_grad():
            mean_loss = self._mean_delta_p_loss(self.val_case_geometry)
        if mean_loss is None:
            return 0.0
        return float((self.delta_p_weight * mean_loss).detach().cpu())

    def _single_case_dp_loss(
        self,
        pred_case: torch.Tensor,
        geo: dict,
    ) -> torch.Tensor | None:
        """Compute log-space delta_p error for one case.

        Parameters
        ----------
        pred_case : Tensor [n_stations, 1]
            Raw model predictions in the configured alpha_D target space.
        geo : dict
            Per-case geometry with keys ``z_hat``, ``d_local_over_D``,
            ``L_roi``, ``D_big``, ``delta_p_case``, ``rho``, ``V_bulk``.
        """
        delta_p_gt = geo["delta_p_case"]
        if delta_p_gt <= 0:
            return None

        pred_values = pred_case.squeeze(-1)  # [n_stations]

        d_over_D = geo["d_local_over_D"]  # [n_stations], already on device
        D_big = geo["D_big"]

        alpha_D_bulk = alpha_d_values_to_bulk(
            pred_values,
            target_name=self.alpha_d_target_name,
            d_over_D=d_over_D,
            local_velocity_normalization=self.local_velocity_normalization,
        )

        # dp/dz = alpha_D_bulk * rho * V_bulk^2 / (2 * D_h)
        D_h = d_over_D * D_big
        rho = geo["rho"]
        V_bulk = geo["V_bulk"]
        dp_dz = alpha_D_bulk * rho * V_bulk ** 2 / (2.0 * D_h)

        # Trapezoidal integration over physical z
        z_physical = geo["z_hat"] * geo["L_roi"]
        delta_p_pred = torch.trapezoid(dp_dz, z_physical)

        # Log-space squared error
        log_pred = torch.log(delta_p_pred.clamp(min=1e-8))
        log_gt = math.log(max(delta_p_gt, 1e-8))
        return (log_pred - log_gt) ** 2
