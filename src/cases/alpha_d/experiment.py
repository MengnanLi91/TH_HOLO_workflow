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
from typing import Any

import torch

from training.experiment import Experiment
from cases.alpha_d.physics.targets import alpha_d_values_to_bulk, decode_alpha_d_target


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

    def compute_val_delta_p_metric(self) -> float:
        """Mean squared log-Δp error on the validation set.

        Independent of ``delta_p_weight`` so HPO can use it as an
        objective-side metric (compare a ``delta_p_weight=0`` trial
        fairly against a non-zero one).  Returns 0.0 if no validation
        geometry has been registered.
        """
        if not self.val_case_geometry:
            return 0.0
        self.model.eval()
        with torch.no_grad():
            mean_loss = self._mean_delta_p_loss(self.val_case_geometry)
        if mean_loss is None:
            return 0.0
        return float(mean_loss.detach().cpu())

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
        delta_p_gt = geo.get("delta_p_case", 0.0)
        if delta_p_gt <= 0:
            return None

        pred_values = pred_case.squeeze(-1)  # [n_stations]

        d_over_D = geo["d_local_over_D"]  # [n_stations], already on device
        D_big = geo["D_big"]

        baseline_encoded = geo.get("baseline_encoded")
        if baseline_encoded is not None:
            bl = baseline_encoded[:, 0] if baseline_encoded.dim() == 2 else baseline_encoded
            pred_values = pred_values + bl.to(pred_values.dtype).to(pred_values.device)

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

    # ------------------------------------------------------------------
    # Phase 2a evaluation hooks
    # ------------------------------------------------------------------

    def compute_extended_metrics(
        self,
        eval_dataset,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Pointwise + Δp metrics for the alpha-D adapter.

        Requires a TabularPairDataset / AlphaDProfileDataset (gated by
        ``_row_case_idx``). Other adapters fall through to ``{}``.
        """
        from cases.alpha_d.metrics import (
            compute_delta_p_metrics,
            compute_pointwise_extended_metrics,
        )

        if not hasattr(eval_dataset, "_row_case_idx"):
            return {}

        output_fields = list(getattr(eval_dataset, "output_columns", []))
        if not output_fields:
            return {}

        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)

        metrics = compute_pointwise_extended_metrics(
            cat_preds, cat_targets, eval_dataset, output_fields,
        )

        local_vel_norm = bool(
            getattr(eval_dataset, "local_velocity_normalization", False)
        )
        dp_metrics = compute_delta_p_metrics(
            self.model,
            eval_dataset,
            self.device,
            alpha_d_target_name=str(output_fields[0]),
            local_velocity_normalization=local_vel_norm,
        )
        if dp_metrics:
            metrics["delta_p"] = dp_metrics
        return metrics

    def print_extended_metrics(self, metrics: dict[str, Any]) -> None:
        from cases.alpha_d.metrics import print_extended_metrics as _print
        _print(metrics)

    # ------------------------------------------------------------------
    # Phase 2d training-lifecycle hooks
    # ------------------------------------------------------------------

    def prepare_for_training(
        self,
        train_dataset,
        val_dataset,
        device: torch.device,
    ) -> None:
        """Bind alpha-D-specific state from the datasets onto the experiment."""
        if hasattr(train_dataset, "output_columns") and train_dataset.output_columns:
            self.alpha_d_target_name = str(train_dataset.output_columns[0])

        # y_mean/y_std for consistency-loss denormalisation.
        if self.consistency_weight > 0:
            norm_stats = getattr(train_dataset, "norm_stats", None)
            if norm_stats:
                y_mean = norm_stats.get("y_mean")
                y_std = norm_stats.get("y_std")
                if y_mean is not None:
                    self.y_mean = y_mean.to(device) if hasattr(y_mean, "to") else y_mean
                if y_std is not None:
                    self.y_std = y_std.to(device) if hasattr(y_std, "to") else y_std

        # Per-case geometry for the Δp integral loss step (train side) — only
        # needed when delta_p_weight > 0 since the gradient step is gated.
        if self.delta_p_weight > 0 and hasattr(train_dataset, "_case_meta"):
            self.case_geometry = self._build_case_geometry(train_dataset, device)
            self.local_velocity_normalization = bool(
                getattr(train_dataset, "local_velocity_normalization", False)
            )

        # Val-side geometry is needed by ``compute_val_delta_p_metric`` for
        # HPO scoring even when delta_p_weight=0, so build whenever val
        # data is available with the metadata it requires.
        if val_dataset is not None and hasattr(val_dataset, "_case_meta"):
            self.val_case_geometry = self._build_case_geometry(val_dataset, device)
            if not self.local_velocity_normalization:
                self.local_velocity_normalization = bool(
                    getattr(val_dataset, "local_velocity_normalization", False)
                )

    def on_epoch_end_extra_step(self) -> None:
        """Run the Δp integral loss step when configured (no-op otherwise)."""
        if self.delta_p_weight > 0:
            self.compute_delta_p_loss_step()

    @staticmethod
    def _build_case_geometry(
        ds,
        device: torch.device,
    ) -> dict[int, dict[str, Any]]:
        """Per-case geometry dicts for the Δp integral loss.

        Returns ``{case_idx: {x_full, z_hat, d_local_over_D, n_stations,
        L_roi, D_big, delta_p_case, rho, V_bulk, baseline_encoded}}``.
        Geometry constants come from each case's metadata; older zarrs
        without them fall back to the historical AlphaD-ETL defaults.
        """
        case_geometry: dict[int, dict[str, Any]] = {}
        for ci in range(len(ds._case_ids_unique)):
            mask = ds._row_case_idx == ci
            cm = ds._case_meta[ci]

            D_big = float(cm.get("D_big", 0.2))
            outer_height_m = float(cm.get("outer_height_m", 1.0))
            buffer_diams = float(cm.get("buffer_diams", 1.0))
            rho = float(cm.get("rho", 1.0))
            V_bulk = float(cm.get("V_bulk", 1.0))

            L_roi = float(cm.get("Lr", 0.0)) * outer_height_m + 2.0 * buffer_diams * D_big

            baseline_encoded = getattr(ds, "_baseline_encoded", None)
            case_geometry[ci] = {
                "x_full": ds._x[mask].to(device),
                "z_hat": ds._raw_z_hat[mask].to(device) if ds._raw_z_hat is not None else None,
                "d_local_over_D": (
                    ds._raw_d_local_over_D[mask].to(device)
                    if ds._raw_d_local_over_D is not None else None
                ),
                "n_stations": int(mask.sum()),
                "L_roi": L_roi,
                "D_big": D_big,
                "delta_p_case": float(cm.get("delta_p_case", 0.0)),
                "rho": rho,
                "V_bulk": V_bulk,
                "baseline_encoded": (
                    baseline_encoded[mask].to(device)
                    if baseline_encoded is not None else None
                ),
            }
        return case_geometry

    def decode_for_plotting(
        self,
        values: torch.Tensor,
        dataset,
        field_name: str,
        mask,
    ):
        """Re-add encoded baseline, decode to bulk α_D for profile plotting."""
        from cases.alpha_d.physics.targets import (
            field_values_to_physical,
            is_alpha_d_target,
        )

        values = values.detach().cpu().clone()

        baseline_encoded = getattr(dataset, "_baseline_encoded", None)
        if (
            getattr(dataset, "has_target_baseline", False)
            and baseline_encoded is not None
        ):
            bl = baseline_encoded[mask][:, 0].to(values.dtype)
            values = values + bl

        d_over_D_attr = getattr(dataset, "_raw_d_local_over_D", None)
        if d_over_D_attr is not None:
            d_over_D = d_over_D_attr[mask].detach().cpu()
        else:
            d_over_D = None

        decoded = field_values_to_physical(
            values,
            field_name=field_name,
            d_over_D=d_over_D,
            local_velocity_normalization=bool(
                getattr(dataset, "local_velocity_normalization", False)
            ),
        )
        label = "alpha_D" if is_alpha_d_target(field_name) else field_name
        return decoded.detach().cpu().numpy(), label
