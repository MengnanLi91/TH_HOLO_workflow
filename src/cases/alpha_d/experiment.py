"""Alpha-D experiment.

Specialises :class:`training.experiment.Experiment` with the alpha-D
case's evaluation metrics (per-region pointwise + Δp integral) and the
plotting hooks the runner uses for per-case profile and parity plots.
The training step itself is inherited unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

from training.experiment import Experiment


class AlphaDExperiment(Experiment):
    """Alpha-D-specific evaluation + plotting hooks."""

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        adapter,
        device,
        **kwargs,
    ):
        super().__init__(model, optimizer, loss_fn, adapter, device, **kwargs)
        self.local_velocity_normalization: bool = False
        self.alpha_d_target_name: str = "log_alpha_D"

    # ------------------------------------------------------------------
    # Evaluation hooks
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

        # Profile adapter emits per-case [N_cases, O, S]; the pointwise
        # metrics helper expects flat [N_rows, O] aligned with the inner
        # TabularPairDataset row layout. Scatter back via _case_slices so
        # the same helper covers both adapters.
        if cat_preds.dim() == 3:
            case_slices = getattr(eval_dataset, "_case_slices", None)
            if case_slices is None:
                return {}
            n_rows = sum(len(s) for s in case_slices)
            n_fields = cat_preds.shape[1]
            flat_preds = torch.empty(n_rows, n_fields, dtype=cat_preds.dtype)
            flat_targets = torch.empty(n_rows, n_fields, dtype=cat_targets.dtype)
            for ci, rows in enumerate(case_slices):
                rows_t = torch.as_tensor(rows, dtype=torch.long)
                flat_preds[rows_t] = cat_preds[ci].T  # [S, O]
                flat_targets[rows_t] = cat_targets[ci].T
            cat_preds, cat_targets = flat_preds, flat_targets

        metrics = compute_pointwise_extended_metrics(
            cat_preds,
            cat_targets,
            eval_dataset,
            output_fields,
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

    def compute_hpo_metrics(self, validation_dataset) -> dict[str, float]:
        """Return pressure-drop metrics used by the alpha-D HPO objective."""
        from cases.alpha_d.metrics import compute_delta_p_metrics

        metrics = compute_delta_p_metrics(
            self.model,
            validation_dataset,
            self.device,
            alpha_d_target_name=self.alpha_d_target_name,
            local_velocity_normalization=bool(
                getattr(validation_dataset, "local_velocity_normalization", False)
            ),
        )
        if not metrics:
            raise ValueError(
                "Alpha-D HPO could not compute pressure-drop validation metrics."
            )
        return {
            "delta_p_log_abs_error_mean": float(metrics["log_abs_error_mean"]),
            "delta_p_relative_error_p90": float(metrics["relative_error_p90"]),
            "delta_p_signed_bias": float(metrics["signed_bias"]),
            "max_abs_lr_level_bias": float(metrics["max_abs_lr_level_bias"]),
        }

    # ------------------------------------------------------------------
    # Training-lifecycle hooks
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

        # Track the local-velocity-normalisation flag so decode_for_plotting
        # and the eval-time Δp integral pick the right basis.
        self.local_velocity_normalization = bool(
            getattr(train_dataset, "local_velocity_normalization", False)
        )
        if not self.local_velocity_normalization and val_dataset is not None:
            self.local_velocity_normalization = bool(
                getattr(val_dataset, "local_velocity_normalization", False)
            )

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
        values = dataset.add_baseline_to_encoded(values, row_mask=mask, field_idx=0)

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

    def baseline_for_plotting(self, dataset, field_name: str, mask):
        """Decode the analytical alpha-D baseline for the masked stations.

        Reuses ``decode_for_plotting`` with a zero residual so the physical
        decode pipeline (baseline re-add + unit conversion) stays in one
        place.
        """
        if not getattr(dataset, "has_target_baseline", False):
            return None
        if getattr(dataset, "_baseline_encoded", None) is None:
            return None
        n = int(mask.sum().item())
        zero = torch.zeros(n, dtype=torch.float32)
        return self.decode_for_plotting(zero, dataset, field_name, mask)
