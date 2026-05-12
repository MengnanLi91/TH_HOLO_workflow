"""Shape contract for ``AlphaDExperiment.compute_extended_metrics``.

Phase 2a moves the alpha-D-specific pointwise + delta_p metrics out of
``runner.evaluate()`` and onto the experiment. This test pins the dict
shape: per-field R²/MAE for the alpha-D target, with the same key names
the plotting layer and `eval_metrics.json` consumers already read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cases.alpha_d.experiment import AlphaDExperiment
from cases.alpha_d.feature_data import engineered_features_spec
from training.datasets_tabular import TabularPairDataset


def test_alpha_d_compute_extended_metrics_shape(alpha_d_zarr_dir: Path) -> None:
    eng_names, eng_builder = engineered_features_spec()
    ds = TabularPairDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
        engineered_feature_names=eng_names,
        engineered_feature_builder=eng_builder,
    )

    # Trivial model that returns zeros; just need it to be callable for the
    # delta_p forward inside compute_extended_metrics.
    class _ZeroModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

    experiment = AlphaDExperiment(
        model=_ZeroModel(),
        optimizer=None,
        loss_fn=None,
        adapter=None,
        device=torch.device("cpu"),
    )

    preds_batch = torch.zeros((len(ds), 1), dtype=torch.float32)
    targets_batch = ds._y.clone()

    metrics = experiment.compute_extended_metrics(
        ds, [preds_batch], [targets_batch],
    )

    assert isinstance(metrics, dict)
    assert "per_field" in metrics
    per_field = metrics["per_field"]
    assert isinstance(per_field, list) and len(per_field) == 1
    entry = per_field[0]
    assert entry["name"] == "log_alpha_D"
    assert "r2" in entry and "mae" in entry
