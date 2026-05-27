"""Tests for the case-agnostic Experiment hook API.

The base ``Experiment`` exposes ``compute_extended_metrics`` /
``print_extended_metrics`` so the runner can delegate case-specific
evaluation without inline ``is_alpha_d_target`` checks. Both default to
no-ops; case experiments override.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.experiment import Experiment


def _make_experiment() -> Experiment:
    return Experiment(
        model=None,
        optimizer=None,
        loss_fn=None,
        adapter=None,
        device=torch.device("cpu"),
    )


def test_compute_extended_metrics_default_returns_empty_dict() -> None:
    exp = _make_experiment()
    result = exp.compute_extended_metrics(
        eval_dataset=object(),
        all_preds=[torch.zeros(2, 1)],
        all_targets=[torch.zeros(2, 1)],
    )
    assert result == {}


def test_print_extended_metrics_default_is_noop(capsys: pytest.CaptureFixture) -> None:
    exp = _make_experiment()
    exp.print_extended_metrics({"per_field": [{"name": "x", "r2": 0.5, "mae": 0.1}]})
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_prepare_for_training_default_is_noop() -> None:
    exp = _make_experiment()
    result = exp.prepare_for_training(
        train_dataset=object(),
        val_dataset=None,
        device=torch.device("cpu"),
    )
    assert result is None


def test_on_epoch_end_extra_step_default_is_noop() -> None:
    exp = _make_experiment()
    assert exp.on_epoch_end_extra_step() is None


def test_baseline_for_plotting_default_returns_none() -> None:
    exp = _make_experiment()
    result = exp.baseline_for_plotting(
        dataset=object(),
        field_name="anything",
        mask=None,
    )
    assert result is None
