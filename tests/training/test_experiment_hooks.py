"""Tests for the case-agnostic Experiment hook API.

Phase 2a introduces ``compute_extended_metrics`` and ``print_extended_metrics``
on the base ``Experiment`` so the runner can delegate case-specific evaluation
without inline ``is_alpha_d_target`` checks. Both default to no-ops.
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
