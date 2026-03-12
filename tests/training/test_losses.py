"""Unit tests for configured training losses."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from training.losses import get_loss_fn


def test_mse_loss() -> None:
    pred = torch.tensor([1.0, 3.0, 5.0])
    target = torch.tensor([1.0, 2.0, 7.0])
    loss = get_loss_fn("mse")(pred, target)
    assert loss.item() == pytest.approx((0.0 + 1.0 + 4.0) / 3.0)


def test_l1_loss() -> None:
    pred = torch.tensor([1.0, 3.0, 5.0])
    target = torch.tensor([1.0, 2.0, 7.0])
    loss = get_loss_fn("l1")(pred, target)
    assert loss.item() == pytest.approx((0.0 + 1.0 + 2.0) / 3.0)


def test_relative_l2_loss() -> None:
    pred = torch.tensor([2.0, 0.0])
    target = torch.tensor([1.0, 0.0])
    loss = get_loss_fn("relative_l2")(pred, target)
    assert loss.item() == pytest.approx(1.0)


def test_unknown_loss_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown loss"):
        get_loss_fn("not_a_loss")
