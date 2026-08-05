"""Tests for shared optimizer, scheduler, and DataLoader construction."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.runtime import build_dataloader, build_optimizer, build_scheduler


@pytest.mark.parametrize(
    ("weight_decay", "optimizer_type"),
    [(0.0, torch.optim.Adam), (1.0e-4, torch.optim.AdamW)],
)
def test_build_optimizer_selects_adam_variant(weight_decay: float, optimizer_type: type) -> None:
    model = torch.nn.Linear(2, 1)

    optimizer = build_optimizer(model.parameters(), {"lr": 2.0e-3, "weight_decay": weight_decay})

    assert isinstance(optimizer, optimizer_type)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(weight_decay)


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)


def test_build_scheduler_returns_none_when_disabled() -> None:
    assert build_scheduler(_optimizer(), {}, epochs=5) is None


def test_build_scheduler_builds_cosine_schedule() -> None:
    scheduler = build_scheduler(_optimizer(), {"lr_scheduler": "cosine"}, epochs=5)

    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 5
    assert scheduler.eta_min == pytest.approx(1.0e-7)


def test_build_scheduler_builds_warmup_then_cosine() -> None:
    optimizer = _optimizer()
    scheduler = build_scheduler(
        optimizer,
        {"lr_scheduler": "cosine", "lr_warmup_epochs": 2},
        epochs=5,
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)

    learning_rates = []
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[:2] == pytest.approx([0.5005, 1.0])
    assert learning_rates[2:] == pytest.approx([0.750000025, 0.250000075, 1.0e-7])


@pytest.mark.parametrize(
    ("batch_size", "num_workers", "message"),
    [(0, 0, "training.batch_size"), (1, -1, "training.num_workers")],
)
def test_build_dataloader_validates_configuration(
    batch_size: int, num_workers: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_dataloader(
            [1, 2],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            device=torch.device("cpu"),
            config_prefix="training",
        )


def test_build_dataloader_applies_batching_and_collation() -> None:
    loader = build_dataloader(
        [1, 2, 3],
        batch_size=2,
        shuffle=False,
        num_workers=0,
        device=torch.device("cpu"),
        collate_fn=sum,
    )

    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
    assert loader.pin_memory is False
    assert loader.persistent_workers is False
    assert list(loader) == [3, 3]


def test_build_dataloader_applies_shuffle_and_worker_settings() -> None:
    loader = build_dataloader(
        [1, 2, 3],
        batch_size=2,
        shuffle=True,
        num_workers=1,
        device=torch.device("cuda"),
    )

    assert isinstance(loader.sampler, torch.utils.data.RandomSampler)
    assert loader.pin_memory is True
    assert loader.persistent_workers is True
