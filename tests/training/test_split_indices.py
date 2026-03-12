"""Unit tests for dataset split strategies."""

from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
_ = torch

from training.datasets import split_indices


def test_sequential_split_indices() -> None:
    sim_names = [f"sim_{i}" for i in range(5)]
    train_idx, test_idx, train_sims, test_sims = split_indices(
        num_cases=len(sim_names),
        split_cfg={"strategy": "sequential", "train_ratio": 0.6},
        sim_names=sim_names,
    )

    assert train_idx == [0, 1, 2]
    assert test_idx == [3, 4]
    assert train_sims == ["sim_0", "sim_1", "sim_2"]
    assert test_sims == ["sim_3", "sim_4"]


def test_random_split_is_deterministic_with_seed() -> None:
    sim_names = [f"sim_{i}" for i in range(8)]
    split_cfg = {"strategy": "random", "train_ratio": 0.5, "seed": 123}

    first = split_indices(len(sim_names), split_cfg, sim_names)
    second = split_indices(len(sim_names), split_cfg, sim_names)
    assert first == second


def test_file_split_indices(tmp_path: Path) -> None:
    sim_names = ["sim_a", "sim_b", "sim_c", "sim_d"]
    train_file = tmp_path / "train.txt"
    test_file = tmp_path / "test.txt"

    train_file.write_text("sim_a\nsim_c\n", encoding="utf-8")
    test_file.write_text("sim_b\nsim_d\n", encoding="utf-8")

    train_idx, test_idx, train_sims, test_sims = split_indices(
        num_cases=len(sim_names),
        split_cfg={
            "strategy": "file",
            "train_file": str(train_file),
            "test_file": str(test_file),
        },
        sim_names=sim_names,
    )

    assert train_idx == [0, 2]
    assert test_idx == [1, 3]
    assert train_sims == ["sim_a", "sim_c"]
    assert test_sims == ["sim_b", "sim_d"]


def test_file_split_rejects_unknown_names(tmp_path: Path) -> None:
    sim_names = ["sim_a", "sim_b", "sim_c", "sim_d"]
    train_file = tmp_path / "train.txt"
    test_file = tmp_path / "test.txt"

    train_file.write_text("sim_a\nunknown\n", encoding="utf-8")
    test_file.write_text("sim_b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown"):
        split_indices(
            num_cases=len(sim_names),
            split_cfg={
                "strategy": "file",
                "train_file": str(train_file),
                "test_file": str(test_file),
            },
            sim_names=sim_names,
        )
