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


def _alpha_d_grid_sim_names() -> list[str]:
    """Mimic the alpha_d parametric grid used in production runs."""
    res = [5000, 7722, 11927, 18420, 28449, 43938, 67860, 104807, 161870, 250000]
    drs = [0.333, 0.428, 0.522, 0.617, 0.711, 0.806, 0.9]
    lrs = [0.01, 0.052, 0.116, 0.179, 0.2]
    names: list[str] = []
    for re in res:
        for dr in drs:
            for lr in lrs:
                dr_str = f"{dr:.3f}".rstrip("0").rstrip(".").replace(".", "p")
                lr_str = f"{lr:.3f}".rstrip("0").rstrip(".").replace(".", "p")
                names.append(f"Re_{re}__Dr_{dr_str}__Lr_{lr_str}")
    return names


def test_stratified_split_covers_dr_extremes() -> None:
    """Both Dr=0.333 and Dr=0.9 corners must appear in train and test."""
    sim_names = _alpha_d_grid_sim_names()
    train_idx, test_idx, train_sims, test_sims = split_indices(
        num_cases=len(sim_names),
        split_cfg={
            "strategy": "stratified",
            "train_ratio": 0.8,
            "seed": 42,
            "n_bins": 3,
        },
        sim_names=sim_names,
    )

    assert set(train_idx).isdisjoint(test_idx)
    assert len(train_idx) + len(test_idx) == len(sim_names)

    for dr_token in ("Dr_0p333", "Dr_0p9"):
        assert any(dr_token in s for s in train_sims), f"{dr_token} missing from train"
        assert any(dr_token in s for s in test_sims), f"{dr_token} missing from test"


def test_stratified_split_is_deterministic_with_seed() -> None:
    sim_names = _alpha_d_grid_sim_names()
    cfg = {"strategy": "stratified", "train_ratio": 0.8, "seed": 7, "n_bins": 3}
    first = split_indices(len(sim_names), cfg, sim_names)
    second = split_indices(len(sim_names), cfg, sim_names)
    assert first == second


def test_stratified_split_ignores_lr() -> None:
    """Stratum key is (Dr, Re); changing Lr alone must not change the split."""
    base = _alpha_d_grid_sim_names()
    # Permute Lr labels deterministically across cases — same (Dr, Re), different Lr.
    lr_perm = ["0p01", "0p052", "0p116", "0p179", "0p2"]
    permuted = []
    for i, name in enumerate(base):
        prefix = name.rsplit("__Lr_", 1)[0]
        permuted.append(f"{prefix}__Lr_{lr_perm[i % len(lr_perm)]}")

    cfg = {"strategy": "stratified", "train_ratio": 0.8, "seed": 42, "n_bins": 3}
    base_train_idx, base_test_idx, *_ = split_indices(len(base), cfg, base)
    perm_train_idx, perm_test_idx, *_ = split_indices(len(permuted), cfg, permuted)

    assert base_train_idx == perm_train_idx
    assert base_test_idx == perm_test_idx
