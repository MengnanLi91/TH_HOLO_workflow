"""Tests for the Optuna HPO module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
zarr = pytest.importorskip("zarr")
optuna = pytest.importorskip("optuna")

FEATURE_NAMES = ["log10_Re", "Dr", "Lr", "z_hat", "d_local_over_D", "is_throat"]
TARGET_NAMES = ["log_alpha_D"]
NUM_CASES = 10
ROWS_PER_CASE = 20


@pytest.fixture()
def synthetic_zarr_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    for i in range(NUM_CASES):
        case_name = f"case_{i:03d}"
        store_path = out_dir / f"{case_name}.zarr"
        root = zarr.open(store=str(store_path), mode="w")
        root.create_array("features", data=rng.standard_normal((ROWS_PER_CASE, len(FEATURE_NAMES))).astype(np.float32))
        root.create_array("targets", data=rng.standard_normal((ROWS_PER_CASE, len(TARGET_NAMES))).astype(np.float32))
        meta = root.require_group("metadata")
        meta.attrs["case_id"] = case_name
        meta.attrs["feature_names"] = json.dumps(FEATURE_NAMES)
        meta.attrs["target_names"] = json.dumps(TARGET_NAMES)
    return out_dir


# ---------------------------------------------------------------------------
# search_space tests
# ---------------------------------------------------------------------------


class TestSearchSpace:
    def test_sample_from_search_space(self) -> None:
        from training.hpo.search_space import sample_from_search_space

        search_space = {
            "training.lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "model.params.num_layers": {"type": "int", "low": 2, "high": 10},
            "model.params.activation_fn": {"type": "categorical", "choices": ["silu", "relu"]},
        }
        study = optuna.create_study()
        trial = study.ask()
        sampled = sample_from_search_space(trial, search_space)

        assert "training.lr" in sampled
        assert 1e-5 <= sampled["training.lr"] <= 1e-1
        assert isinstance(sampled["model.params.num_layers"], int)
        assert 2 <= sampled["model.params.num_layers"] <= 10
        assert sampled["model.params.activation_fn"] in {"silu", "relu"}

    def test_apply_overrides_valid(self) -> None:
        from training.hpo.search_space import apply_overrides

        base = {"training": {"lr": 0.001, "epochs": 10}, "model": {"params": {"size": 128}}}
        result = apply_overrides(base, {"training.lr": 0.01, "model.params.size": 256})
        assert result["training"]["lr"] == 0.01
        assert result["model"]["params"]["size"] == 256
        # Original unchanged
        assert base["training"]["lr"] == 0.001

    def test_apply_overrides_rejects_typo(self) -> None:
        from training.hpo.search_space import apply_overrides

        base = {"training": {"lr": 0.001}}
        with pytest.raises(KeyError, match="lrr"):
            apply_overrides(base, {"training.lrr": 0.01})

    def test_validate_rejects_unsafe_prefix(self) -> None:
        from training.hpo.search_space import validate_search_space

        base = {"data": {"zarr_dir": "/tmp"}, "training": {"lr": 0.001}}
        with pytest.raises(ValueError, match="not allowed"):
            validate_search_space({"data.zarr_dir": {"type": "categorical", "choices": ["/a"]}}, base)

    def test_validate_rejects_model_name(self) -> None:
        from training.hpo.search_space import validate_search_space

        base = {"model": {"name": "mlp"}, "training": {"lr": 0.001}}
        with pytest.raises(ValueError, match="not allowed"):
            validate_search_space({"model.name": {"type": "categorical", "choices": ["fno"]}}, base)

    def test_validate_rejects_nonexistent_path(self) -> None:
        from training.hpo.search_space import validate_search_space

        base = {"training": {"lr": 0.001}}
        with pytest.raises(KeyError, match="does not exist"):
            validate_search_space({"training.lrr": {"type": "float", "low": 0.0, "high": 1.0}}, base)


# ---------------------------------------------------------------------------
# compute_val_loss tests
# ---------------------------------------------------------------------------


class TestComputeValLoss:
    def test_empty_loader_raises(self) -> None:
        from training.runner import compute_val_loss
        from training.experiment import Experiment

        model = torch.nn.Linear(2, 1)
        exp = Experiment(model=model, optimizer=None, loss_fn=torch.nn.MSELoss(), adapter=None, device=torch.device("cpu"))
        empty_loader = torch.utils.data.DataLoader([], batch_size=1)
        with pytest.raises(RuntimeError, match="zero batches"):
            compute_val_loss(exp, empty_loader)

    def test_includes_epoch_level_validation_term(self) -> None:
        from training.runner import compute_val_loss

        class DummyExperiment:
            def validation_step(self, batch) -> float:
                return float(batch)

            def validation_epoch_loss(self, val_loader) -> float:
                _ = val_loader
                return 10.0

        loader = torch.utils.data.DataLoader([1.0, 2.0, 3.0], batch_size=1)
        loss = compute_val_loss(DummyExperiment(), loader)
        assert loss == pytest.approx(12.0)


class TestMetricsOutPath:
    def test_auto_metrics_path_uses_checkpoint_directory(self) -> None:
        from training.runner import _resolve_metrics_out_path

        checkpoint = Path("/tmp/example_case/model.mdlus")
        resolved = _resolve_metrics_out_path({"metrics_out": "auto"}, checkpoint)

        assert resolved == checkpoint.with_name("eval_metrics.json")


# ---------------------------------------------------------------------------
# Objective integration test
# ---------------------------------------------------------------------------


class TestObjective:
    def test_make_objective_returns_float(self, synthetic_zarr_dir: Path) -> None:
        from training.hpo.objective import make_objective
        from training.runner import prepare_training, normalize_split_cfg
        from training.datasets import split_indices
        import random

        base_cfg = {
            "model": {"name": "mlp", "params": {"layer_size": 16, "num_layers": 2, "activation_fn": "silu", "skip_connections": False}},
            "data": {"zarr_dir": str(synthetic_zarr_dir), "split": {"strategy": "random", "train_ratio": 0.8, "seed": 42}},
            "training": {"epochs": 1, "batch_size": 32, "lr": 0.001, "seed": 42, "device": "cpu", "loss": "mse", "experiment": None},
            "output": {},
        }

        prepared = prepare_training(base_cfg)
        dataset = prepared["dataset"]
        split_cfg = normalize_split_cfg(dict(base_cfg["data"]["split"]), default_seed=42)
        train_idx, test_idx, _, _ = split_indices(
            num_cases=len(dataset.sim_names), split_cfg=split_cfg, sim_names=dataset.sim_names,
        )

        rng = random.Random(42)
        shuffled = list(train_idx)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * 0.25))
        val_idx = sorted(shuffled[:n_val])
        train_inner = sorted(shuffled[n_val:])

        objective = make_objective(
            base_cfg=base_cfg,
            search_space={},  # No search space -- use base config as-is
            hpo_cfg={"validation": {"split_ratio": 0.25, "seed": 42}},
            prepared=prepared,
            train_inner_idx=train_inner,
            val_idx=val_idx,
        )

        study = optuna.create_study()
        study.optimize(objective, n_trials=1)
        assert len(study.trials) == 1
        assert isinstance(study.best_value, float)
        assert study.best_value > 0


# ---------------------------------------------------------------------------
# Train pool guard test
# ---------------------------------------------------------------------------


class TestTrainPoolGuard:
    def test_empty_train_inner_raises(self, tmp_path: Path) -> None:
        """With only 2 train cases and val_ratio=1.0, inner train is empty."""
        # Create a minimal dataset with exactly 3 cases (2 train + 1 test)
        rng = np.random.default_rng(99)
        small_dir = tmp_path / "small"
        small_dir.mkdir()
        for i in range(3):
            sp = small_dir / f"c{i}.zarr"
            root = zarr.open(store=str(sp), mode="w")
            root.create_array("features", data=rng.standard_normal((5, len(FEATURE_NAMES))).astype(np.float32))
            root.create_array("targets", data=rng.standard_normal((5, len(TARGET_NAMES))).astype(np.float32))
            meta = root.require_group("metadata")
            meta.attrs["case_id"] = f"c{i}"
            meta.attrs["feature_names"] = json.dumps(FEATURE_NAMES)
            meta.attrs["target_names"] = json.dumps(TARGET_NAMES)

        from training.hpo.study import run_hpo

        cfg = {
            "model": {"name": "mlp", "params": {"layer_size": 8, "num_layers": 2, "activation_fn": "silu", "skip_connections": False}},
            "data": {"zarr_dir": str(small_dir), "split": {"strategy": "sequential", "train_ratio": 0.7, "seed": 42}},
            "training": {"epochs": 1, "batch_size": 8, "lr": 0.001, "seed": 42, "device": "cpu", "loss": "mse", "experiment": None},
            "output": {},
            "hpo": {
                "study_name": "guard_test",
                "direction": "minimize",
                "n_trials": 1,
                "seed": 42,
                "validation": {"split_ratio": 1.0, "seed": 42},
                "search_space": {},
                "output_dir": str(tmp_path / "hpo_guard_test"),
            },
        }
        # 3 cases, train_ratio=0.7 -> 2 train. val_ratio=1.0 -> all go to val, train_inner empty
        with pytest.raises(ValueError, match="empty"):
            run_hpo(cfg)
