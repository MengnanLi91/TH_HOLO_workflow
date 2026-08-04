"""Tests for the Optuna HPO module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
zarr = pytest.importorskip("zarr")
optuna = pytest.importorskip("optuna")

FEATURE_NAMES = [
    "log10_Re",
    "Dr",
    "Lr",
    "z_hat",
    "d_local_over_D",
    "V_local_over_V_bulk",
    "is_throat",
]
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
        root.create_array(
            "features",
            data=rng.standard_normal((ROWS_PER_CASE, len(FEATURE_NAMES))).astype(
                np.float32
            ),
        )
        root.create_array(
            "targets",
            data=rng.standard_normal((ROWS_PER_CASE, len(TARGET_NAMES))).astype(
                np.float32
            ),
        )
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
            "model.params.activation_fn": {
                "type": "categorical",
                "choices": ["silu", "relu"],
            },
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

        base = {
            "training": {"lr": 0.001, "epochs": 10},
            "model": {"params": {"size": 128}},
        }
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
            validate_search_space(
                {"data.zarr_dir": {"type": "categorical", "choices": ["/a"]}}, base
            )

    def test_validate_rejects_model_name(self) -> None:
        from training.hpo.search_space import validate_search_space

        base = {"model": {"name": "mlp"}, "training": {"lr": 0.001}}
        with pytest.raises(ValueError, match="not allowed"):
            validate_search_space(
                {"model.name": {"type": "categorical", "choices": ["fno"]}}, base
            )

    def test_validate_rejects_nonexistent_path(self) -> None:
        from training.hpo.search_space import validate_search_space

        base = {"training": {"lr": 0.001}}
        with pytest.raises(KeyError, match="does not exist"):
            validate_search_space(
                {"training.lrr": {"type": "float", "low": 0.0, "high": 1.0}}, base
            )


# ---------------------------------------------------------------------------
# compute_val_loss tests
# ---------------------------------------------------------------------------


class TestComputeValLoss:
    def test_empty_loader_raises(self) -> None:
        from training.experiment import Experiment
        from training.runner import compute_val_loss

        model = torch.nn.Linear(2, 1)
        exp = Experiment(
            model=model,
            optimizer=None,
            loss_fn=torch.nn.MSELoss(),
            adapter=None,
            device=torch.device("cpu"),
        )
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

    def test_old_training_metadata_schema_is_rejected(self) -> None:
        from training.runner import validate_training_run_meta

        with pytest.raises(ValueError, match="requires schema 3"):
            validate_training_run_meta(
                {"training_run_meta_schema": 2}, "old/run_meta.json"
            )


# ---------------------------------------------------------------------------
# Objective integration test
# ---------------------------------------------------------------------------


class TestObjective:
    def test_make_objective_returns_float(self, synthetic_zarr_dir: Path) -> None:
        import random

        from training.datasets import split_indices
        from training.hpo.objective import make_objective
        from training.runner import normalize_split_cfg, prepare_training

        base_cfg = {
            "model": {
                "name": "mlp",
                "params": {
                    "layer_size": 16,
                    "num_layers": 2,
                    "activation_fn": "silu",
                    "skip_connections": False,
                },
            },
            "data": {
                "zarr_dir": str(synthetic_zarr_dir),
                "split": {"strategy": "random", "train_ratio": 0.8, "seed": 42},
            },
            "training": {
                "epochs": 1,
                "batch_size": 32,
                "lr": 0.001,
                "seed": 42,
                "device": "cpu",
                "loss": "mse",
                "experiment": None,
            },
            "output": {},
        }

        prepared = prepare_training(base_cfg)
        dataset = prepared["dataset"]
        split_cfg = normalize_split_cfg(
            dict(base_cfg["data"]["split"]), default_seed=42
        )
        train_idx, test_idx, _, _ = split_indices(
            num_cases=len(dataset.sim_names),
            split_cfg=split_cfg,
            sim_names=dataset.sim_names,
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
            phase_cfg={
                "max_epochs": 1,
                "early_stopping": {"patience": 1, "min_delta": 0.0},
            },
            objective_weights={"profile_val_loss": 1.0},
            prepared=prepared,
            train_indices=train_inner,
            val_indices=val_idx,
            seed=42,
        )

        study = optuna.create_study()
        study.optimize(objective, n_trials=1)
        assert len(study.trials) == 1
        assert isinstance(study.best_value, float)
        assert study.best_value > 0


# ---------------------------------------------------------------------------
# Clean schema tests
# ---------------------------------------------------------------------------


class TestCleanHpoSchema:
    @staticmethod
    def valid_config() -> dict:
        return {
            "study_name": "test",
            "direction": "minimize",
            "storage": None,
            "load_if_exists": True,
            "retrain_best": False,
            "screening": {
                "n_trials": 2,
                "max_epochs": 2,
                "early_stopping": {"patience": 1, "min_delta": 0.0},
            },
            "validation": {
                "splitter_entrypoint": "training.hpo.splits:random_case_folds",
                "n_folds": 2,
                "screening_fold": 0,
                "seed": 42,
            },
            "objective": {"weights": {"profile_val_loss": 1.0}},
            "confirmation": {
                "top_k": 1,
                "max_epochs": 2,
                "early_stopping": {"patience": 1, "min_delta": 0.0},
                "aggregate_std_weight": 0.5,
                "guard_metric": None,
                "guard_reference": None,
            },
            "enqueue_trials": [],
            "search_space": {},
        }

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            (lambda cfg: cfg.update(n_trials=2), "Obsolete hpo.n_trials"),
            (
                lambda cfg: cfg["validation"].update(split_ratio=0.2),
                "split_ratio",
            ),
            (lambda cfg: cfg.pop("objective"), "missing required"),
        ],
    )
    def test_rejects_obsolete_or_incomplete_shapes(self, mutation, message) -> None:
        from training.hpo.config import validate_hpo_config

        cfg = self.valid_config()
        mutation(cfg)
        with pytest.raises(ValueError, match=message):
            validate_hpo_config(cfg)

    def test_accepts_clean_shape(self) -> None:
        from training.hpo.config import validate_hpo_config

        validate_hpo_config(self.valid_config())

    def test_rejects_pre_contract_optuna_database(self, tmp_path: Path) -> None:
        from training.hpo.study import create_study

        storage = f"sqlite:///{tmp_path / 'old.db'}"
        old_study = optuna.create_study(
            study_name="old-study",
            direction="minimize",
            storage=storage,
        )
        old_study.optimize(lambda trial: 1.0, n_trials=1)
        cfg = self.valid_config()
        cfg.update(study_name="old-study", storage=storage)

        with pytest.raises(ValueError, match="predates the clean-break HPO contract"):
            create_study(cfg)


def test_composite_score_rejects_missing_and_nonfinite_metrics() -> None:
    from training.hpo.objective import composite_score

    assert composite_score({"a": 2.0, "b": 4.0}, {"a": 1.0, "b": 0.5}) == 4.0
    with pytest.raises(ValueError, match="required"):
        composite_score({"a": 2.0}, {"a": 1.0, "b": 0.5})
    with pytest.raises(ValueError, match="non-finite"):
        composite_score({"a": float("nan")}, {"a": 1.0})


def test_confirmation_aggregation_uses_std_and_best_control_guard() -> None:
    from training.hpo.study import aggregate_confirmation

    candidates = [
        {"candidate_id": "trial-1", "source": "sampled", "params": {}},
        {"candidate_id": "control-a", "source": "control", "params": {}},
    ]
    rows = [
        {"candidate_id": "trial-1", "composite_score": 0.8, "guard": 0.3},
        {"candidate_id": "trial-1", "composite_score": 1.2, "guard": 0.4},
        {"candidate_id": "control-a", "composite_score": 1.1, "guard": 0.2},
        {"candidate_id": "control-a", "composite_score": 1.1, "guard": 0.2},
    ]
    aggregates, winner = aggregate_confirmation(
        candidates,
        rows,
        std_weight=0.5,
        guard_metric="guard",
        guard_reference="best_control",
    )

    sampled = next(row for row in aggregates if row["source"] == "sampled")
    assert sampled["rank_score"] == pytest.approx(1.1)
    assert sampled["guard_passed"] is False
    assert winner["candidate_id"] == "control-a"


def _fake_objective_runtime(monkeypatch, validation_losses):
    import training.hpo.objective as objective_module

    class Adapter:
        supports_fold_normalization = False

        @staticmethod
        def collate_fn():
            return None

    class FakeExperiment:
        def __init__(self, model):
            self.model = model

        def prepare_for_training(self, train_dataset, val_dataset, device):
            del train_dataset, val_dataset, device

        def on_epoch_end(self, epoch, avg_loss):
            del epoch, avg_loss

        def on_epoch_end_extra_step(self):
            return None

        def compute_hpo_metrics(self, validation_dataset):
            del validation_dataset
            return {"restored_weight": float(self.model.weight.detach().item())}

    def build_model(_params, _dataset_info):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        return model

    def build_experiment(**kwargs):
        return FakeExperiment(kwargs["model"])

    def train_epoch(experiment, _loader):
        with torch.no_grad():
            experiment.model.weight.add_(1.0)
        return float(experiment.model.weight.item())

    losses = iter(validation_losses)
    monkeypatch.setattr(
        objective_module,
        "get_build_fn_and_adapter",
        lambda _cfg: (build_model, "fake"),
    )
    monkeypatch.setattr(objective_module, "build_experiment", build_experiment)
    monkeypatch.setattr(objective_module, "train_one_epoch", train_epoch)
    monkeypatch.setattr(
        objective_module, "compute_val_loss", lambda _exp, _loader: next(losses)
    )
    dataset = torch.utils.data.TensorDataset(
        torch.arange(6, dtype=torch.float32).unsqueeze(1)
    )
    return objective_module, {
        "dataset": dataset,
        "dataset_info": {"in_features": 1},
        "adapter": Adapter(),
        "adapter_name": "fake",
        "device": torch.device("cpu"),
        "data_cfg": {"normalize": False},
    }


def test_train_and_score_restores_best_validation_weights(monkeypatch) -> None:
    objective_module, prepared = _fake_objective_runtime(
        monkeypatch, [3.0, 1.0, 2.0, 1.0]
    )

    result = objective_module.train_and_score(
        base_cfg={
            "model": {"name": "fake", "params": {}},
            "data": {},
            "training": {"lr": 0.001, "batch_size": 2, "loss": "mse"},
        },
        params={},
        phase_cfg={
            "max_epochs": 3,
            "early_stopping": {"patience": 3, "min_delta": 0.0},
        },
        objective_weights={"profile_val_loss": 1.0, "restored_weight": 0.0},
        prepared=prepared,
        train_indices=[0, 1, 2, 3],
        val_indices=[4, 5],
        seed=42,
    )

    assert result["best_epoch"] == 2
    assert result["epochs_trained"] == 3
    assert result["metrics"]["restored_weight"] == pytest.approx(2.0)
    assert result["metrics"]["profile_val_loss"] == pytest.approx(1.0)


def test_train_and_score_reports_then_prunes(monkeypatch) -> None:
    objective_module, prepared = _fake_objective_runtime(monkeypatch, [3.0])

    class PruningTrial:
        def __init__(self):
            self.reports = []

        def report(self, value, step):
            self.reports.append((value, step))

        @staticmethod
        def should_prune():
            return True

    trial = PruningTrial()
    with pytest.raises(optuna.TrialPruned):
        objective_module.train_and_score(
            base_cfg={
                "model": {"name": "fake", "params": {}},
                "data": {},
                "training": {"lr": 0.001, "batch_size": 2, "loss": "mse"},
            },
            params={},
            phase_cfg={
                "max_epochs": 3,
                "early_stopping": {"patience": 3, "min_delta": 0.0},
            },
            objective_weights={"profile_val_loss": 1.0},
            prepared=prepared,
            train_indices=[0, 1, 2, 3],
            val_indices=[4, 5],
            seed=42,
            trial=trial,
        )

    assert trial.reports == [(3.0, 1)]
