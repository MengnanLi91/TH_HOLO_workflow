"""Tests for the case-level pressure-drop regression workflow."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from case_pressure_drop.data import CANDIDATE_FEATURES, CasePressureDropDataset
from case_pressure_drop.workflow import split_case_indices

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
JOBLIB_AVAILABLE = importlib.util.find_spec("joblib") is not None


def _fmt_param(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _case_name(re_value: float, dr_value: float, lr_value: float) -> str:
    return (
        f"Re_{int(round(re_value))}"
        f"__Dr_{_fmt_param(dr_value)}"
        f"__Lr_{_fmt_param(lr_value)}"
    )


def _delta_p_formula(re_value: float, dr_value: float, lr_value: float) -> float:
    return (
        25.0
        + 0.0012 * re_value * (1.0 - dr_value)
        + 120.0 * lr_value
        + 35.0 / max(dr_value, 1e-8)
        + 0.015 * re_value * dr_value * lr_value
        + 4.0 * math.log10(re_value)
    )


def _write_case_store(
    root: Path,
    *,
    re_value: float,
    dr_value: float,
    lr_value: float,
) -> str:
    case_name = _case_name(re_value, dr_value, lr_value)
    store_path = root / f"{case_name}.zarr" / "metadata"
    store_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "attributes": {
            "case_id": case_name,
            "Re": re_value,
            "Dr": dr_value,
            "Lr": lr_value,
            "delta_p_case": _delta_p_formula(re_value, dr_value, lr_value),
        },
        "zarr_format": 3,
        "node_type": "group",
    }
    (store_path / "zarr.json").write_text(json.dumps(payload), encoding="utf-8")
    return case_name


@pytest.fixture()
def synthetic_case_dataset(tmp_path: Path) -> Path:
    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    re_values = [5_000.0, 12_000.0, 50_000.0, 150_000.0]
    dr_values = [0.2, 0.5, 0.8]
    lr_values = [0.03, 0.12]

    for re_value in re_values:
        for dr_value in dr_values:
            for lr_value in lr_values:
                _write_case_store(
                    out_dir,
                    re_value=re_value,
                    dr_value=dr_value,
                    lr_value=lr_value,
                )
    return out_dir


def test_case_dataset_loads_scalar_metadata(synthetic_case_dataset: Path) -> None:
    dataset = CasePressureDropDataset.from_zarr_dir(synthetic_case_dataset)

    assert len(dataset) == 24
    assert dataset.sim_names[0].startswith("Re_")
    assert np.all(dataset.delta_p_case > 0.0)

    matrix = dataset.build_feature_matrix(CANDIDATE_FEATURES)
    assert matrix.shape == (24, len(CANDIDATE_FEATURES))

    selected = dataset.build_feature_matrix(["Dr", "inv_Dr", "Dr_times_Lr"])
    assert selected.shape == (24, 3)
    assert np.allclose(selected[:, 0] * selected[:, 1], 1.0)


def test_split_case_indices_is_reproducible_and_disjoint(
    synthetic_case_dataset: Path,
) -> None:
    dataset = CasePressureDropDataset.from_zarr_dir(synthetic_case_dataset)
    split_cfg = {"strategy": "stratified", "train_ratio": 0.75, "seed": 7, "n_bins": 3}

    first = split_case_indices(dataset.sim_names, split_cfg)
    second = split_case_indices(dataset.sim_names, split_cfg)

    assert first == second
    train_idx, test_idx, train_sims, test_sims = first
    assert set(train_idx).isdisjoint(test_idx)
    assert set(train_sims).isdisjoint(test_sims)
    assert sorted(train_idx + test_idx) == list(range(len(dataset)))


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn required")
def test_feature_selection_uses_only_train_cases(
    synthetic_case_dataset: Path,
    tmp_path: Path,
) -> None:
    from case_pressure_drop.feature_selection import run_feature_selection

    dataset = CasePressureDropDataset.from_zarr_dir(synthetic_case_dataset)
    train_idx, test_idx, train_sims, test_sims = split_case_indices(
        dataset.sim_names,
        {"strategy": "stratified", "train_ratio": 0.75, "seed": 11, "n_bins": 3},
    )
    train_dataset = dataset.subset_by_case_indices(train_idx)
    _ = test_idx

    result = run_feature_selection(
        train_dataset,
        feature_names=list(CANDIDATE_FEATURES),
        methods=["f_regression", "lasso", "random_forest_permutation"],
        top_k=3,
        n_splits=3,
        seed=11,
        stability_min=0.0,
        mutual_info_n_seeds=3,
        output_dir=tmp_path / "feature_selection",
        config={"data": {"zarr_dir": str(synthetic_case_dataset)}},
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert set(result.used_case_ids) == set(train_sims)
    assert set(report["dataset"]["case_ids"]) == set(train_sims)
    assert set(report["dataset"]["case_ids"]).isdisjoint(test_sims)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn required")
@pytest.mark.skipif(not JOBLIB_AVAILABLE, reason="joblib required")
def test_training_and_evaluation_smoke(
    synthetic_case_dataset: Path,
    tmp_path: Path,
) -> None:
    from case_pressure_drop.workflow import (
        evaluate_case_pressure_drop,
        train_case_pressure_drop,
    )

    run_dir = tmp_path / "case_pressure_drop"
    cfg = {
        "data": {
            "zarr_dir": str(synthetic_case_dataset),
            "split": {
                "strategy": "stratified",
                "train_ratio": 0.75,
                "seed": 13,
                "n_bins": 3,
            },
        },
        "feature_selection": {
            "enabled": True,
            "candidate_features": list(CANDIDATE_FEATURES),
            "methods": ["f_regression", "lasso", "random_forest_permutation"],
            "top_k": 3,
            "n_splits": 3,
            "seed": 13,
            "stability_min": 0.0,
            "mutual_info_n_seeds": 3,
        },
        "models": {
            "cv": {"n_splits": 3, "seed": 13},
            "linear_regression": {},
            "random_forest": {
                "n_estimators": 25,
                "max_depth": 6,
                "min_samples_leaf": 1,
                "n_jobs": 1,
            },
            "mlp": {
                "hidden_layer_sizes": [12],
                "activation": "relu",
                "alpha": 1.0e-4,
                "learning_rate_init": 5.0e-3,
                "max_iter": 300,
                "early_stopping": False,
                "validation_fraction": 0.15,
                "n_iter_no_change": 20,
            },
        },
        "output": {
            "case_dir": str(run_dir),
            "feature_selection_dir": str(run_dir / "feature_selection"),
            "model_dir": str(run_dir / "models"),
            "run_meta": str(run_dir / "run_meta.json"),
            "metrics_out": "auto",
            "table_out": "auto",
            "plot_dir": str(run_dir / "plots"),
        },
    }

    train_result = train_case_pressure_drop(cfg)
    run_meta_path = Path(train_result["run_meta"])
    assert run_meta_path.exists()

    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    assert set(run_meta["models"]) == {
        "linear_regression",
        "random_forest",
        "mlp",
    }
    assert run_meta["best_model"]["name"] in run_meta["models"]
    assert run_meta["data"]["selected_features"]
    assert Path(run_meta["feature_selection"]["report_path"]).exists()
    assert Path(run_meta["feature_selection"]["selected_features_path"]).exists()
    assert set(run_meta["feature_selection"]["case_ids_used"]) == set(
        run_meta["split"]["train_sims"]
    )
    assert set(run_meta["feature_selection"]["case_ids_used"]).isdisjoint(
        run_meta["split"]["test_sims"]
    )
    for model_meta in run_meta["models"].values():
        assert Path(model_meta["artifact"]).exists()

    eval_payload = evaluate_case_pressure_drop(
        {
            "eval": {
                "run_meta": str(run_meta_path),
                "save_plots": False,
                "save_table": True,
            },
            "output": {
                "metrics_out": "auto",
                "table_out": "auto",
                "plot_dir": str(run_dir / "plots"),
            },
        }
    )

    assert eval_payload["n_test_cases"] == len(run_meta["split"]["test_sims"])
    assert set(eval_payload["per_model"]) == {
        "linear_regression",
        "random_forest",
        "mlp",
    }
    assert Path(eval_payload["artifacts"]["metrics_json"]).exists()
    assert Path(eval_payload["artifacts"]["comparison_table"]).exists()
    assert eval_payload["artifacts"]["plots"] == []
