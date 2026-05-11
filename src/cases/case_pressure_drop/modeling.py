"""Model training helpers for the case-level pressure-drop workflow."""

import math
from pathlib import Path
from typing import Any

import numpy as np

from case_pressure_drop.data import CasePressureDropDataset


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for case-level pressure-drop training."
        ) from exc


def _require_joblib():
    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "joblib is required to save case-level pressure-drop models."
        ) from exc
    return joblib


def transform_target(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(values, dtype=np.float64))


def inverse_transform_target(values: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(values, dtype=np.float64))


def compute_metrics(
    y_true_pa: np.ndarray,
    y_pred_pa: np.ndarray,
    *,
    y_true_log: np.ndarray | None = None,
    y_pred_log: np.ndarray | None = None,
) -> dict[str, float]:
    _require_sklearn()
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true_pa = np.asarray(y_true_pa, dtype=np.float64)
    y_pred_pa = np.asarray(y_pred_pa, dtype=np.float64)

    rmse_pa = math.sqrt(mean_squared_error(y_true_pa, y_pred_pa))
    mae_pa = float(mean_absolute_error(y_true_pa, y_pred_pa))
    r2_pa = float(r2_score(y_true_pa, y_pred_pa))
    mape = float(
        np.mean(np.abs((y_pred_pa - y_true_pa) / np.clip(np.abs(y_true_pa), 1e-8, None)))
    )

    payload = {
        "rmse_pa": float(rmse_pa),
        "mae_pa": float(mae_pa),
        "r2_pa": float(r2_pa),
        "mape": float(mape),
    }
    if y_true_log is not None and y_pred_log is not None:
        payload["rmse_log1p"] = float(
            math.sqrt(mean_squared_error(y_true_log, y_pred_log))
        )
    return payload


def build_estimator(model_name: str, model_cfg: dict[str, Any], seed: int):
    _require_sklearn()
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_name == "linear_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(model_cfg.get("n_estimators", 400)),
            max_depth=(
                int(model_cfg["max_depth"])
                if model_cfg.get("max_depth") is not None
                else None
            ),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
            n_jobs=int(model_cfg.get("n_jobs", -1)),
            random_state=int(seed),
        )
    if model_name == "mlp":
        hidden = tuple(int(v) for v in model_cfg.get("hidden_layer_sizes", [64, 64]))
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=hidden,
                        activation=str(model_cfg.get("activation", "relu")),
                        alpha=float(model_cfg.get("alpha", 1.0e-4)),
                        learning_rate_init=float(
                            model_cfg.get("learning_rate_init", 1.0e-3)
                        ),
                        max_iter=int(model_cfg.get("max_iter", 2000)),
                        early_stopping=bool(model_cfg.get("early_stopping", True)),
                        validation_fraction=float(
                            model_cfg.get("validation_fraction", 0.15)
                        ),
                        n_iter_no_change=int(model_cfg.get("n_iter_no_change", 30)),
                        random_state=int(seed),
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model name: {model_name}")


def cross_validate_models(
    dataset: CasePressureDropDataset,
    *,
    feature_names: list[str],
    model_cfg: dict[str, Any],
    cv_cfg: dict[str, Any],
) -> dict[str, Any]:
    _require_sklearn()
    from sklearn.model_selection import GroupKFold

    X = dataset.build_feature_matrix(feature_names)
    y_pa = np.asarray(dataset.delta_p_case, dtype=np.float64)
    y_log = transform_target(y_pa)
    groups = dataset.groups()

    requested_splits = int(cv_cfg.get("n_splits", 5))
    effective_splits = min(requested_splits, len(dataset))
    if effective_splits < 2:
        raise ValueError("Need at least 2 cases to run model cross-validation.")

    seed = int(cv_cfg.get("seed", 42))
    splitter = GroupKFold(n_splits=effective_splits)
    results: dict[str, Any] = {}

    for model_name in ("linear_regression", "random_forest", "mlp"):
        per_fold: list[dict[str, float]] = []
        for fold_idx, (tr, va) in enumerate(splitter.split(X, y_log, groups=groups)):
            estimator = build_estimator(model_name, dict(model_cfg.get(model_name) or {}), seed + fold_idx)
            estimator.fit(X[tr], y_log[tr])
            pred_log = np.asarray(estimator.predict(X[va]), dtype=np.float64)
            pred_pa = inverse_transform_target(pred_log)
            fold_metrics = compute_metrics(
                y_pa[va],
                pred_pa,
                y_true_log=y_log[va],
                y_pred_log=pred_log,
            )
            per_fold.append(fold_metrics)

        summary = {
            "artifact_name": f"{model_name}.joblib",
            "params": dict(model_cfg.get(model_name) or {}),
            "cv": {
                "n_splits_requested": int(requested_splits),
                "n_splits_effective": int(effective_splits),
                "seed": int(seed),
                "per_fold": per_fold,
                "rmse_pa_mean": float(np.mean([row["rmse_pa"] for row in per_fold])),
                "rmse_pa_std": float(np.std([row["rmse_pa"] for row in per_fold])),
                "mae_pa_mean": float(np.mean([row["mae_pa"] for row in per_fold])),
                "r2_pa_mean": float(np.mean([row["r2_pa"] for row in per_fold])),
                "mape_mean": float(np.mean([row["mape"] for row in per_fold])),
                "rmse_log1p_mean": float(
                    np.mean([row["rmse_log1p"] for row in per_fold])
                ),
            },
        }
        results[model_name] = summary

    return results


def fit_and_save_models(
    dataset: CasePressureDropDataset,
    *,
    feature_names: list[str],
    model_cfg: dict[str, Any],
    seed: int,
    model_dir: str | Path,
) -> dict[str, Any]:
    joblib = _require_joblib()
    model_dir = Path(model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    X = dataset.build_feature_matrix(feature_names)
    y_log = transform_target(dataset.delta_p_case)

    saved: dict[str, Any] = {}
    for offset, model_name in enumerate(("linear_regression", "random_forest", "mlp")):
        estimator = build_estimator(
            model_name,
            dict(model_cfg.get(model_name) or {}),
            int(seed) + offset,
        )
        estimator.fit(X, y_log)
        artifact_path = model_dir / f"{model_name}.joblib"
        payload = {
            "model_name": model_name,
            "feature_names": list(feature_names),
            "target_transform": "log1p",
            "estimator": estimator,
        }
        joblib.dump(payload, artifact_path)
        saved[model_name] = {
            "artifact": str(artifact_path),
            "feature_names": list(feature_names),
        }
    return saved


def load_saved_model(path: str | Path) -> dict[str, Any]:
    joblib = _require_joblib()
    artifact = joblib.load(Path(path).expanduser().resolve())
    if not isinstance(artifact, dict) or "estimator" not in artifact:
        raise ValueError(f"Unexpected model artifact format: {path}")
    return artifact
