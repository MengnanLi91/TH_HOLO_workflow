"""Feature-selection helpers for the case-level pressure-drop workflow."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from case_pressure_drop.data import CasePressureDropDataset
from feature_analysis.manifest import build_manifest, write_manifest


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for case-level feature selection."
        ) from exc


def _effective_n_splits(requested: int, n_cases: int) -> int:
    effective = min(int(requested), int(n_cases))
    if effective < 2:
        raise ValueError(
            f"Need at least 2 cases for cross-validation, found {n_cases}."
        )
    return effective


def _scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    unique_vals, inv = np.unique(scores, return_inverse=True)
    for value_idx in range(len(unique_vals)):
        mask = inv == value_idx
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def _scale(X_tr: np.ndarray, X_va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_va)


def _fallback_scores(X_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Use absolute correlation when a fold is too small for nested CV."""
    X_tr = np.asarray(X_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    if X_tr.shape[0] < 2:
        return np.zeros(X_tr.shape[1], dtype=np.float64)

    out = np.zeros(X_tr.shape[1], dtype=np.float64)
    y_std = float(np.std(y_tr))
    if y_std <= 1e-12:
        return out
    for idx in range(X_tr.shape[1]):
        x_col = X_tr[:, idx]
        x_std = float(np.std(x_col))
        if x_std <= 1e-12:
            out[idx] = 0.0
            continue
        out[idx] = abs(float(np.corrcoef(x_col, y_tr)[0, 1]))
        if not np.isfinite(out[idx]):
            out[idx] = 0.0
    return out


def _score_f_regression(X_tr, y_tr, X_va, y_va, rng, **_kwargs) -> np.ndarray:
    from sklearn.feature_selection import f_regression

    scores, _ = f_regression(X_tr, y_tr)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def _score_mutual_info(
    X_tr,
    y_tr,
    X_va,
    y_va,
    rng,
    *,
    n_seeds: int = 10,
    **_kwargs,
) -> np.ndarray:
    from sklearn.feature_selection import mutual_info_regression

    acc = np.zeros(X_tr.shape[1], dtype=np.float64)
    for _ in range(max(1, int(n_seeds))):
        acc += mutual_info_regression(
            X_tr,
            y_tr,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
    return acc / max(1, int(n_seeds))


def _score_lasso(X_tr, y_tr, X_va, y_va, rng, **_kwargs) -> np.ndarray:
    from sklearn.linear_model import LassoCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cv_splits = min(5, len(X_tr))
    if cv_splits < 2:
        return _fallback_scores(X_tr, y_tr)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lasso",
                LassoCV(
                    cv=cv_splits,
                    alphas=30,
                    max_iter=5000,
                    random_state=0,
                    precompute=False,
                ),
            ),
        ]
    )
    pipe.fit(np.asarray(X_tr, dtype=np.float64), np.asarray(y_tr, dtype=np.float64))
    return np.abs(pipe.named_steps["lasso"].coef_)


def _score_random_forest_permutation(
    X_tr,
    y_tr,
    X_va,
    y_va,
    rng,
    **_kwargs,
) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=int(rng.integers(0, 2**31 - 1)),
        n_jobs=1,
    )
    model.fit(X_tr, y_tr)
    result = permutation_importance(
        model,
        X_va,
        y_va,
        n_repeats=10,
        random_state=int(rng.integers(0, 2**31 - 1)),
        n_jobs=1,
    )
    return result.importances_mean


def _score_rfecv_ridge(X_tr, y_tr, X_va, y_va, rng, **_kwargs) -> np.ndarray:
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    cv_splits = min(5, len(X_tr))
    if cv_splits < 2:
        return _fallback_scores(X_tr, y_tr)

    X_tr_s, _ = _scale(X_tr, X_va)
    selector = RFECV(
        estimator=Ridge(alpha=1.0),
        cv=KFold(n_splits=cv_splits, shuffle=True, random_state=0),
        scoring="neg_mean_squared_error",
        min_features_to_select=1,
    )
    selector.fit(X_tr_s, y_tr)
    return (selector.ranking_.max() - selector.ranking_ + 1).astype(np.float64)


def _score_sequential_ridge(X_tr, y_tr, X_va, y_va, rng, **_kwargs) -> np.ndarray:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score

    X_tr_s, _ = _scale(X_tr, X_va)
    d = X_tr_s.shape[1]
    cv_splits = min(5, len(X_tr_s))
    if cv_splits < 2:
        return _fallback_scores(X_tr, y_tr)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=0)
    remaining = list(range(d))
    order: list[int] = []

    while remaining:
        best_idx = None
        best_score = -np.inf
        for idx in remaining:
            cols = order + [idx]
            scores = cross_val_score(
                Ridge(alpha=1.0),
                X_tr_s[:, cols],
                y_tr,
                cv=cv,
                scoring="neg_mean_squared_error",
                n_jobs=1,
            )
            mean_score = float(scores.mean())
            if mean_score > best_score:
                best_score = mean_score
                best_idx = idx
        assert best_idx is not None
        order.append(best_idx)
        remaining.remove(best_idx)

    out = np.zeros(d, dtype=np.float64)
    for rank, idx in enumerate(order):
        out[idx] = d - rank
    return out


METHODS: dict[str, Callable[..., np.ndarray]] = {
    "f_regression": _score_f_regression,
    "mutual_info": _score_mutual_info,
    "lasso": _score_lasso,
    "random_forest_permutation": _score_random_forest_permutation,
    "rfecv_ridge": _score_rfecv_ridge,
    "sequential_ridge": _score_sequential_ridge,
}


@dataclass
class MethodResult:
    mean_score: np.ndarray
    mean_rank: np.ndarray
    per_fold_ranks: np.ndarray
    stability: np.ndarray


@dataclass
class SelectionResult:
    selected_features: list[str]
    report: dict[str, Any]
    report_path: Path
    manifest_path: Path
    selected_features_path: Path
    used_case_ids: list[str]


def _rank_transform(X: np.ndarray) -> np.ndarray:
    """Per-column rank transform with tie averaging.

    Rank-space makes monotonic transforms (``log``, ``1/x``) look linear, so
    downstream redundancy checks are invariant to such reparameterisations.
    """
    X = np.asarray(X, dtype=np.float64)
    out = np.empty_like(X, dtype=np.float64)
    for col in range(X.shape[1]):
        values = X[:, col]
        order = np.argsort(values, kind="stable")
        ranks = np.empty_like(values, dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        # Average ranks for ties.
        unique_vals, inv = np.unique(values, return_inverse=True)
        for uv_idx in range(len(unique_vals)):
            mask = inv == uv_idx
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
        out[:, col] = ranks
    return out


def _multivariate_redundancy_r2(
    X_ranks: np.ndarray,
    candidate_idx: int,
    predictor_indices: list[int],
    *,
    n_neighbors: int = 3,
    cv_splits: int = 5,
) -> float:
    """Cross-validated R² when predicting a candidate from a set of features.

    Uses k-nearest-neighbors on rank-transformed inputs.  KNN catches
    nonlinear multivariate compositions (e.g. ``inv_Lr = Dr / Dr_times_Lr``)
    that pairwise correlation misses.  Rank-transforming makes monotonic
    reparameterisations identical (e.g. ``Dr`` vs ``inv_Dr``).

    Returns R² clipped to ``[0, 1]``.  Higher means more redundant.
    """
    if not predictor_indices:
        return 0.0

    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    X_sub = X_ranks[:, predictor_indices]
    y = X_ranks[:, candidate_idx]

    # Degenerate case: candidate has no variance.
    if np.std(y) < 1e-12:
        return 1.0

    scaler = StandardScaler().fit(X_sub)
    X_scaled = scaler.transform(X_sub)

    k = max(1, min(int(n_neighbors), len(y) - 1))
    folds = max(2, min(int(cv_splits), len(y)))
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
    cv = KFold(n_splits=folds, shuffle=True, random_state=0)
    try:
        scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring="r2")
    except Exception:
        return 0.0
    return float(np.clip(scores.mean(), 0.0, 1.0))


def _select_with_redundancy_filter(
    candidate_order: np.ndarray,
    X_ranks: np.ndarray,
    *,
    top_k: int,
    redundancy_threshold: float,
    feature_names: list[str],
    preselected: list[int] | None = None,
) -> tuple[list[int], list[tuple[str, str, float]]]:
    """Pick ``top_k`` features greedily, skipping multivariate-redundant ones.

    Walks ``candidate_order`` and adds features to the picked set unless they
    can be predicted with CV R² >= ``redundancy_threshold`` from the features
    already in the set.  This catches both pairwise monotonic redundancy
    (e.g. ``Dr`` vs ``1/Dr``) and multivariate algebraic compositions
    (e.g. ``inv_Lr = Dr / Dr_times_Lr``).

    The set starts as ``preselected`` (already-picked features that block
    redundant candidates but do not count against ``top_k``).

    Returns ``(newly_picked_indices, dropped_log)``.  ``dropped_log`` records
    ``(dropped_feature, summary_of_blockers, r2)`` for diagnostics.
    """
    picked_against: list[int] = list(preselected or [])
    newly: list[int] = []
    dropped: list[tuple[str, str, float]] = []
    for idx in candidate_order.tolist():
        if len(newly) >= top_k:
            break
        if idx in picked_against:
            continue
        r2 = _multivariate_redundancy_r2(X_ranks, idx, picked_against)
        if r2 < redundancy_threshold:
            picked_against.append(idx)
            newly.append(idx)
        else:
            blocker_label = " + ".join(feature_names[p] for p in picked_against)
            dropped.append((feature_names[idx], blocker_label, float(r2)))
    return newly, dropped


def run_feature_selection(
    dataset: CasePressureDropDataset,
    *,
    feature_names: list[str],
    methods: list[str],
    top_k: int,
    n_splits: int,
    seed: int,
    stability_min: float,
    mutual_info_n_seeds: int,
    output_dir: str | Path,
    config: dict[str, Any],
    redundancy_threshold: float | None = 0.95,
) -> SelectionResult:
    """Rank candidate features on the training split and write artifacts."""
    _require_sklearn()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    X = dataset.build_feature_matrix(feature_names)
    y = dataset.target_log1p()
    groups = dataset.groups()

    effective_splits = _effective_n_splits(n_splits, len(dataset))
    top_k_effective = max(1, min(int(top_k), len(feature_names)))

    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=effective_splits)
    splits = list(gkf.split(X, y, groups=groups))

    unknown = [name for name in methods if name not in METHODS]
    if unknown:
        raise ValueError(f"Unknown feature-selection method(s): {unknown}")
    if not methods:
        raise ValueError("feature_selection.methods must contain at least one method.")

    per_method_scores: dict[str, list[np.ndarray]] = {name: [] for name in methods}
    per_method_ranks: dict[str, list[np.ndarray]] = {name: [] for name in methods}

    for fold_idx, (tr, va) in enumerate(splits):
        rng = np.random.default_rng(int(seed) + fold_idx)
        for method_name in methods:
            kwargs: dict[str, Any] = {}
            if method_name == "mutual_info":
                kwargs["n_seeds"] = int(mutual_info_n_seeds)
            scores = np.asarray(
                METHODS[method_name](X[tr], y[tr], X[va], y[va], rng, **kwargs),
                dtype=np.float64,
            )
            if scores.shape != (len(feature_names),):
                raise ValueError(
                    f"{method_name} returned shape {scores.shape}, "
                    f"expected ({len(feature_names)},)"
                )
            per_method_scores[method_name].append(scores)
            per_method_ranks[method_name].append(_scores_to_ranks(scores))

    results: dict[str, MethodResult] = {}
    for method_name in methods:
        scores = np.stack(per_method_scores[method_name], axis=0)
        ranks = np.stack(per_method_ranks[method_name], axis=0)
        results[method_name] = MethodResult(
            mean_score=scores.mean(axis=0),
            mean_rank=ranks.mean(axis=0),
            per_fold_ranks=ranks,
            stability=(ranks <= top_k_effective).mean(axis=0),
        )

    borda = np.stack([result.mean_rank for result in results.values()], axis=0).sum(axis=0)
    mean_stability = np.stack(
        [result.stability for result in results.values()],
        axis=0,
    ).mean(axis=0)
    order = np.argsort(borda, kind="stable")

    # Pre-filter by stability, then apply redundancy filter when requested.
    stable_order = np.array(
        [idx for idx in order if mean_stability[idx] >= float(stability_min)],
        dtype=np.int64,
    )
    if stable_order.size == 0:
        stable_order = order

    redundancy_dropped: list[tuple[str, str, float]] = []
    if redundancy_threshold is not None and redundancy_threshold < 1.0:
        X_ranks = _rank_transform(X)
        # First pass: prefer stable features.
        selected_indices, redundancy_dropped = _select_with_redundancy_filter(
            stable_order,
            X_ranks,
            top_k=top_k_effective,
            redundancy_threshold=float(redundancy_threshold),
            feature_names=feature_names,
        )
        # If fewer than top_k survive, relax stability but keep redundancy.
        # top_k is a hard target; stability is a soft preference.
        if len(selected_indices) < top_k_effective:
            extra, extra_dropped = _select_with_redundancy_filter(
                order,
                X_ranks,
                top_k=top_k_effective - len(selected_indices),
                redundancy_threshold=float(redundancy_threshold),
                feature_names=feature_names,
                preselected=selected_indices,
            )
            selected_indices.extend(extra)
            redundancy_dropped.extend(extra_dropped)
    else:
        selected_indices = stable_order[:top_k_effective].tolist()

    selected = [feature_names[idx] for idx in selected_indices]

    per_method_report: dict[str, Any] = {}
    for method_name, result in results.items():
        ranked = sorted(enumerate(result.mean_rank), key=lambda pair: pair[1])
        per_method_report[method_name] = {
            "ranking": [
                {
                    "feature": feature_names[idx],
                    "mean_rank": float(rank),
                    "mean_score": float(result.mean_score[idx]),
                    "stability": float(result.stability[idx]),
                }
                for idx, rank in ranked
            ]
        }

    report = {
        "dataset": {
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_cases": len(dataset),
            "case_ids": list(dataset.sim_names),
            "feature_names": list(feature_names),
            "target_name": "log1p_delta_p_case",
        },
        "top_k": int(top_k_effective),
        "n_splits_requested": int(n_splits),
        "n_splits_effective": int(effective_splits),
        "per_method": per_method_report,
        "consensus": {
            "selected": list(selected),
            "borda_order": [feature_names[idx] for idx in order],
            "borda_score": {feature_names[idx]: float(borda[idx]) for idx in range(len(feature_names))},
            "mean_stability": {
                feature_names[idx]: float(mean_stability[idx])
                for idx in range(len(feature_names))
            },
            "stability_min": float(stability_min),
            "redundancy_threshold": (
                float(redundancy_threshold) if redundancy_threshold is not None else None
            ),
            "redundancy_dropped": [
                {"feature": dropped, "blocked_by": blocker, "r2": round(r2, 4)}
                for dropped, blocker, r2 in redundancy_dropped
            ],
        },
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    selected_path = output_dir / "selected_features.txt"
    selected_path.write_text("\n".join(selected) + "\n", encoding="utf-8")

    manifest = build_manifest(
        config=config,
        zarr_dir=dataset.zarr_dir,
        feature_names=list(feature_names),
        target_name="log1p_delta_p_case",
        n_rows=len(dataset),
        n_cases=len(dataset),
        seeds={"feature_selection": int(seed)},
    )
    manifest_path = write_manifest(manifest, output_dir)

    return SelectionResult(
        selected_features=list(selected),
        report=report,
        report_path=report_path,
        manifest_path=manifest_path,
        selected_features_path=selected_path,
        used_case_ids=list(dataset.sim_names),
    )
