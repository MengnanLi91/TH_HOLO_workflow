"""Feature-selection methods + consensus + baseline + report writing.

All methods operate on a single CV fold (train half) and return a
per-feature score where higher = more important. The runner converts
scores to ranks (1 = best) for Borda aggregation.

Grouped-feature semantics
-------------------------
Blocks declared in ``GROUPED_FEATURES`` (e.g. ``region_onehot``) are
treated as atomic units for the consensus decision:

  block_rank_per_fold     = min(member_ranks)
  block_stability         = fraction of folds where block_rank <= block_rank_top_k
  block kept (mode="any") = block_stability >= block_stability_min

Members still rank individually in per-method outputs; only the
consensus "selected" set collapses them into one in-or-out decision.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import (
    RFECV,
    f_regression as _f_regression,
    mutual_info_regression,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_analysis.data_loader import GROUPED_FEATURES, FeatureAnalysisData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-method scorers. Each returns a [D]-length array of importance scores.
# Higher = more important. They receive train/validation slices + the RNG.
# ---------------------------------------------------------------------------

def _scale(X_tr: np.ndarray, X_va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = StandardScaler().fit(X_tr)
    return s.transform(X_tr), s.transform(X_va)


def score_f_regression(X_tr, y_tr, X_va, y_va, rng, **_):
    F, _p = _f_regression(X_tr, y_tr)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def score_mutual_info(X_tr, y_tr, X_va, y_va, rng, *, n_seeds: int = 10, **_):
    acc = np.zeros(X_tr.shape[1], dtype=np.float64)
    for _ in range(max(1, int(n_seeds))):
        seed = int(rng.integers(0, 2**31 - 1))
        acc += mutual_info_regression(X_tr, y_tr, random_state=seed)
    return acc / max(1, int(n_seeds))


def score_lasso(X_tr, y_tr, X_va, y_va, rng, **_):
    # Avoid sklearn Gram-matrix validation mismatches seen with float32
    # and auto-precompute in some versions/environments.
    X_tr = np.asarray(X_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        (
            "lasso",
            LassoCV(
                cv=5,
                alphas=30,
                max_iter=5000,
                random_state=0,
                precompute=False,
            ),
        ),
    ])
    pipe.fit(X_tr, y_tr)
    return np.abs(pipe.named_steps["lasso"].coef_)


def score_gbr_permutation(X_tr, y_tr, X_va, y_va, rng, **_):
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    gbr.fit(X_tr, y_tr)
    res = permutation_importance(
        gbr, X_va, y_va, n_repeats=10,
        random_state=int(rng.integers(0, 2**31 - 1)), n_jobs=1,
    )
    return res.importances_mean


def score_rfecv_ridge(X_tr, y_tr, X_va, y_va, rng, **_):
    # RFECV's internal CV is not group-aware; acceptable because we are
    # already inside an outer GroupKFold fold. Inner splitter is 5-fold
    # plain KFold on the train half, which is fine for ranking.
    X_tr_s, _ = _scale(X_tr, X_va)
    sel = RFECV(
        estimator=Ridge(alpha=1.0),
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        scoring="neg_mean_squared_error",
        min_features_to_select=1,
    )
    sel.fit(X_tr_s, y_tr)
    # ranking_: 1 = kept to the end, larger = eliminated earlier.
    # Invert so best features have the largest score.
    return (sel.ranking_.max() - sel.ranking_ + 1).astype(float)


def score_sequential_ridge(X_tr, y_tr, X_va, y_va, rng, **_):
    # Manual forward stepwise: greedily add the feature that most improves
    # cv-MSE until all features are ranked. Gives a full ordering (rank 1
    # = added first), unlike sklearn's SequentialFeatureSelector which
    # only exposes a binary support_ mask.
    from sklearn.model_selection import cross_val_score

    X_tr_s, _ = _scale(X_tr, X_va)
    d = X_tr_s.shape[1]
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    remaining = list(range(d))
    order: list[int] = []

    while remaining:
        best_idx = None
        best_score = -np.inf
        for idx in remaining:
            cols = order + [idx]
            scores = cross_val_score(
                Ridge(alpha=1.0),
                X_tr_s[:, cols], y_tr,
                cv=cv, scoring="neg_mean_squared_error", n_jobs=1,
            )
            mean_score = float(scores.mean())
            if mean_score > best_score:
                best_score = mean_score
                best_idx = idx
        order.append(best_idx)
        remaining.remove(best_idx)

    # rank-1 feature → highest score; rank-D → lowest.
    out = np.zeros(d, dtype=np.float64)
    for r, idx in enumerate(order):
        out[idx] = d - r
    return out


METHODS: dict[str, Callable] = {
    "f_regression": score_f_regression,
    "mutual_info": score_mutual_info,
    "lasso": score_lasso,
    "gbr_permutation": score_gbr_permutation,
    "rfecv_ridge": score_rfecv_ridge,
    "sequential_ridge": score_sequential_ridge,
}


# ---------------------------------------------------------------------------
# CV loop + per-method aggregation
# ---------------------------------------------------------------------------

def _scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """Convert importance scores → ranks where 1 = best. Ties → average rank."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average-rank ties so tied features don't get artificial precedence.
    unique_vals, inv = np.unique(scores, return_inverse=True)
    for v_idx in range(len(unique_vals)):
        mask = inv == v_idx
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


@dataclass
class MethodResult:
    mean_score: np.ndarray    # [D] scores averaged across folds
    mean_rank: np.ndarray     # [D] ranks averaged across folds (1 = best)
    per_fold_ranks: np.ndarray  # [n_folds, D]
    top_k_in_fold: np.ndarray   # [n_folds, D] bool — rank ≤ top_k
    stability: np.ndarray     # [D] fraction of folds where feature was in top_k


def run_methods(
    data: FeatureAnalysisData,
    *,
    methods: list[str],
    cv_cfg: dict,
    top_k: int,
    mi_cfg: dict,
) -> dict[str, MethodResult]:
    """Run all methods across GroupKFold folds; return per-method aggregates."""
    n_splits = int(cv_cfg.get("n_splits", 5))
    seed = int(cv_cfg.get("seed", 42))

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(data.X, data.y, groups=data.groups))

    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Known: {sorted(METHODS)}")

    D = data.X.shape[1]
    per_method_scores: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    per_method_ranks: dict[str, list[np.ndarray]] = {m: [] for m in methods}

    for fold, (tr, va) in enumerate(splits):
        rng = np.random.default_rng(seed + fold)
        X_tr, y_tr = data.X[tr], data.y[tr]
        X_va, y_va = data.X[va], data.y[va]

        for m in methods:
            fn = METHODS[m]
            kwargs: dict[str, Any] = {"k": top_k}
            if m == "mutual_info":
                kwargs["n_seeds"] = int(mi_cfg.get("n_seeds", 10))
            try:
                s = np.asarray(fn(X_tr, y_tr, X_va, y_va, rng, **kwargs), dtype=np.float64)
            except Exception as exc:
                logger.exception("method=%s fold=%d failed: %s", m, fold, exc)
                s = np.zeros(D)
            if s.shape != (D,):
                raise ValueError(f"{m} returned shape {s.shape}, expected ({D},)")
            per_method_scores[m].append(s)
            per_method_ranks[m].append(_scores_to_ranks(s))
        logger.info("fold %d/%d done", fold + 1, n_splits)

    results: dict[str, MethodResult] = {}
    for m in methods:
        scores = np.stack(per_method_scores[m], axis=0)  # [n_folds, D]
        ranks = np.stack(per_method_ranks[m], axis=0)
        in_top_k = ranks <= top_k
        results[m] = MethodResult(
            mean_score=scores.mean(axis=0),
            mean_rank=ranks.mean(axis=0),
            per_fold_ranks=ranks,
            top_k_in_fold=in_top_k,
            stability=in_top_k.mean(axis=0),
        )
    return results


# ---------------------------------------------------------------------------
# Borda consensus + grouped-block collapsing
# ---------------------------------------------------------------------------

def borda_consensus(
    results: dict[str, MethodResult],
    feature_names: list[str],
) -> np.ndarray:
    """Sum per-method mean-ranks; lower = better. Returns [D] Borda scores."""
    stack = np.stack([r.mean_rank for r in results.values()], axis=0)  # [M, D]
    return stack.sum(axis=0)


def collapse_blocks_to_selection(
    *,
    feature_names: list[str],
    borda: np.ndarray,
    method_results: dict[str, "MethodResult"],
    grouped_cfg: dict,
    top_k: int,
    stability_min: float,
) -> dict[str, Any]:
    """Decide which features to keep, respecting atomic blocks.

    Block-level rules (from ``grouped_cfg[block].keep_rule``):
      mode: "any"  → per-fold block_rank = min(member_ranks)
      mode: "all"  → per-fold block_rank = max(member_ranks)
      block_rank_top_k    — threshold used per fold (distinct from top_k)
      block_stability_min — fraction of (method × fold) slots where
                            block_rank ≤ block_rank_top_k, required to keep.

    Ungrouped features: kept iff in top-k by Borda AND mean stability
    across methods ≥ ``stability_min``.
    """
    borda_order = np.argsort(borda, kind="stable")  # ascending = best first

    # Mean stability per feature across methods (used for ungrouped features).
    stab_stack = np.stack([r.stability for r in method_results.values()], axis=0)  # [M, D]
    mean_stability = stab_stack.mean(axis=0)

    # Map every feature → block name, or None if ungrouped.
    feature_to_block: dict[str, str | None] = {n: None for n in feature_names}
    enabled_blocks = {
        block_name
        for block_name, block_cfg in grouped_cfg.items()
        if bool((block_cfg or {}).get("enabled", True))
    }
    for block_name, members in GROUPED_FEATURES.items():
        if block_name not in enabled_blocks:
            continue
        for m in members:
            if m in feature_to_block:
                feature_to_block[m] = block_name

    block_decisions: dict[str, dict[str, Any]] = {}
    for block_name, members in GROUPED_FEATURES.items():
        member_idx = [feature_names.index(m) for m in members if m in feature_names]
        if not member_idx:
            continue
        block_cfg = grouped_cfg.get(block_name, {})
        if not bool(block_cfg.get("enabled", True)):
            continue
        keep_rule = dict(block_cfg.get("keep_rule") or {})
        mode = str(keep_rule.get("mode", "any")).lower()
        if mode not in {"any", "all"}:
            raise ValueError(
                f"grouped_features.{block_name}.keep_rule.mode must be "
                f"'any' or 'all', got {mode!r}"
            )
        block_rank_top_k = int(keep_rule.get("block_rank_top_k", top_k))
        block_stab_min = float(keep_rule.get("block_stability_min", stability_min))

        # Per-fold per-method block rank from the full ranks tensor.
        hit_flags: list[bool] = []
        for r in method_results.values():
            member_ranks = r.per_fold_ranks[:, member_idx]  # [n_folds, |members|]
            if mode == "any":
                block_rank = member_ranks.min(axis=1)
            else:
                block_rank = member_ranks.max(axis=1)
            in_top = block_rank <= block_rank_top_k
            hit_flags.extend(in_top.tolist())
        block_stab = float(np.mean(hit_flags)) if hit_flags else 0.0
        block_borda = float(
            np.min(borda[member_idx]) if mode == "any" else np.max(borda[member_idx])
        )
        keep = block_stab >= block_stab_min
        block_decisions[block_name] = {
            "members": list(members),
            "member_indices": member_idx,
            "mode": mode,
            "block_borda": block_borda,
            "block_stability": block_stab,
            "block_rank_top_k": block_rank_top_k,
            "block_stability_min": block_stab_min,
            "keep": bool(keep),
        }

    # Build selected set.
    selected: list[str] = []
    rationale: dict[str, str] = {}

    # First pass: blocks.
    for block_name, dec in block_decisions.items():
        if dec["keep"]:
            selected.extend(dec["members"])
            for m in dec["members"]:
                rationale[m] = f"kept as member of block '{block_name}' (stability={dec['block_stability']:.2f})"

    # Second pass: ungrouped features, keep top-k-by-Borda (excluding already kept).
    kept_set = set(selected)
    for idx in borda_order:
        name = feature_names[idx]
        if name in kept_set:
            continue
        if feature_to_block[name] is not None:
            # Block decision already made (either kept or rejected atomically).
            continue
        if len(kept_set) >= top_k:
            break
        if mean_stability[idx] < stability_min:
            continue
        kept_set.add(name)
        selected.append(name)
        rationale[name] = (
            f"top-{top_k} by Borda (rank={int(np.where(borda_order == idx)[0][0]) + 1}, "
            f"stability={mean_stability[idx]:.2f})"
        )

    return {
        "selected": selected,
        "rationale": rationale,
        "block_decisions": block_decisions,
        "mean_stability": mean_stability.tolist(),
        "borda_score": borda.tolist(),
        "borda_order": [feature_names[i] for i in borda_order],
    }


# ---------------------------------------------------------------------------
# Baseline retrain on the selected feature set
# ---------------------------------------------------------------------------

def run_baseline(
    data: FeatureAnalysisData,
    selected_features: list[str],
    *,
    models: list[str],
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    """Group-wise CV baseline using only the selected feature columns."""
    if not selected_features:
        return {"skipped": True, "reason": "no features selected"}

    sel_idx = [data.feature_names.index(f) for f in selected_features]
    X = data.X[:, sel_idx]
    y = data.y
    groups = data.groups

    gkf = GroupKFold(n_splits=int(n_splits))
    out: dict[str, Any] = {"selected_features": list(selected_features), "per_model": {}}

    builders = {
        "ridge": lambda: Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "gbr": lambda: GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=int(seed),
        ),
    }

    for m in models:
        if m not in builders:
            out["per_model"][m] = {"skipped": True, "reason": f"unknown model '{m}'"}
            continue
        fold_r2: list[float] = []
        fold_rmse: list[float] = []
        for tr, va in gkf.split(X, y, groups=groups):
            est = builders[m]()
            est.fit(X[tr], y[tr])
            pred = est.predict(X[va])
            fold_r2.append(float(r2_score(y[va], pred)))
            fold_rmse.append(float(np.sqrt(mean_squared_error(y[va], pred))))
        out["per_model"][m] = {
            "r2_mean": float(np.mean(fold_r2)),
            "r2_std": float(np.std(fold_r2)),
            "rmse_mean": float(np.mean(fold_rmse)),
            "rmse_std": float(np.std(fold_rmse)),
            "per_fold_r2": fold_r2,
            "per_fold_rmse": fold_rmse,
        }
    return out


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(
    *,
    data: FeatureAnalysisData,
    method_results: dict[str, MethodResult],
    consensus: dict[str, Any],
    baseline: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    per_method_block = {}
    for name, r in method_results.items():
        ranked = sorted(
            enumerate(r.mean_rank), key=lambda pair: pair[1]
        )
        per_method_block[name] = {
            "ranking": [
                {
                    "feature": data.feature_names[idx],
                    "mean_rank": float(rank),
                    "mean_score": float(r.mean_score[idx]),
                    "stability": float(r.stability[idx]),
                }
                for idx, rank in ranked
            ],
        }

    return {
        "dataset": {
            "n_rows": int(data.X.shape[0]),
            "n_features": int(data.X.shape[1]),
            "n_cases": data.n_cases,
            "feature_names": list(data.feature_names),
            "target_name": data.target_name,
            "local_velocity_normalization": bool(data.local_velocity_normalization),
        },
        "top_k": int(top_k),
        "per_method": per_method_block,
        "consensus": consensus,
        "baseline": baseline,
    }
