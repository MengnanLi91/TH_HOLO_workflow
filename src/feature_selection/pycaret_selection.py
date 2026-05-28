"""PyCaret-based feature selection (generic, case-driven).

Callers pass a ``FeatureAnalysisData`` (whose construction stays
case-side, so the case enforces its own ALLOWLIST upstream) together
with the post-selection ``allowlist`` to validate against.

V1 contract
-----------
- The DataFrame handed to PyCaret is built exclusively from a
  ``FeatureAnalysisData`` instance. PyCaret never reads Zarr directly,
  so the caller-supplied allowlist remains the single guard against
  target-adjacent columns leaking back into the selected feature set.

- ``setup()`` locks ``polynomial_features``, ``feature_interaction``,
  ``pca``, and ``group_features`` off. Those settings would synthesize
  columns that ``TabularPairDataset`` cannot reproduce from the Zarr
  stores, which silently breaks the handoff to PhysicsNeMo training via
  ``data.input_columns_file``.

- Case-level train/test split runs *before* ``setup()``. Row-level
  holdout inside PyCaret would place rows from the same case into both
  splits (rows inside a case are spatially correlated, so that leaks).
  The pre-split test frame is passed through via ``test_data``.

- Inside ``setup()``, ``fold_strategy='groupkfold'`` with
  ``fold_groups='case_id'`` keeps internal CV group-safe.
- Output ``selected_features.txt`` is one name per line, no header,
  drop-in for ``data.input_columns_file`` in the MLP training config.
"""

import csv
import json
import logging
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np

from feature_selection.data import FeatureAnalysisData

logger = logging.getLogger(__name__)


# setup() arguments that are locked in v1. PyCaret-generated polynomial
# or PCA columns cannot be reproduced from Zarr by TabularPairDataset;
# group_features would collapse raw columns into a surrogate feature
# TabularPairDataset also cannot reproduce. (PyCaret 2.x's
# ``feature_interaction`` was folded into ``polynomial_features`` in 3.x.)
_V1_LOCKED_SETUP_ARGS: dict[str, Any] = {
    "polynomial_features": False,
    "pca": False,
    "group_features": None,
}

_CASE_ID_COL = "case_id"


# ---------------------------------------------------------------------------
# DataFrame bridge
# ---------------------------------------------------------------------------


def build_dataframe(data: FeatureAnalysisData):
    """Materialize ``FeatureAnalysisData`` as a pandas DataFrame.

    Columns: ``feature_names + [target_name, case_id]``. The ``case_id``
    column is used for GroupKFold inside PyCaret and for the case-level
    holdout split, and is always dropped from the final selection
    artifact.
    """
    import pandas as pd

    if _CASE_ID_COL in data.feature_names:
        raise ValueError(
            f"FeatureAnalysisData.feature_names contains reserved column {_CASE_ID_COL!r}."
        )
    if data.target_name == _CASE_ID_COL:
        raise ValueError(f"target_name collides with reserved column {_CASE_ID_COL!r}.")

    row_case_id = np.empty(data.groups.shape[0], dtype=object)
    for case_idx, case_name in enumerate(data.case_ids):
        row_case_id[data.groups == case_idx] = case_name

    df = pd.DataFrame(data.X, columns=list(data.feature_names))
    df[data.target_name] = np.asarray(data.y)
    df[_CASE_ID_COL] = row_case_id
    return df


def case_level_split(df, *, case_id_col: str, test_ratio: float, seed: int):
    """Split a row-level DataFrame into train/test without crossing cases."""
    from sklearn.model_selection import GroupShuffleSplit

    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1); got {test_ratio}.")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    ((train_idx, test_idx),) = gss.split(df, groups=df[case_id_col].values)
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def enforce_allowlist(selected: list[str], allowlist: Collection[str]) -> None:
    """Raise if any selected feature is outside the caller's allowlist.

    Kept as a top-level function so the v1 contract is unit-testable
    without importing PyCaret. The caller supplies the allowlist; this
    keeps the library case-agnostic.
    """
    allowed = set(allowlist)
    out_of_allowlist = [f for f in selected if f not in allowed]
    if out_of_allowlist:
        raise RuntimeError(
            f"PyCaret selected features outside allowlist: {out_of_allowlist}. "
            "v1 forbids synthesized columns; check _V1_LOCKED_SETUP_ARGS and "
            "setup() kwargs in your config."
        )


# ---------------------------------------------------------------------------
# PyCaret wrappers (lazy-imported)
# ---------------------------------------------------------------------------


def _require_pycaret():
    try:
        from pycaret import regression  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "PyCaret is not available. Install the optional dependency: "
            "`pip install 'pycaret>=3.0'`."
        ) from exc


def _extract_selected_features(exp, target_name: str) -> list[str]:
    """Read selected feature names from PyCaret's post-setup state.

    After ``setup(feature_selection=True, ...)`` the training feature
    matrix (``X_train_transformed``) contains only the columns PyCaret
    kept after low-variance filtering, multicollinearity removal, and
    feature selection.
    """
    X_train = exp.get_config("X_train_transformed")
    cols = [str(c) for c in X_train.columns]
    # The target and case_id should not appear here, but enforce it defensively:
    # the selected_features.txt contract is strict.
    return [c for c in cols if c != target_name and c != _CASE_ID_COL]


def _extract_ranking(exp, ranker_id: str) -> list[dict[str, Any]]:
    """Fit a single ranker and return sorted feature importances.

    The ranker identity is recorded in ``pycaret_setup.json`` so runs
    are reproducible. Uses ``feature_importances_`` for tree models and
    ``|coef_|`` for linear models. Falls back to an unranked list if the
    estimator exposes neither.
    """
    # cross_validation=False fits the ranker once on the full train set
    # instead of running k-fold CV. We only consume feature_importances_ /
    # |coef_|, so CV-validated metrics are wasted compute (typically ~10x
    # slowdown on sklearn GBR with default fold=10). verbose=True surfaces
    # PyCaret's tqdm progress bar for the single fit.
    model = exp.create_model(ranker_id, cross_validation=False, verbose=True)
    X_train = exp.get_config("X_train_transformed")
    names = [str(c) for c in X_train.columns if c != _CASE_ID_COL]

    # PyCaret may return a Pipeline; unwrap the final estimator.
    est = model
    if hasattr(model, "steps"):
        est = model.steps[-1][1]

    importances: np.ndarray | None = None
    if hasattr(est, "feature_importances_"):
        importances = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        if coef.ndim > 1:
            coef = coef[0]
        importances = np.abs(coef)

    if importances is None or len(importances) != len(names):
        return [{"feature": n, "importance": None, "rank": i + 1} for i, n in enumerate(names)]

    order = np.argsort(-importances)
    return [
        {
            "feature": names[i],
            "importance": float(importances[i]),
            "rank": r + 1,
        }
        for r, i in enumerate(order)
    ]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_pycaret_selection(
    data: FeatureAnalysisData,
    *,
    pycaret_cfg: dict[str, Any],
    output_dir: Path,
    allowlist: Collection[str],
) -> dict[str, Any]:
    """Run the v1 PyCaret selection path and write artifacts.

    ``allowlist`` is the set of permitted post-selection feature names;
    callers supply it (typically the case-side ALLOWLIST constant) so
    this library stays alpha-D-agnostic.
    """
    _require_pycaret()
    from pycaret.regression import RegressionExperiment

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(pycaret_cfg.get("seed", 42))
    test_ratio = float(pycaret_cfg.get("test_ratio", 0.2))
    ranker_id = str(pycaret_cfg.get("ranker", "gbr"))

    df = build_dataframe(data)
    train_df, test_df = case_level_split(
        df,
        case_id_col=_CASE_ID_COL,
        test_ratio=test_ratio,
        seed=seed,
    )
    logger.info(
        "PyCaret split: train=%d rows / %d cases, test=%d rows / %d cases",
        len(train_df),
        train_df[_CASE_ID_COL].nunique(),
        len(test_df),
        test_df[_CASE_ID_COL].nunique(),
    )

    user_setup: dict[str, Any] = dict(pycaret_cfg.get("setup") or {})
    locked_overrides = [k for k in _V1_LOCKED_SETUP_ARGS if k in user_setup]
    if locked_overrides:
        raise ValueError(
            f"setup keys locked in v1: {locked_overrides}. Remove them from the "
            "config; see feature_selection.pycaret_selection._V1_LOCKED_SETUP_ARGS."
        )
    # Defaults when unspecified by the user.
    user_setup.setdefault("normalize", True)
    user_setup.setdefault("remove_multicollinearity", True)
    user_setup.setdefault("feature_selection", True)

    setup_kwargs: dict[str, Any] = {
        "data": train_df,
        "target": data.target_name,
        "test_data": test_df,
        "fold_strategy": "groupkfold",
        "fold_groups": _CASE_ID_COL,
        "ignore_features": [_CASE_ID_COL],
        "session_id": seed,
        "html": False,
        "verbose": False,
        **user_setup,
        # Locked kwargs go last so they cannot be silently overridden.
        **_V1_LOCKED_SETUP_ARGS,
    }

    exp = RegressionExperiment()
    exp.setup(**setup_kwargs)

    selected = _extract_selected_features(exp, data.target_name)
    enforce_allowlist(selected, allowlist)
    ranking = _extract_ranking(exp, ranker_id)

    # Trim to n_features_to_select by ranker importance. PyCaret's
    # ``classic`` selection_method can leave extras above the requested
    # count; honour the cap so the artifact matches the user's request.
    top_k = user_setup.get("n_features_to_select")
    if top_k is not None and len(selected) > int(top_k):
        ranked_names = [r["feature"] for r in ranking if r["feature"] in selected]
        selected = ranked_names[: int(top_k)]

    # --- Artifacts -----------------------------------------------------
    write_selected_features(
        output_dir / "selected_features.txt",
        selected,
        allowlist=allowlist,
    )

    with (output_dir / "feature_ranking.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rank", "feature", "importance"])
        writer.writeheader()
        for row in ranking:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "feature": row["feature"],
                    "importance": ("" if row["importance"] is None else f"{row['importance']:.8e}"),
                }
            )

    setup_record = {
        "user_setup": user_setup,
        "locked_setup": _V1_LOCKED_SETUP_ARGS,
        "fold_strategy": "groupkfold",
        "fold_groups": _CASE_ID_COL,
        "ignore_features": [_CASE_ID_COL],
        "ranker": ranker_id,
        "seed": seed,
        "test_ratio": test_ratio,
        "ranking_source": (f"create_model('{ranker_id}') feature_importances_ / |coef_|"),
    }
    (output_dir / "pycaret_setup.json").write_text(
        json.dumps(setup_record, indent=2, sort_keys=True)
    )

    return {
        "selected": selected,
        "ranking": ranking,
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_cases": int(train_df[_CASE_ID_COL].nunique()),
        "n_test_cases": int(test_df[_CASE_ID_COL].nunique()),
    }


def write_selected_features(
    path: Path,
    selected: list[str],
    *,
    allowlist: Collection[str] | None = None,
) -> None:
    """Write ``selected_features.txt``.

    One name per line, no header, no blank lines, trailing newline.
    Drop-in for ``data.input_columns_file`` in the training config. When
    ``allowlist`` is supplied (typically the case-side ALLOWLIST), the
    file is rejected if any name falls outside it.
    """
    if any(not s or s != s.strip() for s in selected):
        raise ValueError("selected_features contains empty or whitespace-padded names.")
    if allowlist is not None:
        enforce_allowlist(selected, allowlist)
    Path(path).write_text("\n".join(selected) + "\n")
