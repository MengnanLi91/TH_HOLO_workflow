"""PyCaret-based feature selection for the case-level pressure-drop workflow.

V1 contract
-----------
- Builds a DataFrame from ``CasePressureDropDataset`` via the deterministic
  ``build_feature_matrix(feature_names)``. The downstream sklearn pipeline
  rebuilds features by name from the same map, so any name PyCaret returns
  must be in ``CANDIDATE_FEATURES`` — synthesized columns would raise in
  ``build_feature_matrix`` later.

- ``setup()`` locks ``polynomial_features``, ``pca``, and ``group_features``
  off for the same reason as the alpha_D selector.

- One row per case at this granularity, so ``fold_strategy='kfold'`` is
  equivalent to the existing GroupKFold loop. No ``fold_groups`` needed.

- Target is ``log1p(delta_p_case)`` — same transform the trainer optimizes,
  so the selector and the model see the same regression problem.

- Returns ``SelectionResult`` so the call site in ``workflow.py`` is
  plug-compatible with the existing sklearn-based ``run_feature_selection``.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from case_pressure_drop.data import CANDIDATE_FEATURES, CasePressureDropDataset
from case_pressure_drop.feature_selection import SelectionResult
from feature_analysis.manifest import build_manifest, write_manifest


logger = logging.getLogger(__name__)


_TARGET_COL = "log1p_delta_p_case"

# Same lockdown as feature_analysis.pycaret_selection: PyCaret must not
# synthesize columns that the deterministic feature map cannot reproduce.
_V1_LOCKED_SETUP_ARGS: dict[str, Any] = {
    "polynomial_features": False,
    "pca": False,
    "group_features": None,
}


def enforce_candidate_set(selected: list[str], candidates: list[str]) -> None:
    """Raise if PyCaret returned a column outside the candidate pool."""
    allowed = set(candidates)
    bad = [n for n in selected if n not in allowed]
    if bad:
        raise RuntimeError(
            f"PyCaret selected features outside the candidate pool: {bad}. "
            "v1 forbids synthesized columns; check _V1_LOCKED_SETUP_ARGS and "
            "the setup() kwargs in your config."
        )


def build_dataframe(dataset: CasePressureDropDataset, feature_names: list[str]):
    """Materialize the dataset as a pandas DataFrame keyed on candidate names."""
    import pandas as pd

    if _TARGET_COL in feature_names:
        raise ValueError(
            f"feature_names contains reserved target column {_TARGET_COL!r}."
        )

    X = dataset.build_feature_matrix(feature_names)  # raises on unknown names
    y = dataset.target_log1p()
    df = pd.DataFrame(X, columns=list(feature_names))
    df[_TARGET_COL] = np.asarray(y, dtype=np.float64)
    return df


def _require_pycaret() -> None:
    try:
        from pycaret import regression
    except Exception as exc:
        raise RuntimeError(
            "PyCaret is not available. Install with `pip install 'pycaret>=3.3'`."
        ) from exc


def _extract_selected(exp) -> list[str]:
    X_train = exp.get_config("X_train_transformed")
    return [str(c) for c in X_train.columns if str(c) != _TARGET_COL]


def _extract_ranking(exp, ranker_id: str) -> list[dict[str, Any]]:
    model = exp.create_model(ranker_id, verbose=False)
    X_train = exp.get_config("X_train_transformed")
    names = [str(c) for c in X_train.columns if str(c) != _TARGET_COL]

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
        return [
            {"feature": n, "importance": None, "rank": i + 1}
            for i, n in enumerate(names)
        ]

    order = np.argsort(-importances)
    return [
        {"feature": names[i], "importance": float(importances[i]), "rank": r + 1}
        for r, i in enumerate(order)
    ]


def run_pycaret_selection(
    dataset: CasePressureDropDataset,
    *,
    feature_names: list[str],
    top_k: int,
    seed: int,
    output_dir: str | Path,
    config: dict[str, Any],
    pycaret_cfg: dict[str, Any] | None = None,
) -> SelectionResult:
    """PyCaret feature selection on the case-level training subset."""
    _require_pycaret()
    from pycaret.regression import RegressionExperiment

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pycaret_cfg = dict(pycaret_cfg or {})
    ranker = str(pycaret_cfg.get("ranker", "gbr"))
    user_setup = dict(pycaret_cfg.get("setup") or {})

    locked_overrides = [k for k in _V1_LOCKED_SETUP_ARGS if k in user_setup]
    if locked_overrides:
        raise ValueError(
            f"setup keys locked in v1: {locked_overrides}. Remove from config; "
            "see case_pressure_drop.pycaret_selection._V1_LOCKED_SETUP_ARGS."
        )

    user_setup.setdefault("normalize", True)
    user_setup.setdefault("remove_multicollinearity", True)
    user_setup.setdefault("feature_selection", True)
    user_setup.setdefault("n_features_to_select", int(top_k))

    df = build_dataframe(dataset, list(feature_names))

    setup_kwargs: dict[str, Any] = {
        "data": df,
        "target": _TARGET_COL,
        "session_id": int(seed),
        "html": False,
        "verbose": False,
        **user_setup,
        **_V1_LOCKED_SETUP_ARGS,
    }

    exp = RegressionExperiment()
    exp.setup(**setup_kwargs)

    selected = _extract_selected(exp)
    enforce_candidate_set(selected, list(feature_names))
    ranking = _extract_ranking(exp, ranker)

    # Trim to top_k by ranker importance when feature_selection produced
    # more rows than requested. PyCaret's selection_method usually honors
    # n_features_to_select, but "classic" can leave a few extras.
    if len(selected) > int(top_k):
        ranked_names = [r["feature"] for r in ranking if r["feature"] in selected]
        selected = ranked_names[: int(top_k)]

    selected_path = output_dir / "selected_features.txt"
    selected_path.write_text("\n".join(selected) + "\n", encoding="utf-8")

    with (output_dir / "feature_ranking.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rank", "feature", "importance"])
        writer.writeheader()
        for row in ranking:
            writer.writerow({
                "rank": row["rank"],
                "feature": row["feature"],
                "importance": (
                    "" if row["importance"] is None
                    else f"{row['importance']:.8e}"
                ),
            })

    setup_record = {
        "user_setup": user_setup,
        "locked_setup": _V1_LOCKED_SETUP_ARGS,
        "ranker": ranker,
        "seed": int(seed),
        "ranking_source": (
            f"create_model('{ranker}') feature_importances_ / |coef_|"
        ),
    }
    (output_dir / "pycaret_setup.json").write_text(
        json.dumps(setup_record, indent=2, sort_keys=True), encoding="utf-8",
    )

    report = {
        "method": "pycaret",
        "dataset": {
            "n_rows": int(df.shape[0]),
            "n_features": int(len(feature_names)),
            "n_cases": len(dataset),
            "case_ids": list(dataset.sim_names),
            "feature_names": list(feature_names),
            "target_name": _TARGET_COL,
        },
        "top_k": int(top_k),
        "consensus": {
            "selected": list(selected),
            "ranking": ranking,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    manifest = build_manifest(
        config=config,
        zarr_dir=dataset.zarr_dir,
        feature_names=list(feature_names),
        target_name=_TARGET_COL,
        n_rows=len(dataset),
        n_cases=len(dataset),
        seeds={"feature_selection_pycaret": int(seed)},
    )
    try:
        import pycaret
        manifest["versions"]["pycaret"] = getattr(pycaret, "__version__", "unknown")
    except ImportError:
        pass
    manifest_path = write_manifest(manifest, output_dir)

    return SelectionResult(
        selected_features=list(selected),
        report=report,
        report_path=report_path,
        manifest_path=manifest_path,
        selected_features_path=selected_path,
        used_case_ids=list(dataset.sim_names),
    )


__all__ = [
    "build_dataframe",
    "enforce_candidate_set",
    "run_pycaret_selection",
    "CANDIDATE_FEATURES",
]
