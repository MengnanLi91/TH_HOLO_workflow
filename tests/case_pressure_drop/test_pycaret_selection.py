"""Tests for the PyCaret feature-selection bridge in case_pressure_drop.

PyCaret itself is optional; the bridge tests don't import it. The full
pipeline test is gated on ``pytest.importorskip('pycaret')``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from case_pressure_drop.data import CANDIDATE_FEATURES, CasePressureDropDataset
from case_pressure_drop.pycaret_selection import (
    build_dataframe,
    enforce_candidate_set,
)


# ---------------------------------------------------------------------------
# Synthetic case zarr fixture (same shape as test_workflow.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Bridge tests (no PyCaret needed)
# ---------------------------------------------------------------------------


def test_build_dataframe_columns_target_and_log1p_transform(
    synthetic_case_dataset: Path,
) -> None:
    dataset = CasePressureDropDataset.from_zarr_dir(synthetic_case_dataset)
    feature_names = list(CANDIDATE_FEATURES)

    df = build_dataframe(dataset, feature_names)

    assert list(df.columns) == feature_names + ["log1p_delta_p_case"]
    assert len(df) == len(dataset)
    # Target column is log1p of delta_p_case (no leakage of raw scale).
    np.testing.assert_allclose(
        df["log1p_delta_p_case"].to_numpy(),
        np.log1p(dataset.delta_p_case),
        rtol=1e-12,
    )
    # Feature matrix matches the deterministic feature map round-trip.
    np.testing.assert_allclose(
        df[feature_names].to_numpy(),
        dataset.build_feature_matrix(feature_names),
    )


def test_build_dataframe_rejects_target_name_in_features(
    synthetic_case_dataset: Path,
) -> None:
    dataset = CasePressureDropDataset.from_zarr_dir(synthetic_case_dataset)
    with pytest.raises(ValueError, match="log1p_delta_p_case"):
        build_dataframe(dataset, ["Re", "log1p_delta_p_case"])


def test_build_dataframe_rejects_unknown_candidate(
    synthetic_case_dataset: Path,
) -> None:
    dataset = CasePressureDropDataset.from_zarr_dir(synthetic_case_dataset)
    # `build_feature_matrix` is the choke point — synthesized names raise here.
    with pytest.raises(ValueError, match="Unknown candidate"):
        build_dataframe(dataset, ["Re", "synth_pca_1"])


def test_enforce_candidate_set_rejects_out_of_pool():
    enforce_candidate_set(["Re", "Dr"], list(CANDIDATE_FEATURES))  # ok
    with pytest.raises(RuntimeError, match="outside the candidate pool"):
        enforce_candidate_set(["Re", "Dr_squared"], list(CANDIDATE_FEATURES))


def test_workflow_dispatch_rejects_unknown_method(synthetic_case_dataset: Path) -> None:
    """Workflow should raise when feature_selection.method is unknown."""
    from case_pressure_drop.workflow import train_case_pressure_drop

    cfg = {
        "data": {"zarr_dir": str(synthetic_case_dataset)},
        "feature_selection": {"enabled": True, "method": "not_a_real_method"},
        "models": {"cv": {"n_splits": 2, "seed": 0}},
        "output": {"case_dir": str(synthetic_case_dataset.parent / "out")},
    }
    with pytest.raises(ValueError, match="not supported"):
        train_case_pressure_drop(cfg)
