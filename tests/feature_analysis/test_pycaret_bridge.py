"""Tests for the PyCaret feature-selection bridge.

PyCaret itself is optional; tests here only exercise the bridge logic
(DataFrame materialization, case-level split, ALLOWLIST enforcement,
selected_features.txt contract). The full pipeline is covered by the
smoke test guarded on ``pytest.importorskip("pycaret")``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from feature_analysis import ALLOWLIST, FeatureAnalysisData
from feature_analysis.pycaret_selection import (
    build_dataframe,
    case_level_split,
    enforce_allowlist,
    write_selected_features,
)


def _make_data(n_cases: int = 5, rows_per_case: int = 10) -> FeatureAnalysisData:
    """Minimal FeatureAnalysisData with ALLOWLIST-safe feature names."""
    feature_names = ["log10_Re", "Dr", "Lr", "z_hat"]
    rng = np.random.default_rng(0)
    n = n_cases * rows_per_case
    X = rng.standard_normal((n, len(feature_names))).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    groups = np.repeat(np.arange(n_cases, dtype=np.int32), rows_per_case)
    case_ids = [f"case_{i:03d}" for i in range(n_cases)]
    return FeatureAnalysisData(
        X=X,
        y=y,
        groups=groups,
        feature_names=feature_names,
        target_name="log_alpha_D",
        case_ids=case_ids,
        rows_per_case=[rows_per_case] * n_cases,
        local_velocity_normalization=True,
    )


def test_build_dataframe_preserves_names_target_and_row_groups():
    data = _make_data(n_cases=4, rows_per_case=7)
    df = build_dataframe(data)

    assert list(df.columns) == data.feature_names + [data.target_name, "case_id"]
    assert len(df) == data.X.shape[0]
    # Feature matrix round-trips numerically.
    np.testing.assert_allclose(
        df[data.feature_names].to_numpy(dtype=np.float32), data.X
    )
    # Target round-trips.
    np.testing.assert_allclose(
        df[data.target_name].to_numpy(dtype=np.float32), data.y
    )
    # Each row's case_id matches the integer group index via case_ids map.
    expected = np.array(
        [data.case_ids[int(g)] for g in data.groups], dtype=object
    )
    assert np.array_equal(df["case_id"].to_numpy(dtype=object), expected)


def test_build_dataframe_rejects_reserved_column_name():
    data = _make_data()
    data.feature_names.append("case_id")
    # Extend X with a junk column so shapes stay consistent.
    data.X = np.column_stack([data.X, np.zeros(len(data.X), dtype=np.float32)])
    with pytest.raises(ValueError, match="case_id"):
        build_dataframe(data)


def test_case_level_split_never_crosses_cases():
    data = _make_data(n_cases=10, rows_per_case=8)
    df = build_dataframe(data)
    train_df, test_df = case_level_split(
        df, case_id_col="case_id", test_ratio=0.3, seed=123,
    )
    train_cases = set(train_df["case_id"].unique())
    test_cases = set(test_df["case_id"].unique())
    assert train_cases.isdisjoint(test_cases)
    # Every row ends up in exactly one side.
    assert len(train_df) + len(test_df) == len(df)
    # Both sides non-empty with a reasonable split.
    assert len(train_df) > 0 and len(test_df) > 0


def test_enforce_allowlist_rejects_unknown_names():
    enforce_allowlist(["log10_Re", "Dr"])  # inside ALLOWLIST, no raise
    bad_name = "synth_poly_log10_Re_x_Dr"
    assert bad_name not in ALLOWLIST
    with pytest.raises(RuntimeError, match="outside ALLOWLIST"):
        enforce_allowlist([bad_name])


def test_selected_features_txt_roundtrips_through_adapter_contract(tmp_path: Path):
    """selected_features.txt must parse the same way as the training adapter.

    training/adapters.py reads the file with:
        [line.strip() for line in path.read_text().splitlines() if line.strip()]
    This test enforces that contract end-to-end.
    """
    selected = ["log10_Re", "Dr", "Lr", "z_hat"]
    path = tmp_path / "selected_features.txt"
    write_selected_features(path, selected)

    text = path.read_text()
    assert text.endswith("\n")
    assert "\n\n" not in text, "blank line would corrupt adapter parsing"
    assert not text.startswith("#"), "no header allowed"

    roundtripped = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    assert roundtripped == selected


def test_write_selected_features_rejects_non_allowlisted():
    path = Path("/tmp/should_never_be_written.txt")
    with pytest.raises(RuntimeError, match="outside ALLOWLIST"):
        write_selected_features(path, ["delta_p_case"])


def test_write_selected_features_rejects_whitespace():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt") as tf:
        with pytest.raises(ValueError, match="whitespace"):
            write_selected_features(Path(tf.name), ["log10_Re", " Dr"])
