import pytest

from cases.alpha_d.extrapolation import (
    AXES,
)  # noqa: F401  (ensures package import path)

CFD_DERIVED = ("V_local_over_V_bulk", "log10_Re_local")


def assert_no_cfd_features(input_columns):
    leaked = [c for c in input_columns if c in CFD_DERIVED]
    if leaked:
        raise AssertionError(f"CFD-derived features present in input_columns: {leaked}")


def test_guard_flags_leak():
    with pytest.raises(AssertionError):
        assert_no_cfd_features(["z_hat", "V_local_over_V_bulk"])


def test_guard_passes_geometric_only():
    assert_no_cfd_features(["z_hat", "log10_Re_throat", "A_local_over_A", "inv_Dr"])
