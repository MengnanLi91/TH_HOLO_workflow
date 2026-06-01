"""Tests for verify_delta_p.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_read_moose_inlet_pressure(tmp_path):
    """Postprocessor stores integral; verifier divides by RZ inlet area."""
    from cases.alpha_d.verify_delta_p import INLET_AREA_M2, read_moose_inlet_pressure

    csv_path = tmp_path / "out.csv"
    csv_path.write_text("time,inlet-p,outlet-u\n0,1.2345,1.0\n")

    p = read_moose_inlet_pressure(csv_path)
    assert p == pytest.approx(1.2345 / INLET_AREA_M2)


def test_compare_emits_two_relative_errors(tmp_path):
    from cases.alpha_d.verify_delta_p import compare

    sidecar = tmp_path / "forchheimer_profile.meta.json"
    sidecar.write_text(
        json.dumps(
            {
                "delta_p_truth": 100.0,
                "delta_p_surrogate": 105.0,
            }
        )
    )

    out = compare(
        sidecar_path=sidecar,
        delta_p_moose=110.0,
    )
    assert out["delta_p_truth"] == 100.0
    assert out["delta_p_surrogate"] == 105.0
    assert out["delta_p_moose"] == 110.0
    assert out["surrogate_fidelity_relerr"] == pytest.approx(0.05)
    assert out["coupling_fidelity_relerr"] == pytest.approx(110.0 / 105.0 - 1.0)
