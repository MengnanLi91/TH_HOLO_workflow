"""Tests for src/cases/alpha_d/export_friction_profile.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_module_imports():
    from cases.alpha_d import export_friction_profile  # noqa: F401


def test_cli_parses_required_args(tmp_path):
    from cases.alpha_d.export_friction_profile import parse_args

    args = parse_args(
        [
            "--zarr",
            str(tmp_path / "case.zarr"),
            "--checkpoint",
            str(tmp_path / "model.mdlus"),
            "--run-meta",
            str(tmp_path / "run_meta.json"),
            "--output-csv",
            str(tmp_path / "out.csv"),
        ]
    )
    assert args.zarr == tmp_path / "case.zarr"
    assert args.checkpoint == tmp_path / "model.mdlus"
    assert args.run_meta == tmp_path / "run_meta.json"
    assert args.output_csv == tmp_path / "out.csv"
