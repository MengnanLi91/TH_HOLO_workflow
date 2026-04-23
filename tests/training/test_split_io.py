"""Tests for reusable split export helpers."""

from __future__ import annotations

import json
from pathlib import Path

from training.split_io import export_split_files, export_split_files_from_run_meta


def test_export_split_files_writes_clean_names(tmp_path: Path) -> None:
    exported = export_split_files(
        ["sim_a.zarr", "sim_b"],
        ["sim_c.zarr"],
        tmp_path / "splits",
    )

    train_lines = Path(exported["train_file"]).read_text(encoding="utf-8").splitlines()
    test_lines = Path(exported["test_file"]).read_text(encoding="utf-8").splitlines()

    assert train_lines == ["sim_a", "sim_b"]
    assert test_lines == ["sim_c"]


def test_export_split_files_from_run_meta(tmp_path: Path) -> None:
    run_meta_path = tmp_path / "run_meta.json"
    run_meta_path.write_text(
        json.dumps(
            {
                "split": {
                    "train_sims": ["sim_1", "sim_2"],
                    "test_sims": ["sim_3"],
                }
            }
        ),
        encoding="utf-8",
    )

    exported = export_split_files_from_run_meta(run_meta_path, tmp_path / "locked")

    assert Path(exported["train_file"]).read_text(encoding="utf-8") == "sim_1\nsim_2\n"
    assert Path(exported["test_file"]).read_text(encoding="utf-8") == "sim_3\n"
