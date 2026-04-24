"""Helpers for exporting reusable train/test split files."""

import json
from pathlib import Path
from typing import Any


def _clean_sim_names(sim_names: list[str] | tuple[str, ...]) -> list[str]:
    cleaned = [str(name).strip().removesuffix(".zarr") for name in sim_names if str(name).strip()]
    if not cleaned:
        raise ValueError("Expected at least one simulation name.")
    return cleaned


def write_sim_name_list(path: str | Path, sim_names: list[str] | tuple[str, ...]) -> Path:
    """Write one simulation name per line and return the resolved output path."""
    cleaned = _clean_sim_names(sim_names)
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    return output_path


def export_split_files(
    train_sims: list[str] | tuple[str, ...],
    test_sims: list[str] | tuple[str, ...],
    output_dir: str | Path,
    *,
    train_filename: str = "train.txt",
    test_filename: str = "test.txt",
) -> dict[str, str]:
    """Export train/test simulation-name lists into a directory."""
    output_root = Path(output_dir).expanduser().resolve()
    train_path = write_sim_name_list(output_root / train_filename, train_sims)
    test_path = write_sim_name_list(output_root / test_filename, test_sims)
    return {
        "output_dir": str(output_root),
        "train_file": str(train_path),
        "test_file": str(test_path),
    }


def load_run_meta(run_meta_path: str | Path) -> dict[str, Any]:
    """Load a run_meta JSON file."""
    path = Path(run_meta_path).expanduser().resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def export_split_files_from_run_meta(
    run_meta_path: str | Path,
    output_dir: str | Path,
    *,
    train_filename: str = "train.txt",
    test_filename: str = "test.txt",
) -> dict[str, str]:
    """Export reusable split files from a run_meta.json file."""
    run_meta = load_run_meta(run_meta_path)
    split_meta = dict(run_meta.get("split") or {})
    train_sims = split_meta.get("train_sims") or []
    test_sims = split_meta.get("test_sims") or []
    if not train_sims or not test_sims:
        raise ValueError(
            "run_meta.json does not contain both split.train_sims and split.test_sims."
        )

    exported = export_split_files(
        train_sims,
        test_sims,
        output_dir,
        train_filename=train_filename,
        test_filename=test_filename,
    )
    exported["run_meta"] = str(Path(run_meta_path).expanduser().resolve())
    return exported
