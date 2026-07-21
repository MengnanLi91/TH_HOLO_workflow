"""Case wrapper around the generic, group-safe PyCaret selector."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from cases.template_case.feature_data import ALLOWLIST, load_feature_matrix
from omegaconf import DictConfig, OmegaConf

from feature_selection.manifest import build_manifest, write_manifest
from feature_selection.pycaret_selection import run_pycaret_selection
from training.case_lists import load_case_selection


@hydra.main(version_base="1.3", config_path="configs", config_name="pycaret")
def main(cfg: DictConfig) -> None:
    """Select reproducible input columns and persist their provenance."""
    config = OmegaConf.to_container(cfg, resolve=True)
    data_cfg = config["data"]
    output_dir = Path(config["output"]["dir"]).expanduser().resolve()
    exclude_cases = load_case_selection(
        data_cfg.get("exclude_cases"),
        data_cfg.get("exclude_cases_file"),
        label="exclude_cases",
    )
    data = load_feature_matrix(
        data_cfg["zarr_dir"],
        target=data_cfg["target"],
        selected_from_allowlist=data_cfg.get("selected_from_allowlist"),
        exclude_cases=exclude_cases,
    )
    result = run_pycaret_selection(
        data,
        pycaret_cfg=config["pycaret"],
        output_dir=output_dir,
        allowlist=ALLOWLIST,
    )
    manifest = build_manifest(
        config=config,
        zarr_dir=data_cfg["zarr_dir"],
        feature_names=data.feature_names,
        target_name=data.target_name,
        n_rows=int(data.X.shape[0]),
        n_cases=data.n_cases,
        seeds={"pycaret": int(config["pycaret"].get("seed", 42))},
    )
    manifest["config"]["data"]["exclude_cases"] = exclude_cases
    write_manifest(manifest, output_dir)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
