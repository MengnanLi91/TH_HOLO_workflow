"""PyCaret-based feature selection entry point.

Reuses ``cases.alpha_d.feature_data.load_feature_matrix`` (and its
``ALLOWLIST``) to build the DataFrame, runs PyCaret regression
``setup()`` with GroupKFold by ``case_id``, and writes a
``selected_features.txt`` that drops in to
``data.input_columns_file`` in the MLP training config.

Usage (from src/ directory):
    python cases/alpha_d/run_feature_selection_pycaret.py
    python cases/alpha_d/run_feature_selection_pycaret.py pycaret.setup.n_features_to_select=8
    python cases/alpha_d/run_feature_selection_pycaret.py output.dir=../data/feature_analysis/pycaret/run_1
"""

import json
import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from cases.alpha_d.feature_data import ALLOWLIST, load_feature_matrix
from feature_selection.manifest import build_manifest, write_manifest
from feature_selection.pycaret_selection import run_pycaret_selection
from training.case_lists import load_case_selection

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="pycaret",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    data_cfg = cfg_dict["data"]
    output_cfg = cfg_dict["output"]
    pycaret_cfg = cfg_dict.get("pycaret") or {}

    exclude_cases = load_case_selection(
        data_cfg.get("exclude_cases"),
        data_cfg.get("exclude_cases_file"),
        label="exclude_cases",
    )
    data = load_feature_matrix(
        zarr_dir=data_cfg["zarr_dir"],
        target=data_cfg.get("target", "log_alpha_D"),
        selected_from_allowlist=data_cfg.get("selected_from_allowlist"),
        local_velocity_normalization=bool(data_cfg.get("local_velocity_normalization", True)),
        min_Dr=data_cfg.get("min_Dr"),
        exclude_cases=exclude_cases,
    )
    logger.info(
        "Loaded %d rows across %d cases; %d candidate features.",
        data.X.shape[0],
        data.n_cases,
        data.X.shape[1],
    )

    out_dir = Path(output_cfg["dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_pycaret_selection(
        data,
        pycaret_cfg=pycaret_cfg,
        output_dir=out_dir,
        allowlist=ALLOWLIST,
    )
    logger.info(
        "Selected %d features: %s",
        len(result["selected"]),
        result["selected"],
    )

    manifest = build_manifest(
        config=cfg_dict,
        zarr_dir=data_cfg["zarr_dir"],
        feature_names=data.feature_names,
        target_name=data.target_name,
        n_rows=int(data.X.shape[0]),
        n_cases=data.n_cases,
        seeds={"pycaret": int(pycaret_cfg.get("seed", 42))},
    )
    manifest["config"]["data"]["exclude_cases"] = exclude_cases
    try:
        import pycaret

        manifest["versions"]["pycaret"] = getattr(pycaret, "__version__", "unknown")
    except ImportError:
        pass
    write_manifest(manifest, out_dir)

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote PyCaret feature-selection artifacts to {out_dir}")


if __name__ == "__main__":
    main()
