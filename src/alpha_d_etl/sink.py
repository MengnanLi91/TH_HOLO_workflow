"""AlphaDZarrSink: writes per-case alpha_D profiles to Zarr stores.

Zarr store layout per case:

    {case_name}.zarr/
        features    float32 [N_stations, D_in]
        targets     float32 [N_stations, D_out]
        metadata/
            attrs: case_id, feature_names, target_names,
                   Re, Dr, Lr, delta_p_case
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.storage import LocalStore

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig

logger = logging.getLogger(__name__)


class AlphaDZarrSink(DataSource):
    """Writes alpha_D profile data to per-case Zarr stores.

    Args:
        cfg                : ProcessingConfig.
        output_dir         : Directory where .zarr stores will be written.
        overwrite_existing : Overwrite existing stores (default True).
    """

    def __init__(
        self,
        cfg: ProcessingConfig,
        output_dir: str,
        overwrite_existing: bool = True,
    ):
        super().__init__(cfg)
        self.output_dir = Path(output_dir)
        self.overwrite_existing = overwrite_existing
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Read stubs (write-only) ---

    def get_file_list(self) -> list[str]:
        raise NotImplementedError("AlphaDZarrSink is write-only.")

    def read_file(self, filename: str) -> dict[str, Any]:
        raise NotImplementedError("AlphaDZarrSink is write-only.")

    def _get_output_path(self, filename: str) -> Path:
        case_dir = Path(filename).parent
        case_name = case_dir.name
        return self.output_dir / f"{case_name}.zarr"

    def should_skip(self, filename: str) -> bool:
        if not self.overwrite_existing:
            return self._get_output_path(filename).exists()
        return False

    def cleanup_temp_files(self) -> None:
        for temp_store in self.output_dir.glob("*.zarr_temp"):
            self.logger.warning("Removing orphaned temp Zarr store: %s", temp_store)
            import shutil
            shutil.rmtree(temp_store)

    # --- Write ---

    def _write_impl_temp_file(self, data: dict[str, Any], output_path: Path) -> None:
        case_name: str = data["case_name"]
        self.logger.info("Writing alpha_D Zarr for '%s' → %s", case_name, output_path)

        store = LocalStore(str(output_path))
        root = zarr.group(store=store)

        root.create_array("features", data=data["features"], overwrite=True)
        root.create_array("targets", data=data["targets"], overwrite=True)

        if "sample_weight" in data:
            root.create_array("sample_weight", data=data["sample_weight"], overwrite=True)

        meta = root.require_group("metadata")
        meta.attrs["case_id"] = case_name
        meta.attrs["feature_names"] = json.dumps(data["feature_names"])
        meta.attrs["target_names"] = json.dumps(data["target_names"])
        meta.attrs["Re"] = data.get("Re", 0.0)
        meta.attrs["Dr"] = data.get("Dr", 0.0)
        meta.attrs["Lr"] = data.get("Lr", 0.0)
        meta.attrs["delta_p_case"] = data.get("delta_p_case", 0.0)

        self.logger.info("  Wrote %d stations to %s", data["features"].shape[0], output_path)
