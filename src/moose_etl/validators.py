"""MooseDatasetValidator: validates processed Zarr dataset structure.

Checks that every .zarr store in the output directory has the expected
group hierarchy and required arrays before ML training begins.
"""

import logging
from pathlib import Path

import zarr

from physicsnemo_curator.etl.dataset_validators import (
    DatasetValidator,
    ValidationError,
    ValidationLevel,
)
from physicsnemo_curator.etl.processing_config import ProcessingConfig

logger = logging.getLogger(__name__)

# Required top-level groups in every Zarr store
_REQUIRED_GROUPS = ["mesh", "fields", "probes", "grid", "metadata"]

# Required arrays inside each group (probes group is dynamic — checked separately)
_REQUIRED_ARRAYS: dict[str, list[str]] = {
    "mesh": ["coords", "connectivity", "edge_src", "edge_dst"],
    "grid": ["x", "y"],
    "metadata": ["time_steps"],
}

# Required metadata attributes
_REQUIRED_META_ATTRS = ["field_names", "probe_columns", "sim_name"]


class MooseDatasetValidator(DatasetValidator):
    """Validates a directory of processed Zarr stores.

    Args:
        cfg        : ProcessingConfig.
        output_dir : Directory containing .zarr stores to validate.
    """

    def __init__(self, cfg: ProcessingConfig, output_dir: str):
        super().__init__(cfg)
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # DatasetValidator interface
    # ------------------------------------------------------------------

    def validate(self) -> list[ValidationError]:
        """Validate all .zarr stores in output_dir."""
        stores = sorted(self.output_dir.glob("*.zarr"))
        if not stores:
            logger.warning("No .zarr stores found in %s", self.output_dir)
            return []

        errors: list[ValidationError] = []
        for store_path in stores:
            errors.extend(self.validate_single_item(store_path))

        if errors:
            logger.error("%d validation error(s) found.", len(errors))
        else:
            logger.info("All %d Zarr store(s) validated successfully.", len(stores))

        return errors

    def validate_single_item(self, item: Path) -> list[ValidationError]:
        """Validate a single .zarr store.

        Checks:
          - Required groups exist.
          - Required arrays within each group exist.
          - Required metadata attributes are present.
          - fields/ group contains at least one dataset.
          - Shape consistency (coords, connectivity).
        """
        errors: list[ValidationError] = []

        try:
            root = zarr.open(str(item), mode="r")
        except Exception as exc:
            errors.append(
                ValidationError(
                    path=item,
                    message=f"Cannot open Zarr store: {exc}",
                    level=ValidationLevel.STRUCTURE,
                )
            )
            return errors

        # --- Group presence ---
        for group_name in _REQUIRED_GROUPS:
            if group_name not in root:
                errors.append(
                    ValidationError(
                        path=item,
                        message=f"Missing required group: '{group_name}'",
                        level=ValidationLevel.STRUCTURE,
                    )
                )

        if errors:
            return errors  # skip field checks if groups are missing

        # --- Array presence ---
        for group_name, required_arrays in _REQUIRED_ARRAYS.items():
            grp = root[group_name]
            for arr_name in required_arrays:
                if arr_name not in grp:
                    errors.append(
                        ValidationError(
                            path=item,
                            message=f"Missing array '{arr_name}' in group '{group_name}'",
                            level=ValidationLevel.FIELDS,
                        )
                    )

        # --- fields/ has at least one array ---
        if len(root["fields"]) == 0:
            errors.append(
                ValidationError(
                    path=item,
                    message="Group 'fields' is empty — no solution fields found.",
                    level=ValidationLevel.FIELDS,
                )
            )

        # --- Metadata attributes ---
        meta_attrs = dict(root["metadata"].attrs)
        for attr in _REQUIRED_META_ATTRS:
            if attr not in meta_attrs:
                errors.append(
                    ValidationError(
                        path=item,
                        message=f"Missing metadata attribute: '{attr}'",
                        level=ValidationLevel.FIELDS,
                    )
                )

        # --- Shape consistency ---
        if "mesh" in root and errors == []:
            mesh = root["mesh"]
            if "coords" in mesh and "connectivity" in mesh:
                n_nodes = mesh["coords"].shape[0]
                max_node_idx = int(mesh["connectivity"][:].max())
                if max_node_idx >= n_nodes:
                    errors.append(
                        ValidationError(
                            path=item,
                            message=(
                                f"Connectivity references node {max_node_idx} "
                                f"but coords has only {n_nodes} nodes."
                            ),
                            level=ValidationLevel.FIELDS,
                        )
                    )

        return errors
