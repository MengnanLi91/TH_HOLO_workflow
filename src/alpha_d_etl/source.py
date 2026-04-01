"""AlphaDSource: reads MOOSE simulation_out.e + case_metadata.txt.

Subclasses physicsnemo_curator DataSource.  Each file represents one
parametric-study case.  The source returns raw mesh + field arrays together
with the case parameters (Re, Dr, Lr, geometry dimensions) needed by the
downstream transformation.
"""

import csv
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from read_exdous import ExodusReader

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig

logger = logging.getLogger(__name__)


def _parse_case_metadata(meta_path: Path) -> dict[str, Any]:
    """Parse ``case_metadata.txt`` into a dict of typed values."""
    meta: dict[str, Any] = {}
    for line in meta_path.read_text().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        try:
            meta[key] = float(val)
        except ValueError:
            meta[key] = val
    return meta


def _load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Load ``cases_manifest.csv`` keyed by case_name."""
    manifest: dict[str, dict[str, Any]] = {}
    with manifest_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["case_name"]
            typed: dict[str, Any] = {}
            for k, v in row.items():
                try:
                    typed[k] = float(v)
                except (ValueError, TypeError):
                    typed[k] = v
            manifest[name] = typed
    return manifest


class AlphaDSource(DataSource):
    """Reads ``simulation_out.e`` files from the parametric study.

    Args:
        cfg       : ProcessingConfig.
        input_dir : Root of parametric study (contains case sub-directories).
        manifest  : Path to ``cases_manifest.csv``.
        mesh_scale: Scale factor applied to mesh coordinates (default 1.0).
        exodus_filename: Name of the output Exodus file inside each case dir.
    """

    def __init__(
        self,
        cfg: ProcessingConfig,
        input_dir: str,
        manifest: str | None = None,
        mesh_scale: float = 1.0,
        exodus_filename: str = "simulation_out.e",
    ):
        super().__init__(cfg)
        self.input_dir = Path(input_dir)
        self.mesh_scale = mesh_scale
        self.exodus_filename = exodus_filename
        self._reader = ExodusReader(use_rich=False)

        if manifest:
            self._manifest = _load_manifest(Path(manifest))
        else:
            manifest_path = self.input_dir / "cases_manifest.csv"
            if manifest_path.exists():
                self._manifest = _load_manifest(manifest_path)
            else:
                self._manifest = {}

    def get_file_list(self) -> list[str]:
        """Return paths to simulation_out.e files in case sub-directories."""
        files: list[str] = []
        for case_dir in sorted(self.input_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            exodus_path = case_dir / self.exodus_filename
            if exodus_path.exists():
                files.append(str(exodus_path))
        if not files:
            logger.warning("No '%s' files found under %s", self.exodus_filename, self.input_dir)
        return files

    def read_file(self, filename: str) -> dict[str, Any]:
        """Read one case's Exodus output and metadata."""
        from netCDF4 import Dataset as NC4Dataset

        path = Path(filename)
        case_dir = path.parent
        case_name = case_dir.name
        self.logger.info("Reading case: %s", case_name)

        # --- Case parameters ---
        meta_path = case_dir / "case_metadata.txt"
        if meta_path.exists():
            case_meta = _parse_case_metadata(meta_path)
        elif case_name in self._manifest:
            case_meta = dict(self._manifest[case_name])
        else:
            raise FileNotFoundError(
                f"No case_metadata.txt or manifest entry for case '{case_name}'."
            )

        # --- Exodus data ---
        ds = NC4Dataset(str(path), "r")
        try:
            coords, connectivity, field_names, fields, time_steps = self._extract(ds)
        finally:
            ds.close()

        # Apply mesh scaling to coordinates
        coords = coords * self.mesh_scale

        self.logger.info(
            "  nodes=%d  elements=%d  fields=%s  time_steps=%d",
            coords.shape[0],
            connectivity.shape[0],
            field_names,
            len(time_steps),
        )

        return {
            "case_name": case_name,
            "case_meta": case_meta,
            "coords": coords,
            "connectivity": connectivity,
            "field_names": field_names,
            "fields": fields,
            "time_steps": time_steps,
        }

    def _extract(self, ds):
        """Extract geometry and fields from an open netCDF4 Dataset."""
        coordx = np.array(ds.variables["coordx"][:], dtype=np.float64)
        coordy = np.array(ds.variables["coordy"][:], dtype=np.float64)
        coordz = np.array(ds.variables["coordz"][:], dtype=np.float64)
        coords = np.stack([coordx, coordy, coordz], axis=1)

        # Element connectivity (multi-block support)
        conn_blocks = []
        for var_name in sorted(ds.variables):
            if var_name.startswith("connect") and not var_name.endswith("names"):
                arr = np.array(ds.variables[var_name][:], dtype=np.int32) - 1
                conn_blocks.append(arr)
        if conn_blocks:
            connectivity = np.concatenate(conn_blocks, axis=0)
        else:
            connectivity = np.empty((0, 4), dtype=np.int32)

        time_steps = np.array(ds.variables["time_whole"][:], dtype=np.float64)

        names_by_kind = self._reader.build_name_lookup(ds)
        elem_field_names: list[str] = names_by_kind.get("element", [])

        num_time = len(time_steps)
        num_elem = connectivity.shape[0]
        num_fields = len(elem_field_names)

        fields = np.zeros((num_time, num_elem, num_fields), dtype=np.float64)
        for fi, _ in enumerate(elem_field_names):
            # Try all element blocks and concatenate
            block_arrays: list[np.ndarray] = []
            block_id = 1
            while True:
                var_name = f"vals_elem_var{fi + 1}eb{block_id}"
                if var_name not in ds.variables:
                    break
                block_arrays.append(np.array(ds.variables[var_name][:], dtype=np.float64))
                block_id += 1

            if block_arrays:
                fields[:, :, fi] = np.concatenate(block_arrays, axis=1)
            else:
                var_name = f"vals_elem_var{fi + 1}"
                if var_name in ds.variables:
                    fields[:, :, fi] = np.array(ds.variables[var_name][:], dtype=np.float64)

        return coords, connectivity, elem_field_names, fields, time_steps

    # --- Write stubs (read-only source) ---

    def _get_output_path(self, filename: str) -> Path:
        raise NotImplementedError("AlphaDSource is read-only.")

    def _write_impl_temp_file(self, data: dict[str, Any], output_path: Path) -> None:
        raise NotImplementedError("AlphaDSource is read-only.")
