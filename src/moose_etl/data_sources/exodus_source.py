"""ExodusDataSource: reads MOOSE Exodus (.e) simulation output files.

Subclasses physicsnemo_curator's DataSource ABC.

Each Exodus file represents one simulation run.  The reader:
  1. Extracts mesh geometry (node coordinates, element connectivity).
  2. Extracts element solution fields for every time step.
  3. Optionally co-reads matching CSV line-probe files via CSVProbeSource.

The returned dict is keyed so that MooseDataTransformation can consume it
directly without field-name guessing.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the src/ directory is on the path so we can import read_exdous helpers
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parents[2]  # src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Reuse decoding helpers from read_exdous.py
from read_exdous import ExodusReader

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig

from moose_etl.data_sources.csv_source import CSVProbeSource
from moose_etl.schemas import MooseRawData

logger = logging.getLogger(__name__)


class ExodusDataSource(DataSource):
    """Reads MOOSE Exodus files and co-reads matching CSV probe files.

    Args:
        cfg        : ProcessingConfig from the curator framework.
        input_dir  : Directory containing Exodus (.e) files.
        data_dir   : Directory containing CSV probe files.
                     If omitted, defaults to input_dir.
    """

    def __init__(
        self,
        cfg: ProcessingConfig,
        input_dir: str,
        data_dir: str | None = None,
    ):
        super().__init__(cfg)
        self.input_dir = Path(input_dir)
        self.data_dir = Path(data_dir) if data_dir else self.input_dir
        self._csv_source = CSVProbeSource(self.data_dir)
        self._exodus_reader = ExodusReader(use_rich=False)

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def get_file_list(self) -> list[str]:
        """Return sorted list of Exodus file paths."""
        files = sorted(self.input_dir.glob("**/*.e"))
        if not files:
            logger.warning("No Exodus (.e) files found in %s", self.input_dir)
        return [str(f) for f in files]

    def read_file(self, filename: str) -> dict[str, Any]:
        """Read one Exodus file and its associated CSV probes.

        Returns a dict that can be passed directly to MooseDataTransformation.
        """
        from netCDF4 import Dataset as NC4Dataset  # local import — not always installed

        path = Path(filename)
        sim_name = path.stem  # e.g. 'lid-driven-segregated_out'
        self.logger.info("Reading Exodus file: %s", path.name)

        ds = NC4Dataset(str(path), "r")
        try:
            raw = self._extract_exodus(ds, sim_name)
        finally:
            ds.close()

        # Co-read CSV probes
        probe_data, probe_columns = self._csv_source.read_all(sim_name)
        raw.probe_data = probe_data
        raw.probe_columns = probe_columns

        self.logger.info(
            "  nodes=%d  elements=%d  time_steps=%d  probes=%d",
            raw.coords.shape[0],
            raw.connectivity.shape[0],
            len(raw.time_steps),
            len(raw.probe_data),
        )

        # Return as plain dict for the curator pipeline
        return {
            "coords": raw.coords,
            "connectivity": raw.connectivity,
            "field_names": raw.field_names,
            "fields": raw.fields,
            "time_steps": raw.time_steps,
            "probe_data": raw.probe_data,
            "probe_columns": raw.probe_columns,
            "sim_name": raw.sim_name,
        }

    # ------------------------------------------------------------------
    # Sink stubs — this class is read-only
    # ------------------------------------------------------------------

    def _get_output_path(self, filename: str) -> Path:
        raise NotImplementedError("ExodusDataSource is read-only; use MooseZarrSink for writing.")

    def _write_impl_temp_file(self, data: dict[str, Any], output_path: Path) -> None:
        raise NotImplementedError("ExodusDataSource is read-only; use MooseZarrSink for writing.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_exodus(self, ds, sim_name: str) -> MooseRawData:
        """Extract geometry, fields, and time from an open netCDF4 Dataset."""

        # --- Coordinates ---
        coordx = np.array(ds.variables["coordx"][:], dtype=np.float32)
        coordy = np.array(ds.variables["coordy"][:], dtype=np.float32)
        if "coordz" in ds.variables:
            coordz = np.array(ds.variables["coordz"][:], dtype=np.float32)
            coords = np.stack([coordx, coordy, coordz], axis=1)
        else:
            coords = np.stack([coordx, coordy], axis=1)

        # --- Connectivity (0-indexed) ---
        # Exodus stores 1-indexed connectivity; subtract 1 for 0-indexed.
        connectivity_1 = np.array(ds.variables["connect1"][:], dtype=np.int32)
        connectivity = connectivity_1 - 1  # [E, K]

        # --- Time steps ---
        time_steps = np.array(ds.variables["time_whole"][:], dtype=np.float32)

        # --- Element variable names ---
        names_by_kind = self._exodus_reader.build_name_lookup(ds)
        elem_field_names: list[str] = names_by_kind.get("element", [])

        # --- Element variable arrays ---
        # Exodus names element vars: vals_elem_var{i}eb{block}
        # We collect all blocks but typically MOOSE writes one block (eb1).
        num_time = len(time_steps)
        num_elem = connectivity.shape[0]
        num_fields = len(elem_field_names)

        if num_fields == 0:
            logger.warning("No element variables found in %s", sim_name)
            fields = np.empty((num_time, num_elem, 0), dtype=np.float32)
        else:
            fields = np.zeros((num_time, num_elem, num_fields), dtype=np.float32)
            for fi, _ in enumerate(elem_field_names):
                # Try block 1 first, then unqualified variable name
                var_name = f"vals_elem_var{fi + 1}eb1"
                if var_name not in ds.variables:
                    var_name = f"vals_elem_var{fi + 1}"
                if var_name in ds.variables:
                    fields[:, :, fi] = np.array(
                        ds.variables[var_name][:], dtype=np.float32
                    )
                else:
                    logger.warning(
                        "Element variable index %d not found in %s", fi + 1, sim_name
                    )

        return MooseRawData(
            coords=coords,
            connectivity=connectivity,
            field_names=elem_field_names,
            fields=fields,
            time_steps=time_steps,
            probe_data={},       # filled by caller
            probe_columns=[],    # filled by caller
            sim_name=sim_name,
        )
