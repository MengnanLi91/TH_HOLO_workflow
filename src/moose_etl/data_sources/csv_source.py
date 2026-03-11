"""CSVProbeSource: reads MOOSE CSV line-probe output files.

CSV files produced by MOOSE VectorPostprocessors follow the naming pattern:
    {sim_prefix}_out_{probe_name}_{timestep:04d}.csv

All CSVs belonging to the same simulation run share the same {sim_prefix}.
Each file holds a column-per-field table (TKE, TKED, id, pressure,
vel_x, vel_y, x, y, z, ...) with one row per sample point along the probe.

This helper is called by ExodusDataSource.read_file() — it is not a
DataSource subclass because it does not manage its own file list.
"""

import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Filename pattern: {prefix}_out_{probe_name}_{timestep:04d}.csv
_PROBE_PATTERN = re.compile(
    r"^(?P<prefix>.+)_out_(?P<probe>.+?)_(?P<ts>\d+)\.csv$"
)


def find_probe_files(sim_prefix: str, data_dir: Path) -> dict[str, list[Path]]:
    """Find all CSV probe files that belong to a simulation run.

    Args:
        sim_prefix: Stem of the Exodus file (e.g. 'lid-driven-segregated_out'
                    stripped of the trailing '_out' is *not* needed — just
                    pass the full exodus stem without extension).
        data_dir:   Directory to search for CSV files.

    Returns:
        Mapping from probe name to sorted list of CSV file paths
        (one entry per time step).
    """
    probes: dict[str, list[Path]] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        m = _PROBE_PATTERN.match(csv_path.name)
        if m is None:
            continue
        # Match files whose prefix is a prefix of sim_prefix or vice-versa.
        # MOOSE names: exodus stem = "case_out", CSV prefix = "case_out"
        # Accept any CSV whose extracted prefix starts with the sim name root.
        file_prefix = m.group("prefix")
        # Simple heuristic: accept if either is a substring of the other.
        if sim_prefix not in file_prefix and file_prefix not in sim_prefix:
            # Try stripping trailing '_out' from either side
            sp_root = sim_prefix.replace("_out", "")
            fp_root = file_prefix.replace("_out", "")
            if sp_root not in fp_root and fp_root not in sp_root:
                continue
        probe_name = m.group("probe")
        probes.setdefault(probe_name, []).append(csv_path)

    # Sort each probe's file list by time step index
    for probe_name in probes:
        probes[probe_name].sort(
            key=lambda p: int(_PROBE_PATTERN.match(p.name).group("ts"))
        )

    return probes


class CSVProbeSource:
    """Reads and aggregates MOOSE CSV line-probe files for one simulation run."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def read_all(
        self, sim_prefix: str
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Read all probe CSVs for a simulation run.

        Returns:
            probe_data   : dict mapping probe_name → numpy array [Np, C]
                           where Np = number of sample points, C = columns.
                           When multiple time steps are found, data from
                           the *last* time step is used (steady-state typical).
            probe_columns: ordered list of column names (shared across probes).
        """
        probe_files = find_probe_files(sim_prefix, self.data_dir)

        if not probe_files:
            logger.warning(
                "No CSV probe files found for sim_prefix='%s' in %s",
                sim_prefix,
                self.data_dir,
            )
            return {}, []

        probe_data: dict[str, np.ndarray] = {}
        probe_columns: list[str] = []

        for probe_name, file_list in probe_files.items():
            # Use the last time step file (typically steady state)
            csv_path = file_list[-1]
            try:
                arr, columns = read_csv(csv_path)
                probe_data[probe_name] = arr
                if not probe_columns:
                    probe_columns = columns
            except Exception as exc:
                logger.error("Failed to read probe '%s' from %s: %s", probe_name, csv_path, exc)

        return probe_data, probe_columns


def read_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    """Read a MOOSE output CSV file into a numpy array.

    Returns:
        arr     : [Np, C] float32 array
        columns : list of column name strings
    """
    with open(path) as fh:
        header = fh.readline().strip()
    columns = [c.strip() for c in header.split(",")]

    # Skip header row; load remaining rows as float
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]  # single-row file

    return arr, columns
