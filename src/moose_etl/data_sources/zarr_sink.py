"""MooseZarrSink: writes processed MOOSE simulation data to Zarr format.

Subclasses physicsnemo_curator's DataSource ABC (write-only role).

Zarr store layout per simulation run
-------------------------------------
{sim_name}.zarr/
├── mesh/
│   ├── coords          float32 [N, D]   node coordinates
│   ├── connectivity    int32   [E, K]   element→node (0-indexed)
│   ├── edge_src        int32   [M]      graph edge source nodes
│   └── edge_dst        int32   [M]      graph edge destination nodes
├── fields/
│   ├── {field_name}    float32 [T, E]   normalized element solution field
│   └── ...
├── probes/
│   ├── {probe_name}    float32 [Np, C]  CSV line-probe values
│   └── ...
├── grid/
│   ├── x               float32 [Nx]     grid x-coordinates
│   ├── y               float32 [Ny]     grid y-coordinates
│   └── {field_name}    float32 [T,Nx,Ny] interpolated field on regular grid
└── metadata/
    ├── time_steps      float32 [T]
    ├── field_names     str attrs on /metadata
    ├── probe_columns   str attrs on /metadata
    └── norm_stats/{field_name}  attrs: mean, std
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


class MooseZarrSink(DataSource):
    """Writes processed MOOSE data to per-simulation Zarr stores.

    Args:
        cfg                : ProcessingConfig.
        output_dir         : Directory where .zarr stores will be written.
        overwrite_existing : If True (default) overwrite existing stores.
        compression_level  : Blosc compression level 1-9.
        compression_method : Blosc codec name (default 'zstd').
        chunk_size_mb      : Target chunk size in MB.
    """

    def __init__(
        self,
        cfg: ProcessingConfig,
        output_dir: str,
        overwrite_existing: bool = True,
        compression_level: int = 3,
        compression_method: str = "zstd",
        chunk_size_mb: float = 1.0,
    ):
        super().__init__(cfg)
        self.output_dir = Path(output_dir)
        self.overwrite_existing = overwrite_existing
        self.compression_level = compression_level
        self.compression_method = compression_method
        self.chunk_size_mb = chunk_size_mb

        self.compressor = zarr.codecs.BloscCodec(
            cname=compression_method,
            clevel=compression_level,
            shuffle=zarr.codecs.BloscShuffle.shuffle,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # DataSource interface — read side is not used
    # ------------------------------------------------------------------

    def get_file_list(self) -> list[str]:
        raise NotImplementedError("MooseZarrSink is write-only.")

    def read_file(self, filename: str) -> dict[str, Any]:
        raise NotImplementedError("MooseZarrSink is write-only.")

    def _get_output_path(self, filename: str) -> Path:
        """Map an Exodus filename to its output .zarr path."""
        stem = Path(filename).stem  # e.g. 'lid-driven-segregated_out'
        return self.output_dir / f"{stem}.zarr"

    def should_skip(self, filename: str) -> bool:
        if not self.overwrite_existing:
            return self._get_output_path(filename).exists()
        return False

    def cleanup_temp_files(self) -> None:
        """Remove orphaned *.zarr_temp directories from interrupted runs."""
        for temp_store in self.output_dir.glob("*.zarr_temp"):
            self.logger.warning("Removing orphaned temp Zarr store: %s", temp_store)
            import shutil
            shutil.rmtree(temp_store)

    # ------------------------------------------------------------------
    # Write implementation
    # ------------------------------------------------------------------

    def _write_impl_temp_file(self, data: dict[str, Any], output_path: Path) -> None:
        """Write processed data to a Zarr store at output_path (temporary location).

        The base class handles the atomic temp→final rename after this returns.
        """
        sim_name: str = data["sim_name"]
        self.logger.info("Writing Zarr store for '%s' → %s", sim_name, output_path)

        store = LocalStore(str(output_path))
        root = zarr.group(store=store)

        # --- mesh/ ---
        mesh = root.require_group("mesh")
        self._write_array(mesh, "coords", data["coords"])
        self._write_array(mesh, "connectivity", data["connectivity"])
        self._write_array(mesh, "edge_src", data["edge_src"])
        self._write_array(mesh, "edge_dst", data["edge_dst"])

        # --- fields/ ---
        fields_grp = root.require_group("fields")
        field_names: list[str] = data["field_names"]
        fields: np.ndarray = data["fields"]  # [T, E, F]
        for fi, name in enumerate(field_names):
            self._write_array(fields_grp, name, fields[:, :, fi])

        # --- probes/ ---
        probes_grp = root.require_group("probes")
        for probe_name, arr in data["probe_data"].items():
            self._write_array(probes_grp, probe_name, arr)

        # --- grid/ ---
        grid_grp = root.require_group("grid")
        self._write_array(grid_grp, "x", data["grid_x"])
        self._write_array(grid_grp, "y", data["grid_y"])
        grid_fields: np.ndarray = data["grid_fields"]  # [T, Nx, Ny, F]
        for fi, name in enumerate(field_names):
            self._write_array(grid_grp, name, grid_fields[:, :, :, fi])

        # --- metadata/ ---
        meta = root.require_group("metadata")
        self._write_array(meta, "time_steps", data["time_steps"])

        # Store lists and dicts as JSON attributes on the metadata group
        meta.attrs["field_names"] = json.dumps(field_names)
        meta.attrs["probe_columns"] = json.dumps(data["probe_columns"])
        meta.attrs["sim_name"] = sim_name

        # Normalization statistics
        norm_grp = meta.require_group("norm_stats")
        for field_name, stats in data["norm_stats"].items():
            field_norm = norm_grp.require_group(field_name)
            field_norm.attrs["mean"] = stats["mean"]
            field_norm.attrs["std"] = stats["std"]

        self.logger.info("  Zarr store written: %s", output_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calculate_chunks(self, arr: np.ndarray) -> tuple:
        """Calculate chunk sizes targeting self.chunk_size_mb."""
        target_bytes = int(self.chunk_size_mb * 1024 * 1024)
        item_size = arr.itemsize
        shape = arr.shape

        if len(shape) == 1:
            chunk = (min(shape[0], max(1, target_bytes // item_size)),)
        elif len(shape) == 2:
            # Chunk along first dim, keep second dim whole
            rows = min(shape[0], max(1, target_bytes // (item_size * shape[1])))
            chunk = (rows, shape[1])
        else:
            # For 3-D+, chunk the last two dims whole and first dim proportionally
            inner = 1
            for d in shape[1:]:
                inner *= d
            first = min(shape[0], max(1, target_bytes // (item_size * inner)))
            chunk = (first,) + shape[1:]

        return chunk

    def _write_array(self, group: zarr.Group, name: str, arr: np.ndarray) -> None:
        """Write a numpy array to a zarr group with compression and chunking."""
        chunks = self._calculate_chunks(arr)
        group.create_array(
            name,
            data=arr,
            chunks=chunks,
            compressors=self.compressor,
            overwrite=True,
        )
