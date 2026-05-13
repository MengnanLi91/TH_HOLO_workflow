"""MooseDataTransformation: normalize, build graph, interpolate to grid.

Implements the DataTransformation ABC from physicsnemo-curator.

Pipeline:
  1. Per-field mean/std normalization across all time steps and elements.
  2. Graph edge construction from element→node connectivity (all node pairs
     within each element, both directions).
  3. Bilinear interpolation of element-centroid values onto a regular Nx×Ny grid
     using scipy.interpolate.griddata.

Returns a dict containing all fields of MooseProcessedData, ready for
MooseZarrSink to write to disk.
"""

import itertools
import logging
from typing import Any, Optional

import numpy as np
from scipy.interpolate import griddata

from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig

from moose_etl.schemas import MooseProcessedData, NormStats

logger = logging.getLogger(__name__)


class MooseDataTransformation(DataTransformation):
    """Normalize, build graph, and interpolate MOOSE simulation data.

    Args:
        cfg      : ProcessingConfig from the curator framework.
        grid_nx  : Number of grid columns for the regular-grid output.
        grid_ny  : Number of grid rows for the regular-grid output.
        eps      : Small value added to std to avoid division by zero.
    """

    def __init__(
        self,
        cfg: ProcessingConfig,
        grid_nx: int = 64,
        grid_ny: int = 64,
        eps: float = 1e-8,
    ):
        super().__init__(cfg)
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.eps = eps

    # ------------------------------------------------------------------
    # DataTransformation interface
    # ------------------------------------------------------------------

    def transform(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Transform raw MOOSE data into ML-ready form.

        Args:
            data: dict produced by ExodusDataSource.read_file().

        Returns:
            dict of MooseProcessedData fields, or None to skip this sample.
        """
        coords: np.ndarray = data["coords"]           # [N, D]
        connectivity: np.ndarray = data["connectivity"]  # [E, K]
        field_names: list[str] = data["field_names"]
        fields: np.ndarray = data["fields"]           # [T, E, F]
        time_steps: np.ndarray = data["time_steps"]
        probe_data: dict = data["probe_data"]
        probe_columns: list[str] = data["probe_columns"]
        sim_name: str = data["sim_name"]

        if fields.size == 0:
            logger.warning("Skipping %s: no element fields found.", sim_name)
            return None

        # 1. Normalize fields
        norm_fields, norm_stats = self._normalize(fields, field_names)

        # 2. Build undirected graph edges from element connectivity
        edge_src, edge_dst = self._build_edges(connectivity)

        # 3. Interpolate to regular grid
        grid_fields, grid_x, grid_y = self._interpolate_to_grid(
            coords, connectivity, norm_fields
        )

        processed = MooseProcessedData(
            coords=coords,
            connectivity=connectivity,
            edge_src=edge_src,
            edge_dst=edge_dst,
            fields=norm_fields,
            field_names=field_names,
            norm_stats=norm_stats,
            probe_data=probe_data,
            probe_columns=probe_columns,
            grid_fields=grid_fields,
            grid_x=grid_x,
            grid_y=grid_y,
            time_steps=time_steps,
            sim_name=sim_name,
        )

        # Return as plain dict for the curator pipeline / zarr sink
        return {
            "coords": processed.coords,
            "connectivity": processed.connectivity,
            "edge_src": processed.edge_src,
            "edge_dst": processed.edge_dst,
            "fields": processed.fields,
            "field_names": processed.field_names,
            "norm_stats": {
                name: {"mean": float(s.mean), "std": float(s.std)}
                for name, s in processed.norm_stats.items()
            },
            "probe_data": processed.probe_data,
            "probe_columns": processed.probe_columns,
            "grid_fields": processed.grid_fields,
            "grid_x": processed.grid_x,
            "grid_y": processed.grid_y,
            "time_steps": processed.time_steps,
            "sim_name": processed.sim_name,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(
        self, fields: np.ndarray, field_names: list[str]
    ) -> tuple[np.ndarray, dict[str, NormStats]]:
        """Z-score normalize each field independently.

        fields: [T, E, F]
        Returns normalized array (same shape) and per-field stats.
        """
        norm_fields = fields.copy()
        norm_stats: dict[str, NormStats] = {}

        for fi, name in enumerate(field_names):
            vals = fields[:, :, fi]  # [T, E]
            mean = float(vals.mean())
            std = float(vals.std())
            norm_fields[:, :, fi] = (vals - mean) / (std + self.eps)
            norm_stats[name] = NormStats(mean=mean, std=std)

        return norm_fields, norm_stats

    def _build_edges(
        self, connectivity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build undirected graph edges from element-to-node connectivity.

        For each element, connect all pairs of its nodes in both directions.
        Duplicate edges (shared faces/edges between elements) are removed.

        connectivity: [E, K] 0-indexed node indices per element.
        Returns edge_src, edge_dst each of shape [M].
        """
        edge_set: set[tuple[int, int]] = set()
        num_elem, nodes_per_elem = connectivity.shape

        for e in range(num_elem):
            nodes = connectivity[e]  # [K]
            for i, j in itertools.combinations(range(nodes_per_elem), 2):
                n_i, n_j = int(nodes[i]), int(nodes[j])
                edge_set.add((n_i, n_j))
                edge_set.add((n_j, n_i))  # undirected → both directions

        if not edge_set:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        edges = np.array(sorted(edge_set), dtype=np.int32)  # [M, 2]
        edge_src = edges[:, 0]
        edge_dst = edges[:, 1]
        return edge_src, edge_dst

    def _interpolate_to_grid(
        self,
        coords: np.ndarray,
        connectivity: np.ndarray,
        fields: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate element-centroid values to a regular Nx×Ny grid.

        Element centroids are computed as the mean position of their nodes.
        scipy.interpolate.griddata (linear method) fills the grid.

        coords      : [N, D]     node coordinates (only first two dims used)
        connectivity: [E, K]     element→node connectivity (0-indexed)
        fields      : [T, E, F]  normalized element values

        Returns:
            grid_fields : [T, Nx, Ny, F]
            grid_x      : [Nx]   column x-coordinates
            grid_y      : [Ny]   row y-coordinates
        """
        # Element centroids: mean of node positions [E, 2]
        elem_xy = coords[:, :2][connectivity].mean(axis=1)  # [E, 2]

        x_min, y_min = elem_xy[:, 0].min(), elem_xy[:, 1].min()
        x_max, y_max = elem_xy[:, 0].max(), elem_xy[:, 1].max()

        grid_x = np.linspace(x_min, x_max, self.grid_nx, dtype=np.float32)
        grid_y = np.linspace(y_min, y_max, self.grid_ny, dtype=np.float32)
        gx, gy = np.meshgrid(grid_x, grid_y, indexing="ij")  # [Nx, Ny]

        num_time = fields.shape[0]
        num_fields = fields.shape[2]
        grid_fields = np.zeros(
            (num_time, self.grid_nx, self.grid_ny, num_fields), dtype=np.float32
        )

        for t in range(num_time):
            for fi in range(num_fields):
                values = fields[t, :, fi]  # [E]
                grid_fields[t, :, :, fi] = griddata(
                    points=elem_xy,
                    values=values,
                    xi=(gx, gy),
                    method="linear",
                    fill_value=0.0,
                ).astype(np.float32)

        return grid_fields, grid_x, grid_y
