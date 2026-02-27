"""MooseDataset: PyTorch Dataset over processed Zarr simulation stores.

Supports three representation modes for different PhysicsNeMo model families:

  "graph"       GNN / MeshGraphNet
  ─────────────────────────────────────────────────────────────────────────
  coords        float32 [N, D]      node spatial coordinates
  edge_index    int64   [2, M]      COO edge list (src / dst)
  node_fields   float32 [T, N, F]  per-node fields (interpolated from elements)
  elem_fields   float32 [T, E, F]  per-element fields (raw)
  probe_data    dict    probe_name → float32 [Np, C]

  "point_cloud" PointNet / Transformer
  ─────────────────────────────────────────────────────────────────────────
  coords        float32 [N, D]      node spatial coordinates
  node_fields   float32 [T, N, F]  per-node fields

  "grid"        CNN (U-Net, FNO)
  ─────────────────────────────────────────────────────────────────────────
  grid_x        float32 [Nx]        column x-coordinates
  grid_y        float32 [Ny]        row y-coordinates
  grid_fields   float32 [T, Nx, Ny, F]  fields on regular grid

All modes also include:
  field_names   list[str]           field name for each F index
  norm_stats    dict                field_name → {"mean": float, "std": float}
  sim_name      str                 unique simulation identifier
  time_steps    float32 [T]

If time_idx is given (≥ 0), only that time step is returned (T-dim removed).

Denormalization
───────────────
    dataset.denormalize("pressure", tensor)
returns a tensor in original physical units.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MooseDataset(Dataset):
    """Dataset over a directory of processed MOOSE Zarr stores.

    Args:
        zarr_dir  : Path to the directory containing *.zarr stores.
        mode      : One of "graph", "point_cloud", "grid".
        time_idx  : If ≥ 0, return only this time step (removes T dimension).
                    If -1 (default), return all time steps.
    """

    MODES = ("graph", "point_cloud", "grid")

    def __init__(
        self,
        zarr_dir: str | Path,
        mode: str = "graph",
        time_idx: int = -1,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")
        self.zarr_dir = Path(zarr_dir)
        self.mode = mode
        self.time_idx = time_idx

        self.sim_paths: list[Path] = sorted(self.zarr_dir.glob("*.zarr"))
        if not self.sim_paths:
            raise FileNotFoundError(f"No .zarr stores found in {self.zarr_dir}")

        logger.info(
            "MooseDataset: found %d simulation(s), mode='%s'",
            len(self.sim_paths),
            mode,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sim_paths)

    def __getitem__(self, idx: int) -> dict:
        import zarr

        store_path = self.sim_paths[idx]
        root = zarr.open(str(store_path), mode="r")

        # --- Common metadata ---
        meta = root["metadata"]
        field_names: list[str] = json.loads(meta.attrs["field_names"])
        probe_columns: list[str] = json.loads(meta.attrs["probe_columns"])
        sim_name: str = str(meta.attrs["sim_name"])
        time_steps = torch.from_numpy(np.array(meta["time_steps"], dtype=np.float32))

        norm_stats = _load_norm_stats(meta)

        sample: dict = {
            "field_names": field_names,
            "norm_stats": norm_stats,
            "sim_name": sim_name,
            "time_steps": time_steps,
        }

        if self.mode == "graph":
            sample.update(self._load_graph(root, field_names, probe_columns))
        elif self.mode == "point_cloud":
            sample.update(self._load_point_cloud(root, field_names))
        elif self.mode == "grid":
            sample.update(self._load_grid(root, field_names))

        return sample

    # ------------------------------------------------------------------
    # Mode-specific loaders
    # ------------------------------------------------------------------

    def _load_graph(
        self, root, field_names: list[str], probe_columns: list[str]
    ) -> dict:
        mesh = root["mesh"]
        coords = _to_tensor(mesh["coords"])                   # [N, D]
        connectivity = torch.from_numpy(np.array(mesh["connectivity"], dtype=np.int64))
        edge_src = torch.from_numpy(np.array(mesh["edge_src"], dtype=np.int64))
        edge_dst = torch.from_numpy(np.array(mesh["edge_dst"], dtype=np.int64))
        edge_index = torch.stack([edge_src, edge_dst], dim=0)  # [2, M]

        # Element fields [T, E, F]
        elem_fields = _load_fields(root["fields"], field_names)
        elem_fields = _slice_time(elem_fields, self.time_idx)

        # Interpolate element fields to nodes via simple centroid averaging
        node_fields = _elem_to_node(elem_fields, connectivity, coords.shape[0])

        # Probes
        probes_grp = root["probes"]
        probe_data = {
            name: _to_tensor(probes_grp[name])
            for name in probes_grp.array_keys()
        }

        return {
            "coords": coords,
            "edge_index": edge_index,
            "elem_fields": elem_fields,
            "node_fields": node_fields,
            "probe_data": probe_data,
        }

    def _load_point_cloud(self, root, field_names: list[str]) -> dict:
        mesh = root["mesh"]
        coords = _to_tensor(mesh["coords"])  # [N, D]
        connectivity = torch.from_numpy(np.array(mesh["connectivity"], dtype=np.int64))

        elem_fields = _load_fields(root["fields"], field_names)  # [T, E, F]
        elem_fields = _slice_time(elem_fields, self.time_idx)
        node_fields = _elem_to_node(elem_fields, connectivity, coords.shape[0])

        return {
            "coords": coords,
            "node_fields": node_fields,
        }

    def _load_grid(self, root, field_names: list[str]) -> dict:
        grid_grp = root["grid"]
        grid_x = _to_tensor(grid_grp["x"])  # [Nx]
        grid_y = _to_tensor(grid_grp["y"])  # [Ny]

        # Load each field and stack: [T, Nx, Ny, F]
        field_arrays = [
            _to_tensor(grid_grp[name]).unsqueeze(-1)  # [T, Nx, Ny, 1]
            for name in field_names
            if name in grid_grp
        ]
        grid_fields = torch.cat(field_arrays, dim=-1)  # [T, Nx, Ny, F]
        grid_fields = _slice_time(grid_fields, self.time_idx)

        return {
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_fields": grid_fields,
        }

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def denormalize(self, field_name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Reverse the z-score normalization for a single field.

        Args:
            field_name: Name of the field (must be in norm_stats).
            tensor    : Normalized tensor of any shape.

        Returns:
            Tensor in original physical units.
        """
        # Load stats from the first store (stats are per-simulation; use idx=0
        # here as a convention — override per-sample if needed).
        import zarr

        root = zarr.open(str(self.sim_paths[0]), mode="r")
        norm_stats = _load_norm_stats(root["metadata"])
        if field_name not in norm_stats:
            raise KeyError(f"Field '{field_name}' not found in norm_stats.")
        stats = norm_stats[field_name]
        mean = torch.tensor(stats["mean"], dtype=tensor.dtype, device=tensor.device)
        std = torch.tensor(stats["std"], dtype=tensor.dtype, device=tensor.device)
        return tensor * std + mean


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_tensor(arr) -> torch.Tensor:
    """Convert a zarr array to a float32 torch tensor."""
    return torch.from_numpy(np.array(arr, dtype=np.float32))


def _load_fields(fields_grp, field_names: list[str]) -> torch.Tensor:
    """Load and stack element fields from a zarr group → [T, E, F]."""
    arrays = [
        _to_tensor(fields_grp[name]).unsqueeze(-1)   # [T, E, 1]
        for name in field_names
        if name in fields_grp
    ]
    if not arrays:
        raise ValueError("No matching field arrays found in 'fields' group.")
    return torch.cat(arrays, dim=-1)  # [T, E, F]


def _slice_time(tensor: torch.Tensor, time_idx: int) -> torch.Tensor:
    """If time_idx >= 0, select that time step and remove the T dimension."""
    if time_idx >= 0:
        return tensor[time_idx]  # removes T dim
    return tensor


def _elem_to_node(
    elem_fields: torch.Tensor,
    connectivity: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Average element fields onto nodes (scatter mean over element→node map).

    elem_fields  : [..., E, F]  (leading dims may include T)
    connectivity : [E, K]       0-indexed node indices
    n_nodes      : N

    Returns node_fields : [..., N, F]
    """
    leading = elem_fields.shape[:-2]
    E, F = elem_fields.shape[-2], elem_fields.shape[-1]
    K = connectivity.shape[1]

    # Flatten leading dims for scatter
    flat_fields = elem_fields.reshape(-1, E, F)   # [B, E, F]
    B = flat_fields.shape[0]

    # node_fields accumulator
    node_acc = torch.zeros(B, n_nodes, F, dtype=elem_fields.dtype)
    node_cnt = torch.zeros(n_nodes, dtype=torch.float32)

    for k in range(K):
        node_idx = connectivity[:, k]  # [E]
        node_acc.scatter_add_(
            1,
            node_idx.unsqueeze(0).unsqueeze(-1).expand(B, E, F),
            flat_fields,
        )
        ones = torch.ones(E, dtype=torch.float32)
        node_cnt.scatter_add_(0, node_idx, ones)

    # Avoid division by zero (nodes not referenced by any element)
    node_cnt = node_cnt.clamp(min=1.0)
    node_fields = node_acc / node_cnt.unsqueeze(0).unsqueeze(-1)  # [B, N, F]

    return node_fields.reshape(*leading, n_nodes, F)


def _load_norm_stats(meta_grp) -> dict[str, dict[str, float]]:
    """Read per-field normalization stats from metadata/norm_stats/."""
    stats: dict[str, dict[str, float]] = {}
    if "norm_stats" not in meta_grp:
        return stats
    norm_grp = meta_grp["norm_stats"]
    for field_name in norm_grp.group_keys():
        field_grp = norm_grp[field_name]
        stats[field_name] = {
            "mean": float(field_grp.attrs["mean"]),
            "std": float(field_grp.attrs["std"]),
        }
    return stats
