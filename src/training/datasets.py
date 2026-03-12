"""Dataset wrappers and split helpers for training workflows."""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from dataset.moose_dataset import MooseDataset
from training import _require_pyg


def parse_field_list(raw: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    """Parse field selections from CLI-style strings or list values."""
    if raw is None:
        return None

    if isinstance(raw, str):
        names = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        names = [str(item).strip() for item in raw if str(item).strip()]

    if not names:
        raise ValueError("Field list is empty. Provide at least one field name.")
    return names


def resolve_time_idx(time_idx: int, num_steps: int, label: str) -> int:
    """Resolve negative time indexing and validate the index."""
    resolved = time_idx if time_idx >= 0 else num_steps + time_idx
    if resolved < 0 or resolved >= num_steps:
        raise ValueError(
            f"{label}={time_idx} is out of range for {num_steps} time step(s)."
        )
    return resolved


class GridPairDataset(Dataset):
    """Build supervised grid pairs `(x, y)` from MooseDataset grid samples."""

    def __init__(
        self,
        zarr_dir: str | Path,
        input_fields: list[str] | None,
        output_fields: list[str] | None,
        input_time_idx: int,
        target_time_idx: int,
    ):
        self.zarr_dir = Path(zarr_dir)
        self.base = MooseDataset(zarr_dir=self.zarr_dir, mode="grid", time_idx=-1)
        reference = self.base[0]

        self.field_names = list(reference["field_names"])
        self.field_to_index = {name: idx for idx, name in enumerate(self.field_names)}
        self.sim_names = [path.stem for path in self.base.sim_paths]

        self.input_fields = input_fields or list(self.field_names)
        self.output_fields = output_fields or list(self.input_fields)

        missing_inputs = [
            name for name in self.input_fields if name not in self.field_to_index
        ]
        if missing_inputs:
            raise ValueError(f"Unknown input field(s): {missing_inputs}")

        missing_outputs = [
            name for name in self.output_fields if name not in self.field_to_index
        ]
        if missing_outputs:
            raise ValueError(f"Unknown output field(s): {missing_outputs}")

        grid = reference["grid_fields"]
        if grid.ndim != 4:
            raise ValueError(
                "Expected grid_fields with shape [T, Nx, Ny, F], "
                f"got {tuple(grid.shape)}"
            )

        self.num_time_steps = int(grid.shape[0])
        self.spatial_shape = (int(grid.shape[1]), int(grid.shape[2]))
        self.input_time_idx = resolve_time_idx(
            input_time_idx, self.num_time_steps, "input_time_idx"
        )
        self.target_time_idx = resolve_time_idx(
            target_time_idx, self.num_time_steps, "target_time_idx"
        )

        self.input_indices = [self.field_to_index[name] for name in self.input_fields]
        self.output_indices = [self.field_to_index[name] for name in self.output_fields]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.base[idx]
        if list(sample["field_names"]) != self.field_names:
            raise ValueError(
                f"Field names differ across simulations. Expected {self.field_names}, "
                f"got {sample['field_names']} in {sample['sim_name']}."
            )

        grid = sample["grid_fields"]
        if grid.ndim != 4:
            raise ValueError(
                "Expected grid_fields with shape [T, Nx, Ny, F], "
                f"got {tuple(grid.shape)}"
            )

        spatial_shape = (int(grid.shape[1]), int(grid.shape[2]))
        if spatial_shape != self.spatial_shape:
            raise ValueError(
                f"Grid size mismatch for {sample['sim_name']}: expected "
                f"{self.spatial_shape}, got {spatial_shape}."
            )

        x = self._select_channels(grid, self.input_time_idx, self.input_indices)
        y = self._select_channels(grid, self.target_time_idx, self.output_indices)
        return x, y

    @staticmethod
    def _select_channels(
        grid: torch.Tensor, time_idx: int, field_indices: list[int]
    ) -> torch.Tensor:
        tensor = grid[time_idx, :, :, field_indices]
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        return tensor.permute(2, 0, 1).contiguous()


class GraphPairDataset(Dataset):
    """Build supervised PyG Data objects from MooseDataset graph samples."""

    def __init__(
        self,
        zarr_dir: str | Path,
        input_fields: list[str] | None,
        output_fields: list[str] | None,
        input_time_idx: int,
        target_time_idx: int,
    ):
        _require_pyg()
        from torch_geometric.data import Data

        self._pyg_data_cls = Data
        self.zarr_dir = Path(zarr_dir)
        self.base = MooseDataset(zarr_dir=self.zarr_dir, mode="graph", time_idx=-1)
        reference = self.base[0]

        self.field_names = list(reference["field_names"])
        self.field_to_index = {name: idx for idx, name in enumerate(self.field_names)}
        self.sim_names = [path.stem for path in self.base.sim_paths]

        self.input_fields = input_fields or list(self.field_names)
        self.output_fields = output_fields or list(self.input_fields)

        missing_inputs = [
            name for name in self.input_fields if name not in self.field_to_index
        ]
        if missing_inputs:
            raise ValueError(f"Unknown input field(s): {missing_inputs}")

        missing_outputs = [
            name for name in self.output_fields if name not in self.field_to_index
        ]
        if missing_outputs:
            raise ValueError(f"Unknown output field(s): {missing_outputs}")

        node_fields = reference["node_fields"]
        if node_fields.ndim != 3:
            raise ValueError(
                "Expected node_fields with shape [T, N, F], "
                f"got {tuple(node_fields.shape)}"
            )

        coords = reference["coords"]
        if coords.ndim != 2:
            raise ValueError(
                f"Expected coords with shape [N, D], got {tuple(coords.shape)}"
            )

        self.num_time_steps = int(node_fields.shape[0])
        self.coord_dim = int(coords.shape[1])
        self.edge_dim = self.coord_dim + 1

        self.input_time_idx = resolve_time_idx(
            input_time_idx, self.num_time_steps, "input_time_idx"
        )
        self.target_time_idx = resolve_time_idx(
            target_time_idx, self.num_time_steps, "target_time_idx"
        )

        self.input_indices = [self.field_to_index[name] for name in self.input_fields]
        self.output_indices = [self.field_to_index[name] for name in self.output_fields]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        if list(sample["field_names"]) != self.field_names:
            raise ValueError(
                f"Field names differ across simulations. Expected {self.field_names}, "
                f"got {sample['field_names']} in {sample['sim_name']}."
            )

        node_fields = sample["node_fields"]
        if node_fields.ndim != 3:
            raise ValueError(
                "Expected node_fields with shape [T, N, F], "
                f"got {tuple(node_fields.shape)}"
            )

        coords = sample["coords"].float().contiguous()
        edge_index = sample["edge_index"].long().contiguous()

        x = self._select_channels(node_fields, self.input_time_idx, self.input_indices)
        y = self._select_channels(node_fields, self.target_time_idx, self.output_indices)
        edge_attr = self._build_edge_attr(coords, edge_index)

        data = self._pyg_data_cls(
            x=x.float().contiguous(),
            y=y.float().contiguous(),
            edge_index=edge_index,
            edge_attr=edge_attr.float().contiguous(),
            pos=coords,
        )
        data.num_nodes = int(coords.shape[0])
        return data

    @staticmethod
    def _select_channels(
        node_fields: torch.Tensor, time_idx: int, field_indices: list[int]
    ) -> torch.Tensor:
        tensor = node_fields[time_idx, :, field_indices]
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        return tensor.contiguous()

    @staticmethod
    def _build_edge_attr(coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        displacement = coords[dst] - coords[src]
        distance = torch.linalg.norm(displacement, dim=1, keepdim=True)
        return torch.cat([displacement, distance], dim=1)


def _read_sim_name_list(path: str | Path) -> list[str]:
    entries: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.append(stripped.removesuffix(".zarr"))

    if not entries:
        raise ValueError(f"Split file '{path}' contains no simulation names.")
    return entries


def split_indices(
    num_cases: int,
    split_cfg: dict,
    sim_names: list[str],
) -> tuple[list[int], list[int], list[str], list[str]]:
    """Return split indices and simulation-name lists."""
    if num_cases != len(sim_names):
        raise ValueError(
            f"sim_names length {len(sim_names)} does not match num_cases {num_cases}."
        )
    if num_cases < 2:
        raise ValueError(
            f"Need at least 2 cases to split train/test, but found {num_cases}."
        )

    strategy = str(split_cfg.get("strategy", "sequential"))

    if strategy in {"sequential", "random"}:
        train_ratio = float(split_cfg.get("train_ratio", 0.8))
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

        n_train = int(num_cases * train_ratio)
        n_train = max(1, min(num_cases - 1, n_train))

        indices = list(range(num_cases))
        if strategy == "random":
            seed = int(split_cfg.get("seed", 42))
            rng = random.Random(seed)
            rng.shuffle(indices)
            train_idx = sorted(indices[:n_train])
            test_idx = sorted(indices[n_train:])
        else:
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]

    elif strategy == "file":
        train_file = split_cfg.get("train_file")
        test_file = split_cfg.get("test_file")
        if not train_file or not test_file:
            raise ValueError(
                "split.strategy='file' requires both split.train_file and split.test_file"
            )

        train_names = _read_sim_name_list(train_file)
        test_names = _read_sim_name_list(test_file)

        sim_to_idx = {name: idx for idx, name in enumerate(sim_names)}

        unknown_train = [name for name in train_names if name not in sim_to_idx]
        unknown_test = [name for name in test_names if name not in sim_to_idx]
        if unknown_train or unknown_test:
            raise ValueError(
                "Split files contain unknown simulation names. "
                f"unknown_train={unknown_train}, unknown_test={unknown_test}"
            )

        overlap = sorted(set(train_names).intersection(test_names))
        if overlap:
            raise ValueError(
                f"Split files overlap on simulation name(s): {overlap}."
            )

        train_idx = sorted({sim_to_idx[name] for name in train_names})
        test_idx = sorted({sim_to_idx[name] for name in test_names})

        if not train_idx or not test_idx:
            raise ValueError("Both train and test split files must contain at least one case.")

    else:
        raise ValueError(
            "split.strategy must be one of {'sequential', 'random', 'file'}, "
            f"got '{strategy}'."
        )

    train_sims = [sim_names[idx] for idx in train_idx]
    test_sims = [sim_names[idx] for idx in test_idx]
    return train_idx, test_idx, train_sims, test_sims
