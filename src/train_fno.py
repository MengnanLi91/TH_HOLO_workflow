"""Train a PhysicsNeMo FNO model on ETL-produced MOOSE Zarr data.

This script consumes `MooseDataset(mode="grid")` items and builds supervised
input/target tensor pairs for FNO:
  x: [Cin, Nx, Ny]
  y: [Cout, Nx, Ny]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None

# Ensure src/ is importable when running "python train_fno.py" from src/.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset.moose_dataset import MooseDataset
from fno_cli_config import parse_args_with_yaml


def _import_physicsnemo():
    """Import PhysicsNeMo components from installed package or vendored source."""
    try:
        from physicsnemo.metrics.general.mse import mse
        from physicsnemo.models.fno.fno import FNO

        return FNO, mse
    except ModuleNotFoundError:
        vendored_root = Path(__file__).resolve().parents[1] / "physicsnemo"
        if vendored_root.exists() and str(vendored_root) not in sys.path:
            sys.path.insert(0, str(vendored_root))
        try:
            from physicsnemo.metrics.general.mse import mse
            from physicsnemo.models.fno.fno import FNO

            return FNO, mse
        except ModuleNotFoundError as import_error:
            raise ModuleNotFoundError(
                "Could not import physicsnemo. Install `nvidia-physicsnemo` or "
                "run in a container image where PhysicsNeMo is available."
            ) from import_error


def _parse_field_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("Field list is empty. Provide comma-separated names.")
    return names


def _resolve_time_idx(time_idx: int, num_steps: int, label: str) -> int:
    resolved = time_idx if time_idx >= 0 else num_steps + time_idx
    if resolved < 0 or resolved >= num_steps:
        raise ValueError(
            f"{label}={time_idx} is out of range for {num_steps} time step(s)."
        )
    return resolved


class MooseGridPairDataset(Dataset):
    """Create supervised (x, y) pairs from MooseDataset grid tensors."""

    def __init__(
        self,
        zarr_dir: Path,
        input_fields: list[str] | None,
        output_fields: list[str] | None,
        input_time_idx: int,
        target_time_idx: int,
    ):
        self.base = MooseDataset(zarr_dir=zarr_dir, mode="grid", time_idx=-1)
        reference = self.base[0]

        self.field_names = list(reference["field_names"])
        self.field_to_index = {name: idx for idx, name in enumerate(self.field_names)}

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

        grid = reference["grid_fields"]  # [T, Nx, Ny, F]
        if grid.ndim != 4:
            raise ValueError(
                "Expected grid_fields with shape [T, Nx, Ny, F], "
                f"got {tuple(grid.shape)}"
            )

        self.num_time_steps = int(grid.shape[0])
        self.spatial_shape = (int(grid.shape[1]), int(grid.shape[2]))
        self.input_time_idx = _resolve_time_idx(
            input_time_idx, self.num_time_steps, "input_time_idx"
        )
        self.target_time_idx = _resolve_time_idx(
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

        grid = sample["grid_fields"]  # [T, Nx, Ny, F]
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
        # Slice [Nx, Ny, C] then move channels first -> [C, Nx, Ny].
        tensor = grid[time_idx, :, :, field_indices]
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        return tensor.permute(2, 0, 1).contiguous()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but no CUDA device is available.")
    return torch.device(device_arg)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a PhysicsNeMo FNO model from ETL-generated Zarr stores."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file. CLI args override YAML values.",
    )
    parser.add_argument(
        "--zarr-dir",
        type=Path,
        default=None,
        help="Directory containing ETL output *.zarr stores.",
    )
    parser.add_argument(
        "--input-fields",
        type=str,
        default=None,
        help="Comma-separated field names for model inputs (default: all fields).",
    )
    parser.add_argument(
        "--output-fields",
        type=str,
        default=None,
        help="Comma-separated field names for model targets (default: same as input).",
    )
    parser.add_argument(
        "--input-time-idx",
        type=int,
        default=0,
        help="Time index used for model input (supports negative indexing).",
    )
    parser.add_argument(
        "--target-time-idx",
        type=int,
        default=-1,
        help="Time index used for model target (supports negative indexing).",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device.",
    )

    parser.add_argument(
        "--latent-channels", type=int, default=32, help="FNO latent channel width."
    )
    parser.add_argument(
        "--num-fno-layers", type=int, default=4, help="Number of FNO spectral layers."
    )
    parser.add_argument(
        "--num-fno-modes", type=int, default=12, help="Fourier modes per dimension."
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Spatial padding used by spectral layers.",
    )
    parser.add_argument(
        "--decoder-layers", type=int, default=1, help="Number of decoder MLP layers."
    )
    parser.add_argument(
        "--decoder-layer-size",
        type=int,
        default=32,
        help="Hidden size for decoder MLP layers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../data/models/fno_model.mdlus"),
        help="Output path for the saved .mdlus model.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parse_args_with_yaml(parser)

    if args.zarr_dir is None:
        raise ValueError("Missing zarr_dir. Set it in --config or pass --zarr-dir.")

    _set_seed(args.seed)
    device = _resolve_device(args.device)
    input_fields = _parse_field_list(args.input_fields)
    output_fields = _parse_field_list(args.output_fields)

    dataset = MooseGridPairDataset(
        zarr_dir=args.zarr_dir,
        input_fields=input_fields,
        output_fields=output_fields,
        input_time_idx=args.input_time_idx,
        target_time_idx=args.target_time_idx,
    )

    pin_memory = device.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    fno_cls, mse_fn = _import_physicsnemo()
    model = fno_cls(
        in_channels=len(dataset.input_indices),
        out_channels=len(dataset.output_indices),
        dimension=2,
        latent_channels=args.latent_channels,
        num_fno_layers=args.num_fno_layers,
        num_fno_modes=args.num_fno_modes,
        padding=args.padding,
        decoder_layers=args.decoder_layers,
        decoder_layer_size=args.decoder_layer_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"Training FNO on {len(dataset)} case(s) "
        f"with input fields={dataset.input_fields} "
        f"(t={dataset.input_time_idx}) and output fields={dataset.output_fields} "
        f"(t={dataset.target_time_idx}) on device={device}."
    )

    epoch_iter = range(1, args.epochs + 1)
    epoch_progress = None
    if tqdm is not None:
        epoch_progress = tqdm(
            epoch_iter,
            total=args.epochs,
            desc="training",
            unit="epoch",
        )
        epoch_iter = epoch_progress

    for epoch in epoch_iter:
        model.train()
        running_loss = 0.0
        for x, y in dataloader:
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)

            pred = model(x)
            loss = mse_fn(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        avg_loss = running_loss / len(dataloader)
        if epoch_progress is not None:
            epoch_progress.set_postfix(loss=f"{avg_loss:.3e}")
        else:
            print(f"epoch {epoch}/{args.epochs}: loss={avg_loss:.6e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
