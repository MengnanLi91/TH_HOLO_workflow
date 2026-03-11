"""Evaluate a trained PhysicsNeMo FNO checkpoint on ETL-produced Zarr data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure src/ is importable when running "python eval_fno.py" from src/.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fno_cli_config import parse_args_with_yaml
from train_fno import MooseGridPairDataset, _parse_field_list, _resolve_device


def _import_physicsnemo():
    """Import PhysicsNeMo components from installed package or vendored source."""
    try:
        from physicsnemo.core.module import Module
        from physicsnemo.metrics.general.mse import mse

        return Module, mse
    except ModuleNotFoundError:
        vendored_root = Path(__file__).resolve().parents[1] / "physicsnemo"
        if vendored_root.exists() and str(vendored_root) not in sys.path:
            sys.path.insert(0, str(vendored_root))
        try:
            from physicsnemo.core.module import Module
            from physicsnemo.metrics.general.mse import mse

            return Module, mse
        except ModuleNotFoundError as import_error:
            raise ModuleNotFoundError(
                "Could not import physicsnemo. Install `nvidia-physicsnemo` or "
                "run in a container image where PhysicsNeMo is available."
            ) from import_error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a PhysicsNeMo .mdlus checkpoint on Zarr grid data."
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
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained PhysicsNeMo .mdlus checkpoint.",
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
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Evaluation device.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional path to write metrics JSON.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parse_args_with_yaml(parser)

    if args.zarr_dir is None:
        raise ValueError("Missing zarr_dir. Set it in --config or pass --zarr-dir.")
    if args.checkpoint is None:
        raise ValueError(
            "Missing checkpoint path. Set it in --config or pass --checkpoint."
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

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
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    module_cls, mse_fn = _import_physicsnemo()
    model = module_cls.from_checkpoint(str(args.checkpoint)).to(device)
    model.eval()

    num_batches = 0
    total_mse = 0.0
    per_field_mse = torch.zeros(len(dataset.output_fields), device=device)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)

            pred = model(x)
            if pred.shape != y.shape:
                raise ValueError(
                    f"Prediction shape {tuple(pred.shape)} does not match "
                    f"target shape {tuple(y.shape)}."
                )

            total_mse += float(mse_fn(pred, y).detach().cpu())
            per_field_mse += torch.mean((pred - y) ** 2, dim=(0, 2, 3))
            num_batches += 1

    if num_batches == 0:
        raise RuntimeError("No evaluation batches were produced.")

    avg_mse = total_mse / num_batches
    avg_rmse = math.sqrt(avg_mse)
    per_field_mse = per_field_mse / num_batches
    per_field_rmse = torch.sqrt(per_field_mse)

    print(
        f"Evaluated {len(dataset)} case(s) from {args.zarr_dir} "
        f"with checkpoint {args.checkpoint}"
    )
    print(f"overall: mse={avg_mse:.6e}, rmse={avg_rmse:.6e}")
    for name, mse_value, rmse_value in zip(
        dataset.output_fields,
        per_field_mse.detach().cpu().tolist(),
        per_field_rmse.detach().cpu().tolist(),
    ):
        print(f"{name}: mse={mse_value:.6e}, rmse={rmse_value:.6e}")

    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "zarr_dir": str(args.zarr_dir),
            "checkpoint": str(args.checkpoint),
            "num_cases": len(dataset),
            "num_batches": num_batches,
            "overall": {"mse": avg_mse, "rmse": avg_rmse},
            "per_field": [
                {"name": name, "mse": mse_value, "rmse": rmse_value}
                for name, mse_value, rmse_value in zip(
                    dataset.output_fields,
                    per_field_mse.detach().cpu().tolist(),
                    per_field_rmse.detach().cpu().tolist(),
                )
            ],
        }
        args.metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved metrics JSON to {args.metrics_out}")


if __name__ == "__main__":
    main()
