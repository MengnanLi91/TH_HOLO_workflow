"""Export an alpha_D surrogate prediction as a MOOSE Forchheimer profile.

For a single target case, this script:
  1. Loads features from the case's existing ETL `.zarr` store.
  2. Loads the trained Conv1D checkpoint.
  3. Runs forward inference to get the per-station alpha_D profile.
  4. Maps alpha_D(z) to MOOSE PINSFV Forchheimer coefficient C_F(z) using
     the throat-only 1D plug-flow equivalence.
  5. Writes forchheimer_profile.csv (z, F) plus a sidecar metadata.json.

The CSV is consumed by MOOSE's PiecewiseLinear function backing the
ADGenericVectorFunctorMaterial on block 2 (the throat).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zarr


@dataclass
class Args:
    zarr: Path
    checkpoint: Path
    run_meta: Path
    output_csv: Path


def parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to the target case's .zarr store."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to the .mdlus checkpoint."
    )
    parser.add_argument(
        "--run-meta",
        type=Path,
        required=True,
        help="Path to the run_meta.json sibling of the checkpoint.",
    )
    parser.add_argument(
        "--output-csv", type=Path, required=True, help="Output CSV path (Forchheimer profile)."
    )
    ns = parser.parse_args(argv)
    return Args(
        zarr=ns.zarr,
        checkpoint=ns.checkpoint,
        run_meta=ns.run_meta,
        output_csv=ns.output_csv,
    )


@dataclass
class CaseData:
    case_id: str
    features: np.ndarray  # (50, n_feat)
    targets: np.ndarray  # (50, n_tgt)
    feature_names: list[str]
    target_names: list[str]
    Re: float
    Dr: float
    Lr: float
    D_big: float  # outer pipe diameter, m
    outer_height_m: float
    buffer_diams: float
    rho: float
    V_bulk: float
    delta_p_truth: float = field(default=0.0)


def load_case_from_zarr(zarr_path: Path) -> CaseData:
    """Read features, targets, and geometry metadata for one case."""
    if not zarr_path.exists():
        raise FileNotFoundError(f"Case zarr not found: {zarr_path}")

    root = zarr.open(store=str(zarr_path), mode="r")
    meta = root["metadata"]

    feature_names = list(json.loads(meta.attrs["feature_names"]))
    target_names = list(json.loads(meta.attrs["target_names"]))

    return CaseData(
        case_id=str(meta.attrs["case_id"]),
        features=np.asarray(root["features"][:], dtype=np.float32),
        targets=np.asarray(root["targets"][:], dtype=np.float32),
        feature_names=feature_names,
        target_names=target_names,
        Re=float(meta.attrs["Re"]),
        Dr=float(meta.attrs["Dr"]),
        Lr=float(meta.attrs["Lr"]),
        D_big=float(meta.attrs.get("D_big", 0.2)),
        outer_height_m=float(meta.attrs.get("outer_height_m", 1.0)),
        buffer_diams=float(meta.attrs.get("buffer_diams", 1.0)),
        rho=float(meta.attrs.get("rho", 1.0)),
        V_bulk=float(meta.attrs.get("V_bulk", 1.0)),
        delta_p_truth=float(meta.attrs.get("delta_p_case", 0.0)),
    )


def main(argv: list[str] | None = None) -> int:
    _args = parse_args(argv)
    raise NotImplementedError(
        "export_friction_profile pipeline not yet implemented; "
        "see plan docs/superpowers/plans/2026-05-28-alpha-d-moose-coupling.md."
    )


if __name__ == "__main__":
    raise SystemExit(main())
