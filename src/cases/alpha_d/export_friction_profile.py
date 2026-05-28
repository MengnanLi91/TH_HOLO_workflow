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
from dataclasses import dataclass
from pathlib import Path


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


def main(argv: list[str] | None = None) -> int:
    _args = parse_args(argv)
    raise NotImplementedError(
        "export_friction_profile pipeline not yet implemented; "
        "see plan docs/superpowers/plans/2026-05-28-alpha-d-moose-coupling.md."
    )


if __name__ == "__main__":
    raise SystemExit(main())
