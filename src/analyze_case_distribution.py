"""CLI to preview training-dataset distribution before running HPO / training.

Usage (inside the container):
    # Inspect the raw zarr directory
    python analyze_case_distribution.py \\
        --zarr-dir ../data/flow_contraction_expansion/parametric_study/processed

    # Apply the same filters the training pipeline will use
    python analyze_case_distribution.py \\
        --zarr-dir ../data/flow_contraction_expansion/parametric_study/processed \\
        --min-Dr 0.333

    # Show the train / test split recorded in a previous run
    python analyze_case_distribution.py \\
        --run-meta ../data/models/case_pressure_drop/run_meta.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from case_pressure_drop.distribution import (
    AXES,
    load_sim_names_from_zarr,
    load_split_from_run_meta,
    print_distribution_rich,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Preview the distribution of cases by Dr, Re, Lr before training.",
    )
    parser.add_argument(
        "--zarr-dir",
        default=None,
        help=(
            "Directory of processed *.zarr case stores.  If omitted but "
            "--run-meta is given, the zarr_dir recorded in run_meta.json is used."
        ),
    )
    parser.add_argument(
        "--run-meta",
        default=None,
        help=(
            "Optional path to a run_meta.json; when provided, Train/Test columns "
            "are populated from its recorded split."
        ),
    )
    parser.add_argument(
        "--min-Dr",
        type=float,
        default=None,
        help="Exclude cases whose Dr is below this value (matches TabularPairDataset).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Case name to exclude (repeatable).",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        default=list(AXES),
        choices=list(AXES),
        help="Which parameter axes to report.  Default: Dr Re Lr.",
    )
    args = parser.parse_args(argv)

    train_sims: list[str] = []
    test_sims: list[str] = []
    zarr_dir: str | None = None

    if args.run_meta:
        train_sims, test_sims = load_split_from_run_meta(args.run_meta)
        # If the user did not pass --zarr-dir, try to read it from run_meta.
        if not args.zarr_dir:
            import json

            meta = json.loads(Path(args.run_meta).expanduser().resolve().read_text())
            zarr_dir = meta.get("data", {}).get("zarr_dir")

    if args.zarr_dir:
        zarr_dir = args.zarr_dir

    if zarr_dir:
        all_sims = load_sim_names_from_zarr(
            zarr_dir,
            exclude_cases=args.exclude,
            min_Dr=args.min_Dr,
        )
    elif train_sims or test_sims:
        all_sims = sorted(set(train_sims) | set(test_sims))
    else:
        parser.error("Provide --zarr-dir and/or --run-meta.")

    print_distribution_rich(
        all_sims=all_sims,
        train_sims=train_sims or None,
        test_sims=test_sims or None,
        zarr_dir=zarr_dir,
        axes=tuple(args.axes),
    )


if __name__ == "__main__":
    main()
