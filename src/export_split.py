"""Export reusable train/test split files from run metadata."""

import argparse
import os
import sys

# Ensure src/ is importable when running from src/ as a script.
sys.path.insert(0, os.path.dirname(__file__))

from training.split_io import export_split_files_from_run_meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export train/test split files from a run_meta.json artifact."
    )
    parser.add_argument(
        "--run-meta",
        default="../data/models/run_meta.json",
        help="Path to the run_meta.json file (default: ../data/models/run_meta.json).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where train.txt and test.txt will be written.",
    )
    parser.add_argument(
        "--train-filename",
        default="train.txt",
        help="Filename to use for the exported training split list.",
    )
    parser.add_argument(
        "--test-filename",
        default="test.txt",
        help="Filename to use for the exported test split list.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    exported = export_split_files_from_run_meta(
        run_meta_path=args.run_meta,
        output_dir=args.output_dir,
        train_filename=args.train_filename,
        test_filename=args.test_filename,
    )
    print(f"Run metadata: {exported['run_meta']}")
    print(f"Train split:  {exported['train_file']}")
    print(f"Test split:   {exported['test_file']}")


if __name__ == "__main__":
    main()
