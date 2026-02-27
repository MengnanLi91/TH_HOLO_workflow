"""MOOSE Simulation ETL pipeline entry point.

Reads Exodus + CSV simulation outputs, normalizes fields, builds graph
connectivity and regular-grid interpolations, and writes Zarr stores
ready for PhysicsNeMo ML training.

Usage (from the src/ directory):
    python run_etl.py \\
        etl.source.input_dir=../data \\
        etl.source.data_dir=../data \\
        etl.sink.output_dir=../data/processed

Override any Hydra config key on the CLI, e.g.:
    python run_etl.py etl.processing.num_processes=8 \\
                      etl.transformations.moose_transform.grid_nx=128
"""

import os
import sys

# Ensure src/ is importable so moose_etl and read_exdous are found.
sys.path.insert(0, os.path.dirname(__file__))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from physicsnemo_curator.etl.etl_orchestrator import ETLOrchestrator
from physicsnemo_curator.etl.processing_config import ProcessingConfig
from physicsnemo_curator.utils import utils as curator_utils


@hydra.main(version_base="1.3", config_path="moose_etl/config", config_name="moose_etl")
def main(cfg: DictConfig) -> None:
    """Run the MOOSE ETL pipeline."""

    # Configure multiprocessing (spawn for safety across platforms)
    curator_utils.setup_multiprocessing()

    # Build processing config
    processing_config = ProcessingConfig(**cfg.etl.processing)

    # Optional validator
    validator = None
    if "validator" in cfg.etl:
        validator = instantiate(cfg.etl.validator, processing_config)

    # Instantiate source (ExodusDataSource)
    source = instantiate(cfg.etl.source, processing_config)

    # Instantiate sink (MooseZarrSink)
    sink = instantiate(cfg.etl.sink, processing_config)

    # Instantiate transformations — curator pattern passes cfg as positional arg
    cfgs = {k: {"_args_": [processing_config]} for k in cfg.etl.transformations.keys()}
    transformations = instantiate(cfg.etl.transformations, **cfgs)

    # Run orchestrator
    orchestrator = ETLOrchestrator(
        source=source,
        sink=sink,
        transformations=transformations,
        processing_config=processing_config,
        validator=validator,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
