"""Alpha-D ETL pipeline entry point.

Reads MOOSE CFD simulation outputs from the parametric study, extracts
Darcy resistance coefficient profiles from the contraction region, and
writes per-case Zarr stores for MLP surrogate training.

Usage (from src/ directory):
    python run_alpha_d_etl.py \\
        etl.source.input_dir=../data/flow_contraction_expansion/parametric_study \\
        etl.sink.output_dir=../data/flow_contraction_expansion/parametric_study/processed
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from physicsnemo_curator.etl.etl_orchestrator import ETLOrchestrator
from physicsnemo_curator.etl.processing_config import ProcessingConfig
from physicsnemo_curator.utils import utils as curator_utils


@hydra.main(version_base="1.3", config_path="alpha_d_etl/config", config_name="alpha_d_etl")
def main(cfg: DictConfig) -> None:
    """Run the alpha-D extraction ETL pipeline."""
    curator_utils.setup_multiprocessing()

    processing_config = ProcessingConfig(**cfg.etl.processing)

    source = instantiate(cfg.etl.source, processing_config)
    sink = instantiate(cfg.etl.sink, processing_config)

    cfgs = {k: {"_args_": [processing_config]} for k in cfg.etl.transformations.keys()}
    transformations = instantiate(cfg.etl.transformations, **cfgs)

    orchestrator = ETLOrchestrator(
        source=source,
        sink=sink,
        transformations=transformations,
        processing_config=processing_config,
        validator=None,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
