"""Discoverability wrapper for the moose-grid ETL.

Equivalent to ``python run_etl.py`` at the src/ root (whose default
``config_name`` is also ``etl``). Override with ``--config-name`` to
switch to another variant under ``configs/``.
"""

import os
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from physicsnemo_curator.etl.etl_orchestrator import ETLOrchestrator
from physicsnemo_curator.etl.processing_config import ProcessingConfig
from physicsnemo_curator.utils import utils as curator_utils


@hydra.main(version_base="1.3", config_path="configs", config_name="etl")
def main(cfg: DictConfig) -> None:
    curator_utils.setup_multiprocessing()

    processing_config = ProcessingConfig(**cfg.etl.processing)

    validator = None
    if "validator" in cfg.etl:
        validator = instantiate(cfg.etl.validator, processing_config)

    source = instantiate(cfg.etl.source, processing_config)
    sink = instantiate(cfg.etl.sink, processing_config)

    cfgs = {k: {"_args_": [processing_config]} for k in cfg.etl.transformations.keys()}
    transformations = instantiate(cfg.etl.transformations, **cfgs)

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
