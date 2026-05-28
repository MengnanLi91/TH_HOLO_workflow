"""Discoverability wrapper for alpha-D model training.

Calls :func:`training.runner.train_or_hpo`, which honours an ``hpo``
block in the config (dispatching Optuna + retrain-best) or trains
directly when the block is absent or set to ``null``. Behaviour is
identical to the top-level ``src/train.py``; this wrapper only fixes
``config_path`` and the default ``config_name`` for the alpha-D case.

Default config is ``train_mlp``; override with ``--config-name train_conv1d``.
"""

import os
import sys

import hydra
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.runner import train_or_hpo


@hydra.main(version_base="1.3", config_path="configs", config_name="train_mlp")
def main(cfg: DictConfig) -> None:
    train_or_hpo(cfg)


if __name__ == "__main__":
    main()
