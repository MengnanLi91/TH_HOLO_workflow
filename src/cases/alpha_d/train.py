"""Discoverability wrapper for alpha-D model training.

Invokes :func:`training.runner.train` against this case's configs. For
HPO, use the canonical entry point instead — it threads through Optuna
and the retrain-best step:

    python train.py --config-path cases/alpha_d/configs --config-name train_mlp

Default config is ``train_mlp``; override with ``--config-name train_conv1d``.
"""

import os
import sys

import hydra
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.runner import train


@hydra.main(version_base="1.3", config_path="configs", config_name="train_mlp")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
