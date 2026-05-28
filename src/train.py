"""Hydra entry point for model training (with optional HPO).

The actual train-vs-HPO dispatch lives in
:func:`training.runner.train_or_hpo` — when the config contains an
``hpo`` block with a non-empty ``search_space``, Optuna runs first
(plus a retrain-best step); otherwise the model trains directly.
Set ``hpo=null`` on the CLI to force the direct path.

Usage (from src/ directory):
    python train.py --config-name alpha_d_mlp              # HPO + retrain
    python train.py --config-name alpha_d_mlp hpo=null     # direct training
    python train.py --config-name fno                      # direct (no hpo)
"""

import os
import sys

import hydra
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(__file__))

from training.runner import train_or_hpo


@hydra.main(version_base="1.3", config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    train_or_hpo(cfg)


if __name__ == "__main__":
    main()
