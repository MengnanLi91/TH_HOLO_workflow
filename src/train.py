"""Hydra entry point for generic supervised model training."""

import os
import sys

import hydra
from omegaconf import DictConfig

# Ensure src/ is importable when running from src/ as a script.
sys.path.insert(0, os.path.dirname(__file__))

from training.runner import train


@hydra.main(version_base="1.3", config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
