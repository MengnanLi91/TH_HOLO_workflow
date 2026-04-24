"""Hydra entry point for case-level pressure-drop model training."""

import os
import sys

import hydra
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(__file__))

from case_pressure_drop.workflow import train_case_pressure_drop


@hydra.main(version_base="1.3", config_path="config", config_name="case_pressure_drop")
def main(cfg: DictConfig) -> None:
    train_case_pressure_drop(cfg)


if __name__ == "__main__":
    main()
