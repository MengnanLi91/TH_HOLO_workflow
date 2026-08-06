"""Hydra entrypoint for direct training or Optuna HPO in this case."""

import hydra
from omegaconf import DictConfig

from training.runner import train_or_hpo


@hydra.main(version_base="1.3", config_path="configs", config_name="train_model")
def main(cfg: DictConfig) -> None:
    """Run HPO when a search space is configured; otherwise train once."""
    train_or_hpo(cfg)


if __name__ == "__main__":
    main()
