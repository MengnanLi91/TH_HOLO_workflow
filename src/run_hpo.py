"""Hydra entry point for hyperparameter optimization.

Usage (from src/ directory):
    python run_hpo.py --config-name hpo_alpha_d_mlp
    python run_hpo.py --config-name hpo_alpha_d_mlp hpo.n_trials=200
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import hydra
from omegaconf import DictConfig, OmegaConf

from training.hpo.study import run_hpo


@hydra.main(version_base="1.3", config_path="config", config_name="hpo_default")
def main(cfg: DictConfig) -> None:
    """Run Optuna hyperparameter optimization."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    results = run_hpo(cfg_dict)

    n_complete = results.get("n_complete", 0)
    n_pruned = results.get("n_pruned", 0)
    n_trials = results.get("n_trials", 0)
    print(f"\nHPO complete: {n_complete} finished, {n_pruned} pruned, {n_trials} total")

    if n_complete > 0:
        print(f"Best trial #{results['best_trial_number']}: "
              f"val_loss={results['best_value']:.6e}")
        print(f"Best params: {json.dumps(results['best_params'], indent=2)}")
        print(f"Artifacts saved to: {results['output_dir']}")

        if "retrain" in results:
            retrain = results["retrain"]
            print(f"\nRetrained model saved to: {retrain['checkpoint']}")
            print(f"Final train loss: {retrain['final_train_loss']:.6e}")


if __name__ == "__main__":
    main()
