"""Hydra entry point for model training (with optional HPO).

When the config contains an ``hpo`` section with a non-empty
``search_space``, Optuna hyperparameter optimization runs first and
(by default) retrains the best configuration.  Set ``hpo=null`` on the
CLI to skip HPO and train directly.

Usage (from src/ directory):
    python train.py --config-name alpha_d_mlp              # HPO + retrain
    python train.py --config-name alpha_d_mlp hpo=null     # direct training
    python train.py --config-name fno                      # direct (no hpo)
"""

import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure src/ is importable when running from src/ as a script.
sys.path.insert(0, os.path.dirname(__file__))

from training.runner import train


def _has_hpo(cfg: DictConfig) -> bool:
    """Check whether the config requests HPO."""
    hpo = cfg.get("hpo")
    if hpo is None:
        return False
    if isinstance(hpo, DictConfig) and hpo.get("search_space"):
        return True
    if isinstance(hpo, dict) and hpo.get("search_space"):
        return True
    return False


@hydra.main(version_base="1.3", config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    if _has_hpo(cfg):
        import json
        from training.hpo.study import run_hpo

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
    else:
        train(cfg)


if __name__ == "__main__":
    main()
