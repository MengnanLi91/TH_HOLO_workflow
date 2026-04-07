"""Study creation, orchestration, and artifact saving."""

import json
import logging
import random
from pathlib import Path
from typing import Any

import optuna

from training.datasets import split_indices
from training.hpo.objective import make_objective
from training.hpo.search_space import validate_search_space
from training.hpo.visualize import save_study_plots
from training.runner import normalize_split_cfg, prepare_training, train

logger = logging.getLogger(__name__)


def create_study(hpo_cfg: dict) -> optuna.Study:
    """Create or resume an Optuna study from config."""
    sampler_cfg = hpo_cfg.get("sampler", {})
    sampler_cls = getattr(optuna.samplers, sampler_cfg.get("name", "TPESampler"))
    sampler = sampler_cls(**sampler_cfg.get("params", {}))

    pruner_cfg = hpo_cfg.get("pruner", {})
    pruner_cls = getattr(optuna.pruners, pruner_cfg.get("name", "MedianPruner"))
    pruner = pruner_cls(**pruner_cfg.get("params", {}))

    storage = hpo_cfg.get("storage")
    if storage:
        storage_dir = Path(storage.replace("sqlite:///", "")).parent
        storage_dir.mkdir(parents=True, exist_ok=True)

    study_name = hpo_cfg.get("study_name", "hpo_study")
    load_if_exists = hpo_cfg.get("load_if_exists", True)
    try:
        return optuna.create_study(
            study_name=study_name,
            direction=hpo_cfg.get("direction", "minimize"),
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=load_if_exists,
        )
    except optuna.exceptions.DuplicatedStudyError:
        db_path = storage.replace("sqlite:///", "") if storage else "<storage>"
        print(
            f"\n[HPO Error] Study '{study_name}' already exists in the database.\n"
            f"  This happens when hpo.load_if_exists=false but the study name is already\n"
            f"  registered in: {db_path}\n"
            f"\n"
            f"  Fix options:\n"
            f"    1) Resume the existing study (recommended):\n"
            f"         hpo.load_if_exists=true\n"
            f"\n"
            f"    2) Start fresh under a new name (old study preserved):\n"
            f"         hpo.study_name={study_name}_v2\n"
            f"\n"
            f"    3) Delete the database and start over:\n"
            f"         rm {db_path}\n"
        )
        raise


def run_hpo(cfg_dict: dict) -> dict[str, Any]:
    """Run hyperparameter optimization.

    Parameters
    ----------
    cfg_dict : dict
        Full Hydra config (resolved) with ``hpo`` section + base training config.

    Returns
    -------
    dict
        Summary with best trial info, study stats, saved artifacts.
    """
    hpo_cfg = dict(cfg_dict.get("hpo") or {})
    base_cfg = {k: v for k, v in cfg_dict.items() if k != "hpo"}
    search_space = dict(hpo_cfg.get("search_space", {}))

    # --- Validate search space before doing any work ---
    validate_search_space(search_space, base_cfg)

    # --- Build dataset + adapter once (cached for all trials) ---
    prepared = prepare_training(base_cfg)
    dataset = prepared["dataset"]
    seed = prepared["seed"]

    # --- Outer train/test split (same as normal training) ---
    data_cfg = prepared["data_cfg"]
    split_cfg = normalize_split_cfg(
        dict(data_cfg.get("split") or {}), default_seed=seed
    )
    num_cases = len(dataset.sim_names) if hasattr(dataset, "sim_names") else len(dataset)
    train_idx, test_idx, train_sims, test_sims = split_indices(
        num_cases=num_cases,
        split_cfg=split_cfg,
        sim_names=dataset.sim_names,
    )

    # --- Inner train/val split from the training pool ---
    val_cfg = hpo_cfg.get("validation", {})
    val_ratio = float(val_cfg.get("split_ratio", 0.2))
    val_seed = int(val_cfg.get("seed", seed))

    rng = random.Random(val_seed)
    shuffled_train = list(train_idx)
    rng.shuffle(shuffled_train)
    n_val = max(1, int(len(shuffled_train) * val_ratio))
    val_idx = sorted(shuffled_train[:n_val])
    train_inner_idx = sorted(shuffled_train[n_val:])

    # --- Guard: non-empty splits ---
    if not train_inner_idx:
        raise ValueError(
            f"Inner training split is empty after reserving {n_val} validation case(s) "
            f"from {len(train_idx)} training case(s). Reduce hpo.validation.split_ratio "
            f"(currently {val_ratio}) or provide more data."
        )
    if not val_idx:
        raise ValueError(
            f"Validation split is empty. Only {len(train_idx)} training case(s) "
            f"available with val_ratio={val_ratio}. Provide more data."
        )

    val_sims = [dataset.sim_names[i] for i in val_idx]
    train_inner_sims = [dataset.sim_names[i] for i in train_inner_idx]

    logger.info(
        "HPO splits: %d train_inner, %d val, %d test (held out)",
        len(train_inner_idx), len(val_idx), len(test_idx),
    )

    # --- Create study and objective ---
    study = create_study(hpo_cfg)
    objective = make_objective(
        base_cfg=base_cfg,
        search_space=search_space,
        hpo_cfg=hpo_cfg,
        prepared=prepared,
        train_inner_idx=train_inner_idx,
        val_idx=val_idx,
    )

    # --- Optimize ---
    n_trials = int(hpo_cfg.get("n_trials", 50))
    timeout = hpo_cfg.get("timeout")
    show_progress_bar = bool(hpo_cfg.get("show_progress_bar", True))
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
    )

    # --- Results ---
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    if not completed:
        logger.warning("No trials completed successfully.")
        return {"n_trials": len(study.trials), "n_complete": 0, "n_pruned": len(pruned)}

    best = study.best_trial

    # --- Save artifacts ---
    output_dir = Path(hpo_cfg.get("output_dir", "hpo_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # best_params.json
    params_path = output_dir / "best_params.json"
    params_path.write_text(json.dumps(best.params, indent=2), encoding="utf-8")

    # best_config.yaml -- train-ready config (no hpo section)
    from training.hpo.search_space import apply_overrides

    best_config = apply_overrides(base_cfg, best.params)
    best_config_path = output_dir / "best_config.yaml"
    try:
        from omegaconf import OmegaConf

        OmegaConf.save(OmegaConf.create(best_config), str(best_config_path))
    except ImportError:
        import yaml

        best_config_path.write_text(
            yaml.dump(best_config, default_flow_style=False), encoding="utf-8"
        )

    # split_metadata.json
    split_meta = {
        "outer_train_sims": train_sims,
        "outer_test_sims": test_sims,
        "inner_train_sims": train_inner_sims,
        "inner_val_sims": val_sims,
    }
    split_path = output_dir / "split_metadata.json"
    split_path.write_text(json.dumps(split_meta, indent=2), encoding="utf-8")

    # Visualization (non-fatal)
    plot_files = save_study_plots(study, output_dir)

    results = {
        "study_name": study.study_name,
        "best_trial_number": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
        "n_complete": len(completed),
        "n_pruned": len(pruned),
        "output_dir": str(output_dir),
        "plots": plot_files,
    }

    # --- Optional: retrain best on the original outer train split ---
    if hpo_cfg.get("retrain_best", False):
        logger.info("Retraining best config on the original outer train split...")
        retrain_result = train(best_config)
        results["retrain"] = retrain_result

    return results
