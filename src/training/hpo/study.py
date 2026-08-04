"""Screening and multi-fold confirmation orchestration for HPO."""

from __future__ import annotations

import csv
import importlib
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import optuna

from training.datasets import split_indices
from training.hpo.config import validate_hpo_config
from training.hpo.objective import make_objective, train_and_score
from training.hpo.search_space import apply_overrides, validate_search_space
from training.hpo.splits import validate_folds
from training.hpo.visualize import save_study_plots
from training.runner import (
    normalize_split_cfg,
    prepare_fold_dataset,
    prepare_training,
    train,
)

logger = logging.getLogger(__name__)
HPO_STUDY_CONTRACT_SCHEMA = 1


def _load_splitter(entrypoint: str):
    module_name, callable_name = entrypoint.rsplit(":", 1)
    splitter = getattr(importlib.import_module(module_name), callable_name)
    if not callable(splitter):
        raise TypeError(f"HPO splitter '{entrypoint}' is not callable.")
    return splitter


def create_study(hpo_cfg: dict) -> optuna.Study:
    """Create or resume an Optuna screening study from config."""
    sampler_cfg = hpo_cfg.get("sampler", {})
    sampler_cls = getattr(optuna.samplers, sampler_cfg.get("name", "TPESampler"))
    sampler = sampler_cls(**sampler_cfg.get("params", {}))
    pruner_cfg = hpo_cfg.get("pruner", {})
    pruner_cls = getattr(optuna.pruners, pruner_cfg.get("name", "MedianPruner"))
    pruner = pruner_cls(**pruner_cfg.get("params", {}))

    storage = hpo_cfg["storage"]
    if storage:
        database_path = str(storage).replace("sqlite:///", "")
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=str(hpo_cfg["study_name"]),
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=bool(hpo_cfg["load_if_exists"]),
    )
    stored_schema = study.user_attrs.get("hpo_study_contract_schema")
    if stored_schema is None and study.trials:
        raise ValueError(
            f"Optuna study '{study.study_name}' predates the clean-break HPO contract; "
            "use a new study name or storage database."
        )
    if stored_schema not in {None, HPO_STUDY_CONTRACT_SCHEMA}:
        raise ValueError(
            f"Unsupported HPO study contract schema {stored_schema!r} for "
            f"study '{study.study_name}'."
        )
    study.set_user_attr("hpo_study_contract_schema", HPO_STUDY_CONTRACT_SCHEMA)
    return study


def _canonical_params(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _enqueue_controls(study: optuna.Study, controls: list[dict[str, Any]]) -> None:
    existing_controls = {
        trial.user_attrs.get("control_name")
        for trial in study.trials
        if trial.user_attrs.get("control_name") is not None
    }
    for control in controls:
        if control["name"] not in existing_controls:
            study.enqueue_trial(
                dict(control["params"]),
                user_attrs={
                    "control_name": str(control["name"]),
                    "control_params": dict(control["params"]),
                },
                skip_if_exists=True,
            )


def _screening_records(study: optuna.Study) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trial in study.trials:
        record: dict[str, Any] = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "control_name": trial.user_attrs.get("control_name"),
            "composite_score": trial.value,
            "best_epoch": trial.user_attrs.get("best_epoch"),
            "epochs_trained": trial.user_attrs.get("epochs_trained"),
            "params": dict(trial.user_attrs.get("control_params") or trial.params),
            "metrics": {
                key.removeprefix("metric:"): value
                for key, value in trial.user_attrs.items()
                if key.startswith("metric:")
            },
        }
        records.append(record)
    return records


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _confirmation_candidates(
    completed: list[optuna.trial.FrozenTrial],
    controls: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    sampled = sorted(
        (trial for trial in completed if trial.user_attrs.get("control_name") is None),
        key=lambda trial: float(trial.value),
    )[:top_k]
    candidates = [
        {
            "candidate_id": f"trial-{trial.number}",
            "source": "sampled",
            "screening_trial": trial.number,
            "params": dict(trial.params),
        }
        for trial in sampled
    ]
    candidates.extend(
        {
            "candidate_id": f"control-{control['name']}",
            "source": "control",
            "screening_trial": None,
            "params": dict(control["params"]),
        }
        for control in controls
    )

    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _canonical_params(candidate["params"])
        if key not in seen:
            seen.add(key)
            deduplicated.append(candidate)
    return deduplicated


def aggregate_confirmation(
    candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    std_weight: float,
    guard_metric: str | None,
    guard_reference: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate fold results and select the guarded confirmed winner."""
    aggregates: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_rows = [
            row for row in rows if row["candidate_id"] == candidate["candidate_id"]
        ]
        scores = [float(row["composite_score"]) for row in candidate_rows]
        mean_score = statistics.fmean(scores)
        std_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        aggregate = {
            **candidate,
            "objective_mean": mean_score,
            "objective_std": std_score,
            "rank_score": mean_score + std_weight * std_score,
            "worst_guard_metric": (
                max(abs(float(row[guard_metric])) for row in candidate_rows)
                if guard_metric is not None
                else None
            ),
            "guard_passed": True,
        }
        aggregates.append(aggregate)

    if guard_reference == "best_control":
        controls = [row for row in aggregates if row["source"] == "control"]
        if not controls:
            raise ValueError(
                "Confirmation guard requested best_control but no control was run."
            )
        best_control = min(controls, key=lambda row: row["rank_score"])
        threshold = float(best_control["worst_guard_metric"])
        for aggregate in aggregates:
            aggregate["guard_passed"] = (
                float(aggregate["worst_guard_metric"]) <= threshold
            )

    eligible = [row for row in aggregates if row["guard_passed"]]
    if not eligible:
        raise RuntimeError("Every confirmation candidate failed the configured guard.")
    winner = min(eligible, key=lambda row: row["rank_score"])
    return aggregates, winner


def run_hpo(cfg_dict: dict) -> dict[str, Any]:
    """Screen candidates on one fold, then confirm finalists on every fold."""
    hpo_cfg = dict(cfg_dict.get("hpo") or {})
    validate_hpo_config(hpo_cfg)
    base_cfg = {key: value for key, value in cfg_dict.items() if key != "hpo"}
    search_space = dict(hpo_cfg["search_space"])
    validate_search_space(search_space, base_cfg)
    for control in hpo_cfg["enqueue_trials"]:
        apply_overrides(base_cfg, control["params"])

    prepared = prepare_training(base_cfg)
    dataset = prepared["dataset"]
    split_cfg = normalize_split_cfg(
        dict(prepared["data_cfg"].get("split") or {}),
        default_seed=prepared["seed"],
    )
    train_indices, test_indices, train_sims, test_sims = split_indices(
        num_cases=len(dataset.sim_names),
        split_cfg=split_cfg,
        sim_names=dataset.sim_names,
    )

    validation_cfg = dict(hpo_cfg["validation"])
    splitter = _load_splitter(str(validation_cfg["splitter_entrypoint"]))
    folds = splitter(
        list(dataset.sim_names),
        list(train_indices),
        int(validation_cfg["n_folds"]),
        int(validation_cfg["seed"]),
    )
    validate_folds(folds, train_indices, int(validation_cfg["n_folds"]))
    screening_fold = int(validation_cfg["screening_fold"])
    screen_train, screen_val = folds[screening_fold]
    screening_prepared = prepare_fold_dataset(prepared, screen_train)

    study = create_study(hpo_cfg)
    controls = list(hpo_cfg["enqueue_trials"])
    _enqueue_controls(study, controls)
    objective = make_objective(
        base_cfg=base_cfg,
        search_space=search_space,
        phase_cfg=dict(hpo_cfg["screening"]),
        objective_weights=dict(hpo_cfg["objective"]["weights"]),
        prepared=screening_prepared,
        train_indices=screen_train,
        val_indices=screen_val,
        seed=int(validation_cfg["seed"]),
    )
    terminal_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    terminal_count = sum(trial.state in terminal_states for trial in study.trials)
    remaining = max(0, int(hpo_cfg["screening"]["n_trials"]) - terminal_count)
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=hpo_cfg.get("timeout"),
            show_progress_bar=bool(hpo_cfg.get("show_progress_bar", True)),
        )

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED
    ]
    if not completed and not controls:
        raise RuntimeError("No HPO screening trial completed successfully.")

    output_dir = Path(hpo_cfg.get("output_dir", "hpo_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    screening_records = _screening_records(study)
    (output_dir / "screening.json").write_text(
        json.dumps(screening_records, indent=2), encoding="utf-8"
    )
    screening_rows = [
        {
            "trial_number": record["trial_number"],
            "state": record["state"],
            "control_name": record["control_name"],
            "composite_score": record["composite_score"],
            "best_epoch": record["best_epoch"],
            "epochs_trained": record["epochs_trained"],
            **record["params"],
            **record["metrics"],
        }
        for record in screening_records
    ]
    _write_rows(output_dir / "screening.csv", screening_rows)

    confirmation_cfg = dict(hpo_cfg["confirmation"])
    candidates = _confirmation_candidates(
        completed, controls, int(confirmation_cfg["top_k"])
    )
    if not candidates:
        raise RuntimeError(
            "No sampled candidates or controls are available for confirmation."
        )

    fold_prepared = {
        fold_index: prepare_fold_dataset(prepared, fold[0])
        for fold_index, fold in enumerate(folds)
    }
    confirmation_rows: list[dict[str, Any]] = []
    base_seed = int(validation_cfg["seed"])
    guard_metric = confirmation_cfg["guard_metric"]
    for candidate in candidates:
        for fold_index, (fold_train, fold_val) in enumerate(folds):
            result = train_and_score(
                base_cfg=base_cfg,
                params=candidate["params"],
                phase_cfg=confirmation_cfg,
                objective_weights=dict(hpo_cfg["objective"]["weights"]),
                prepared=fold_prepared[fold_index],
                train_indices=fold_train,
                val_indices=fold_val,
                seed=base_seed + fold_index,
            )
            if guard_metric is not None and guard_metric not in result["metrics"]:
                raise ValueError(
                    f"Experiment did not provide confirmation guard metric '{guard_metric}'."
                )
            confirmation_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "source": candidate["source"],
                    "screening_trial": candidate["screening_trial"],
                    "fold": fold_index,
                    "seed": base_seed + fold_index,
                    "best_epoch": result["best_epoch"],
                    "epochs_trained": result["epochs_trained"],
                    "composite_score": result["score"],
                    **result["metrics"],
                }
            )

    aggregates, winner = aggregate_confirmation(
        candidates,
        confirmation_rows,
        std_weight=float(confirmation_cfg["aggregate_std_weight"]),
        guard_metric=guard_metric,
        guard_reference=confirmation_cfg["guard_reference"],
    )
    confirmation_payload = {
        "fold_runs": confirmation_rows,
        "candidates": aggregates,
        "winner": winner,
    }
    (output_dir / "confirmation.json").write_text(
        json.dumps(confirmation_payload, indent=2), encoding="utf-8"
    )
    _write_rows(output_dir / "confirmation.csv", confirmation_rows)

    best_params = dict(winner["params"])
    (output_dir / "best_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )
    best_config = apply_overrides(base_cfg, best_params)
    best_config_path = output_dir / "best_config.yaml"
    from omegaconf import OmegaConf

    OmegaConf.save(OmegaConf.create(best_config), str(best_config_path))
    split_payload = {
        "outer_train_sims": train_sims,
        "outer_test_sims": test_sims,
        "folds": [
            {
                "fold": fold_index,
                "train_sims": [dataset.sim_names[index] for index in fold_train],
                "validation_sims": [dataset.sim_names[index] for index in fold_val],
            }
            for fold_index, (fold_train, fold_val) in enumerate(folds)
        ],
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(split_payload, indent=2), encoding="utf-8"
    )
    plot_files = save_study_plots(study, output_dir)

    results: dict[str, Any] = {
        "study_name": study.study_name,
        "best_trial_number": winner["screening_trial"],
        "best_candidate_id": winner["candidate_id"],
        "best_value": winner["rank_score"],
        "best_params": best_params,
        "n_trials": len(study.trials),
        "n_complete": len(completed),
        "n_pruned": len(pruned),
        "output_dir": str(output_dir),
        "plots": plot_files,
    }
    if bool(hpo_cfg["retrain_best"]):
        logger.info("Retraining the confirmed winner on the outer training split.")
        results["retrain"] = train(best_config)
    return results
