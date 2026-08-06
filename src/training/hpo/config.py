"""Validation for the clean-break HPO configuration contract."""

from __future__ import annotations

import math
from typing import Any

REQUIRED_HPO_KEYS = {
    "study_name",
    "direction",
    "storage",
    "load_if_exists",
    "retrain_best",
    "screening",
    "validation",
    "objective",
    "confirmation",
    "enqueue_trials",
    "search_space",
}


def _require_mapping(parent: dict, key: str) -> dict:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"hpo.{key} must be a mapping.")
    return value


def _validate_early_stopping(parent: dict, path: str) -> None:
    cfg = parent.get("early_stopping")
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}.early_stopping must be a mapping.")
    if set(cfg) != {"patience", "min_delta"}:
        raise ValueError(f"{path}.early_stopping requires exactly patience and min_delta.")
    if int(cfg["patience"]) < 1:
        raise ValueError(f"{path}.early_stopping.patience must be >= 1.")
    if float(cfg["min_delta"]) < 0.0:
        raise ValueError(f"{path}.early_stopping.min_delta must be >= 0.")


def validate_hpo_config(hpo_cfg: dict[str, Any]) -> None:
    """Reject obsolete or incomplete HPO configurations before execution."""
    if "n_trials" in hpo_cfg:
        raise ValueError("Obsolete hpo.n_trials is not supported; use hpo.screening.n_trials.")
    missing = sorted(REQUIRED_HPO_KEYS - set(hpo_cfg))
    if missing:
        raise ValueError(f"hpo is missing required key(s): {missing}")
    if hpo_cfg["direction"] != "minimize":
        raise ValueError("hpo.direction must be 'minimize' for composite objectives.")

    screening = _require_mapping(hpo_cfg, "screening")
    if set(screening) != {"n_trials", "max_epochs", "early_stopping"}:
        raise ValueError("hpo.screening requires exactly n_trials, max_epochs, and early_stopping.")
    if int(screening["n_trials"]) < 1 or int(screening["max_epochs"]) < 1:
        raise ValueError("hpo.screening n_trials and max_epochs must be >= 1.")
    _validate_early_stopping(screening, "hpo.screening")

    validation = _require_mapping(hpo_cfg, "validation")
    if "split_ratio" in validation:
        raise ValueError(
            "Obsolete hpo.validation.split_ratio is not supported; configure a fold builder."
        )
    required_validation = {"splitter_entrypoint", "n_folds", "screening_fold", "seed"}
    if set(validation) != required_validation:
        raise ValueError(f"hpo.validation requires exactly {sorted(required_validation)}.")
    n_folds = int(validation["n_folds"])
    screening_fold = int(validation["screening_fold"])
    if n_folds < 2:
        raise ValueError("hpo.validation.n_folds must be >= 2.")
    if screening_fold < 0 or screening_fold >= n_folds:
        raise ValueError("hpo.validation.screening_fold must index an available fold.")
    if ":" not in str(validation["splitter_entrypoint"]):
        raise ValueError("hpo.validation.splitter_entrypoint must be '<module>:<callable>'.")

    objective = _require_mapping(hpo_cfg, "objective")
    if set(objective) != {"weights"} or not isinstance(objective["weights"], dict):
        raise ValueError("hpo.objective requires exactly a non-empty weights mapping.")
    if not objective["weights"]:
        raise ValueError("hpo.objective.weights must not be empty.")
    for name, weight in objective["weights"].items():
        if not str(name).strip() or not math.isfinite(float(weight)):
            raise ValueError("hpo.objective.weights names and values must be finite.")

    confirmation = _require_mapping(hpo_cfg, "confirmation")
    required_confirmation = {
        "top_k",
        "max_epochs",
        "early_stopping",
        "aggregate_std_weight",
        "guard_metric",
        "guard_reference",
    }
    if set(confirmation) != required_confirmation:
        raise ValueError(f"hpo.confirmation requires exactly {sorted(required_confirmation)}.")
    if int(confirmation["top_k"]) < 1 or int(confirmation["max_epochs"]) < 1:
        raise ValueError("hpo.confirmation top_k and max_epochs must be >= 1.")
    if float(confirmation["aggregate_std_weight"]) < 0.0:
        raise ValueError("hpo.confirmation.aggregate_std_weight must be >= 0.")
    _validate_early_stopping(confirmation, "hpo.confirmation")

    enqueued = hpo_cfg["enqueue_trials"]
    if not isinstance(enqueued, list):
        raise ValueError("hpo.enqueue_trials must be a list.")
    control_names: set[str] = set()
    for index, control in enumerate(enqueued):
        if not isinstance(control, dict) or set(control) != {"name", "params"}:
            raise ValueError(
                f"hpo.enqueue_trials[{index}] requires exactly name and params mappings."
            )
        name = str(control["name"])
        if not name or name in control_names or not isinstance(control["params"], dict):
            raise ValueError("HPO control names must be unique and params must be mappings.")
        control_names.add(name)

    guard_metric = confirmation["guard_metric"]
    guard_reference = confirmation["guard_reference"]
    if (guard_metric is None) != (guard_reference is None):
        raise ValueError("confirmation guard_metric and guard_reference must both be set or null.")
    if guard_reference is not None:
        if guard_reference != "best_control":
            raise ValueError("hpo.confirmation.guard_reference must be 'best_control' or null.")
        if not enqueued:
            raise ValueError("A best_control guard requires at least one enqueued control.")
