"""Parse YAML search space definitions into Optuna trial suggestions."""

import copy
from typing import Any

import optuna


# Dot-path prefixes that must never be overridden during HPO because they
# change the dataset or model identity rather than tuning hyperparameters.
UNSAFE_PREFIXES = (
    "data",
    "model.name",
    "model.entrypoint",
    "model.adapter",
)


def validate_search_space(search_space: dict[str, dict], base_cfg: dict) -> None:
    """Check that all search-space keys are safe and exist in base_cfg.

    Raises ``ValueError`` for unsafe prefixes and ``KeyError`` for
    dot-paths that do not exist in *base_cfg* (catches typos).
    """
    for dot_path in search_space:
        for prefix in UNSAFE_PREFIXES:
            if dot_path == prefix or dot_path.startswith(prefix + "."):
                raise ValueError(
                    f"Search-space key '{dot_path}' is not allowed. "
                    f"Overriding '{prefix}' changes the dataset or model identity. "
                    "Only training.* and model.params.* paths are supported."
                )

        keys = dot_path.split(".")
        node = base_cfg
        for i, key in enumerate(keys):
            if not isinstance(node, dict) or key not in node:
                partial = ".".join(keys[: i + 1])
                raise KeyError(
                    f"Search-space path '{dot_path}' is invalid: "
                    f"'{partial}' does not exist in the base config. "
                    "Check for typos."
                )
            node = node[key]


def sample_from_search_space(
    trial: optuna.Trial,
    search_space: dict[str, dict],
) -> dict[str, Any]:
    """Sample hyperparameters from a YAML search-space definition.

    Each entry in *search_space* maps a dot-path config key to a spec dict::

        training.lr:
          type: float
          low: 1e-5
          high: 1e-2
          log: true

    Supported types: ``float``, ``int``, ``categorical``.

    Returns a dict mapping dot-path keys to sampled values.
    """
    sampled: dict[str, Any] = {}
    for param_path, spec in search_space.items():
        param_type = spec["type"]
        if param_type == "float":
            sampled[param_path] = trial.suggest_float(
                param_path,
                low=float(spec["low"]),
                high=float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif param_type == "int":
            sampled[param_path] = trial.suggest_int(
                param_path,
                low=int(spec["low"]),
                high=int(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif param_type == "categorical":
            sampled[param_path] = trial.suggest_categorical(
                param_path, spec["choices"],
            )
        else:
            raise ValueError(
                f"Unknown search-space type '{param_type}' for '{param_path}'. "
                "Must be 'float', 'int', or 'categorical'."
            )
    return sampled


def apply_overrides(base_cfg: dict, overrides: dict[str, Any]) -> dict:
    """Deep-copy *base_cfg* and set values at dot-paths.

    Raises ``KeyError`` if any intermediate or leaf key does not already
    exist in the base config.  This prevents silent creation of new keys
    from typos.
    """
    cfg = copy.deepcopy(base_cfg)
    for dot_path, value in overrides.items():
        keys = dot_path.split(".")
        node = cfg
        for i, key in enumerate(keys[:-1]):
            if not isinstance(node, dict) or key not in node:
                partial = ".".join(keys[: i + 1])
                raise KeyError(
                    f"Cannot apply override '{dot_path}': "
                    f"'{partial}' does not exist in config."
                )
            node = node[key]
        leaf = keys[-1]
        if leaf not in node:
            raise KeyError(
                f"Cannot apply override '{dot_path}': "
                f"key '{leaf}' does not exist in config at '{'.'.join(keys[:-1])}'."
            )
        node[leaf] = value
    return cfg
