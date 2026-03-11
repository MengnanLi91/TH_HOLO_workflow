"""Helpers for loading YAML defaults into argparse-based FNO scripts."""

import argparse
from pathlib import Path
from typing import Any


def _normalize_key(key: str) -> str:
    """Support YAML keys written as `batch-size` or `batch_size`."""
    return key.replace("-", "_")


def _load_yaml_defaults(
    config_path: Path, parser: argparse.ArgumentParser
) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError as import_error:
        raise ModuleNotFoundError(
            "YAML config parsing requires `omegaconf` (installed with hydra-core)."
        ) from import_error

    loaded = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping/object.")

    allowed = {action.dest for action in parser._actions if action.dest != "help"}
    normalized: dict[str, Any] = {}
    unknown_keys: list[str] = []

    for raw_key, value in loaded.items():
        key = _normalize_key(str(raw_key))
        if key not in allowed:
            unknown_keys.append(str(raw_key))
            continue
        normalized[key] = value

    if unknown_keys:
        unknown_keys = sorted(unknown_keys)
        raise ValueError(
            "Unknown config key(s): "
            + ", ".join(unknown_keys)
            + ". Keys must match CLI argument names."
        )

    return normalized


def parse_args_with_yaml(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse args with optional `--config` YAML defaults.

    Precedence order:
      1) Parser defaults
      2) YAML values from --config
      3) Explicit CLI args
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, remaining = pre_parser.parse_known_args()

    if pre_args.config is not None:
        yaml_defaults = _load_yaml_defaults(pre_args.config, parser)
        parser.set_defaults(**yaml_defaults)

    parser.set_defaults(config=pre_args.config)
    return parser.parse_args(remaining)
