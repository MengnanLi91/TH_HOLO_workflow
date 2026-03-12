"""Shared helpers for generic training and evaluation workflows."""

import importlib
import sys
from pathlib import Path


def _add_vendored_physicsnemo() -> None:
    """Add local vendored physicsnemo package to sys.path when available."""
    vendored_root = Path(__file__).resolve().parents[2] / "physicsnemo"
    if vendored_root.exists() and str(vendored_root) not in sys.path:
        sys.path.insert(0, str(vendored_root))


def import_physicsnemo_module(module_path: str):
    """Import a physicsnemo module from installed package or vendored source."""
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError:
        _add_vendored_physicsnemo()
        try:
            return importlib.import_module(module_path)
        except ModuleNotFoundError as import_error:
            raise ModuleNotFoundError(
                "Could not import physicsnemo. Install `nvidia-physicsnemo` or "
                "run in an environment where physicsnemo is available."
            ) from import_error


def import_physicsnemo_attr(module_path: str, attr_name: str):
    """Import a single symbol from a physicsnemo module."""
    module = import_physicsnemo_module(module_path)
    if not hasattr(module, attr_name):
        raise AttributeError(f"Module '{module_path}' has no attribute '{attr_name}'.")
    return getattr(module, attr_name)


def _require_pyg():
    """Require torch_geometric only when graph workflows are used."""
    try:
        import torch_geometric

        return torch_geometric
    except ModuleNotFoundError as import_error:
        raise ImportError(
            "Graph models require torch_geometric. Install with:\n"
            "  uv pip install torch_geometric torch_scatter torch_sparse\n"
            "See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
        ) from import_error
