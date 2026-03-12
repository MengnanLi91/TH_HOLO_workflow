"""Model registry and entrypoint resolution for training workflows."""

import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ModelEntry:
    build_fn: Callable
    adapter: str


MODEL_REGISTRY: dict[str, ModelEntry] = {}


def register_model(name: str, build_fn: Callable, adapter: str) -> None:
    if adapter not in {"grid", "graph"}:
        raise ValueError(f"adapter must be 'grid' or 'graph', got '{adapter}'.")
    if name in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is already registered.")
    MODEL_REGISTRY[name] = ModelEntry(build_fn=build_fn, adapter=adapter)


def resolve_entrypoint(entrypoint: str) -> Callable:
    if ":" not in entrypoint:
        raise ValueError(
            f"Invalid entrypoint '{entrypoint}'. Expected format 'module.path:callable'."
        )
    module_path, callable_name = entrypoint.rsplit(":", 1)
    module = importlib.import_module(module_path)
    if not hasattr(module, callable_name):
        raise AttributeError(
            f"Entrypoint callable '{callable_name}' not found in module '{module_path}'."
        )
    build_fn = getattr(module, callable_name)
    if not callable(build_fn):
        raise TypeError(f"Entrypoint '{entrypoint}' is not callable.")
    return build_fn


def _validate_build_signature(build_fn: Callable, entrypoint: str) -> None:
    """Ensure a build function can accept `(model_cfg, dataset_info)` arguments."""
    signature = inspect.signature(build_fn)
    parameters = list(signature.parameters.values())

    positional_count = sum(
        1
        for param in parameters
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    has_varargs = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters)

    if positional_count < 2 and not has_varargs:
        raise TypeError(
            f"Model build function '{entrypoint}' must accept (model_cfg, dataset_info), "
            f"but has signature: {signature}"
        )


def get_build_fn_and_adapter(model_cfg: dict) -> tuple[Callable, str]:
    """Resolve model build function and adapter.

    Built-in models derive adapter from registry. Custom entrypoints must specify
    `model.adapter` explicitly.
    """
    entrypoint = model_cfg.get("entrypoint")
    if entrypoint:
        build_fn = resolve_entrypoint(str(entrypoint))
        _validate_build_signature(build_fn, str(entrypoint))

        adapter_name = model_cfg.get("adapter")
        if not adapter_name:
            raise ValueError(
                "model.adapter is required when using model.entrypoint. "
                "Set model.adapter to 'grid' or 'graph'."
            )
        if adapter_name not in {"grid", "graph"}:
            raise ValueError(
                f"model.adapter must be 'grid' or 'graph', got '{adapter_name}'."
            )
        return build_fn, adapter_name

    name = model_cfg.get("name")
    if not name:
        raise ValueError("model.name is required when model.entrypoint is not set.")
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Registered models: {sorted(MODEL_REGISTRY)}"
        )

    entry = MODEL_REGISTRY[name]
    user_adapter = model_cfg.get("adapter")
    if user_adapter and user_adapter != entry.adapter:
        raise ValueError(
            f"model.adapter='{user_adapter}' conflicts with registered adapter "
            f"'{entry.adapter}' for model '{name}'. Remove model.adapter or fix the mismatch."
        )

    return entry.build_fn, entry.adapter


def model_entrypoint_string(model_cfg: dict, build_fn: Callable) -> str:
    entrypoint = model_cfg.get("entrypoint")
    if entrypoint:
        return str(entrypoint)
    return f"{build_fn.__module__}:{build_fn.__name__}"


def _load_builtins() -> None:
    from training.models import afno, fno, meshgraphnet, pix2pix

    _ = (afno, fno, meshgraphnet, pix2pix)


_load_builtins()
