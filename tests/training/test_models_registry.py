"""Unit tests for model registry and entrypoint resolution."""

from __future__ import annotations

import sys
import types

import pytest

from training.models import MODEL_REGISTRY, get_build_fn_and_adapter


@pytest.mark.parametrize("name,adapter", [("fno", "grid"), ("afno", "grid"), ("pix2pix", "grid"), ("meshgraphnet", "graph")])
def test_builtin_models_registered(name: str, adapter: str) -> None:
    assert name in MODEL_REGISTRY
    build_fn, adapter_name = get_build_fn_and_adapter({"name": name})
    assert callable(build_fn)
    assert adapter_name == adapter


def test_builtin_adapter_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="conflicts with registered adapter"):
        get_build_fn_and_adapter({"name": "fno", "adapter": "graph"})


def test_custom_entrypoint_requires_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("tmp_custom_model")

    def build(model_cfg, dataset_info):
        return (model_cfg, dataset_info)

    module.build = build
    monkeypatch.setitem(sys.modules, "tmp_custom_model", module)

    with pytest.raises(ValueError, match="model.adapter is required"):
        get_build_fn_and_adapter({"entrypoint": "tmp_custom_model:build"})


def test_custom_entrypoint_signature_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("tmp_bad_signature")

    def build_only_one_arg(model_cfg):
        return model_cfg

    module.build = build_only_one_arg
    monkeypatch.setitem(sys.modules, "tmp_bad_signature", module)

    with pytest.raises(TypeError, match=r"must accept \(model_cfg, dataset_info\)"):
        get_build_fn_and_adapter(
            {"entrypoint": "tmp_bad_signature:build", "adapter": "grid"}
        )


def test_custom_entrypoint_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("tmp_valid_custom")

    def build(model_cfg, dataset_info):
        return {"cfg": model_cfg, "info": dataset_info}

    module.build = build
    monkeypatch.setitem(sys.modules, "tmp_valid_custom", module)

    build_fn, adapter = get_build_fn_and_adapter(
        {"entrypoint": "tmp_valid_custom:build", "adapter": "graph"}
    )
    assert build_fn is build
    assert adapter == "graph"
