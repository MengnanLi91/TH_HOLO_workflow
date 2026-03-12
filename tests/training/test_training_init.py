"""Unit tests for training package helper imports."""

from __future__ import annotations

import sys
import types

from training import _require_pyg, import_physicsnemo_attr


def test_import_physicsnemo_attr_returns_symbol() -> None:
    sqrt = import_physicsnemo_attr("math", "sqrt")
    assert sqrt(9.0) == 3.0


def test_require_pyg_uses_loaded_module(monkeypatch) -> None:
    fake_pyg = types.ModuleType("torch_geometric")
    monkeypatch.setitem(sys.modules, "torch_geometric", fake_pyg)
    assert _require_pyg() is fake_pyg
