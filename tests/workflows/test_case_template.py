from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

from workflows.runner import validate_workflow

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PACKAGE = REPO_ROOT / "templates" / "case" / "src" / "cases" / "template_case"


def _load_study_module():
    path = TEMPLATE_PACKAGE / "study_workflow.py"
    spec = importlib.util.spec_from_file_location("template_case_study_workflow", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _study_config() -> dict:
    with (TEMPLATE_PACKAGE / "configs" / "study.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_case_template_python_files_compile_and_configs_are_present():
    for path in TEMPLATE_PACKAGE.rglob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    assert (TEMPLATE_PACKAGE / "configs" / "pycaret.yaml").is_file()
    training = (TEMPLATE_PACKAGE / "configs" / "train_model.yaml").read_text(
        encoding="utf-8"
    )
    assert "retrain_best: true" in training
    assert "search_space:" in training
    assert "input_columns_file:" in training


def test_case_template_builds_expected_dag_and_method_commands():
    module = _load_study_module()
    definition = module.build_workflow(_study_config(), REPO_ROOT)

    assert [stage.name for stage in validate_workflow(definition)] == [
        "prepare_data",
        "plan_cases",
        "select_features",
        "train_model",
        "summarize",
    ]
    assert "train or tune mlp" in definition.stage_map()["train_model"].description
    assert definition.input_paths(_study_config(), REPO_ROOT) == [
        REPO_ROOT / "data" / "template_case" / "processed"
    ]


def test_case_template_rejects_unsafe_artifact_path():
    module = _load_study_module()
    config = _study_config()
    config["training"]["method"]["checkpoint"] = "../outside.mdlus"

    with pytest.raises(ValueError, match="must stay beneath"):
        module.build_workflow(config, REPO_ROOT)
