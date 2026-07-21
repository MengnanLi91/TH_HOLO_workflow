"""Case-owned workflow template for feature selection and model HPO.

Copy this module with the rest of ``cases.template_case`` and replace only
the stage logic that is specific to the new case.  The generic workflow
package remains responsible for ordering, execution, manifests, and resume.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from workflows import (
    Artifact,
    Command,
    RunContext,
    Stage,
    StageResult,
    WorkflowDefinition,
)


def _required_string(mapping: dict[str, Any], key: str, section: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section}.{key} must be a non-empty string")
    return value.strip()


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _artifact_name(method: dict[str, Any], key: str) -> Path:
    path = Path(_required_string(method, key, "training.method"))
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ValueError(
            f"training.method.{key} must stay beneath the model artifact directory"
        )
    return path


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_case_file(path: Path, cases: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{case}\n" for case in cases), encoding="utf-8")


def _read_case_file(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _zarr_dir(context: RunContext) -> Path:
    inputs = context.config.get("inputs") or {}
    return _repo_path(context.repo_root, _required_string(inputs, "zarr_dir", "inputs"))


def workflow_input_paths(config: dict[str, Any], repo_root: Path) -> list[Path]:
    """Declare external inputs whose fingerprints bind a workflow run ID."""
    inputs = config.get("inputs") or {}
    return [_repo_path(repo_root, _required_string(inputs, "zarr_dir", "inputs"))]


def _python_command(context: RunContext, *arguments: str | Path, label: str) -> Command:
    study = context.config.get("study") or {}
    return Command(
        argv=tuple(str(value) for value in ("python", *arguments)),
        executor=str(study.get("python_executor", "python")),
        cwd=context.repo_root,
        env={"PYTHONPATH": str(context.repo_root / "src")},
        label=label,
    )


def _prepare_data(context: RunContext) -> StageResult:
    stores = sorted(_zarr_dir(context).glob("*.zarr"))
    if len(stores) < 2:
        raise ValueError(
            f"The template requires at least two .zarr cases under {_zarr_dir(context)}"
        )
    summary = context.run_dir / "data" / "input_summary.json"
    _atomic_json(
        summary,
        {
            "schema_version": 1,
            "zarr_dir": str(_zarr_dir(context)),
            "case_count": len(stores),
            "case_ids": [path.stem for path in stores],
        },
    )
    return StageResult(
        artifacts=[Artifact("data/input_summary.json", "processed input summary")],
        details={"case_count": len(stores)},
    )


def _prepare_data_valid(context: RunContext) -> bool:
    path = context.run_dir / "data" / "input_summary.json"
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return int(summary.get("case_count", 0)) >= 2


def _plan_cases(context: RunContext) -> StageResult:
    cases = sorted(path.stem for path in _zarr_dir(context).glob("*.zarr"))
    study = context.config.get("study") or {}
    ratio = float(study.get("test_fraction", 0.2))
    if not 0.0 < ratio < 1.0:
        raise ValueError("study.test_fraction must be between zero and one")

    shuffled = list(cases)
    random.Random(int(study.get("seed", 42))).shuffle(shuffled)
    test_count = min(len(cases) - 1, max(1, round(len(cases) * ratio)))
    heldout = sorted(shuffled[:test_count])
    training = sorted(set(cases) - set(heldout))
    split_dir = context.run_dir / "splits"
    _write_case_file(split_dir / "heldout_cases.txt", heldout)
    _write_case_file(split_dir / "train_cases.txt", training)
    _atomic_json(
        split_dir / "split.json",
        {
            "schema_version": 1,
            "seed": int(study.get("seed", 42)),
            "test_fraction": ratio,
            "train_cases": training,
            "heldout_cases": heldout,
        },
    )
    return StageResult(
        artifacts=[Artifact("splits", "canonical case split")],
        details={"train_cases": len(training), "heldout_cases": len(heldout)},
    )


def _case_split_valid(context: RunContext) -> bool:
    train_path = context.run_dir / "splits" / "train_cases.txt"
    heldout_path = context.run_dir / "splits" / "heldout_cases.txt"
    if not train_path.is_file() or not heldout_path.is_file():
        return False
    training = _read_case_file(train_path)
    heldout = _read_case_file(heldout_path)
    return bool(training and heldout and set(training).isdisjoint(heldout))


def _select_features(context: RunContext) -> StageResult:
    feature = context.config.get("feature_selection") or {}
    output_dir = context.run_dir / "features"
    context.run(
        _python_command(
            context,
            "-m",
            _required_string(feature, "module", "feature_selection"),
            "--config-name",
            _required_string(feature, "config_name", "feature_selection"),
            f"data.zarr_dir={_zarr_dir(context)}",
            "data.exclude_cases=[]",
            f"data.exclude_cases_file={context.run_dir / 'splits' / 'heldout_cases.txt'}",
            f"output.dir={output_dir}",
            label="feature_selection",
        )
    )
    return StageResult(artifacts=[Artifact("features", "selected input columns")])


def _features_valid(context: RunContext) -> bool:
    selected = context.run_dir / "features" / "selected_features.txt"
    manifest = context.run_dir / "features" / "manifest.json"
    return bool(selected.is_file() and _read_case_file(selected) and manifest.is_file())


def _method(context: RunContext) -> dict[str, Any]:
    training = context.config.get("training") or {}
    method = training.get("method")
    if not isinstance(method, dict):
        raise ValueError("Configuration requires a [training.method] section")
    return method


def _model_dir(context: RunContext) -> Path:
    return (
        context.run_dir
        / "models"
        / _required_string(_method(context), "id", "training.method")
    )


def _train_model(context: RunContext) -> StageResult:
    method = _method(context)
    model_dir = _model_dir(context)
    checkpoint = model_dir / _artifact_name(method, "checkpoint")
    run_meta = model_dir / _artifact_name(method, "run_meta")
    context.run(
        _python_command(
            context,
            "-m",
            _required_string(method, "runner_module", "training.method"),
            "--config-name",
            _required_string(method, "config_name", "training.method"),
            f"data.zarr_dir={_zarr_dir(context)}",
            f"data.input_columns_file={context.run_dir / 'features' / 'selected_features.txt'}",
            "data.split.strategy=file",
            f"data.split.train_file={context.run_dir / 'splits' / 'train_cases.txt'}",
            f"data.split.test_file={context.run_dir / 'splits' / 'heldout_cases.txt'}",
            f"output.root_dir={context.run_dir / 'models'}",
            f"output.case_name={_required_string(method, 'id', 'training.method')}",
            f"output.checkpoint={checkpoint}",
            f"output.run_meta={run_meta}",
            label="train_or_hpo",
        )
    )
    return StageResult(
        artifacts=[Artifact(model_dir.relative_to(context.run_dir), "trained method")],
        details={
            "method_id": _required_string(method, "id", "training.method"),
            "hydra_config": _required_string(method, "config_name", "training.method"),
        },
    )


def _model_valid(context: RunContext) -> bool:
    method = _method(context)
    checkpoint = _model_dir(context) / _artifact_name(method, "checkpoint")
    run_meta = _model_dir(context) / _artifact_name(method, "run_meta")
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size == 0
        or not run_meta.is_file()
    ):
        return False
    try:
        metadata = json.loads(run_meta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        isinstance(metadata, dict)
        and metadata.get("training_run_meta_schema") == 2
        and bool(metadata.get("model_name"))
        and bool(metadata.get("entrypoint"))
    )


def _summarize(context: RunContext) -> StageResult:
    method = _method(context)
    report = context.run_dir / "report" / "summary.json"
    _atomic_json(
        report,
        {
            "schema_version": 1,
            "method": {
                "id": _required_string(method, "id", "training.method"),
                "runner_module": _required_string(
                    method, "runner_module", "training.method"
                ),
                "config_name": _required_string(
                    method, "config_name", "training.method"
                ),
            },
            "train_cases": _read_case_file(
                context.run_dir / "splits" / "train_cases.txt"
            ),
            "heldout_cases": _read_case_file(
                context.run_dir / "splits" / "heldout_cases.txt"
            ),
            "checkpoint": str(
                (
                    _model_dir(context) / _artifact_name(method, "checkpoint")
                ).relative_to(context.run_dir)
            ),
            "run_meta": str(
                (_model_dir(context) / _artifact_name(method, "run_meta")).relative_to(
                    context.run_dir
                )
            ),
        },
    )
    return StageResult(artifacts=[Artifact("report/summary.json", "run summary")])


def build_workflow(config: dict[str, Any], repo_root: Path) -> WorkflowDefinition:
    """Build this case's workflow; called by ``multifid-workflow``."""
    del repo_root
    workflow = config.get("workflow") or {}
    _required_string(config.get("inputs") or {}, "zarr_dir", "inputs")
    feature = config.get("feature_selection") or {}
    _required_string(feature, "module", "feature_selection")
    _required_string(feature, "config_name", "feature_selection")
    method = _method_from_config(config)
    _artifact_name(method, "checkpoint")
    _artifact_name(method, "run_meta")

    return WorkflowDefinition(
        workflow_id=_required_string(workflow, "id", "workflow"),
        version=int(workflow.get("version", 1)),
        input_paths=workflow_input_paths,
        stages=(
            Stage(
                "prepare_data",
                _prepare_data,
                description="validate the processed case stores",
                validator=_prepare_data_valid,
            ),
            Stage(
                "plan_cases",
                _plan_cases,
                dependencies=("prepare_data",),
                description="write one deterministic train/held-out split",
                validator=_case_split_valid,
            ),
            Stage(
                "select_features",
                _select_features,
                dependencies=("plan_cases",),
                description="run group-safe PyCaret feature selection",
                validator=_features_valid,
            ),
            Stage(
                "train_model",
                _train_model,
                dependencies=("select_features",),
                description=(
                    f"train or tune {_required_string(method, 'id', 'training.method')} "
                    "using its Hydra profile"
                ),
                validator=_model_valid,
            ),
            Stage(
                "summarize",
                _summarize,
                dependencies=("train_model",),
                description="record the selected method and final artifacts",
            ),
        ),
    )


def _method_from_config(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training") or {}
    method = training.get("method")
    if not isinstance(method, dict):
        raise ValueError("Configuration requires a [training.method] section")
    for key in ("id", "runner_module", "config_name", "checkpoint", "run_meta"):
        _required_string(method, key, "training.method")
    return method
