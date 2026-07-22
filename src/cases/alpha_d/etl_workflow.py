"""ETL-only workflow for turning an alpha-D Exodus campaign into Zarr stores."""

from __future__ import annotations

import json
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


def _repo_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(str(raw).format(repo_root=repo_root)).expanduser()
    return path if path.is_absolute() else repo_root / path


def _raw_dir(context: RunContext) -> Path:
    inputs = context.config.get("inputs") or {}
    return _repo_path(context.repo_root, _required_string(inputs, "raw_dir", "inputs"))


def workflow_input_paths(config: dict[str, Any], repo_root: Path) -> list[Path]:
    """Return the raw campaign tree that binds an ETL run ID."""
    inputs = config.get("inputs") or {}
    return [_repo_path(repo_root, _required_string(inputs, "raw_dir", "inputs"))]


def _python_command(context: RunContext, *arguments: str | Path, label: str) -> Command:
    etl = context.config.get("etl") or {}
    return Command(
        argv=tuple(str(value) for value in ("python", *arguments)),
        executor=str(etl.get("python_executor", "python")),
        cwd=context.repo_root,
        env={"PYTHONPATH": str(context.repo_root / "src")},
        label=label,
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _raw_cases(raw_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in raw_dir.iterdir()
        if path.is_dir() and (path / "simulation_out.e").is_file()
    )


def _extract(context: RunContext) -> StageResult:
    raw_dir = _raw_dir(context)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw alpha-D campaign directory not found: {raw_dir}")
    raw_cases = _raw_cases(raw_dir)
    if not raw_cases:
        raise FileNotFoundError(
            f"No case directories containing simulation_out.e found under {raw_dir}"
        )

    etl = context.config.get("etl") or {}
    output_dir = context.run_dir / "data" / "processed"
    context.run(
        _python_command(
            context,
            "-m",
            _required_string(etl, "runner_module", "etl"),
            f"etl.source.input_dir={raw_dir}",
            f"etl.sink.output_dir={output_dir}",
            f"etl.processing.num_processes={int(etl.get('num_processes', 4))}",
            label="extract_alpha_d",
        )
    )

    stores = sorted(path.stem for path in output_dir.glob("*.zarr"))
    if not stores:
        raise FileNotFoundError(f"ETL produced no .zarr stores under {output_dir}")
    missing_cases = sorted(set(raw_cases) - set(stores))
    summary = context.run_dir / "data" / "etl_summary.json"
    _atomic_json(
        summary,
        {
            "schema_version": 1,
            "raw_dir": str(raw_dir),
            "processed_dir": "data/processed",
            "raw_case_count": len(raw_cases),
            "processed_case_count": len(stores),
            "missing_cases": missing_cases,
        },
    )
    return StageResult(
        status="partial" if missing_cases else "succeeded",
        artifacts=[
            Artifact("data/processed", "processed alpha-D Zarr stores"),
            Artifact("data/etl_summary.json", "ETL coverage summary"),
        ],
        details={
            "raw_case_count": len(raw_cases),
            "processed_case_count": len(stores),
            "missing_case_count": len(missing_cases),
        },
    )


def _extract_valid(context: RunContext) -> bool:
    summary_path = context.run_dir / "data" / "etl_summary.json"
    output_dir = context.run_dir / "data" / "processed"
    if not summary_path.is_file() or not output_dir.is_dir():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        summary.get("schema_version") == 1
        and int(summary.get("processed_case_count", 0)) > 0
        and bool(list(output_dir.glob("*.zarr")))
    )


def build_workflow(config: dict[str, Any], repo_root: Path) -> WorkflowDefinition:
    """Build the alpha-D raw-Exodus-to-Zarr workflow."""
    del repo_root
    workflow = config.get("workflow") or {}
    _required_string(workflow, "id", "workflow")
    _required_string(config.get("inputs") or {}, "raw_dir", "inputs")
    etl = config.get("etl") or {}
    _required_string(etl, "runner_module", "etl")
    workers = int(etl.get("num_processes", 4))
    if workers < 1:
        raise ValueError("etl.num_processes must be at least one")

    return WorkflowDefinition(
        workflow_id=str(workflow["id"]),
        version=int(workflow.get("version", 1)),
        stages=(
            Stage(
                "extract",
                _extract,
                description="extract alpha-D profiles from raw Exodus cases",
                validator=_extract_valid,
            ),
        ),
        input_paths=workflow_input_paths,
    )
