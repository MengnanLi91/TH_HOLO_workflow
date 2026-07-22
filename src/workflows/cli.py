"""Command-line interface for planning and running study workflows."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any

from workflows.runner import WorkflowRunner, validate_workflow


def _repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate repository root from {start}")


def _load_entrypoint(value: str):
    if ":" not in value:
        raise ValueError("workflow.entrypoint must use 'module.path:callable' format")
    module_name, callable_name = value.rsplit(":", 1)
    return getattr(importlib.import_module(module_name), callable_name)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _with_etl_input_dir(
    config: dict[str, Any], input_dir: Path | None, repo_root: Path
) -> dict[str, Any]:
    """Return ``config`` with an optional absolute raw-campaign override."""
    if input_dir is None:
        return config
    raw_dir = input_dir.expanduser().resolve()
    updated = dict(config)
    inputs = dict(config.get("inputs") or {})
    inputs["raw_dir"] = str(raw_dir)
    updated["inputs"] = inputs
    if not raw_dir.is_relative_to(repo_root):
        executors = dict(config.get("executors") or {})
        python = dict(executors.get("python") or {})
        binds = [str(value) for value in python.get("binds") or []]
        raw_bind = f"{raw_dir}:{raw_dir}"
        if raw_bind not in binds:
            binds.append(raw_bind)
        python["binds"] = binds
        executors["python"] = python
        updated["executors"] = executors
    return updated


def _require_etl_workflow(config: dict[str, Any]) -> None:
    """Reject using the ETL command with a multi-stage study configuration."""
    workflow = config.get("workflow") or {}
    if workflow.get("kind") != "etl":
        raise ValueError(
            "The etl command requires workflow.kind = 'etl'; use the run command "
            "for a complete study workflow."
        )


def _definition(config: dict[str, Any], repo_root: Path):
    workflow_cfg = config.get("workflow") or {}
    entrypoint = workflow_cfg.get("entrypoint")
    if not entrypoint:
        raise ValueError("Configuration requires workflow.entrypoint")
    return _load_entrypoint(str(entrypoint))(config, repo_root)


def _run_dir(config: dict[str, Any], repo_root: Path, run_id: str) -> Path:
    workflow_cfg = config.get("workflow") or {}
    output_root = Path(str(workflow_cfg.get("output_root", "data/workflows")))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    return output_root / str(workflow_cfg["id"]) / run_id


def _selected_stages(definition, target: str | None):
    ordered = validate_workflow(definition)
    if target is None:
        return ordered

    names = {target}
    stage_map = definition.stage_map()
    if target not in stage_map:
        raise ValueError(f"Unknown target stage {target!r}")

    def add_dependencies(name: str) -> None:
        for dependency in stage_map[name].dependencies:
            if dependency not in names:
                names.add(dependency)
                add_dependencies(dependency)

    add_dependencies(target)
    return [stage for stage in ordered if stage.name in names]


def _dependency_summary(dependencies: tuple[str, ...]) -> str:
    """Return a compact dependency label suitable for a terminal table."""
    if not dependencies:
        return "-"
    if len(dependencies) <= 2:
        return ", ".join(dependencies)

    leaf_names = {name.rsplit(".", 1)[-1] for name in dependencies}
    if len(leaf_names) == 1:
        return f"{len(dependencies)} {leaf_names.pop()} stages"
    if all(name.startswith("panel.") for name in dependencies):
        return f"{len(dependencies)} panel stages"
    return f"{len(dependencies)} upstream stages"


def _wrap_stage_name(name: str, width: int) -> list[str]:
    """Wrap dotted stage names at component boundaries when possible."""
    if len(name) <= width:
        return [name]

    lines: list[str] = []
    current = ""
    for component in name.split("."):
        candidate = f"{current}.{component}" if current else component
        if current and len(candidate) > width:
            lines.append(current)
            current = component
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _wrap_cell(value: str, width: int) -> list[str]:
    return textwrap.wrap(value, width=width, break_long_words=False) or [""]


def _wrap_dependencies(value: str, width: int) -> list[str]:
    """Wrap individual stage dependencies at dotted-name boundaries."""
    names = value.split(", ")
    if len(names) == 1:
        return _wrap_stage_name(value, width)

    lines: list[str] = []
    for index, name in enumerate(names):
        suffix = "," if index < len(names) - 1 else ""
        if "." in name:
            wrapped = _wrap_stage_name(name, width)
        else:
            wrapped = _wrap_cell(name, width)
        wrapped[-1] += suffix
        lines.extend(wrapped)
    return lines


def _print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Print a compact, terminal-width-aware table without extra dependencies."""
    terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    number_width = max(2, len(headers[0]), *(len(row[0]) for row in rows))
    stage_width = min(
        42,
        max(22, min(max(len(row[1]) for row in rows), terminal_width // 3)),
    )
    dependency_width = min(
        30,
        max(18, min(max(len(row[2]) for row in rows), terminal_width // 4)),
    )
    description_width = max(
        24, terminal_width - number_width - stage_width - dependency_width - 6
    )
    widths = (number_width, stage_width, dependency_width, description_width)

    def render(values: tuple[str, ...]) -> None:
        cells = (
            _wrap_cell(values[0], widths[0]),
            _wrap_stage_name(values[1], widths[1]),
            _wrap_dependencies(values[2], widths[2]),
            _wrap_cell(values[3], widths[3]),
        )
        for index in range(max(len(cell) for cell in cells)):
            print(
                f"{cells[0][index] if index < len(cells[0]) else '':>{widths[0]}}  "
                f"{cells[1][index] if index < len(cells[1]) else '':<{widths[1]}}  "
                f"{cells[2][index] if index < len(cells[2]) else '':<{widths[2]}}  "
                f"{cells[3][index] if index < len(cells[3]) else ''}"
            )

    render(headers)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        render(row)


def _print_plan(definition, target: str | None) -> None:
    ordered = _selected_stages(definition, target)
    target_label = "all stages" if target is None else f"{target} and dependencies"
    print("Workflow plan")
    print("=" * len("Workflow plan"))
    print(
        f"{definition.workflow_id} | schema {definition.version} | "
        f"{len(ordered)} stage{'s' if len(ordered) != 1 else ''}"
    )
    print(f"Target: {target_label}\n")

    rows = [
        (
            str(index),
            stage.name,
            _dependency_summary(stage.dependencies),
            stage.description or "-",
        )
        for index, stage in enumerate(ordered, 1)
    ]
    _print_table(("#", "Stage", "Depends on", "Description"), rows)


def _status(run_dir: Path) -> int:
    path = run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print("Workflow status")
    print("=" * len("Workflow status"))
    print(f"{manifest['workflow_id']} / {manifest['run_id']}: {manifest['status']}\n")
    rows = [
        (
            str(index),
            name,
            str(record["status"]),
            str(record.get("description") or "-"),
        )
        for index, (name, record) in enumerate(manifest["stages"].items(), 1)
    ]
    _print_table(("#", "Stage", "Status", "Description"), rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multifid-workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Validate and display the stage graph")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--target")
    run = subparsers.add_parser("run", help="Execute or resume a workflow")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--target")
    etl = subparsers.add_parser("etl", help="Execute or resume an ETL-only workflow")
    etl.add_argument("--config", type=Path, required=True)
    etl.add_argument("--run-id", required=True)
    etl.add_argument(
        "--input-dir",
        type=Path,
        help="Override inputs.raw_dir with this raw campaign directory",
    )
    status = subparsers.add_parser("status", help="Display a saved run manifest")
    status.add_argument("--run-dir", type=Path, required=True)
    publish = subparsers.add_parser(
        "publish", help="Publish or check declared docs artifacts"
    )
    publish.add_argument("--run-dir", type=Path, required=True)
    publish.add_argument("--check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            return _status(args.run_dir.resolve())
        if args.command == "publish":
            run_dir = args.run_dir.resolve()
            config = json.loads(
                (run_dir / "resolved_config.json").read_text(encoding="utf-8")
            )
            repo_root = _repo_root(run_dir)
            definition = _definition(config, repo_root)
            WorkflowRunner(
                definition, config=config, repo_root=repo_root, run_dir=run_dir
            ).publish(check=args.check)
            return 0
        config_path = args.config.resolve()
        config = load_config(config_path)
        repo_root = _repo_root(config_path.parent)
        if args.command == "etl":
            config = _with_etl_input_dir(config, args.input_dir, repo_root)
            _require_etl_workflow(config)
        definition = _definition(config, repo_root)
        if args.command == "plan":
            _print_plan(definition, args.target)
            return 0
        run_dir = _run_dir(config, repo_root, args.run_id)
        WorkflowRunner(
            definition, config=config, repo_root=repo_root, run_dir=run_dir
        ).run(target=getattr(args, "target", None))
        print(run_dir)
        return 0
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"multifid-workflow: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
