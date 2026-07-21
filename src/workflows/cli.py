"""Command-line interface for planning and running study workflows."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
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


def _print_plan(definition, target: str | None) -> None:
    ordered = validate_workflow(definition)
    if target:
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
        ordered = [stage for stage in ordered if stage.name in names]
    print(f"Workflow: {definition.workflow_id} (schema {definition.version})")
    for index, stage in enumerate(ordered, 1):
        dependencies = ", ".join(stage.dependencies) or "none"
        suffix = f" — {stage.description}" if stage.description else ""
        print(f"{index:>2}. {stage.name} [depends: {dependencies}]{suffix}")


def _status(run_dir: Path) -> int:
    path = run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(f"{manifest['workflow_id']} / {manifest['run_id']}: {manifest['status']}")
    for name, record in manifest["stages"].items():
        print(f"  {record['status']:<10} {name}")
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
        definition = _definition(config, repo_root)
        if args.command == "plan":
            _print_plan(definition, args.target)
            return 0
        run_dir = _run_dir(config, repo_root, args.run_id)
        WorkflowRunner(
            definition, config=config, repo_root=repo_root, run_dir=run_dir
        ).run(target=args.target)
        print(run_dir)
        return 0
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"multifid-workflow: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
