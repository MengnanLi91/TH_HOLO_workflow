"""Command-line interface for planning and running study workflows."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from workflows.runner import WorkflowRunner, _target_stages, validate_workflow


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


def apply_config_overrides(
    config: dict[str, Any], overrides: list[str] | None
) -> dict[str, Any]:
    """Apply repeatable TOML-scalar overrides to existing config leaves."""
    resolved = copy.deepcopy(config)
    for expression in overrides or []:
        if "=" not in expression:
            raise ValueError(
                f"Invalid --set {expression!r}; expected existing.path=value"
            )
        dot_path, raw_value = expression.split("=", 1)
        keys = dot_path.split(".")
        if not dot_path or any(not key for key in keys):
            raise ValueError(f"Invalid --set path {dot_path!r}")
        try:
            value = tomllib.loads(f"value = {raw_value}\n")["value"]
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"Invalid TOML scalar for --set {dot_path}: {raw_value}"
            ) from exc
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"--set {dot_path} must use a TOML scalar value")

        node: Any = resolved
        for key in keys[:-1]:
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"Unknown --set path {dot_path!r}")
            node = node[key]
        leaf = keys[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(f"Unknown --set path {dot_path!r}")
        if not isinstance(node[leaf], (str, int, float, bool)):
            raise ValueError(f"--set {dot_path} cannot replace a table or array")
        node[leaf] = value
    return resolved


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


def _dependency_summary(dependencies: tuple[str, ...]) -> str:
    """Return a compact dependency label suitable for a terminal table."""
    if not dependencies:
        return "-"
    if len(dependencies) <= 2:
        return ", ".join(dependencies)

    leaf_names = {name.rsplit(".", 1)[-1] for name in dependencies}
    if len(leaf_names) == 1:
        return f"{len(dependencies)} {leaf_names.pop()}"
    return f"{len(dependencies)} upstream"


_STATUS_STYLES = {
    "pending": "dim",
    "running": "bold cyan",
    "succeeded": "bold green",
    "partial": "bold yellow",
    "failed": "bold red",
    "skipped": "bold magenta",
    "completed": "bold green",
    "completed_with_partial_results": "bold yellow",
    "partial_run": "bold yellow",
}
_STATUS_SYMBOLS = {
    "pending": "○",
    "running": "●",
    "succeeded": "✓",
    "partial": "!",
    "failed": "✗",
    "skipped": "–",
    "completed": "✓",
    "completed_with_partial_results": "!",
    "partial_run": "!",
}


def _status_text(status: str) -> Text:
    """Return a consistently styled, human-readable workflow status."""
    label = status.replace("_", " ").title()
    marker = _STATUS_SYMBOLS.get(status, "?")
    return Text(f"{marker} {label}", style=_STATUS_STYLES.get(status, "bold"))


def _stage_counts(records: dict[str, Any]) -> Text:
    """Return a compact color-coded count of stages by state."""
    counts = Counter(
        str(record.get("status", "unknown")) for record in records.values()
    )
    parts: list[Text] = []
    for status in sorted(counts):
        if parts:
            parts.append(Text("  "))
        value = _status_text(status)
        value.append(f" {counts[status]}")
        parts.append(value)
    return Text.assemble(*parts) if parts else Text("No stages", style="dim")


def _summary_table(rows: list[tuple[str, Text | str]]) -> Table:
    """Build the small key/value table used in CLI summary panels."""
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold bright_blue", no_wrap=True)
    table.add_column()
    for label, value in rows:
        table.add_row(label, value)
    return table


def _stage_table(*, status: bool) -> Table:
    """Build the shared stage table for workflow plans and saved runs."""
    table = Table(
        box=box.SIMPLE_HEAVY,
        header_style="bold bright_blue",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("#", justify="right", style="dim", width=3, no_wrap=True)
    table.add_column(
        "Stage",
        style="cyan",
        width=28,
        no_wrap=True,
        overflow="ellipsis",
    )
    if status:
        table.add_column("State", no_wrap=True)
    else:
        table.add_column(
            "Depends on",
            style="magenta",
            width=19,
            no_wrap=True,
            overflow="ellipsis",
        )
    table.add_column(
        "Description", min_width=18, ratio=1, no_wrap=True, overflow="ellipsis"
    )
    return table


def _workflow_tree(definition, ordered) -> Tree:
    """Return a dependency tree with explicit references for shared DAG edges."""
    numbers = {stage.name: index for index, stage in enumerate(ordered, 1)}
    names = set(numbers)
    children = {stage.name: [] for stage in ordered}
    roots = []
    extra_dependencies: dict[str, list[str]] = {}
    for stage in ordered:
        dependencies = [name for name in stage.dependencies if name in names]
        if not dependencies:
            roots.append(stage)
            continue
        children[dependencies[0]].append(stage)
        extra_dependencies[stage.name] = dependencies[1:]

    root = Tree(
        Text(
            f"{definition.workflow_id} · {len(ordered)} selected stages",
            style="bold cyan",
        ),
        guide_style="bold bright_blue",
    )

    def add_stage(parent: Tree, stage) -> None:
        label = Text(f"#{numbers[stage.name]:02d} ", style="bold bright_yellow")
        label.append(stage.name, style="bold bright_cyan")
        additional = extra_dependencies.get(stage.name, [])
        if additional:
            label.append("  also needs ", style="bold bright_magenta")
            label.append(
                ", ".join(f"#{numbers[name]}" for name in additional),
                style="bold bright_yellow",
            )
        node = parent.add(label, guide_style="bold bright_blue")
        for child in children[stage.name]:
            add_stage(node, child)

    for stage in roots:
        add_stage(root, stage)
    return root


def _print_plan(definition, target: str | None, *, tree: bool = True) -> None:
    ordered = _target_stages(validate_workflow(definition), target)
    console = Console()
    target_label = "All stages" if target is None else f"{target} and dependencies"
    console.print(
        Panel(
            _summary_table(
                [
                    ("Workflow", Text(definition.workflow_id, style="bold cyan")),
                    ("Schema", str(definition.version)),
                    ("Stages", f"{len(ordered)} selected"),
                    ("Target", Text(target_label, style="cyan")),
                    ("Mode", Text("Validation only; no commands run", style="dim")),
                ]
            ),
            title="[bold bright_blue]Workflow plan[/bold bright_blue]",
            border_style="bright_blue",
            expand=False,
        )
    )
    if tree:
        console.print(
            "[dim]Each stage appears once beneath its first dependency; "
            "shared DAG edges are marked as 'also needs'.[/]"
        )
        console.print(_workflow_tree(definition, ordered))
        return

    table = _stage_table(status=False)
    for index, stage in enumerate(ordered, 1):
        table.add_row(
            str(index),
            Text(stage.name),
            Text(_dependency_summary(stage.dependencies)),
            Text(
                stage.description or "-", style="dim" if not stage.description else ""
            ),
        )
    console.print(table)


def _status(run_dir: Path) -> int:
    path = run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    console = Console()
    overall_status = str(manifest["status"])
    console.print(
        Panel(
            _summary_table(
                [
                    ("Workflow", Text(str(manifest["workflow_id"]), style="bold cyan")),
                    ("Run ID", Text(str(manifest["run_id"]), style="cyan")),
                    ("Overall", _status_text(overall_status)),
                    ("Stages", _stage_counts(manifest["stages"])),
                    ("Updated", str(manifest.get("updated_utc", "-"))),
                ]
            ),
            title="[bold bright_blue]Workflow status[/bold bright_blue]",
            border_style=_STATUS_STYLES.get(overall_status, "bright_blue"),
            expand=False,
        )
    )
    table = _stage_table(status=True)
    for index, (name, record) in enumerate(manifest["stages"].items(), 1):
        description = str(record.get("description") or "-")
        if record.get("error"):
            description = f"{description}\nError: {record['error']}"
        table.add_row(
            str(index),
            Text(name),
            _status_text(str(record.get("status", "unknown"))),
            Text(description, style="dim" if description == "-" else ""),
        )
    console.print(table)
    return 0


def _run_with_progress(runner: WorkflowRunner, *, target: str | None) -> dict[str, Any]:
    """Run selected stages with a Rich progress bar driven by runner events."""
    selected = _target_stages(runner.ordered, target)
    complete_label = "Target complete" if target is not None else "Workflow complete"
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=Console(),
        transient=False,
    ) as progress:
        task = progress.add_task("Preparing workflow", total=len(selected))

        def on_stage(stage, state: str) -> None:
            if state == "running":
                progress.update(task, description=f"[cyan]Running[/] {stage.name}")
                return
            if state == "failed":
                progress.update(task, description=f"[bold red]Failed[/] {stage.name}")
                return

            labels = {
                "reused": "[dim]Reusing[/]",
                "succeeded": "[green]Completed[/]",
                "partial": "[yellow]Partial[/]",
            }
            progress.advance(task)
            progress.update(task, description=f"{labels[state]} {stage.name}")

        manifest = runner.run(target=target, on_stage=on_stage)
        if manifest["status"] == "completed_with_partial_results":
            final_label = f"[yellow]! {complete_label} with partial results[/]"
        elif manifest["status"] == "partial_run":
            final_label = (
                f"[yellow]! {complete_label}; workflow remains partially run[/]"
            )
        elif manifest["status"] == "failed":
            final_label = "[bold red]✗ Workflow remains failed[/]"
        else:
            final_label = f"[green]✓ {complete_label}[/]"
        progress.update(task, completed=len(selected), description=final_label)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multifid-workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Validate and display the stage graph")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--target")
    plan.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="EXISTING.PATH=VALUE",
    )
    plan.add_argument(
        "--table",
        action="store_true",
        help="Display the stage graph as a numbered table instead of a dependency tree",
    )
    run = subparsers.add_parser("run", help="Execute or resume a workflow")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--target")
    run.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="EXISTING.PATH=VALUE",
    )
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
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            raw_repo_root = manifest.get("repo_root")
            if not isinstance(raw_repo_root, str) or not raw_repo_root.strip():
                raise ValueError("Run manifest requires a non-empty repo_root")
            recorded_repo_root = Path(raw_repo_root).expanduser()
            if not recorded_repo_root.exists():
                raise FileNotFoundError(
                    f"Run manifest repo_root does not exist: {recorded_repo_root}"
                )
            recorded_repo_root = recorded_repo_root.resolve()
            repo_root = _repo_root(recorded_repo_root)
            if repo_root != recorded_repo_root:
                raise ValueError(
                    f"Run manifest repo_root is not a repository root: {recorded_repo_root}"
                )
            config = json.loads(
                (run_dir / "resolved_config.json").read_text(encoding="utf-8")
            )
            definition = _definition(config, repo_root)
            WorkflowRunner(
                definition, config=config, repo_root=repo_root, run_dir=run_dir
            ).publish(check=args.check)
            return 0
        config_path = args.config.resolve()
        config = load_config(config_path)
        repo_root = _repo_root(config_path.parent)
        if args.command in {"plan", "run"}:
            config = apply_config_overrides(config, args.set_overrides)
        if args.command == "etl":
            config = _with_etl_input_dir(config, args.input_dir, repo_root)
            _require_etl_workflow(config)
        definition = _definition(config, repo_root)
        if args.command == "plan":
            _print_plan(definition, args.target, tree=not args.table)
            return 0
        run_dir = _run_dir(config, repo_root, args.run_id)
        runner = WorkflowRunner(
            definition, config=config, repo_root=repo_root, run_dir=run_dir
        )
        if args.command == "run":
            _run_with_progress(runner, target=args.target)
        else:
            runner.run(target=getattr(args, "target", None))
        print(run_dir)
        return 0
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"multifid-workflow: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
