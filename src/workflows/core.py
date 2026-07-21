"""Public workflow contracts shared by the runner and case definitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

StageStatus = Literal["pending", "running", "succeeded", "partial", "failed", "skipped"]


@dataclass(frozen=True)
class Artifact:
    """A file or directory produced by a stage.

    Relative paths are resolved beneath the run directory. External absolute
    paths are allowed for declared inputs but should not be used for generated
    outputs.
    """

    path: str | Path
    label: str = ""


@dataclass(frozen=True)
class Command:
    """One subprocess invocation executed without a shell."""

    argv: tuple[str, ...]
    executor: str = "local"
    cwd: str | Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    label: str = "command"


@dataclass(frozen=True)
class CommandResult:
    """Recorded result of a command invocation."""

    argv: tuple[str, ...]
    returncode: int
    log_path: Path
    command_path: Path


@dataclass
class StageResult:
    """Result returned by a stage action."""

    status: Literal["succeeded", "partial"] = "succeeded"
    artifacts: list[Artifact] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class Executor(Protocol):
    """Execution backend used by :class:`RunContext`."""

    def execute(
        self, command: Command, *, log_path: Path, command_path: Path
    ) -> CommandResult:
        """Execute ``command`` and return its recorded result."""


@dataclass
class RunContext:
    """Runtime state supplied to each case-owned stage action."""

    repo_root: Path
    run_dir: Path
    config: dict[str, Any]
    stage_name: str
    executors: Mapping[str, Executor]
    _command_index: int = 0

    def resolve(self, path: str | Path) -> Path:
        """Resolve a generated-artifact path beneath the run directory."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.run_dir / candidate

    def run(self, command: Command, *, check: bool = True) -> CommandResult:
        """Execute and record a command through its named executor."""
        if command.executor not in self.executors:
            raise KeyError(
                f"Unknown executor {command.executor!r}; available: {sorted(self.executors)}"
            )
        self._command_index += 1
        safe_stage = self.stage_name.replace("/", "_")
        safe_label = command.label.replace("/", "_")
        stem = f"{safe_stage}.{self._command_index:02d}.{safe_label}"
        log_dir = self.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        result = self.executors[command.executor].execute(
            command,
            log_path=log_dir / f"{stem}.log",
            command_path=log_dir / f"{stem}.command.txt",
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {result.returncode}; see {result.log_path}"
            )
        return result


StageAction = Callable[[RunContext], StageResult | None]
StageValidator = Callable[[RunContext], bool]
Publisher = Callable[[RunContext, bool], StageResult | None]
InputPaths = Callable[[dict[str, Any], Path], list[str | Path]]


@dataclass(frozen=True)
class Stage:
    """A case-owned unit of work in a workflow DAG."""

    name: str
    action: StageAction
    dependencies: tuple[str, ...] = ()
    description: str = ""
    validator: StageValidator | None = None


@dataclass(frozen=True)
class WorkflowDefinition:
    """Complete case workflow returned by ``build_workflow``."""

    workflow_id: str
    version: int
    stages: tuple[Stage, ...]
    publisher: Publisher | None = None
    input_paths: InputPaths | None = None

    def stage_map(self) -> dict[str, Stage]:
        return {stage.name: stage for stage in self.stages}
