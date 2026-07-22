"""Subprocess executors used by the generic workflow runner."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from workflows.core import Command, CommandResult


def _write_command(path: Path, argv: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(shlex.join(argv) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class LocalExecutor:
    """Execute commands directly on the host."""

    repo_root: Path

    def execute(self, command: Command, *, log_path: Path, command_path: Path) -> CommandResult:
        argv = [str(value) for value in command.argv]
        cwd = Path(command.cwd) if command.cwd is not None else self.repo_root
        if not cwd.is_absolute():
            cwd = self.repo_root / cwd
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in command.env.items()})
        _write_command(command_path, argv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        return CommandResult(tuple(argv), completed.returncode, log_path, command_path)


@dataclass(frozen=True)
class ApptainerExecutor:
    """Execute commands inside a configured Apptainer image."""

    repo_root: Path
    image: Path | None = None
    image_env: str | None = None
    binds: tuple[str, ...] = ()
    use_nv: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def resolved_image(self) -> Path:
        raw = os.environ.get(self.image_env) if self.image_env else None
        image = Path(raw).expanduser() if raw else self.image
        if image is None:
            raise ValueError(
                "Apptainer executor requires an image; set "
                f"{self.image_env!r} or configure executors.<name>.image"
            )
        return image

    def execute(self, command: Command, *, log_path: Path, command_path: Path) -> CommandResult:
        cwd = Path(command.cwd) if command.cwd is not None else self.repo_root
        if not cwd.is_absolute():
            cwd = self.repo_root / cwd
        argv = ["apptainer", "exec"]
        if self.use_nv:
            argv.append("--nv")
        for bind in self.binds:
            argv.extend(("--bind", bind))
        argv.extend(("--pwd", str(cwd)))
        argv.extend(self.extra_args)
        argv.append(str(self.resolved_image()))
        if command.env:
            argv.append("env")
            argv.extend(f"{key}={value}" for key, value in command.env.items())
        argv.extend(str(value) for value in command.argv)
        _write_command(command_path, argv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                argv,
                cwd=self.repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        return CommandResult(tuple(argv), completed.returncode, log_path, command_path)


def build_executors(config: dict, repo_root: Path):
    """Construct named executors from a resolved workflow configuration."""
    executors: dict[str, LocalExecutor | ApptainerExecutor] = {"local": LocalExecutor(repo_root)}
    for name, raw in (config.get("executors") or {}).items():
        kind = str(raw.get("kind", "local"))
        if kind == "local":
            executors[name] = LocalExecutor(repo_root)
            continue
        if kind != "apptainer":
            raise ValueError(f"Unsupported executor kind {kind!r} for {name!r}")
        image = raw.get("image")
        image_env = raw.get("image_env")
        if image_env:
            image = os.environ.get(str(image_env), image)
        binds = list(raw.get("binds") or [])
        bind_env = raw.get("bind_env")
        if bind_env and os.environ.get(str(bind_env)):
            binds.append(os.environ[str(bind_env)])
        executors[name] = ApptainerExecutor(
            repo_root=repo_root,
            image=Path(str(image)).expanduser() if image else None,
            image_env=str(image_env) if image_env else None,
            binds=tuple(str(value).format(repo_root=repo_root) for value in binds),
            use_nv=bool(raw.get("use_nv", False)),
            extra_args=tuple(str(value) for value in raw.get("extra_args") or []),
        )
    return executors
