"""DAG validation, manifests, and resumable workflow execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workflows.core import Artifact, RunContext, Stage, StageResult, WorkflowDefinition
from workflows.executors import build_executors
from workflows.fingerprint import fingerprint_path, load_cache, write_cache

MANIFEST_SCHEMA = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_version(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_root), *args],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).strip()
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ):
            return ""

    def run_bytes(*args: str) -> bytes:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_root), *args],
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ):
            return b""

    status = run("status", "--porcelain")
    diff = run("diff", "--binary", "HEAD")
    dirty_digest = hashlib.sha256()
    dirty_digest.update(status.encode("utf-8"))
    dirty_digest.update(b"\0")
    dirty_digest.update(diff.encode("utf-8"))
    for raw_path in run_bytes("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        path = repo_root / relative
        dirty_digest.update(raw_path)
        dirty_digest.update(b"\0")
        if path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    dirty_digest.update(chunk)
    return {
        "sha": run("rev-parse", "HEAD") or None,
        "dirty": bool(status),
        "dirty_sha256": dirty_digest.hexdigest() if status else None,
    }


def validate_workflow(definition: WorkflowDefinition) -> list[Stage]:
    """Validate ``definition`` and return its dependency-ordered stages."""
    stage_map = definition.stage_map()
    if len(stage_map) != len(definition.stages):
        raise ValueError("Workflow stage names must be unique")
    for stage in definition.stages:
        if not stage.name or stage.name.strip() != stage.name:
            raise ValueError(f"Invalid stage name {stage.name!r}")
        missing = sorted(set(stage.dependencies) - set(stage_map))
        if missing:
            raise ValueError(f"Stage {stage.name!r} has missing dependencies: {missing}")

    ordered: list[Stage] = []
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(name: str) -> None:
        if name in permanent:
            return
        if name in temporary:
            raise ValueError(f"Workflow contains a dependency cycle at {name!r}")
        temporary.add(name)
        for dependency in stage_map[name].dependencies:
            visit(dependency)
        temporary.remove(name)
        permanent.add(name)
        ordered.append(stage_map[name])

    for stage in definition.stages:
        visit(stage.name)
    return ordered


def _target_stages(ordered: list[Stage], target: str | None) -> list[Stage]:
    if target is None:
        return ordered
    stage_map = {stage.name: stage for stage in ordered}
    if target not in stage_map:
        raise ValueError(f"Unknown target stage {target!r}; available: {sorted(stage_map)}")
    needed: set[str] = set()

    def add(name: str) -> None:
        if name in needed:
            return
        needed.add(name)
        for dependency in stage_map[name].dependencies:
            add(dependency)

    add(target)
    return [stage for stage in ordered if stage.name in needed]


def _artifact_record(artifact: Artifact, run_dir: Path) -> dict[str, Any]:
    raw_path = Path(artifact.path)
    path = raw_path if raw_path.is_absolute() else run_dir / raw_path
    fingerprint, _ = fingerprint_path(path)
    try:
        recorded_path = path.relative_to(run_dir).as_posix()
    except ValueError:
        recorded_path = str(path)
    return {"path": recorded_path, "label": artifact.label, "fingerprint": fingerprint}


def _artifacts_valid(records: list[dict[str, Any]], run_dir: Path) -> bool:
    for record in records:
        raw = Path(record["path"])
        path = raw if raw.is_absolute() else run_dir / raw
        if not path.exists():
            return False
        current, _ = fingerprint_path(path)
        if current["sha256"] != record.get("fingerprint", {}).get("sha256"):
            return False
    return True


def _dependency_fingerprints(manifest: dict[str, Any], stage: Stage) -> dict[str, str]:
    """Hash dependency outputs so changed upstream artifacts invalidate a stage."""
    fingerprints: dict[str, str] = {}
    for name in stage.dependencies:
        record = manifest["stages"][name]
        value = {
            "status": record.get("status"),
            "artifacts": record.get("artifacts") or [],
        }
        fingerprints[name] = _canonical_hash(value)
    return fingerprints


class WorkflowRunner:
    """Execute a :class:`workflows.core.WorkflowDefinition` with a manifest."""

    def __init__(
        self,
        definition: WorkflowDefinition,
        *,
        config: dict[str, Any],
        repo_root: Path,
        run_dir: Path,
    ):
        self.definition = definition
        self.config = config
        self.repo_root = repo_root.resolve()
        self.run_dir = run_dir.resolve()
        self.ordered = validate_workflow(definition)
        self.executors = build_executors(config, self.repo_root)
        self.manifest_path = self.run_dir / "run_manifest.json"

    def _input_fingerprints(self) -> list[dict[str, Any]]:
        if self.definition.input_paths is not None:
            paths = list(self.definition.input_paths(self.config, self.repo_root))
        else:
            paths = list((self.config.get("workflow") or {}).get("fingerprint_paths") or [])
        cache_path = self.run_dir / "fingerprint_cache.json"
        cache = load_cache(cache_path)
        refreshed: dict[str, Any] = {}
        records = []
        for raw in paths:
            formatted = str(raw).format(repo_root=self.repo_root)
            path = Path(formatted)
            if not path.is_absolute():
                path = self.repo_root / path
            record, entries = fingerprint_path(path, cache=cache)
            refreshed.update(entries)
            records.append(record)
        if paths:
            write_cache(cache_path, refreshed)
        return records

    def _binding(self) -> dict[str, Any]:
        return {
            "config_sha256": _canonical_hash(self.config),
            "code": _git_version(self.repo_root),
            "inputs": self._input_fingerprints(),
        }

    def _new_manifest(self, binding: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA,
            "workflow_id": self.definition.workflow_id,
            "workflow_version": self.definition.version,
            "run_id": self.run_dir.name,
            "repo_root": str(self.repo_root),
            "created_utc": _now(),
            "updated_utc": _now(),
            "status": "pending",
            "binding": binding,
            "stages": {
                stage.name: {
                    "status": "pending",
                    "dependencies": list(stage.dependencies),
                    "description": stage.description,
                    "artifacts": [],
                }
                for stage in self.ordered
            },
        }

    def _load_or_create_manifest(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        binding = self._binding()
        if self.manifest_path.is_file():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("binding") != binding:
                raise ValueError(
                    "Run directory is bound to different configuration, code, or inputs; "
                    "choose a new --run-id."
                )
            return manifest
        manifest = self._new_manifest(binding)
        _atomic_json(self.manifest_path, manifest)
        _atomic_json(self.run_dir / "resolved_config.json", self.config)
        return manifest

    def run(
        self,
        *,
        target: str | None = None,
        on_stage: Callable[[Stage, str], None] | None = None,
    ) -> dict[str, Any]:
        """Execute or resume stages, optionally reporting their lifecycle events.

        ``on_stage`` receives each selected stage and one of ``running``,
        ``reused``, ``succeeded``, ``partial``, or ``failed``. Reused stages
        passed their recorded artifact and semantic validation checks.
        """
        manifest = self._load_or_create_manifest()
        selected = _target_stages(self.ordered, target)
        manifest["status"] = "running"
        manifest["updated_utc"] = _now()
        _atomic_json(self.manifest_path, manifest)
        any_partial = False
        for stage in selected:
            record = manifest["stages"][stage.name]
            dependency_fingerprints = _dependency_fingerprints(manifest, stage)
            context = RunContext(
                repo_root=self.repo_root,
                run_dir=self.run_dir,
                config=self.config,
                stage_name=stage.name,
                executors=self.executors,
            )
            if (
                record.get("status") in ("succeeded", "partial")
                and _artifacts_valid(record.get("artifacts") or [], self.run_dir)
                and record.get("dependency_fingerprints", {}) == dependency_fingerprints
                and (stage.validator is None or stage.validator(context))
            ):
                any_partial = any_partial or record["status"] == "partial"
                if on_stage is not None:
                    on_stage(stage, "reused")
                continue
            record.update({"status": "running", "started_utc": _now(), "error": None})
            _atomic_json(self.manifest_path, manifest)
            if on_stage is not None:
                on_stage(stage, "running")
            try:
                result = stage.action(context) or StageResult()
                if stage.validator is not None and not stage.validator(context):
                    raise ValueError(f"Stage {stage.name!r} outputs failed semantic validation")
                artifact_records = [
                    _artifact_record(artifact, self.run_dir) for artifact in result.artifacts
                ]
                record.update(
                    {
                        "status": result.status,
                        "completed_utc": _now(),
                        "artifacts": artifact_records,
                        "dependency_fingerprints": dependency_fingerprints,
                        "details": result.details,
                    }
                )
                any_partial = any_partial or result.status == "partial"
            except Exception as exc:
                record.update({"status": "failed", "completed_utc": _now(), "error": str(exc)})
                manifest["status"] = "failed"
                manifest["updated_utc"] = _now()
                _atomic_json(self.manifest_path, manifest)
                if on_stage is not None:
                    on_stage(stage, "failed")
                raise
            manifest["updated_utc"] = _now()
            _atomic_json(self.manifest_path, manifest)
            if on_stage is not None:
                on_stage(stage, result.status)
        all_terminal = all(
            item.get("status") in ("succeeded", "partial") for item in manifest["stages"].values()
        )
        if all_terminal:
            manifest["status"] = "completed_with_partial_results" if any_partial else "completed"
        else:
            manifest["status"] = "partial_run"
        manifest["updated_utc"] = _now()
        _atomic_json(self.manifest_path, manifest)
        return manifest

    def publish(self, *, check: bool) -> StageResult:
        if self.definition.publisher is None:
            raise ValueError(f"Workflow {self.definition.workflow_id!r} has no publisher")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Run manifest not found: {self.manifest_path}")
        context = RunContext(
            repo_root=self.repo_root,
            run_dir=self.run_dir,
            config=self.config,
            stage_name="publish",
            executors=self.executors,
        )
        return self.definition.publisher(context, check) or StageResult()
