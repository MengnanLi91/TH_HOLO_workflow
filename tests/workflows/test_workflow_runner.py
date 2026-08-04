from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows import (
    Artifact,
    Command,
    Stage,
    StageResult,
    WorkflowDefinition,
    WorkflowRunner,
)
from workflows.core import RunContext
from workflows.executors import ApptainerExecutor, build_executors
from workflows.runner import validate_workflow


def test_validate_workflow_orders_dependencies_and_rejects_cycles():
    noop = lambda _context: StageResult()  # noqa: E731
    definition = WorkflowDefinition(
        workflow_id="example",
        version=1,
        stages=(
            Stage("report", noop, dependencies=("prepare",)),
            Stage("prepare", noop),
        ),
    )

    assert [stage.name for stage in validate_workflow(definition)] == [
        "prepare",
        "report",
    ]

    cyclic = WorkflowDefinition(
        workflow_id="bad",
        version=1,
        stages=(
            Stage("a", noop, dependencies=("b",)),
            Stage("b", noop, dependencies=("a",)),
        ),
    )
    with pytest.raises(ValueError, match="dependency cycle"):
        validate_workflow(cyclic)


def test_runner_resumes_valid_stages_and_reruns_changed_artifacts(tmp_path):
    calls = {"prepare": 0, "report": 0}

    def prepare(context: RunContext):
        calls["prepare"] += 1
        path = context.resolve("prepared.txt")
        path.write_text(f"prepared {calls['prepare']}\n", encoding="utf-8")
        return StageResult(artifacts=[Artifact("prepared.txt")])

    def report(context: RunContext):
        calls["report"] += 1
        path = context.resolve("report.txt")
        path.write_text("report\n", encoding="utf-8")
        return StageResult(status="partial", artifacts=[Artifact("report.txt")])

    definition = WorkflowDefinition(
        workflow_id="example",
        version=1,
        stages=(
            Stage("prepare", prepare),
            Stage("report", report, dependencies=("prepare",)),
        ),
    )
    runner = WorkflowRunner(
        definition,
        config={"workflow": {"id": "example"}},
        repo_root=Path(__file__).resolve().parents[2],
        run_dir=tmp_path / "run",
    )

    events: list[tuple[str, str]] = []
    manifest = runner.run(on_stage=lambda stage, state: events.append((stage.name, state)))
    assert manifest["status"] == "completed_with_partial_results"
    assert calls == {"prepare": 1, "report": 1}
    assert events == [
        ("prepare", "running"),
        ("prepare", "succeeded"),
        ("report", "running"),
        ("report", "partial"),
    ]

    events = []
    runner.run(on_stage=lambda stage, state: events.append((stage.name, state)))
    assert calls == {"prepare": 1, "report": 1}
    assert events == [("prepare", "reused"), ("report", "reused")]

    (tmp_path / "run" / "prepared.txt").write_text("changed\n", encoding="utf-8")
    targeted = runner.run(target="prepare")
    assert targeted["status"] == "completed_with_partial_results"
    assert targeted["stages"]["report"]["status"] == "partial"
    assert calls == {"prepare": 2, "report": 1}

    runner.run()
    assert calls == {"prepare": 2, "report": 2}


def test_runner_reruns_when_stage_validator_fails(tmp_path):
    calls = 0
    validator_calls = 0
    valid = True

    def produce(context: RunContext):
        nonlocal calls
        calls += 1
        context.resolve("value.txt").write_text(str(calls), encoding="utf-8")
        return StageResult(artifacts=[Artifact("value.txt")])

    def validate(_context: RunContext):
        nonlocal validator_calls
        validator_calls += 1
        return valid

    definition = WorkflowDefinition(
        "validated",
        1,
        (Stage("produce", produce, validator=validate),),
    )
    runner = WorkflowRunner(
        definition,
        config={"workflow": {"id": "validated"}},
        repo_root=Path(__file__).resolve().parents[2],
        run_dir=tmp_path / "run",
    )

    runner.run()
    assert validator_calls == 1
    runner.run()
    assert calls == 1
    assert validator_calls == 2

    valid = False
    with pytest.raises(ValueError, match="failed semantic validation"):
        runner.run()
    assert calls == 2
    assert validator_calls == 4


def test_runner_validates_outputs_before_marking_first_run_successful(tmp_path):
    def produce(context: RunContext):
        context.resolve("value.txt").write_text("invalid", encoding="utf-8")
        return StageResult(artifacts=[Artifact("value.txt")])

    definition = WorkflowDefinition(
        "validated",
        1,
        (
            Stage(
                "produce",
                produce,
                validator=lambda context: context.resolve("value.txt").read_text() == "valid",
            ),
        ),
    )
    runner = WorkflowRunner(
        definition,
        config={"workflow": {"id": "validated"}},
        repo_root=Path(__file__).resolve().parents[2],
        run_dir=tmp_path / "run",
    )

    with pytest.raises(ValueError, match="failed semantic validation"):
        runner.run()

    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["stages"]["produce"]["status"] == "failed"


def test_runner_records_failure_and_resumes_it(tmp_path):
    attempts = 0

    def flaky(context: RunContext):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("expected failure")
        path = context.resolve("done.txt")
        path.write_text("done\n", encoding="utf-8")
        return StageResult(artifacts=[Artifact("done.txt")])

    definition = WorkflowDefinition("flaky", 1, (Stage("flaky", flaky),))
    runner = WorkflowRunner(
        definition,
        config={"workflow": {"id": "flaky"}},
        repo_root=Path(__file__).resolve().parents[2],
        run_dir=tmp_path / "run",
    )

    events: list[tuple[str, str]] = []
    with pytest.raises(RuntimeError, match="expected failure"):
        runner.run(on_stage=lambda stage, state: events.append((stage.name, state)))
    failed = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert failed["stages"]["flaky"]["status"] == "failed"
    assert events == [("flaky", "running"), ("flaky", "failed")]

    completed = runner.run()
    assert completed["status"] == "completed"
    assert attempts == 2


def test_targeted_run_preserves_unselected_failure_status(tmp_path):
    def fail(_context: RunContext):
        raise RuntimeError("expected failure")

    definition = WorkflowDefinition(
        "failed-target",
        1,
        (
            Stage("prepare", lambda _context: None),
            Stage("report", fail, dependencies=("prepare",)),
        ),
    )
    runner = WorkflowRunner(
        definition,
        config={"workflow": {"id": "failed-target"}},
        repo_root=Path(__file__).resolve().parents[2],
        run_dir=tmp_path / "run",
    )

    with pytest.raises(RuntimeError, match="expected failure"):
        runner.run()

    targeted = runner.run(target="prepare")
    assert targeted["status"] == "failed"
    assert targeted["stages"]["report"]["status"] == "failed"


def test_initial_targeted_run_with_pending_stage_is_partial_run(tmp_path):
    definition = WorkflowDefinition(
        "initial-target",
        1,
        (
            Stage("prepare", lambda _context: None),
            Stage("report", lambda _context: None, dependencies=("prepare",)),
        ),
    )
    runner = WorkflowRunner(
        definition,
        config={"workflow": {"id": "initial-target"}},
        repo_root=Path(__file__).resolve().parents[2],
        run_dir=tmp_path / "run",
    )

    manifest = runner.run(target="prepare")

    assert manifest["status"] == "partial_run"
    assert manifest["stages"]["prepare"]["status"] == "succeeded"
    assert manifest["stages"]["report"]["status"] == "pending"


def test_runner_flushes_input_and_artifact_fingerprints_after_failure(tmp_path):
    input_path = tmp_path / "input.txt"
    input_path.write_text("input\n", encoding="utf-8")

    def produce(context: RunContext):
        context.resolve("artifact.txt").write_text("artifact\n", encoding="utf-8")
        return StageResult(artifacts=[Artifact("artifact.txt")])

    def fail(_context: RunContext):
        raise RuntimeError("stop after artifact")

    definition = WorkflowDefinition(
        "cached-failure",
        1,
        (
            Stage("produce", produce),
            Stage("fail", fail, dependencies=("produce",)),
        ),
        input_paths=lambda _config, _repo_root: [input_path],
    )
    run_dir = tmp_path / "run"
    runner = WorkflowRunner(
        definition,
        config={"workflow": {"id": "cached-failure"}},
        repo_root=tmp_path,
        run_dir=run_dir,
    )

    with pytest.raises(RuntimeError, match="stop after artifact"):
        runner.run()

    cache = json.loads((run_dir / "fingerprint_cache.json").read_text(encoding="utf-8"))
    assert str(input_path.resolve()) in cache
    assert str((run_dir / "artifact.txt").resolve()) in cache


def test_runner_reuses_persisted_artifact_fingerprint_cache(monkeypatch, tmp_path):
    calls = 0

    def produce(context: RunContext):
        nonlocal calls
        calls += 1
        context.resolve("artifact.txt").write_text("artifact\n", encoding="utf-8")
        return StageResult(artifacts=[Artifact("artifact.txt")])

    definition = WorkflowDefinition("cached-artifact", 1, (Stage("produce", produce),))
    config = {"workflow": {"id": "cached-artifact"}}
    run_dir = tmp_path / "run"
    WorkflowRunner(
        definition,
        config=config,
        repo_root=tmp_path,
        run_dir=run_dir,
    ).run()

    from workflows import fingerprint

    original_file_digest = fingerprint._file_digest
    cache_hits: list[bool] = []

    def record_cache_hit(path, cached):
        cache_hits.append(bool(cached and str(path.resolve()) in cached))
        return original_file_digest(path, cached)

    monkeypatch.setattr(fingerprint, "_file_digest", record_cache_hit)
    WorkflowRunner(
        definition,
        config=config,
        repo_root=tmp_path,
        run_dir=run_dir,
    ).run()

    assert calls == 1
    assert cache_hits == [True]


def test_interrupted_workflow_resumes_without_repeating_expensive_stage(tmp_path):
    calls = {"prepare": 0, "expensive": 0, "summarize": 0}

    def write_stage(name):
        def action(context: RunContext):
            calls[name] += 1
            context.resolve(f"{name}.txt").write_text(name, encoding="utf-8")
            return StageResult(artifacts=[Artifact(f"{name}.txt")])

        return action

    def summarize(context: RunContext):
        calls["summarize"] += 1
        if calls["summarize"] == 1:
            raise RuntimeError("simulated interruption")
        context.resolve("summary.txt").write_text("summary", encoding="utf-8")
        return StageResult(artifacts=[Artifact("summary.txt")])

    definition = WorkflowDefinition(
        "resume",
        1,
        (
            Stage("prepare", write_stage("prepare")),
            Stage("expensive", write_stage("expensive"), dependencies=("prepare",)),
            Stage("summarize", summarize, dependencies=("expensive",)),
        ),
    )
    assert [stage.name for stage in validate_workflow(definition)] == [
        "prepare",
        "expensive",
        "summarize",
    ]
    runner = WorkflowRunner(
        definition,
        config={"workflow": {"id": "resume"}},
        repo_root=Path(__file__).resolve().parents[2],
        run_dir=tmp_path / "run",
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.run()
    runner.run()

    assert calls == {"prepare": 1, "expensive": 1, "summarize": 2}


def test_runner_rejects_reusing_run_id_with_different_config(tmp_path):
    definition = WorkflowDefinition("example", 1, (Stage("noop", lambda _ctx: None),))
    repo_root = Path(__file__).resolve().parents[2]
    run_dir = tmp_path / "run"
    WorkflowRunner(
        definition,
        config={"workflow": {"id": "example"}, "value": 1},
        repo_root=repo_root,
        run_dir=run_dir,
    ).run()

    with pytest.raises(ValueError, match="choose a new --run-id"):
        WorkflowRunner(
            definition,
            config={"workflow": {"id": "example"}, "value": 2},
            repo_root=repo_root,
            run_dir=run_dir,
        ).run()


def test_local_executor_records_command_and_log(tmp_path):
    def command_stage(context: RunContext):
        result = context.run(
            Command(
                argv=(sys.executable, "-c", "print('workflow command')"),
                label="probe",
            )
        )
        assert result.returncode == 0
        return StageResult(artifacts=[Artifact(result.log_path)])

    definition = WorkflowDefinition("command", 1, (Stage("command", command_stage),))
    runner = WorkflowRunner(
        definition,
        config={"workflow": {"id": "command"}},
        repo_root=Path(__file__).resolve().parents[2],
        run_dir=tmp_path / "run",
    )

    runner.run()

    log = tmp_path / "run" / "logs" / "command.01.probe.log"
    command = tmp_path / "run" / "logs" / "command.01.probe.command.txt"
    assert log.read_text(encoding="utf-8") == "workflow command\n"
    assert "-c" in command.read_text(encoding="utf-8")


def test_apptainer_executor_constructs_argument_array(monkeypatch, tmp_path):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("workflows.executors.subprocess.run", fake_run)
    executor = ApptainerExecutor(
        repo_root=tmp_path,
        image=Path("/images/python.sif"),
        binds=(f"{tmp_path}:{tmp_path}",),
    )
    result = executor.execute(
        Command(
            argv=("python", "-c", "print('ok')"),
            cwd=tmp_path,
            env={"PYTHONPATH": str(tmp_path / "src")},
        ),
        log_path=tmp_path / "command.log",
        command_path=tmp_path / "command.txt",
    )

    assert result.returncode == 0
    assert captured["argv"][:2] == ["apptainer", "exec"]
    assert "/images/python.sif" in captured["argv"]
    assert captured["argv"][-3:] == ["python", "-c", "print('ok')"]
    assert "shell" not in captured["kwargs"]


def test_apptainer_image_environment_is_resolved_at_use_time(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKFLOW_TEST_IMAGE", "/images/from-environment.sif")
    executor = build_executors(
        {
            "executors": {
                "python": {
                    "kind": "apptainer",
                    "image": "/images/from-config.sif",
                    "image_env": "WORKFLOW_TEST_IMAGE",
                }
            }
        },
        tmp_path,
    )["python"]

    monkeypatch.delenv("WORKFLOW_TEST_IMAGE")
    assert isinstance(executor, ApptainerExecutor)
    assert executor.resolved_image() == Path("/images/from-config.sif")

    monkeypatch.setenv("WORKFLOW_TEST_IMAGE", "/images/set-after-construction.sif")
    assert executor.resolved_image() == Path("/images/set-after-construction.sif")

    monkeypatch.setenv("WORKFLOW_TEST_IMAGE", "/images/changed-after-construction.sif")
    assert executor.resolved_image() == Path("/images/changed-after-construction.sif")
