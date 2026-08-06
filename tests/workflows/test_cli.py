from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflows import Stage, WorkflowDefinition, WorkflowRunner
from workflows.cli import (
    _dependency_summary,
    _print_plan,
    _run_with_progress,
    _status,
    apply_config_overrides,
    build_parser,
    main,
)


def test_plan_table_option_uses_compact_dependency_summaries(capsys):
    noop = lambda _context: None  # noqa: E731
    exports = tuple(f"panel.case_{index}.export_closure" for index in range(3))
    definition = WorkflowDefinition(
        workflow_id="display_example",
        version=1,
        stages=(
            Stage("prepare", noop, description="prepare inputs"),
            *(
                Stage(name, noop, dependencies=("prepare",), description="export")
                for name in exports
            ),
            Stage("solve", noop, dependencies=exports, description="run solver"),
        ),
    )

    _print_plan(definition, target=None, tree=False)
    output = capsys.readouterr().out

    assert "Workflow plan" in output
    assert "display_example" in output
    assert "5 selected" in output
    assert "Depends on" in output
    assert "3 export_closure" in output
    assert "panel.case_0" in output
    assert "export_closure" in output
    assert "[depends:" not in output


def test_plan_tree_marks_shared_dependencies(capsys):
    noop = lambda _context: None  # noqa: E731
    definition = WorkflowDefinition(
        workflow_id="display_example",
        version=1,
        stages=(
            Stage("prepare", noop),
            Stage("select", noop, dependencies=("prepare",)),
            Stage("tune", noop, dependencies=("select",)),
            Stage("train", noop, dependencies=("prepare", "tune")),
        ),
    )

    _print_plan(definition, target=None)
    output = capsys.readouterr().out

    assert "display_example · 4 selected stages" in output
    assert "#01 prepare" in output
    assert "#04 train" in output
    assert "also needs #3" in output


def test_plan_progress_and_runner_share_target_selection(tmp_path, capsys):
    calls: list[str] = []

    def record(name):
        def action(_context):
            calls.append(name)

        return action

    definition = WorkflowDefinition(
        "target-example",
        1,
        (
            Stage("prepare", record("prepare")),
            Stage("unrelated", record("unrelated")),
            Stage("target", record("target"), dependencies=("prepare",)),
        ),
    )
    runner = WorkflowRunner(
        definition,
        config={"workflow": {"id": "target-example"}},
        repo_root=tmp_path,
        run_dir=tmp_path / "run",
    )

    _print_plan(definition, target="target")
    plan_output = capsys.readouterr().out
    assert "2 selected" in plan_output
    assert "#01 prepare" in plan_output
    assert "#02 target" in plan_output
    assert "unrelated" not in plan_output

    manifest = _run_with_progress(runner, target="target")
    progress_output = capsys.readouterr().out
    assert calls == ["prepare", "target"]
    assert "2/2" in progress_output
    assert manifest["stages"]["unrelated"]["status"] == "pending"

    expected = "Unknown target stage 'missing'; available: ['prepare', 'target', 'unrelated']"
    errors = []
    for invoke in (
        lambda: _print_plan(definition, target="missing"),
        lambda: _run_with_progress(runner, target="missing"),
        lambda: runner.run(target="missing"),
    ):
        with pytest.raises(ValueError) as exc_info:
            invoke()
        errors.append(str(exc_info.value))
    assert errors == [expected, expected, expected]


def test_dependency_summary_does_not_infer_case_semantics_from_stage_names():
    assert (
        _dependency_summary(
            ("panel.example.train", "panel.control.export", "panel.validation.check")
        )
        == "3 upstream"
    )


def test_plan_parser_defaults_to_tree_and_accepts_table_option():
    default = build_parser().parse_args(["plan", "--config", "workflow.toml"])
    table = build_parser().parse_args(["plan", "--config", "workflow.toml", "--table"])

    assert default.table is False
    assert table.table is True


def test_config_overrides_parse_toml_scalars_without_mutating_source():
    config = {"training": {"alpha": {"include_acceleration_head": True, "epochs": 5}}}

    resolved = apply_config_overrides(
        config,
        ["training.alpha.include_acceleration_head=false", "training.alpha.epochs=12"],
    )

    assert resolved["training"]["alpha"] == {
        "include_acceleration_head": False,
        "epochs": 12,
    }
    assert config["training"]["alpha"]["include_acceleration_head"] is True


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("training.missing=1", "Unknown"),
        ("training.alpha=1", "table or array"),
        ("training.alpha.epochs=[1, 2]", "TOML scalar"),
    ],
)
def test_config_overrides_reject_unknown_or_non_scalar_paths(override, message):
    config = {"training": {"alpha": {"epochs": 5}}}

    with pytest.raises((KeyError, ValueError), match=message):
        apply_config_overrides(config, [override])


def test_status_output_uses_color_coded_state_summary(tmp_path, capsys):
    run_dir = tmp_path / "display_example" / "run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        """{
  "workflow_id": "display_example",
  "run_id": "run-001",
  "status": "completed_with_partial_results",
  "updated_utc": "2026-07-22T12:00:00+00:00",
  "stages": {
    "prepare": {"status": "succeeded", "description": "Prepare inputs"},
    "summarize": {"status": "partial", "description": "Summarize results"},
    "publish": {"status": "failed", "description": "Publish docs", "error": "missing figure"}
  }
}
""",
        encoding="utf-8",
    )

    assert _status(run_dir) == 0
    output = capsys.readouterr().out

    assert "Workflow status" in output
    assert "Completed With Partial Results" in output
    assert "✓ Succeeded 1" in output
    assert "! Partial 1" in output
    assert "✗ Failed 1" in output
    assert "State" in output
    assert "missing figure" in output


def test_run_progress_tracks_reused_and_partial_stages(monkeypatch, capsys):
    noop = lambda _context: None  # noqa: E731
    definition = WorkflowDefinition(
        workflow_id="display_example",
        version=1,
        stages=(
            Stage("prepare", noop),
            Stage("summarize", noop, dependencies=("prepare",)),
        ),
    )

    class Runner:
        def __init__(self):
            self.definition = definition
            self.ordered = list(definition.stages)
            self.events: list[tuple[str, str]] = []

        def run(self, *, target, on_stage):
            on_stage(definition.stages[0], "reused")
            on_stage(definition.stages[1], "running")
            on_stage(definition.stages[1], "partial")
            self.events = [("prepare", "reused"), ("summarize", "partial")]
            return {"status": "completed_with_partial_results"}

    runner = Runner()
    monkeypatch.setattr(
        "workflows.cli.validate_workflow",
        lambda _definition: pytest.fail("progress display revalidated the workflow"),
    )
    _run_with_progress(runner, target=None)
    output = capsys.readouterr().out

    assert runner.events == [("prepare", "reused"), ("summarize", "partial")]
    assert "Workflow complete with partial results" in output


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("partial_run", "Target complete; workflow remains partially run"),
        ("failed", "Workflow remains failed"),
    ],
)
def test_run_progress_reports_noncomplete_overall_state(status, expected, capsys):
    definition = WorkflowDefinition(
        workflow_id="display_example",
        version=1,
        stages=(Stage("prepare", lambda _context: None),),
    )

    class Runner:
        ordered = list(definition.stages)

        @staticmethod
        def run(*, target, on_stage):
            on_stage(definition.stages[0], "reused")
            return {"status": status}

    _run_with_progress(Runner(), target="prepare")

    assert expected in capsys.readouterr().out


def test_publish_uses_manifest_repo_root_for_external_run_dir(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    run_dir = tmp_path / "external" / "run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"repo_root": str(repo_root)}), encoding="utf-8"
    )
    (run_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
    published = {}

    def publisher(context, check):
        published["repo_root"] = context.repo_root
        published["check"] = check

    definition = WorkflowDefinition(
        "publish-example",
        1,
        (Stage("prepare", lambda _context: None),),
        publisher=publisher,
    )
    monkeypatch.setattr("workflows.cli._definition", lambda _config, _repo_root: definition)

    assert main(["publish", "--run-dir", str(run_dir), "--check"]) == 0
    assert published == {"repo_root": repo_root, "check": True}


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({}, "requires a non-empty repo_root"),
        ({"repo_root": None}, "requires a non-empty repo_root"),
        ({"repo_root": "/path/that/does/not/exist"}, "repo_root does not exist"),
    ],
)
def test_publish_rejects_invalid_manifest_repo_root(manifest, message, tmp_path, capsys):
    run_dir = tmp_path / "run-001"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert main(["publish", "--run-dir", str(run_dir)]) == 2
    assert message in capsys.readouterr().err


def test_run_command_renders_stage_progress(monkeypatch, tmp_path, capsys):
    definition = WorkflowDefinition(
        workflow_id="display_example",
        version=1,
        stages=(Stage("prepare", lambda _context: None),),
    )
    config_path = tmp_path / "workflow.toml"
    config_path.write_text("[workflow]\nid = 'display_example'\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr("workflows.cli._repo_root", lambda _start: repo_root)
    monkeypatch.setattr("workflows.cli._definition", lambda _config, _repo_root: definition)
    monkeypatch.setattr("workflows.cli._run_dir", lambda _config, _repo_root, _run_id: run_dir)

    assert main(["run", "--config", str(config_path), "--run-id", "run-001"]) == 0
    output = capsys.readouterr().out

    assert "Workflow complete" in output
    assert str(run_dir) in output


def test_run_fingerprints_and_persists_resolved_overrides(monkeypatch, tmp_path):
    from workflows.runner import _canonical_hash

    definition = WorkflowDefinition(
        workflow_id="display_example",
        version=1,
        stages=(Stage("prepare", lambda _context: None),),
    )
    config_path = tmp_path / "workflow.toml"
    config_path.write_text(
        """[workflow]
id = "display_example"

[training.alpha]
include_acceleration_head = true
""",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr("workflows.cli._repo_root", lambda _start: repo_root)
    monkeypatch.setattr("workflows.cli._definition", lambda _config, _repo_root: definition)
    monkeypatch.setattr("workflows.cli._run_dir", lambda _config, _repo_root, _run_id: run_dir)

    assert (
        main(
            [
                "run",
                "--config",
                str(config_path),
                "--run-id",
                "run-001",
                "--set",
                "training.alpha.include_acceleration_head=false",
            ]
        )
        == 0
    )

    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert resolved["training"]["alpha"]["include_acceleration_head"] is False
    assert manifest["binding"]["config_sha256"] == _canonical_hash(resolved)
