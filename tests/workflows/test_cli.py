from __future__ import annotations

from pathlib import Path

from workflows import Stage, WorkflowDefinition
from workflows.cli import _print_plan, _run_with_progress, _status, build_parser, main


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


def test_plan_parser_defaults_to_tree_and_accepts_table_option():
    default = build_parser().parse_args(["plan", "--config", "workflow.toml"])
    table = build_parser().parse_args(["plan", "--config", "workflow.toml", "--table"])

    assert default.table is False
    assert table.table is True


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


def test_run_progress_tracks_reused_and_partial_stages(capsys):
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
            self.events: list[tuple[str, str]] = []

        def run(self, *, target, on_stage):
            on_stage(definition.stages[0], "reused")
            on_stage(definition.stages[1], "running")
            on_stage(definition.stages[1], "partial")
            self.events = [("prepare", "reused"), ("summarize", "partial")]
            return {"status": "completed_with_partial_results"}

    runner = Runner()
    _run_with_progress(runner, target=None)
    output = capsys.readouterr().out

    assert runner.events == [("prepare", "reused"), ("summarize", "partial")]
    assert "Workflow complete with partial results" in output


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
    monkeypatch.setattr(
        "workflows.cli._definition", lambda _config, _repo_root: definition
    )
    monkeypatch.setattr(
        "workflows.cli._run_dir", lambda _config, _repo_root, _run_id: run_dir
    )

    assert main(["run", "--config", str(config_path), "--run-id", "run-001"]) == 0
    output = capsys.readouterr().out

    assert "Workflow complete" in output
    assert str(run_dir) in output
