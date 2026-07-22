from __future__ import annotations

from workflows import Stage, WorkflowDefinition
from workflows.cli import _print_plan


def test_plan_output_uses_compact_dependency_summaries(capsys):
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

    _print_plan(definition, target=None)
    output = capsys.readouterr().out

    assert "Workflow plan" in output
    assert "display_example | schema 1 | 5 stages" in output
    assert "Depends on" in output
    assert "3 export_closure stages" in output
    assert "panel.case_0" in output
    assert "export_closure" in output
    assert "[depends:" not in output
