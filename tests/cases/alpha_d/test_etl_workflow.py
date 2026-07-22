from __future__ import annotations

import json
import tomllib
from pathlib import Path

from cases.alpha_d.etl_workflow import _extract, _extract_valid, build_workflow
from workflows import RunContext
from workflows import cli as workflow_cli
from workflows.runner import validate_workflow

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "src/cases/alpha_d/configs/etl_workflow.toml"


def _config() -> dict:
    with CONFIG_PATH.open("rb") as stream:
        return tomllib.load(stream)


def _raw_case(root: Path, name: str) -> None:
    case = root / name
    case.mkdir(parents=True)
    (case / "simulation_out.e").write_bytes(b"raw exodus")
    (case / "case_metadata.txt").write_text("Re=5000\n", encoding="utf-8")


def test_alpha_d_etl_workflow_has_one_resumable_extraction_stage():
    definition = build_workflow(_config(), REPO_ROOT)

    assert definition.workflow_id == "alpha_d_etl"
    assert [stage.name for stage in validate_workflow(definition)] == ["extract"]
    assert "raw Exodus" in definition.stage_map()["extract"].description


def test_etl_extract_records_command_and_partial_coverage(tmp_path):
    raw_dir = tmp_path / "raw"
    _raw_case(raw_dir, "case-a")
    _raw_case(raw_dir, "case-b")
    config = _config()
    config["inputs"]["raw_dir"] = str(raw_dir)
    context = RunContext(REPO_ROOT, tmp_path / "run", config, "extract", {})
    commands = []

    def record(command, *, check=True):
        del check
        commands.append(command)
        (context.run_dir / "data/processed/case-a.zarr").mkdir(parents=True)
        return None

    context.run = record
    result = _extract(context)

    assert result.status == "partial"
    assert commands[0].argv[2] == "cases.alpha_d.run_etl"
    assert any(value == f"etl.source.input_dir={raw_dir}" for value in commands[0].argv)
    assert any(
        value == f"etl.sink.output_dir={context.run_dir / 'data/processed'}"
        for value in commands[0].argv
    )
    summary = json.loads(
        (context.run_dir / "data/etl_summary.json").read_text(encoding="utf-8")
    )
    assert summary["missing_cases"] == ["case-b"]
    assert _extract_valid(context)


def test_etl_subcommand_overrides_raw_dir_and_binds_external_input(
    monkeypatch, tmp_path, capsys
):
    captured = {}

    class FakeRunner:
        def __init__(self, definition, *, config, repo_root, run_dir):
            captured.update(
                definition=definition,
                config=config,
                repo_root=repo_root,
                run_dir=run_dir,
            )

        def run(self, *, target=None):
            assert target is None
            return {}

    raw_dir = tmp_path / "external-raw"
    monkeypatch.setattr(workflow_cli, "WorkflowRunner", FakeRunner)

    status = workflow_cli.main(
        [
            "etl",
            "--config",
            str(CONFIG_PATH),
            "--run-id",
            "etl-test",
            "--input-dir",
            str(raw_dir),
        ]
    )

    assert status == 0
    assert captured["definition"].workflow_id == "alpha_d_etl"
    assert captured["config"]["inputs"]["raw_dir"] == str(raw_dir.resolve())
    assert (
        f"{raw_dir.resolve()}:{raw_dir.resolve()}"
        in captured["config"]["executors"]["python"]["binds"]
    )
    assert str(captured["run_dir"]).endswith("data/workflows/alpha_d_etl/etl-test")
    assert "alpha_d_etl" in capsys.readouterr().out


def test_etl_subcommand_rejects_a_complete_study_config(capsys):
    study_config = REPO_ROOT / "src/cases/alpha_d/configs/coupling_study.toml"

    status = workflow_cli.main(
        ["etl", "--config", str(study_config), "--run-id", "not-an-etl-run"]
    )

    assert status == 2
    assert "requires workflow.kind = 'etl'" in capsys.readouterr().err
