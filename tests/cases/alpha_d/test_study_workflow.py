from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path

import pytest

from cases.alpha_d.study_workflow import (
    ALPHA_ARTIFACT_CONTRACT,
    ALPHA_EXPORT_CONTRACT,
    Panel,
    _plan_panels,
    _publish,
    _require_alpha_training_contract,
    _require_export_contract,
    _summarize,
    build_workflow,
    parse_alpha_training_method,
)
from workflows import CommandResult, RunContext
from workflows.runner import validate_workflow

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "src/cases/alpha_d/configs/coupling_study.toml"


def _config():
    with CONFIG_PATH.open("rb") as stream:
        return tomllib.load(stream)


def test_summary_stage_writes_pressure_drop_comparison_artifacts(tmp_path):
    class RecordingExecutor:
        command = None

        def execute(self, command, *, log_path, command_path):
            self.command = command
            return CommandResult(command.argv, 0, log_path, command_path)

    executor = RecordingExecutor()
    context = RunContext(
        REPO_ROOT,
        tmp_path,
        _config(),
        "summarize",
        {"python": executor},
    )

    result = _summarize(context)
    command = executor.command.argv
    output_names = {Path(value).name for value in command}

    assert result.artifacts[0].path == "report"
    assert "pressure_drop_comparison.json" in output_names
    assert "pressure_drop_comparison.md" in output_names
    assert "pressure_drop_comparison_errors.svg" in output_names
    assert not any("claim_" in value for value in command)


def test_alpha_d_workflow_has_published_stage_families():
    definition = build_workflow(_config(), REPO_ROOT)
    stages = {stage.name for stage in validate_workflow(definition)}

    assert definition.version == 2
    assert {
        "prepare_data",
        "plan_panels",
        "tune_alpha",
        "solve_moose",
        "summarize",
    } <= stages
    for tag in (
        "indist_panel",
        "Dr_low_pure",
        "Dr_high_guarded",
        "Re_low",
        "Re_high",
        "Lr_low",
        "Lr_high",
    ):
        assert f"panel.{tag}.select_features" in stages
        assert f"panel.{tag}.train_direct" in stages
        assert f"panel.{tag}.train_alpha" in stages
        assert f"panel.{tag}.export_closure" in stages

    assert len(stages) == 33
    assert "conv1d_profile" in definition.stage_map()["tune_alpha"].description
    assert "conv1d_profile" in definition.stage_map()["panel.indist_panel.train_alpha"].description


def test_default_alpha_training_method_is_explicit():
    method = parse_alpha_training_method(_config())

    assert method.method_id == "conv1d_profile"
    assert method.runner_module == "cases.alpha_d.train"
    assert method.config_name == "train_conv1d"
    assert method.artifact_contract == ALPHA_ARTIFACT_CONTRACT
    assert method.checkpoint == Path("model.mdlus")
    assert method.run_meta == Path("run_meta.json")
    assert method.hpo.enabled is True
    assert method.hpo.reference_panel == "indist_panel"
    assert method.export.module == "cases.alpha_d.export_friction_profile"
    assert method.export.contract == ALPHA_EXPORT_CONTRACT


def test_alternate_method_changes_plan_without_workflow_edits():
    config = copy.deepcopy(_config())
    alpha = config["training"]["alpha"]
    alpha.update(
        {
            "id": "custom_cnn",
            "runner_module": "user_methods.train",
            "config_name": "train_cnn",
            "checkpoint": "checkpoints/cnn.mdlus",
            "run_meta": "metadata/cnn.json",
        }
    )
    alpha["hpo"]["enabled"] = False
    alpha["export"]["module"] = "user_methods.export_profile"

    definition = build_workflow(config, REPO_ROOT)
    method = parse_alpha_training_method(config)

    assert method.method_id == "custom_cnn"
    assert method.runner_module == "user_methods.train"
    assert method.config_name == "train_cnn"
    assert method.checkpoint == Path("checkpoints/cnn.mdlus")
    assert method.run_meta == Path("metadata/cnn.json")
    assert method.hpo.enabled is False
    assert method.export.module == "user_methods.export_profile"
    assert "custom_cnn" in definition.stage_map()["tune_alpha"].description
    assert (
        "forchheimer_profile_v1"
        in definition.stage_map()["panel.indist_panel.export_closure"].description
    )


def test_alternate_method_drives_tune_train_and_export_commands(tmp_path):
    config = copy.deepcopy(_config())
    alpha = config["training"]["alpha"]
    alpha.update(
        {
            "id": "custom_cnn",
            "runner_module": "user_methods.train",
            "config_name": "train_cnn",
            "checkpoint": "checkpoints/cnn.mdlus",
            "run_meta": "metadata/cnn.json",
        }
    )
    alpha["export"]["module"] = "user_methods.export_profile"
    definition = build_workflow(config, REPO_ROOT)
    commands = []
    context = RunContext(REPO_ROOT, tmp_path, config, "test", {})

    panel_dir = tmp_path / "panels/indist_panel"
    feature_dir = panel_dir / "artifacts/alpha_feature_selection"
    feature_dir.mkdir(parents=True)
    (feature_dir / "selected_features.txt").write_text("Dr\nz_hat\n", encoding="utf-8")
    (panel_dir / "heldout_cases.txt").write_text("case-a\n", encoding="utf-8")
    (panel_dir / "report_cases.txt").write_text("case-a\n", encoding="utf-8")

    def record(command, *, check=True):
        del check
        commands.append(command)
        if command.label == "tune_alpha":
            path = tmp_path / "tuning/hpo/best_params.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"model.params.width": 16}', encoding="utf-8")
        elif command.label == "train_alpha_indist_panel":
            artifact = panel_dir / "artifacts/alpha"
            checkpoint = artifact / "checkpoints/cnn.mdlus"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
            run_meta = artifact / "metadata/cnn.json"
            run_meta.parent.mkdir(parents=True, exist_ok=True)
            run_meta.write_text(
                json.dumps(
                    {
                        "training_run_meta_schema": 2,
                        "entrypoint": "user_models.cnn:build",
                        "adapter": "profile",
                        "data": {
                            "input_columns": ["Dr", "z_hat"],
                            "output_columns": ["signed_log1p_alpha_D"],
                            "normalize": False,
                            "effective": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
        elif command.label == "export_indist_panel_case-a":
            output = Path(command.argv[command.argv.index("--output-csv") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("z,F\n0.0,1.0\n", encoding="utf-8")
            output.with_suffix(".meta.json").write_text(
                json.dumps(
                    {
                        "case_id": "case-a",
                        "delta_p_surrogate": 1.0,
                        "delta_p_truth": 1.1,
                    }
                ),
                encoding="utf-8",
            )
        return None

    context.run = record
    stages = definition.stage_map()
    stages["tune_alpha"].action(context)
    stages["panel.indist_panel.train_alpha"].action(context)
    stages["panel.indist_panel.export_closure"].action(context)

    tune, train, export = commands
    assert tune.argv[2:5] == ("user_methods.train", "--config-name", "train_cnn")
    assert train.argv[2:5] == ("user_methods.train", "--config-name", "train_cnn")
    assert any(value.endswith("checkpoints/cnn.mdlus") for value in train.argv)
    assert any(value.endswith("metadata/cnn.json") for value in train.argv)
    assert export.argv[2] == "user_methods.export_profile"
    assert export.argv[export.argv.index("--checkpoint") + 1].endswith("checkpoints/cnn.mdlus")
    assert export.argv[export.argv.index("--run-meta") + 1].endswith("metadata/cnn.json")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config.pop("training"), "training.alpha"),
        (
            lambda config: config["training"]["alpha"].update(artifact_contract="unknown"),
            "artifact_contract",
        ),
        (
            lambda config: config["training"]["alpha"]["export"].update(contract="unknown"),
            "export.contract",
        ),
        (
            lambda config: config["training"]["alpha"].update(checkpoint="../outside.mdlus"),
            "artifact directory",
        ),
        (
            lambda config: config["training"]["alpha"].update(runner_module="not-a-module"),
            "dotted Python module",
        ),
    ],
)
def test_invalid_alpha_training_method_fails_planning(mutate, message):
    config = copy.deepcopy(_config())
    mutate(config)

    with pytest.raises(ValueError, match=message):
        build_workflow(config, REPO_ROOT)


def test_unknown_hpo_reference_panel_fails_planning():
    config = copy.deepcopy(_config())
    config["training"]["alpha"]["hpo"]["reference_panel"] = "missing"

    with pytest.raises(ValueError, match="reference_panel.*not a panel"):
        build_workflow(config, REPO_ROOT)


def test_low_dr_direct_floor_and_dr_report_guards_are_explicit():
    config = _config()
    panels = {panel["tag"]: panel for panel in config["panels"]}

    assert panels["Dr_low_pure"]["reg_min_dr"] == 0.1
    assert all(
        panel["reg_min_dr"] == 0.333 for tag, panel in panels.items() if tag != "Dr_low_pure"
    )
    assert panels["Dr_low_pure"]["guard_axes"] == ["Re", "Lr"]
    assert panels["Dr_high_guarded"]["guard_axes"] == ["Re", "Lr"]


def _contract_context(tmp_path: Path) -> tuple[RunContext, Panel]:
    panel = Panel("sample", "indist", 0.333, count=1)
    return (
        RunContext(
            repo_root=REPO_ROOT,
            run_dir=tmp_path,
            config=_config(),
            stage_name="contract",
            executors={},
        ),
        panel,
    )


def _valid_training_artifacts(tmp_path: Path) -> tuple[RunContext, Panel, Path]:
    context, panel = _contract_context(tmp_path)
    alpha = tmp_path / "panels/sample/artifacts/alpha"
    alpha.mkdir(parents=True)
    (alpha / "model.mdlus").write_bytes(b"checkpoint")
    metadata = {
        "training_run_meta_schema": 2,
        "entrypoint": "user_models.cnn:build",
        "adapter": "profile",
        "data": {
            "input_columns": ["Dr", "z_hat"],
            "output_columns": ["signed_log1p_alpha_D"],
            "normalize": True,
            "norm_stats": {"x_mean": [0.5, 0.5], "x_std": [0.2, 0.3]},
            "effective": {"target_transform": "example"},
        },
    }
    run_meta = alpha / "run_meta.json"
    run_meta.write_text(json.dumps(metadata), encoding="utf-8")
    return context, panel, run_meta


def test_alpha_training_contract_accepts_profile_method(tmp_path):
    context, panel, _ = _valid_training_artifacts(tmp_path)

    _require_alpha_training_contract(context, panel, parse_alpha_training_method(context.config))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metadata: metadata.update(adapter="pointwise"), "adapter='profile'"),
        (
            lambda metadata: metadata["data"].update(output_columns=["alpha_D"]),
            "output_columns",
        ),
        (
            lambda metadata: metadata["data"].update(norm_stats={"x_mean": [], "x_std": []}),
            "normalization statistics",
        ),
    ],
)
def test_alpha_training_contract_rejects_incompatible_metadata(tmp_path, mutation, message):
    context, panel, run_meta = _valid_training_artifacts(tmp_path)
    metadata = json.loads(run_meta.read_text(encoding="utf-8"))
    mutation(metadata)
    run_meta.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _require_alpha_training_contract(
            context, panel, parse_alpha_training_method(context.config)
        )


def test_alpha_training_contract_rejects_missing_checkpoint(tmp_path):
    context, panel, _ = _valid_training_artifacts(tmp_path)
    (tmp_path / "panels/sample/artifacts/alpha/model.mdlus").unlink()

    with pytest.raises(ValueError, match="nonempty checkpoint"):
        _require_alpha_training_contract(
            context, panel, parse_alpha_training_method(context.config)
        )


def _valid_export_artifacts(tmp_path: Path) -> tuple[RunContext, Panel, Path]:
    context, panel = _contract_context(tmp_path)
    panel_dir = tmp_path / "panels/sample"
    panel_dir.mkdir(parents=True)
    (panel_dir / "report_cases.txt").write_text("case-a\n", encoding="utf-8")
    output = panel_dir / "coupled/case-a/forchheimer_profile.csv"
    output.parent.mkdir(parents=True)
    output.write_text("z,F\n0.0,1.0\n1.0,2.0\n", encoding="utf-8")
    output.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "case_id": "case-a",
                "delta_p_surrogate": 2.5,
                "delta_p_truth": 2.7,
            }
        ),
        encoding="utf-8",
    )
    return context, panel, output


def test_export_contract_accepts_valid_profile(tmp_path):
    context, panel, _ = _valid_export_artifacts(tmp_path)

    _require_export_contract(context, panel, parse_alpha_training_method(context.config))


@pytest.mark.parametrize("failure", ["empty_csv", "wrong_case", "nonfinite"])
def test_export_contract_rejects_invalid_profile(tmp_path, failure):
    context, panel, output = _valid_export_artifacts(tmp_path)
    if failure == "empty_csv":
        output.write_text("", encoding="utf-8")
    else:
        sidecar = output.with_suffix(".meta.json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if failure == "wrong_case":
            metadata["case_id"] = "other"
        else:
            metadata["delta_p_surrogate"] = "nan"
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises((KeyError, OSError, TypeError, ValueError)):
        _require_export_contract(context, panel, parse_alpha_training_method(context.config))


def test_export_action_replaces_semantically_invalid_existing_profile(tmp_path):
    config = _config()
    definition = build_workflow(config, REPO_ROOT)
    panel_dir = tmp_path / "panels/indist_panel"
    panel_dir.mkdir(parents=True)
    (panel_dir / "report_cases.txt").write_text("case-a\n", encoding="utf-8")
    output = panel_dir / "coupled/case-a/forchheimer_profile.csv"
    output.parent.mkdir(parents=True)
    output.write_text("bad,columns\n0,1\n", encoding="utf-8")
    output.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "case_id": "case-a",
                "delta_p_surrogate": 1.0,
                "delta_p_truth": 1.1,
            }
        ),
        encoding="utf-8",
    )
    calls = 0
    context = RunContext(REPO_ROOT, tmp_path, config, "export", {})

    def record(command, *, check=True):
        nonlocal calls
        del command, check
        calls += 1
        output.write_text("z,F\n0.0,1.0\n", encoding="utf-8")
        return None

    context.run = record
    definition.stage_map()["panel.indist_panel.export_closure"].action(context)

    assert calls == 1


@pytest.mark.skipif(
    not (REPO_ROOT / "data/flow_contraction_expansion/parametric_study/processed").is_dir(),
    reason="Processed alpha-D campaign is not present",
)
def test_panel_plan_matches_published_membership_counts(tmp_path):
    context = RunContext(
        repo_root=REPO_ROOT,
        run_dir=tmp_path,
        config=_config(),
        stage_name="plan_panels",
        executors={},
    )

    _plan_panels(context)
    index = json.loads((tmp_path / "panels/index.json").read_text(encoding="utf-8"))
    counts = {row["tag"]: (row["heldout_count"], row["report_count"]) for row in index["panels"]}

    assert counts == {
        "indist_panel": (36, 36),
        "Dr_low_pure": (126, 50),
        "Dr_high_guarded": (200, 72),
        "Re_low": (136, 136),
        "Re_high": (96, 96),
        "Lr_low": (126, 126),
        "Lr_high": (127, 127),
    }
    manifest = json.loads(
        (tmp_path / "panels/indist_panel/panel_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["panel_manifest_schema"] == 3
    assert manifest["alpha_method"]["id"] == "conv1d_profile"
    assert manifest["alpha_method"]["config_name"] == "train_conv1d"
    assert manifest["alpha_checkpoint"] == "artifacts/alpha/model.mdlus"
    assert manifest["alpha_run_meta"] == "artifacts/alpha/run_meta.json"


def test_publish_is_only_docs_copy_and_check_detects_drift(tmp_path):
    repo = tmp_path / "repo"
    run_dir = repo / "data/workflows/example/run-1"
    report = run_dir / "report"
    report.mkdir(parents=True)
    (report / "pressure_drop_comparison_errors.svg").write_text("<svg/>\n", encoding="utf-8")
    (report / "pressure_drop_comparison.json").write_text(
        json.dumps({"summaries": [{"tag": "panel"}], "conclusion": ["done"]}),
        encoding="utf-8",
    )
    (run_dir / "moose_matrix.json").write_text(
        json.dumps({"attempted": 1, "succeeded": 1, "failed": 0}),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "workflow_id": "example",
                "workflow_version": 1,
                "run_id": "run-1",
                "binding": {
                    "config_sha256": "config",
                    "inputs": [],
                    "code": {"sha": "abc", "dirty": False},
                },
                "stages": {"summarize": {"status": "succeeded"}},
            }
        ),
        encoding="utf-8",
    )
    config = {
        "training": _config()["training"],
        "publish": {
            "manifest": "docs/published.json",
            "files": [
                {
                    "source": "report/pressure_drop_comparison_errors.svg",
                    "destination": "docs/_static/summary.svg",
                }
            ],
        },
    }
    context = RunContext(repo, run_dir, config, "publish", {})

    _publish(context, False)
    published = json.loads((repo / "docs/published.json").read_text(encoding="utf-8"))
    assert published["published_results_schema"] == 2
    assert published["alpha_method"]["id"] == "conv1d_profile"
    _publish(context, True)
    (repo / "docs/_static/summary.svg").write_text("drift\n", encoding="utf-8")

    with pytest.raises(ValueError, match="drifted"):
        _publish(context, True)
