import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from cases.alpha_d.summarize_claim_evidence import main, summarize_study


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_case_artifacts(
    root: Path,
    *,
    force_test=None,
    report_cases=None,
    tag="target",
    kind="indist",
    axis=None,
    side=None,
    include_moose=True,
):
    heldout = [
        "Re_100__Dr_0p5__Lr_0p1",
        "Re_200__Dr_0p5__Lr_0p1",
    ]
    report_cases = heldout if report_cases is None else report_cases
    tag_dir = root / ("indist" if kind == "indist" else "axes") / tag
    direct_dir = tag_dir / "artifacts" / "direct"
    alpha_dir = tag_dir / "artifacts" / "alpha"
    fs_dir = tag_dir / "artifacts" / "alpha_feature_selection"
    coupled_dir = tag_dir / "coupled"
    moose_dir = tag_dir / "moose"
    force_test = heldout if force_test is None else force_test

    (tag_dir / "heldout_cases.txt").parent.mkdir(parents=True, exist_ok=True)
    (tag_dir / "heldout_cases.txt").write_text(
        "\n".join(heldout) + "\n", encoding="utf-8"
    )
    (tag_dir / "heldout_cases_hydra.txt").write_text(
        "[" + ",".join(heldout) + "]\n",
        encoding="utf-8",
    )
    (tag_dir / "report_cases.txt").write_text(
        "\n".join(report_cases) + "\n", encoding="utf-8"
    )
    (tag_dir / "report_cases_hydra.txt").write_text(
        "[" + ",".join(report_cases) + "]\n",
        encoding="utf-8",
    )

    _write_json(
        tag_dir / "manifest.json",
        {
            "claim_evidence_manifest": 1,
            "tag": tag,
            "kind": kind,
            "axis": axis,
            "side": side,
            "heldout_cases": heldout,
            "report_cases": report_cases,
            "heldout_cases_txt": "heldout_cases.txt",
            "heldout_cases_hydra": "heldout_cases_hydra.txt",
            "report_cases_txt": "report_cases.txt",
            "report_cases_hydra": "report_cases_hydra.txt",
            "regressor_run_meta": "artifacts/direct/run_meta.json",
            "regressor_eval_metrics": "artifacts/direct/eval_metrics.json",
            "alpha_feature_manifest": "artifacts/alpha_feature_selection/manifest.json",
            "alpha_run_meta": "artifacts/alpha/run_meta.json",
            "coupled_dir": "coupled",
            "moose_verifier_dir": "moose",
        },
    )
    _write_json(
        direct_dir / "run_meta.json",
        {
            "split": {"force_test": force_test},
            "feature_selection": {"case_ids_used": ["Re_300__Dr_0p5__Lr_0p1"]},
        },
    )
    _write_json(
        direct_dir / "eval_metrics.json",
        {
            "per_case_predictions": [
                {
                    "case": heldout[0],
                    "delta_p_true": 100.0,
                    "linear_regression_pred": 70.0,
                    "random_forest_pred": 80.0,
                    "mlp_pred": 110.0,
                },
                {
                    "case": heldout[1],
                    "delta_p_true": 200.0,
                    "linear_regression_pred": 250.0,
                    "random_forest_pred": 210.0,
                    "mlp_pred": 260.0,
                },
            ]
        },
    )
    _write_json(
        fs_dir / "manifest.json",
        {"config": {"data": {"exclude_cases": heldout}}},
    )
    _write_json(
        alpha_dir / "run_meta.json",
        {
            "data": {"exclude_cases": heldout},
            "split": {"train_sims": ["Re_300__Dr_0p5__Lr_0p1"]},
        },
    )
    _write_json(
        coupled_dir / heldout[0] / "forchheimer_profile.meta.json",
        {"delta_p_truth": 100.0, "delta_p_surrogate": 90.0},
    )
    _write_json(
        coupled_dir / heldout[1] / "forchheimer_profile.meta.json",
        {"delta_p_truth": 200.0, "delta_p_surrogate": 220.0},
    )
    if include_moose:
        case_moose_dir = moose_dir / heldout[0]
        (case_moose_dir / "moose_primary.csv").parent.mkdir(parents=True, exist_ok=True)
        (case_moose_dir / "moose_primary.csv").write_text(
            "time,inlet-p\n0,1.0\n", encoding="utf-8"
        )
        _write_json(
            case_moose_dir / "run_status.json",
            {
                "moose_run_status_schema": 1,
                "case": heldout[0],
                "status": "success",
                "selected_attempt": "primary",
                "output_csv": "moose_primary.csv",
                "verification_json": "verify_delta_p.json",
            },
        )
        _write_json(
            case_moose_dir / "verify_delta_p.json",
            {
                "verification_schema": 1,
                "verification_status": "valid",
                "delta_p_truth": 100.0,
                "delta_p_surrogate": 90.0,
                "delta_p_moose": 91.0,
            },
        )
    return heldout


def test_summarize_study_computes_paired_statistics(tmp_path):
    _write_case_artifacts(tmp_path)

    result = summarize_study(tmp_path)

    assert result["missing_artifacts"] == []
    summary = result["summaries"][0]
    assert summary["n"] == 2
    assert summary["predictors"]["coupled"]["mean_abs_relerr"] == pytest.approx(0.1)
    assert summary["predictors"]["random_forest"]["mean_abs_relerr"] == pytest.approx(
        0.125
    )
    assert summary["paired"]["random_forest"]["coupled_win_rate"] == pytest.approx(0.5)
    assert summary["paired"]["random_forest"]["mean_abs_error_diff"] == pytest.approx(
        -0.025
    )
    assert result["moose_spotchecks"][0][
        "moose_minus_integral_relerr"
    ] == pytest.approx(1.0 / 90.0)


def test_summarize_study_reads_new_panel_and_moose_status_schemas(tmp_path):
    heldout = _write_case_artifacts(tmp_path)
    tag_dir = tmp_path / "indist" / "target"
    legacy_manifest = tag_dir / "manifest.json"
    manifest = json.loads(legacy_manifest.read_text(encoding="utf-8"))
    manifest.pop("claim_evidence_manifest")
    manifest["panel_manifest_schema"] = 2
    legacy_manifest.unlink()
    _write_json(tag_dir / "panel_manifest.json", manifest)

    status_path = tag_dir / "moose" / heldout[0] / "run_status.json"
    _write_json(
        status_path,
        {
            "moose_case_status_schema": 2,
            "case": heldout[0],
            "status": "success",
            "selected_attempt": "primary",
            "attempts": [
                {
                    "name": "primary",
                    "solver_returncode": 0,
                    "output_csv": "moose_primary.csv",
                    "verification_status": "valid",
                }
            ],
        },
    )

    result = summarize_study(tmp_path)

    assert result["claim_evidence_summary_schema"] == 2
    assert result["evidence_classes"] == {
        "direct_scalar_regression": ["linear_regression", "random_forest", "mlp"],
        "direct_alpha_d_gradient_integration": "records",
        "moose_coupled_alpha_d": "moose_spotchecks",
    }
    assert len(result["moose_spotchecks"]) == 1


def test_summarize_study_uses_report_cases_for_statistics(tmp_path):
    heldout = _write_case_artifacts(tmp_path, report_cases=["Re_100__Dr_0p5__Lr_0p1"])

    result = summarize_study(tmp_path)

    assert result["consistency"][0]["heldout_count"] == len(heldout)
    assert result["consistency"][0]["report_count"] == 1
    assert result["summaries"][0]["n"] == 1
    assert result["summaries"][0]["predictors"]["coupled"][
        "mean_abs_relerr"
    ] == pytest.approx(0.1)


def test_summarize_study_fails_when_report_cases_are_not_held_out(tmp_path):
    _write_case_artifacts(
        tmp_path,
        report_cases=["Re_100__Dr_0p5__Lr_0p1", "Re_999__Dr_0p5__Lr_0p1"],
    )

    with pytest.raises(ValueError, match="report cases"):
        summarize_study(tmp_path)


def test_summarize_study_can_be_rerun_from_copied_relative_archive(tmp_path):
    _write_case_artifacts(tmp_path)
    copy_root = tmp_path.with_name(f"{tmp_path.name}_copy")
    if copy_root.exists():
        shutil.rmtree(copy_root)
    shutil.copytree(tmp_path, copy_root)

    result = summarize_study(copy_root)

    assert result["summaries"][0]["n"] == 2
    assert result["missing_artifacts"] == []


def test_summarize_study_falls_back_from_legacy_absolute_paths(tmp_path):
    _write_case_artifacts(tmp_path)
    manifest_path = tmp_path / "indist" / "target" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "regressor_run_meta",
        "regressor_eval_metrics",
        "alpha_feature_manifest",
        "alpha_run_meta",
        "coupled_dir",
        "moose_verifier_dir",
    ):
        manifest[key] = f"/remote/unavailable/{key}"
    _write_json(manifest_path, manifest)

    result = summarize_study(tmp_path)

    assert result["summaries"][0]["n"] == 2
    assert len(result["moose_spotchecks"]) == 1


def test_summarize_cli_writes_outputs(tmp_path):
    _write_case_artifacts(tmp_path)
    out_json = tmp_path / "claim_evidence.json"
    out_md = tmp_path / "claim_evidence.md"
    out_csv = tmp_path / "paired_case_errors.csv"
    out_moose_csv = tmp_path / "moose_paired_case_errors.csv"
    out_svg = tmp_path / "claim_error_summary.svg"

    assert (
        main(
            [
                "--study-root",
                str(tmp_path),
                "--out-json",
                str(out_json),
                "--out-markdown",
                str(out_md),
                "--out-csv",
                str(out_csv),
                "--out-moose-csv",
                str(out_moose_csv),
                "--out-svg",
                str(out_svg),
            ]
        )
        == 0
    )

    markdown = out_md.read_text(encoding="utf-8")
    assert "Surrogate-Model Technical Comparison" in markdown
    assert "direct alpha-D integration MARE" in markdown
    assert "MOOSE-Coupled Surrogate-Closure" in markdown
    assert "coupled_relerr" in out_csv.read_text(encoding="utf-8")
    assert "MOOSE-backed" not in out_csv.read_text(encoding="utf-8")
    assert "MOOSE-backed" in out_moose_csv.read_text(encoding="utf-8")
    assert "viewBox" in out_svg.read_text(encoding="utf-8")
    assert json.loads(out_json.read_text(encoding="utf-8"))["summaries"]


def test_summarize_study_fails_on_holdout_mismatch(tmp_path):
    _write_case_artifacts(tmp_path, force_test=["Re_999__Dr_0p5__Lr_0p1"])

    with pytest.raises(ValueError, match="direct force_test"):
        summarize_study(tmp_path)


def test_summarize_study_reports_missing_coupled_sidecar(tmp_path):
    heldout = _write_case_artifacts(tmp_path)
    sidecar = (
        tmp_path
        / "indist"
        / "target"
        / "coupled"
        / heldout[1]
        / "forchheimer_profile.meta.json"
    )
    sidecar.unlink()

    result = summarize_study(tmp_path)

    assert result["missing_artifacts"] == [
        {"tag": "target", "case": heldout[1], "missing": "coupled_sidecar"}
    ]
    assert result["summaries"][0]["n"] == 1


def test_primary_low_dr_without_moose_is_labeled_integral_only(tmp_path):
    _write_case_artifacts(
        tmp_path,
        tag="Dr_low_pure",
        kind="ood",
        axis="Dr",
        side="low",
        include_moose=False,
    )

    result = summarize_study(tmp_path)

    assert any(
        "No validated primary low-Dr MOOSE-coupled result" in line
        for line in result["conclusion"]
    )


def test_summarize_study_rejects_zero_moose_result(tmp_path):
    heldout = _write_case_artifacts(
        tmp_path,
        tag="Dr_low_pure",
        kind="ood",
        axis="Dr",
        side="low",
    )
    verifier = (
        tmp_path / "axes" / "Dr_low_pure" / "moose" / heldout[0] / "verify_delta_p.json"
    )
    payload = json.loads(verifier.read_text(encoding="utf-8"))
    payload["delta_p_moose"] = 0.0
    _write_json(verifier, payload)

    result = summarize_study(tmp_path)

    assert result["moose_spotchecks"] == []
    assert result["moose_failures"][0]["case"] == heldout[0]
    assert "not finite and positive" in result["moose_failures"][0]["reason"]
    assert any(
        "No validated primary low-Dr MOOSE-coupled result" in line
        for line in result["conclusion"]
    )


def test_summarize_study_requires_moose_run_status(tmp_path):
    heldout = _write_case_artifacts(tmp_path)
    status = tmp_path / "indist" / "target" / "moose" / heldout[0] / "run_status.json"
    status.unlink()

    result = summarize_study(tmp_path)

    assert result["moose_spotchecks"] == []
    assert result["moose_failures"] == [
        {
            "tag": "target",
            "case": heldout[0],
            "reason": "missing run_status.json",
            "path": str(status.parent),
        }
    ]


def test_claim_evidence_runbook_is_valid_bash():
    repo = Path(__file__).resolve().parents[3]
    script = repo / "docs/superpowers/scripts/2026-07-06-claim-evidence-runbook.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_claim_evidence_runbook_stronger_dry_runs():
    repo = Path(__file__).resolve().parents[3]
    script = repo / "docs/superpowers/scripts/2026-07-06-claim-evidence-runbook.sh"
    subprocess.run(
        [
            "bash",
            "-lc",
            (
                f"export CLAIM_DRY_RUN=1 STUDY_TAG=dryrun; source {script}; claim_stronger_matrix"
            ),
        ],
        cwd=repo,
        check=True,
    )


def test_claim_evidence_runbook_moose_and_summarize_dry_run():
    repo = Path(__file__).resolve().parents[3]
    script = repo / "docs/superpowers/scripts/2026-07-06-claim-evidence-runbook.sh"
    subprocess.run(
        [
            "bash",
            "-lc",
            (
                "export CLAIM_DRY_RUN=1 STUDY_TAG=dryrun; "
                f"source {script}; "
                "claim_moose_primary Dr_low_pure; "
                "claim_summarize"
            ),
        ],
        cwd=repo,
        check=True,
    )


def _write_fake_moose_environment(tmp_path: Path, *, inlet_integral: float):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    apptainer = fake_bin / "apptainer"
    apptainer.write_text(
        """#!/usr/bin/env bash
set -e
[ "$1" = exec ] && shift
if [ "$1" = --bind ]; then shift 2; fi
shift
exec "$@"
""",
        encoding="utf-8",
    )
    apptainer.chmod(0o755)
    moose = tmp_path / "fake-moose"
    moose.write_text(
        f"""#!/usr/bin/env bash
set -e
for arg in "$@"; do
  case "$arg" in
    Outputs/file_base=*) output_base=${{arg#*=}} ;;
  esac
done
printf 'time,inlet-p,outlet-u\\n0,{inlet_integral},1.0\\n' > "$output_base.csv"
: > "$output_base.e"
""",
        encoding="utf-8",
    )
    moose.chmod(0o755)
    py_sif = tmp_path / "python.sif"
    moose_sif = tmp_path / "moose.sif"
    py_sif.touch()
    moose_sif.touch()
    return fake_bin, moose, py_sif, moose_sif


def _run_fake_moose_spotcheck(tmp_path: Path, *, inlet_integral: float):
    repo = Path(__file__).resolve().parents[3]
    script = repo / "docs/superpowers/scripts/2026-07-06-claim-evidence-runbook.sh"
    out_root = tmp_path / "study"
    case = "Re_100__Dr_0p5__Lr_0p1"
    tag_dir = out_root / "axes" / "Dr_low_pure"
    coupled = tag_dir / "coupled" / case
    coupled.mkdir(parents=True)
    _write_json(
        coupled / "forchheimer_profile.meta.json",
        {
            "Re": 100.0,
            "Dr": 0.5,
            "Lr": 0.1,
            "D_big": 0.2,
            "delta_p_truth": 100.0,
            "delta_p_surrogate": 90.0,
        },
    )
    (coupled / "forchheimer_profile.csv").write_text(
        "x,forchheimer\n0,1\n0.5,1\n", encoding="utf-8"
    )
    fake_bin, moose, py_sif, moose_sif = _write_fake_moose_environment(
        tmp_path, inlet_integral=inlet_integral
    )
    exports = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REPO": str(repo),
        "OUT_ROOT": str(out_root),
        "STUDY_TAG": "fake",
        "SIF": str(py_sif),
        "MOOSE_SIF": str(moose_sif),
        "MOOSE_EXE": str(moose),
        "BIND": f"{repo}:{repo}",
        "CLAIM_MOOSE_FORCE": "1",
    }
    command = "; ".join(
        [f"export {key}={shlex.quote(value)}" for key, value in exports.items()]
        + [
            f"source {shlex.quote(str(script))}",
            f"claim_moose_spotcheck Dr_low_pure {case}",
        ]
    )
    completed = subprocess.run(["bash", "-c", command], cwd=repo, check=False)
    return completed, tag_dir / "moose" / case


def test_claim_moose_spotcheck_writes_validated_case_artifacts(tmp_path):
    completed, case_dir = _run_fake_moose_spotcheck(tmp_path, inlet_integral=3.14)

    assert completed.returncode == 0
    status = json.loads((case_dir / "run_status.json").read_text(encoding="utf-8"))
    verifier = json.loads(
        (case_dir / "verify_delta_p.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "success"
    assert status["selected_attempt"] == "primary"
    assert verifier["verification_status"] == "valid"
    assert (case_dir / "2d-porous-flow_alphaD.i").is_file()
    assert (case_dir / "forchheimer_profile.csv").is_file()


def test_claim_moose_spotcheck_rejects_zero_output_after_retry(tmp_path):
    completed, case_dir = _run_fake_moose_spotcheck(tmp_path, inlet_integral=0.0)

    assert completed.returncode == 1
    status = json.loads((case_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert [attempt["name"] for attempt in status["attempts"]] == [
        "primary",
        "retry",
    ]
    assert all(attempt["verification_exit_code"] == 2 for attempt in status["attempts"])
    assert not (case_dir / "verify_delta_p.json").exists()
