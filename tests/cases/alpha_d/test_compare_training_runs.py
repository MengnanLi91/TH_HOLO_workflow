from __future__ import annotations

import csv
from pathlib import Path

from cases.alpha_d.compare_training_runs import compare_reports


def _write_report(path: Path, errors: dict[str, list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["tag", "kind", "case", "truth", "coupled_relerr"],
        )
        writer.writeheader()
        for tag, values in errors.items():
            for index, value in enumerate(values):
                writer.writerow(
                    {
                        "tag": tag,
                        "kind": "indist" if tag == "indist_panel" else "ood",
                        "case": f"{tag}-{index}",
                        "truth": 100.0 + index,
                        "coupled_relerr": value,
                    }
                )


def test_publication_gate_passes_only_for_required_improvements(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_report(
        baseline,
        {
            "indist_panel": [-0.1, 0.1],
            "Lr_low": [-0.2, -0.1],
            "Lr_high": [0.2, 0.1],
            "Dr_low_pure": [0.1, 0.2],
        },
    )
    _write_report(
        candidate,
        {
            "indist_panel": [-0.05, 0.05],
            "Lr_low": [-0.1, 0.05],
            "Lr_high": [0.1, -0.05],
            "Dr_low_pure": [0.101, 0.201],
        },
    )

    result = compare_reports(baseline, candidate)

    assert result["passed"] is True
    assert all(gate["passed"] for gate in result["gates"])


def test_publication_gate_rejects_other_panel_regression_over_quarter_point(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    common = {
        "indist_panel": [-0.1, 0.1],
        "Lr_low": [-0.2, -0.1],
        "Lr_high": [0.2, 0.1],
        "Dr_low_pure": [0.1, 0.2],
    }
    _write_report(baseline, common)
    changed = dict(common)
    changed["indist_panel"] = [-0.05, 0.05]
    changed["Lr_low"] = [-0.1, 0.05]
    changed["Lr_high"] = [0.1, -0.05]
    changed["Dr_low_pure"] = [0.104, 0.204]
    _write_report(candidate, changed)

    result = compare_reports(baseline, candidate)

    assert result["passed"] is False
    assert any(
        gate["name"] == "Dr_low_pure.mare.regression_lte_0.25pp" and not gate["passed"]
        for gate in result["gates"]
    )
