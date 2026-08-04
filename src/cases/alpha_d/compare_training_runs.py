"""Apply the alpha-D publication gates to two paired-case reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"tag", "kind", "case", "truth", "coupled_relerr"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["tag"]), str(row["case"]))
        if key in indexed:
            raise ValueError(f"Duplicate paired-case row {key} in {path}")
        error = float(row["coupled_relerr"])
        truth = float(row["truth"])
        if not math.isfinite(error) or not math.isfinite(truth):
            raise ValueError(f"Non-finite paired-case row {key} in {path}")
        indexed[key] = {**row, "truth": truth, "coupled_relerr": error}
    return indexed


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    position = 0.9 * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _panel_statistics(rows: list[dict[str, Any]]) -> dict[str, float | int | str]:
    errors = [float(row["coupled_relerr"]) for row in rows]
    return {
        "kind": str(rows[0]["kind"]),
        "n": len(rows),
        "mare": statistics.fmean(abs(error) for error in errors),
        "p90": _p90([abs(error) for error in errors]),
        "signed_bias": statistics.fmean(errors),
        "absolute_signed_bias": abs(statistics.fmean(errors)),
    }


def compare_reports(baseline_csv: Path, candidate_csv: Path) -> dict[str, Any]:
    """Compare aligned reports and evaluate every publication gate."""
    baseline = _read_rows(baseline_csv)
    candidate = _read_rows(candidate_csv)
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(
            f"Paired reports do not align: missing={missing[:5]} extra={extra[:5]}"
        )
    for key in baseline:
        if not math.isclose(
            float(baseline[key]["truth"]),
            float(candidate[key]["truth"]),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"Truth value changed for paired row {key}")

    tags = sorted({tag for tag, _case in baseline})
    panels: dict[str, dict[str, Any]] = {}
    for tag in tags:
        base_stats = _panel_statistics(
            [row for (row_tag, _), row in baseline.items() if row_tag == tag]
        )
        candidate_stats = _panel_statistics(
            [row for (row_tag, _), row in candidate.items() if row_tag == tag]
        )
        panels[tag] = {"baseline": base_stats, "candidate": candidate_stats}

    in_domain = [tag for tag in tags if panels[tag]["baseline"]["kind"] == "indist"]
    if len(in_domain) != 1:
        raise ValueError(f"Expected exactly one in-domain panel, found {in_domain}")
    required_lr = {"Lr_low", "Lr_high"}
    if not required_lr.issubset(panels):
        raise ValueError(
            f"Reports are missing required Lr panels: {sorted(required_lr - set(panels))}"
        )

    gates: list[dict[str, Any]] = []

    def add_gate(
        name: str, passed: bool, baseline_value: float, candidate_value: float
    ) -> None:
        gates.append(
            {
                "name": name,
                "passed": bool(passed),
                "baseline": baseline_value,
                "candidate": candidate_value,
            }
        )

    in_tag = in_domain[0]
    for metric in ("mare", "p90"):
        before = float(panels[in_tag]["baseline"][metric])
        after = float(panels[in_tag]["candidate"][metric])
        add_gate(f"{in_tag}.{metric}.decreases", after < before, before, after)
    for tag in sorted(required_lr):
        for metric in ("mare", "absolute_signed_bias"):
            before = float(panels[tag]["baseline"][metric])
            after = float(panels[tag]["candidate"][metric])
            add_gate(f"{tag}.{metric}.decreases", after < before, before, after)
    for tag in tags:
        if tag == in_tag or tag in required_lr:
            continue
        before = float(panels[tag]["baseline"]["mare"])
        after = float(panels[tag]["candidate"]["mare"])
        add_gate(
            f"{tag}.mare.regression_lte_0.25pp", after <= before + 0.0025, before, after
        )

    return {
        "alpha_d_training_comparison_schema": 1,
        "baseline_csv": str(baseline_csv),
        "candidate_csv": str(candidate_csv),
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "panels": panels,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Alpha-D training comparison",
        "",
        f"Overall: **{'PASS' if result['passed'] else 'FAIL'}**",
        "",
    ]
    lines.extend(
        f"- {'PASS' if gate['passed'] else 'FAIL'}: {gate['name']} "
        f"({gate['baseline']:.4%} -> {gate['candidate']:.4%})"
        for gate in result["gates"]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_reports(args.baseline_csv, args.candidate_csv)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.out_markdown.write_text(_markdown(result), encoding="utf-8")
    print("PASS" if result["passed"] else "FAIL")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
