"""Summarize technical-comparison artifacts for coupled vs direct delta-P studies.

The study runbook writes one manifest per in-distribution target or
out-of-distribution shell. This module merges those manifests with direct
regressor metrics, alpha-D sidecars, and optional MOOSE verifier JSON files
into technical-report JSON, Markdown, CSV, and a small SVG summary figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cases.alpha_d.extrapolation import parse_case_params

REGRESSOR_MODELS = ("linear_regression", "random_forest", "mlp")
ALL_PREDICTORS = ("coupled",) + REGRESSOR_MODELS


@dataclass(frozen=True)
class Manifest:
    path: Path
    tag: str
    kind: str
    heldout_cases: list[str]
    report_cases: list[str]
    regressor_run_meta: Path
    regressor_eval_metrics: Path
    alpha_feature_manifest: Path
    alpha_run_meta: Path
    coupled_dir: Path
    moose_verifier_dir: Path | None
    axis: str | None = None
    side: str | None = None
    k: int | None = None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw: str | None, *, base: Path) -> Path:
    if not raw:
        return Path("")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    candidate = base / path
    return candidate.resolve()


def _resolve_with_archive_fallback(
    raw: str | None, *, base: Path, archived: str
) -> Path:
    archived_path = (base / archived).resolve()
    if archived_path.exists():
        return archived_path
    return _resolve_path(raw, base=base)


def _read_case_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def discover_manifests(study_root: Path) -> list[Manifest]:
    manifests: list[Manifest] = []
    for path in sorted(study_root.rglob("manifest.json")):
        raw = _load_json(path)
        if raw.get("claim_evidence_manifest") != 1:
            continue
        base = path.parent
        heldout = [str(case) for case in raw.get("heldout_cases") or []]
        if not heldout and raw.get("heldout_cases_txt"):
            heldout = _read_case_list(
                _resolve_path(raw["heldout_cases_txt"], base=base)
            )
        report = [str(case) for case in raw.get("report_cases") or []]
        if not report and raw.get("report_cases_txt"):
            report = _read_case_list(_resolve_path(raw["report_cases_txt"], base=base))
        if not report:
            report = list(heldout)
        manifests.append(
            Manifest(
                path=path,
                tag=str(raw["tag"]),
                kind=str(raw.get("kind", "unknown")),
                axis=raw.get("axis"),
                side=raw.get("side"),
                k=int(raw["k"]) if raw.get("k") is not None else None,
                heldout_cases=heldout,
                report_cases=report,
                regressor_run_meta=_resolve_with_archive_fallback(
                    raw.get("regressor_run_meta"),
                    base=base,
                    archived="artifacts/direct/run_meta.json",
                ),
                regressor_eval_metrics=_resolve_with_archive_fallback(
                    raw.get("regressor_eval_metrics"),
                    base=base,
                    archived="artifacts/direct/eval_metrics.json",
                ),
                alpha_feature_manifest=_resolve_with_archive_fallback(
                    raw.get("alpha_feature_manifest"),
                    base=base,
                    archived="artifacts/alpha_feature_selection/manifest.json",
                ),
                alpha_run_meta=_resolve_with_archive_fallback(
                    raw.get("alpha_run_meta"),
                    base=base,
                    archived="artifacts/alpha/run_meta.json",
                ),
                coupled_dir=_resolve_with_archive_fallback(
                    raw.get("coupled_dir"), base=base, archived="coupled"
                ),
                moose_verifier_dir=(
                    _resolve_with_archive_fallback(
                        raw.get("moose_verifier_dir"), base=base, archived="moose"
                    )
                    if raw.get("moose_verifier_dir")
                    else None
                ),
            )
        )
    if not manifests:
        raise FileNotFoundError(
            f"No technical-study manifests found under {study_root}"
        )
    return manifests


def _assert_same_set(*, label: str, actual: list[str], expected: list[str]) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(
            f"{label} does not match heldout set: missing={missing[:10]} extra={extra[:10]}"
        )


def verify_manifest_consistency(manifest: Manifest) -> dict[str, Any]:
    heldout = manifest.heldout_cases
    result: dict[str, Any] = {
        "tag": manifest.tag,
        "heldout_count": len(heldout),
        "report_count": len(manifest.report_cases),
        "checks": [],
    }
    report_extra = sorted(set(manifest.report_cases) - set(heldout))
    if report_extra:
        raise ValueError(
            f"{manifest.tag}: report cases are not a subset of heldout cases: {report_extra[:10]}"
        )
    result["checks"].append("report cases are subset of heldout")

    reg_meta = _load_json(manifest.regressor_run_meta)
    force_test = [
        str(case) for case in reg_meta.get("split", {}).get("force_test") or []
    ]
    _assert_same_set(
        label=f"{manifest.tag}: direct force_test", actual=force_test, expected=heldout
    )
    result["checks"].append("direct force_test matches heldout")

    fs_cases = [
        str(case)
        for case in reg_meta.get("feature_selection", {}).get("case_ids_used") or []
    ]
    overlap = sorted(set(fs_cases) & set(heldout))
    if overlap:
        raise ValueError(
            f"{manifest.tag}: direct feature selection used heldout cases: {overlap[:10]}"
        )
    result["checks"].append("direct feature selection excludes heldout")

    alpha_fs = _load_json(manifest.alpha_feature_manifest)
    alpha_fs_excluded = [
        str(case)
        for case in alpha_fs.get("config", {}).get("data", {}).get("exclude_cases")
        or []
    ]
    _assert_same_set(
        label=f"{manifest.tag}: alpha-D feature selection exclude_cases",
        actual=alpha_fs_excluded,
        expected=heldout,
    )
    result["checks"].append("alpha-D feature selection excludes heldout")

    alpha_meta = _load_json(manifest.alpha_run_meta)
    alpha_excluded = [
        str(case) for case in alpha_meta.get("data", {}).get("exclude_cases") or []
    ]
    _assert_same_set(
        label=f"{manifest.tag}: alpha-D training exclude_cases",
        actual=alpha_excluded,
        expected=heldout,
    )
    result["checks"].append("alpha-D training excludes heldout")

    alpha_train = [
        str(case) for case in alpha_meta.get("split", {}).get("train_sims") or []
    ]
    alpha_overlap = sorted(set(alpha_train) & set(heldout))
    if alpha_overlap:
        raise ValueError(
            f"{manifest.tag}: alpha-D training split contains heldout cases: {alpha_overlap[:10]}"
        )
    result["checks"].append("alpha-D train split excludes heldout")
    return result


def relerr(pred: float, truth: float) -> float:
    return (pred - truth) / truth


def collect_records(
    manifest: Manifest,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    metrics = _load_json(manifest.regressor_eval_metrics)
    rows = {str(row["case"]): row for row in metrics.get("per_case_predictions", [])}
    records: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for case in manifest.report_cases:
        row = rows.get(case)
        sidecar_path = manifest.coupled_dir / case / "forchheimer_profile.meta.json"
        if row is None:
            missing.append({"case": case, "missing": "regressor_row"})
            continue
        if not sidecar_path.exists():
            missing.append({"case": case, "missing": "coupled_sidecar"})
            continue
        sidecar = _load_json(sidecar_path)
        params = parse_case_params(case)
        truth = float(row["delta_p_true"])
        coupled = float(sidecar["delta_p_surrogate"])
        record: dict[str, Any] = {
            "tag": manifest.tag,
            "kind": manifest.kind,
            "axis": manifest.axis,
            "side": manifest.side,
            "case": case,
            "axis_value": getattr(params, manifest.axis) if manifest.axis else None,
            "truth": truth,
            "coupled": coupled,
            "coupled_relerr": relerr(coupled, truth),
            "source": "integral",
        }
        if "delta_p_truth" in sidecar:
            record["truth_sidecar"] = float(sidecar["delta_p_truth"])
            if not math.isclose(
                record["truth_sidecar"], truth, rel_tol=0.01, abs_tol=1e-9
            ):
                record["truth_warning"] = "regressor_truth_differs_from_sidecar"
        for model in REGRESSOR_MODELS:
            pred = float(row[f"{model}_pred"])
            record[model] = pred
            record[f"{model}_relerr"] = relerr(pred, truth)
        records.append(record)
    return records, missing


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    weight = pos - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def bootstrap_ci(
    values: list[float],
    *,
    seed: int = 42,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> list[float | None]:
    if not values:
        return [None, None]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        boot_means.append(sum(sample) / len(sample))
    return [
        quantile(boot_means, alpha / 2.0),
        quantile(boot_means, 1.0 - alpha / 2.0),
    ]


def _predictor_stats(
    records: list[dict[str, Any]], predictor: str
) -> dict[str, float | None]:
    errs = [float(row[f"{predictor}_relerr"]) for row in records]
    abs_errs = [abs(err) for err in errs]
    return {
        "mean_abs_relerr": mean(abs_errs),
        "median_abs_relerr": median(abs_errs),
        "p90_abs_relerr": quantile(abs_errs, 0.9),
        "signed_bias": mean(errs),
    }


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["tag"]), record.get("axis_value"))
        grouped.setdefault(key, []).append(record)

    summaries: list[dict[str, Any]] = []
    for (tag, axis_value), rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1]),
    ):
        summary: dict[str, Any] = {
            "tag": tag,
            "kind": rows[0]["kind"],
            "axis": rows[0].get("axis"),
            "side": rows[0].get("side"),
            "axis_value": axis_value,
            "n": len(rows),
            "predictors": {
                name: _predictor_stats(rows, name) for name in ALL_PREDICTORS
            },
            "paired": {},
        }
        coupled_abs = [abs(float(row["coupled_relerr"])) for row in rows]
        for model in REGRESSOR_MODELS:
            model_abs = [abs(float(row[f"{model}_relerr"])) for row in rows]
            diffs = [c - m for c, m in zip(coupled_abs, model_abs)]
            wins = [c < m for c, m in zip(coupled_abs, model_abs)]
            summary["paired"][model] = {
                "coupled_win_rate": sum(wins) / len(wins) if wins else None,
                "mean_abs_error_diff": mean(diffs),
                "mean_abs_error_diff_ci95": bootstrap_ci(diffs),
            }
        summaries.append(summary)
    return summaries


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.2f}%"


def _positive_finite(raw: dict[str, Any], key: str) -> bool:
    try:
        value = float(raw[key])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def _moose_artifact_error(
    *,
    manifest: Manifest,
    case_dir: Path,
    verifier: dict[str, Any],
    status: dict[str, Any],
) -> str | None:
    case = case_dir.name
    if case not in set(manifest.report_cases):
        return "case is not in report_cases"
    if status.get("moose_run_status_schema") != 1:
        return "missing supported MOOSE run-status schema"
    if status.get("status") != "success":
        return f"MOOSE run status is {status.get('status')!r}"
    if verifier.get("verification_schema") != 1:
        return "missing supported verifier schema"
    if verifier.get("verification_status") != "valid":
        return f"verifier status is {verifier.get('verification_status')!r}"
    for key in ("delta_p_truth", "delta_p_surrogate", "delta_p_moose"):
        if not _positive_finite(verifier, key):
            return f"{key} is not finite and positive"
    output_csv = status.get("output_csv")
    if not output_csv:
        return "run status does not identify the selected MOOSE CSV"
    output_path = _resolve_path(str(output_csv), base=case_dir)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return f"selected MOOSE CSV is missing or empty: {output_csv}"
    return None


def collect_moose_records(
    manifests: list[Manifest], failures: list[dict[str, str]] | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failure_rows = failures if failures is not None else []
    for manifest in manifests:
        if not manifest.moose_verifier_dir or not manifest.moose_verifier_dir.exists():
            continue
        metrics = _load_json(manifest.regressor_eval_metrics)
        direct_rows = {
            str(row["case"]): row for row in metrics.get("per_case_predictions", [])
        }
        case_dirs = {
            path.parent
            for pattern in ("run_status.json", "verify_delta_p.json")
            for path in manifest.moose_verifier_dir.rglob(pattern)
        }
        for case_dir in sorted(case_dirs):
            case = case_dir.name
            status_path = case_dir / "run_status.json"
            verifier_path = case_dir / "verify_delta_p.json"
            if not status_path.is_file():
                failure_rows.append(
                    {
                        "tag": manifest.tag,
                        "case": case,
                        "reason": "missing run_status.json",
                        "path": str(case_dir),
                    }
                )
                continue
            status = _load_json(status_path)
            if not verifier_path.is_file():
                failure_rows.append(
                    {
                        "tag": manifest.tag,
                        "case": case,
                        "reason": status.get("message")
                        or "missing verify_delta_p.json",
                        "path": str(case_dir),
                    }
                )
                continue
            raw = _load_json(verifier_path)
            artifact_error = _moose_artifact_error(
                manifest=manifest, case_dir=case_dir, verifier=raw, status=status
            )
            if artifact_error:
                failure_rows.append(
                    {
                        "tag": manifest.tag,
                        "case": case,
                        "reason": artifact_error,
                        "path": str(case_dir),
                    }
                )
                continue
            params = parse_case_params(case)
            truth = float(raw["delta_p_truth"])
            integral = float(raw["delta_p_surrogate"])
            moose = float(raw["delta_p_moose"])
            record: dict[str, Any] = {
                "tag": manifest.tag,
                "kind": manifest.kind,
                "axis": manifest.axis,
                "side": manifest.side,
                "axis_value": getattr(params, manifest.axis) if manifest.axis else None,
                "case": case,
                "path": str(verifier_path),
                "run_status_path": str(status_path),
                "selected_attempt": status.get("selected_attempt"),
                "truth": truth,
                "coupled": moose,
                "coupled_integral": integral,
                "delta_p_truth": truth,
                "delta_p_surrogate_integral": integral,
                "delta_p_moose": moose,
                "coupled_relerr": relerr(moose, truth),
                "integral_relerr": relerr(integral, truth),
                "moose_minus_integral_relerr": relerr(moose, integral),
                "moose_vs_truth_relerr": relerr(moose, truth),
                "source": "MOOSE-backed",
            }
            direct = direct_rows.get(case)
            if direct:
                for model in REGRESSOR_MODELS:
                    pred = float(direct[f"{model}_pred"])
                    record[model] = pred
                    record[f"{model}_relerr"] = relerr(pred, truth)
            rows.append(record)
    return rows


def load_moose_spotchecks(manifests: list[Manifest]) -> list[dict[str, Any]]:
    return collect_moose_records(manifests)


def build_conclusion(
    summaries: list[dict[str, Any]],
    moose_rows: list[dict[str, Any]],
    moose_failures: list[dict[str, str]],
    consistency: list[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    indist = [row for row in summaries if row["kind"] == "indist"]
    if indist:
        row = max(indist, key=lambda item: int(item["n"]))
        rf = row["predictors"]["random_forest"]["mean_abs_relerr"]
        coupled = row["predictors"]["coupled"]["mean_abs_relerr"]
        winner = (
            "direct random forest"
            if rf is not None and coupled is not None and rf < coupled
            else "direct alpha-D pressure-gradient integration"
        )
        strength = "panel" if int(row["n"]) >= 20 else "weak panel"
        lines.append(
            f"In-distribution scalar delta-P {strength}: {winner} "
            f"(n={row['n']}, RF={_format_pct(rf)}, "
            f"direct alpha-D integration={_format_pct(coupled)})."
        )
    dr_low = [row for row in summaries if row["tag"] == "Dr_low_pure"]
    if not dr_low:
        dr_low = [
            row
            for row in summaries
            if row.get("axis") == "Dr" and row.get("side") == "low"
        ]
    if dr_low:
        lower_error_fractions = [
            row["paired"]["random_forest"]["coupled_win_rate"]
            for row in dr_low
            if row["paired"]["random_forest"]["coupled_win_rate"] is not None
        ]
        coupled_means = [
            row["predictors"]["coupled"]["mean_abs_relerr"]
            for row in dr_low
            if row["predictors"]["coupled"]["mean_abs_relerr"] is not None
        ]
        rf_means = [
            row["predictors"]["random_forest"]["mean_abs_relerr"]
            for row in dr_low
            if row["predictors"]["random_forest"]["mean_abs_relerr"] is not None
        ]
        lines.append(
            "Primary low-Dr geometry OOD: direct alpha-D integration vs RF "
            "lower-error case percentage ranges "
            f"{_format_pct(min(lower_error_fractions) if lower_error_fractions else None)} "
            f"to {_format_pct(max(lower_error_fractions) if lower_error_fractions else None)} "
            "(direct alpha-D integration mean range "
            f"{_format_pct(min(coupled_means) if coupled_means else None)}-"
            f"{_format_pct(max(coupled_means) if coupled_means else None)}, "
            f"RF mean range {_format_pct(min(rf_means) if rf_means else None)}-"
            f"{_format_pct(max(rf_means) if rf_means else None)})."
        )
    controls = [
        row
        for row in summaries
        if row["kind"] == "ood"
        and row["tag"] != "Dr_low_pure"
        and row.get("axis") in {"Re", "Lr"}
    ]
    if controls:
        lines.append(
            "Re OOD and Lr OOD results are reported separately; they do not support "
            "uniform extrapolation."
        )
    primary_report = next(
        (row["report_count"] for row in consistency if row["tag"] == "Dr_low_pure"),
        None,
    )
    primary_moose = len(
        {row["case"] for row in moose_rows if row["tag"] == "Dr_low_pure"}
    )
    primary_failures = len(
        {row["case"] for row in moose_failures if row["tag"] == "Dr_low_pure"}
    )
    if primary_report:
        if primary_moose >= primary_report:
            lines.append(
                f"Validated primary low-Dr MOOSE evidence covers all {primary_report} "
                "reported case(s)."
            )
        elif primary_moose > 0:
            lines.append(
                f"Validated primary low-Dr MOOSE evidence is partial: "
                f"{primary_moose}/{primary_report} reported case(s); "
                f"{primary_failures} attempted case(s) failed validation."
            )
        else:
            lines.append(
                "No validated primary low-Dr MOOSE-coupled result was found; primary "
                "sweep remains direct alpha-D pressure-gradient integration evidence "
                f"({primary_failures} attempted "
                "case(s) failed validation)."
            )
    elif moose_rows:
        gaps = [abs(float(row["moose_minus_integral_relerr"])) for row in moose_rows]
        lines.append(
            f"MOOSE-coupled spot checks available for {len(moose_rows)} case(s); max "
            f"|MOOSE minus direct integration| gap is "
            f"{_format_pct(max(gaps) if gaps else None)}."
        )
    else:
        lines.append(
            "No MOOSE-coupled spot checks found; sweep results remain direct alpha-D "
            "pressure-gradient integration results."
        )
    return lines


def write_case_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tag",
        "kind",
        "axis",
        "side",
        "axis_value",
        "case",
        "truth",
        "coupled",
        "coupled_relerr",
        "linear_regression",
        "linear_regression_relerr",
        "random_forest",
        "random_forest_relerr",
        "mlp",
        "mlp_relerr",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_moose_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tag",
        "kind",
        "axis",
        "side",
        "axis_value",
        "case",
        "truth",
        "coupled_integral",
        "integral_relerr",
        "coupled",
        "coupled_relerr",
        "moose_minus_integral_relerr",
        "linear_regression",
        "linear_regression_relerr",
        "random_forest",
        "random_forest_relerr",
        "mlp",
        "mlp_relerr",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _summary_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.append(
        "| tag | kind | held-out axis | held-out value | cases (n) | "
        "direct alpha-D integration MARE | RF MARE | MLP MARE | linear MARE | "
        "cases where alpha-D integration has lower error than RF |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    if not rows:
        lines.append("| n/a |  |  |  | 0 | n/a | n/a | n/a | n/a | n/a |")
        return
    for row in rows:
        predictors = row["predictors"]
        paired = row["paired"]
        lines.append(
            "| "
            f"{row['tag']} | {row['kind']} | {row.get('axis') or ''} | "
            f"{'' if row.get('axis_value') is None else row['axis_value']} | {row['n']} | "
            f"{_format_pct(predictors['coupled']['mean_abs_relerr'])} | "
            f"{_format_pct(predictors['random_forest']['mean_abs_relerr'])} | "
            f"{_format_pct(predictors['mlp']['mean_abs_relerr'])} | "
            f"{_format_pct(predictors['linear_regression']['mean_abs_relerr'])} | "
            f"{_format_pct(paired['random_forest']['coupled_win_rate'])} |"
        )


def write_markdown(
    path: Path,
    *,
    summaries: list[dict[str, Any]],
    missing: list[dict[str, str]],
    moose_rows: list[dict[str, Any]],
    moose_failures: list[dict[str, str]],
    conclusion: list[str],
) -> None:
    lines = ["# Surrogate-Model Technical Comparison", ""]
    lines.extend(f"- {line}" for line in conclusion)
    indist = [row for row in summaries if row["kind"] == "indist"]
    primary = [row for row in summaries if row["tag"] == "Dr_low_pure"]
    secondary_dr = [row for row in summaries if row["tag"] == "Dr_high_guarded"]
    controls = [
        row
        for row in summaries
        if row["kind"] == "ood" and row.get("axis") in {"Re", "Lr"}
    ]
    other = [
        row
        for row in summaries
        if row not in indist + primary + secondary_dr + controls
    ]
    lines.extend(["", "## In-Distribution Panel", ""])
    _summary_table(lines, indist)
    lines.extend(["", "## Dr OOD", ""])
    _summary_table(lines, primary)
    if secondary_dr:
        lines.extend(["", "## Additional Dr OOD Results", ""])
        _summary_table(lines, secondary_dr)
    lines.extend(["", "## Re OOD and Lr OOD", ""])
    _summary_table(lines, controls)
    if other:
        lines.extend(["", "## Other Direct Alpha-D Integration Rows", ""])
        _summary_table(lines, other)
    if moose_rows:
        primary_moose = [row for row in moose_rows if row["tag"] == "Dr_low_pure"]
        control_moose = [row for row in moose_rows if row["tag"] != "Dr_low_pure"]
        if primary_moose:
            lines.extend(["", "## MOOSE-Coupled Surrogate-Closure Primary Results", ""])
            _moose_table(lines, primary_moose)
        if control_moose:
            lines.extend(["", "## MOOSE-Coupled Surrogate-Closure Spot Checks", ""])
            _moose_table(lines, control_moose)
    if moose_failures:
        lines.extend(["", "## Failed Or Unvalidated MOOSE Runs", ""])
        lines.append("| tag | case | reason |")
        lines.append("|---|---|---|")
        for row in moose_failures:
            lines.append(f"| {row['tag']} | {row['case']} | {row['reason']} |")
    if missing:
        lines.extend(["", "## Missing Artifacts", ""])
        lines.append("| tag | case | missing |")
        lines.append("|---|---|---|")
        for row in missing:
            lines.append(f"| {row['tag']} | {row['case']} | {row['missing']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _moose_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.append(
        "| tag | case | truth | direct alpha-D integration | MOOSE-coupled closure | "
        "MOOSE minus integration | MOOSE vs truth |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            f"{row['tag']} | {row['case']} | {row['truth']:.6g} | "
            f"{row['coupled_integral']:.6g} | {row['coupled']:.6g} | "
            f"{_format_pct(row['moose_minus_integral_relerr'])} | "
            f"{_format_pct(row['coupled_relerr'])} |"
        )


def write_svg(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = summaries[:12]
    width = 1100
    row_h = 42
    height = 80 + row_h * max(1, len(rows))
    scale_max = 1.0
    for row in rows:
        for predictor in ("coupled", "random_forest", "mlp"):
            value = row["predictors"][predictor]["mean_abs_relerr"]
            if value is not None:
                scale_max = max(scale_max, float(value))
    colors = {"coupled": "#2ca02c", "random_forest": "#d62728", "mlp": "#9467bd"}
    labels = {"coupled": "alpha-D integration", "random_forest": "RF", "mlp": "MLP"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px}.title{font-size:18px;font-weight:700}</style>",
        '<text x="20" y="30" class="title">Surrogate-model comparison: mean absolute relative error</text>',
    ]
    x0 = 260
    bar_max = width - x0 - 120
    for idx, row in enumerate(rows):
        y = 62 + idx * row_h
        level = "" if row.get("axis_value") is None else f"={row['axis_value']:.4g}"
        parts.append(
            f'<text x="20" y="{y + 18}">{row["tag"]} {row.get("axis") or ""}{level}</text>'
        )
        for j, predictor in enumerate(("coupled", "random_forest", "mlp")):
            value = row["predictors"][predictor]["mean_abs_relerr"] or 0.0
            bar_w = max(1.0, bar_max * float(value) / scale_max)
            yy = y + j * 12
            parts.append(
                f'<rect x="{x0}" y="{yy}" width="{bar_w:.2f}" height="9" '
                f'fill="{colors[predictor]}"><title>{labels[predictor]} {_format_pct(value)}</title></rect>'
            )
            parts.append(
                f'<text x="{x0 + bar_w + 5:.2f}" y="{yy + 9}">{labels[predictor]} {_format_pct(value)}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def summarize_study(study_root: Path) -> dict[str, Any]:
    manifests = discover_manifests(study_root)
    consistency = [verify_manifest_consistency(manifest) for manifest in manifests]
    all_records: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for manifest in manifests:
        records, missing_rows = collect_records(manifest)
        all_records.extend(records)
        for row in missing_rows:
            missing.append({"tag": manifest.tag, **row})
    summaries = summarize_records(all_records)
    moose_failures: list[dict[str, str]] = []
    moose_rows = collect_moose_records(manifests, failures=moose_failures)
    conclusion = build_conclusion(summaries, moose_rows, moose_failures, consistency)
    return {
        "study_root": str(study_root),
        "consistency": consistency,
        "summaries": summaries,
        "records": all_records,
        "missing_artifacts": missing,
        "moose_spotchecks": moose_rows,
        "moose_failures": moose_failures,
        "conclusion": conclusion,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-markdown", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-moose-csv", type=Path)
    parser.add_argument("--out-svg", type=Path, required=True)
    args = parser.parse_args(argv)

    result = summarize_study(args.study_root)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_case_csv(args.out_csv, result["records"])
    if args.out_moose_csv:
        write_moose_csv(args.out_moose_csv, result["moose_spotchecks"])
    write_markdown(
        args.out_markdown,
        summaries=result["summaries"],
        missing=result["missing_artifacts"],
        moose_rows=result["moose_spotchecks"],
        moose_failures=result["moose_failures"],
        conclusion=result["conclusion"],
    )
    write_svg(args.out_svg, result["summaries"])
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_markdown}")
    print(f"Wrote {args.out_csv}")
    if args.out_moose_csv:
        print(f"Wrote {args.out_moose_csv}")
    print(f"Wrote {args.out_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
