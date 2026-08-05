"""Reusable workflow definition for the published alpha-D coupling study.

This module owns alpha-D-specific panel construction, subprocess arguments,
MOOSE coupling, and publication.  The generic :mod:`workflows` package knows
nothing about this case.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cases.alpha_d.extrapolation import build_indist_panel, build_split, enumerate_cases
from workflows import (
    Artifact,
    Command,
    RunContext,
    Stage,
    StageResult,
    WorkflowDefinition,
)

PANEL_MANIFEST_SCHEMA = 4
PUBLISHED_RESULTS_SCHEMA = 3
ALPHA_ARTIFACT_CONTRACT = "alpha_d_profile_v1"
ALPHA_EXPORT_CONTRACT = "forchheimer_profile_v1"
ALPHA_INPUT_COLUMNS = [
    "Dr",
    "Lr",
    "log10_Re_throat",
    "z_hat",
    "z_hat_times_Dr",
    "z_hat_times_Lr",
    "dist_to_throat_start",
    "dist_to_throat_end",
    "dist_to_nearest_step",
]


@dataclass(frozen=True)
class Panel:
    tag: str
    kind: str
    reg_min_dr: float
    axis: str | None = None
    side: str | None = None
    k: int | None = None
    count: int | None = None
    guard_axes: tuple[str, ...] = ()
    guard_k: int = 0


@dataclass(frozen=True)
class AlphaHPO:
    """Hyperparameter-search policy selected by the study configuration."""

    enabled: bool


@dataclass(frozen=True)
class AlphaFeatureSelection:
    """PyCaret feature-selection entry point shared by one Conv1D study run."""

    module: str
    config_name: str


@dataclass(frozen=True)
class AlphaExport:
    """Configured alpha-D profile exporter and its output contract."""

    module: str
    contract: str


@dataclass(frozen=True)
class AlphaTrainingMethod:
    """One TOML-selected alpha-D training method used throughout a run."""

    method_id: str
    runner_module: str
    config_name: str
    artifact_contract: str
    checkpoint: Path
    run_meta: Path
    include_acceleration_head: bool
    hpo: AlphaHPO
    feature_selection: AlphaFeatureSelection
    export: AlphaExport

    def manifest(self) -> dict[str, Any]:
        """Return the JSON-safe method identity persisted with study outputs."""
        return {
            "id": self.method_id,
            "runner_module": self.runner_module,
            "config_name": self.config_name,
            "artifact_contract": self.artifact_contract,
            "checkpoint": self.checkpoint.as_posix(),
            "run_meta": self.run_meta.as_posix(),
            "include_acceleration_head": self.include_acceleration_head,
            "hpo": {
                "enabled": self.hpo.enabled,
            },
            "feature_selection": {
                "module": self.feature_selection.module,
                "config_name": self.feature_selection.config_name,
            },
            "export": {
                "module": self.export.module,
                "contract": self.export.contract,
            },
        }


def _required_string(mapping: dict[str, Any], key: str, prefix: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix}.{key} must be a non-empty string")
    return value.strip()


def _module_name(mapping: dict[str, Any], key: str, prefix: str) -> str:
    value = _required_string(mapping, key, prefix)
    if any(not part.isidentifier() for part in value.split(".")):
        raise ValueError(f"{prefix}.{key} must be a dotted Python module name")
    return value


def _artifact_path(mapping: dict[str, Any], key: str, prefix: str) -> Path:
    raw = _required_string(mapping, key, prefix)
    path = Path(raw)
    if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{prefix}.{key} must stay beneath the method artifact directory")
    return path


def parse_alpha_training_method(config: dict[str, Any]) -> AlphaTrainingMethod:
    """Parse and validate the required ``training.alpha`` TOML contract."""
    training = config.get("training")
    if not isinstance(training, dict) or not isinstance(training.get("alpha"), dict):
        raise ValueError("Configuration requires a [training.alpha] section")
    alpha = training["alpha"]
    prefix = "training.alpha"
    artifact_contract = _required_string(alpha, "artifact_contract", prefix)
    if artifact_contract != ALPHA_ARTIFACT_CONTRACT:
        raise ValueError(
            f"Unsupported {prefix}.artifact_contract {artifact_contract!r}; "
            f"expected {ALPHA_ARTIFACT_CONTRACT!r}"
        )

    hpo = alpha.get("hpo")
    if not isinstance(hpo, dict):
        raise ValueError("Configuration requires a [training.alpha.hpo] section")
    if not isinstance(hpo.get("enabled"), bool):
        raise ValueError("training.alpha.hpo.enabled must be true or false")

    feature_selection = alpha.get("feature_selection")
    if not isinstance(feature_selection, dict):
        raise ValueError("Configuration requires a [training.alpha.feature_selection] section")

    export = alpha.get("export")
    if not isinstance(export, dict):
        raise ValueError("Configuration requires a [training.alpha.export] section")
    export_contract = _required_string(export, "contract", f"{prefix}.export")
    if export_contract != ALPHA_EXPORT_CONTRACT:
        raise ValueError(
            f"Unsupported {prefix}.export.contract {export_contract!r}; "
            f"expected {ALPHA_EXPORT_CONTRACT!r}"
        )

    checkpoint = _artifact_path(alpha, "checkpoint", prefix)
    run_meta = _artifact_path(alpha, "run_meta", prefix)
    if checkpoint == run_meta:
        raise ValueError("training.alpha.checkpoint and run_meta must be different")
    if not isinstance(alpha.get("include_acceleration_head"), bool):
        raise ValueError("training.alpha.include_acceleration_head must be true or false")

    return AlphaTrainingMethod(
        method_id=_required_string(alpha, "id", prefix),
        runner_module=_module_name(alpha, "runner_module", prefix),
        config_name=_required_string(alpha, "config_name", prefix),
        artifact_contract=artifact_contract,
        checkpoint=checkpoint,
        run_meta=run_meta,
        include_acceleration_head=alpha["include_acceleration_head"],
        hpo=AlphaHPO(enabled=hpo["enabled"]),
        feature_selection=AlphaFeatureSelection(
            module=_module_name(feature_selection, "module", f"{prefix}.feature_selection"),
            config_name=_required_string(
                feature_selection, "config_name", f"{prefix}.feature_selection"
            ),
        ),
        export=AlphaExport(
            module=_module_name(export, "module", f"{prefix}.export"),
            contract=export_contract,
        ),
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(str(raw).format(repo_root=repo_root)).expanduser()
    return path if path.is_absolute() else repo_root / path


def _panels(config: dict[str, Any]) -> tuple[Panel, ...]:
    panels: list[Panel] = []
    seen: set[str] = set()
    for raw in config.get("panels") or []:
        tag = str(raw["tag"])
        if tag in seen:
            raise ValueError(f"Duplicate panel tag {tag!r}")
        seen.add(tag)
        kind = str(raw["kind"])
        if kind not in {"indist", "ood"}:
            raise ValueError(f"Panel {tag!r} has unsupported kind {kind!r}")
        panel = Panel(
            tag=tag,
            kind=kind,
            reg_min_dr=float(raw.get("reg_min_dr", 0.333)),
            axis=str(raw["axis"]) if raw.get("axis") else None,
            side=str(raw["side"]) if raw.get("side") else None,
            k=int(raw["k"]) if raw.get("k") is not None else None,
            count=int(raw["count"]) if raw.get("count") is not None else None,
            guard_axes=tuple(str(value) for value in raw.get("guard_axes") or []),
            guard_k=int(raw.get("guard_k", 0)),
        )
        if kind == "indist" and panel.count is None:
            raise ValueError(f"In-distribution panel {tag!r} requires count")
        if kind == "ood" and (panel.axis is None or panel.side is None or panel.k is None):
            raise ValueError(f"OOD panel {tag!r} requires axis, side, and k")
        panels.append(panel)
    if not panels:
        raise ValueError("At least one [[panels]] entry is required")
    return tuple(panels)


def _zarr_dir(context: RunContext) -> Path:
    inputs = context.config.get("inputs") or {}
    mode = str(inputs.get("mode", "reuse"))
    if mode == "reuse":
        return _repo_path(context.repo_root, inputs["zarr_dir"])
    if mode == "raw_etl":
        return context.run_dir / "data" / "processed"
    raise ValueError("inputs.mode must be 'reuse' or 'raw_etl'")


def workflow_input_paths(config: dict[str, Any], repo_root: Path) -> list[str | Path]:
    """Return external inputs whose content binds a run ID."""
    inputs = config.get("inputs") or {}
    mode = str(inputs.get("mode", "reuse"))
    if mode == "reuse":
        return [_repo_path(repo_root, inputs["zarr_dir"])]
    if mode == "raw_etl":
        return [_repo_path(repo_root, inputs["raw_dir"])]
    raise ValueError("inputs.mode must be 'reuse' or 'raw_etl'")


def _panel_dir(context: RunContext, panel: Panel) -> Path:
    return context.run_dir / "panels" / panel.tag


def _case_file(context: RunContext, panel: Panel, name: str = "heldout_cases.txt") -> Path:
    return _panel_dir(context, panel) / name


def _python_command(
    context: RunContext,
    *argv: str | Path,
    label: str,
    cwd: Path | None = None,
) -> Command:
    return Command(
        argv=tuple(str(value) for value in ("python", *argv)),
        executor=str((context.config.get("study") or {}).get("python_executor", "python")),
        cwd=cwd or context.repo_root,
        env={"PYTHONPATH": str(context.repo_root / "src")},
        label=label,
    )


def _write_case_file(path: Path, cases: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{case}\n" for case in cases), encoding="utf-8")


def _read_case_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prepare_data(context: RunContext) -> StageResult:
    inputs = context.config.get("inputs") or {}
    mode = str(inputs.get("mode", "reuse"))
    zarr_dir = _zarr_dir(context)
    if mode == "raw_etl":
        raw_dir = _repo_path(context.repo_root, inputs["raw_dir"])
        context.run(
            _python_command(
                context,
                "-m",
                "cases.alpha_d.run_etl",
                f"etl.source.input_dir={raw_dir}",
                f"etl.sink.output_dir={zarr_dir}",
                label="etl",
            )
        )
    stores = sorted(zarr_dir.glob("*.zarr"))
    if not stores:
        raise FileNotFoundError(f"No processed .zarr stores found under {zarr_dir}")
    summary = context.run_dir / "data" / "inputs.json"
    _atomic_json(
        summary,
        {
            "input_schema": 1,
            "mode": mode,
            "zarr_dir": str(zarr_dir),
            "case_count": len(stores),
        },
    )
    artifacts = [Artifact(summary.relative_to(context.run_dir), "resolved data input")]
    if mode == "raw_etl":
        artifacts.append(Artifact(zarr_dir.relative_to(context.run_dir), "processed dataset"))
    return StageResult(artifacts=artifacts, details={"case_count": len(stores)})


def _plan_panels(context: RunContext) -> StageResult:
    cases = enumerate_cases(_zarr_dir(context))
    study = context.config.get("study") or {}
    method = parse_alpha_training_method(context.config)
    dr_floor = float(study.get("alpha_min_dr", 0.333))
    exclude_below_dr = float(study.get("exclude_below_dr", 0.1))
    index: list[dict[str, Any]] = []
    hpo_exclusions: set[str] = set()
    for panel in _panels(context.config):
        if panel.kind == "indist":
            heldout_params = build_indist_panel(
                cases,
                count=int(panel.count),
                guard_k=panel.guard_k,
                dr_floor=dr_floor,
                exclude_below_dr=exclude_below_dr,
            )
            report_params = heldout_params
            shell_values: list[float] = []
        else:
            split = build_split(
                cases,
                axis=str(panel.axis),
                side=str(panel.side),
                k=int(panel.k),
                dr_floor=dr_floor,
                exclude_below_dr=exclude_below_dr,
                report_guard_axes=panel.guard_axes,
                report_guard_k=panel.guard_k,
            )
            heldout_params = split.shell
            report_params = split.report
            shell_values = split.shell_axis_values
        heldout = [case.name for case in heldout_params]
        report = [case.name for case in report_params]
        hpo_exclusions.update(heldout)
        hpo_exclusions.update(report)
        panel_dir = _panel_dir(context, panel)
        _write_case_file(panel_dir / "heldout_cases.txt", heldout)
        _write_case_file(panel_dir / "report_cases.txt", report)
        manifest = {
            "panel_manifest_schema": PANEL_MANIFEST_SCHEMA,
            "workflow_run_id": context.run_dir.name,
            "tag": panel.tag,
            "kind": panel.kind,
            "axis": panel.axis,
            "side": panel.side,
            "k": panel.k,
            "reg_min_dr": panel.reg_min_dr,
            "shell_axis_values": shell_values,
            "heldout_cases": heldout,
            "report_cases": report,
            "heldout_cases_txt": "heldout_cases.txt",
            "report_cases_txt": "report_cases.txt",
            "regressor_run_meta": "artifacts/direct/run_meta.json",
            "regressor_eval_metrics": "artifacts/direct/eval_metrics.json",
            "regressor_feature_selection_dir": "artifacts/direct/feature_selection",
            "alpha_method": method.manifest(),
            "alpha_checkpoint": (Path("artifacts/alpha") / method.checkpoint).as_posix(),
            "alpha_run_meta": (Path("artifacts/alpha") / method.run_meta).as_posix(),
            "hpo_overrides": "../../tuning/best_overrides.txt",
            "coupled_dir": "coupled",
            "moose_verifier_dir": "moose",
        }
        _atomic_json(panel_dir / "panel_manifest.json", manifest)
        index.append(
            {
                "tag": panel.tag,
                "kind": panel.kind,
                "heldout_count": len(heldout),
                "report_count": len(report),
            }
        )
    index_path = context.run_dir / "panels" / "index.json"
    _atomic_json(index_path, {"panel_index_schema": 1, "panels": index})
    hpo_exclusions_path = context.run_dir / "panels" / "hpo_exclude_cases.txt"
    _write_case_file(hpo_exclusions_path, sorted(hpo_exclusions))
    artifacts = [
        Artifact(index_path.relative_to(context.run_dir), "panel index"),
        Artifact(
            hpo_exclusions_path.relative_to(context.run_dir),
            "outer cases excluded from HPO folds",
        ),
    ]
    for panel in _panels(context.config):
        for name in ("heldout_cases.txt", "report_cases.txt", "panel_manifest.json"):
            artifacts.append(
                Artifact((_panel_dir(context, panel) / name).relative_to(context.run_dir))
            )
    return StageResult(
        artifacts=artifacts,
        details={"panels": index},
    )


def _quote_hydra_string(value: str) -> str:
    """Quote one string according to Hydra's quoted-value escaping rules."""
    quote = '"'
    parts = [quote]
    backslashes = 0
    for character in value:
        if character == "\\":
            backslashes += 1
            continue
        if character == quote:
            parts.append("\\" * (2 * backslashes + 1))
        else:
            parts.append("\\" * backslashes)
        parts.append(character)
        backslashes = 0
    parts.append("\\" * (2 * backslashes))
    parts.append(quote)
    return "".join(parts)


def _hydra_override_argv(params: dict[str, Any]) -> tuple[str, ...]:
    """Return deterministic Hydra overrides without a shell round-trip."""
    if not isinstance(params, dict):
        raise TypeError("HPO best parameters must be a JSON object")
    arguments = []
    for key, value in sorted(params.items()):
        if value is not None and not isinstance(value, (bool, int, float, str)):
            raise TypeError(
                f"HPO parameter {key!r} must be a JSON scalar, got {type(value).__name__}"
            )
        rendered = (
            _quote_hydra_string(value)
            if isinstance(value, str)
            else json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        arguments.append(f"{key}={rendered}")
    return tuple(arguments)


def _alpha_feature_dir(context: RunContext) -> Path:
    """Return the one feature-selection artifact shared by Conv1D stages."""
    return context.run_dir / "features" / "alpha"


def _alpha_selected_features_path(context: RunContext) -> Path:
    return _alpha_feature_dir(context) / "selected_features.txt"


def _selected_alpha_features(context: RunContext) -> list[str]:
    """Read and validate the frozen profile feature list for this run."""
    selected = _read_case_file(_alpha_selected_features_path(context))
    if not selected:
        raise ValueError("Alpha-D feature selection produced no input columns")
    if len(set(selected)) != len(selected):
        raise ValueError("Alpha-D feature selection produced duplicate input columns")
    return selected


def _alpha_features_valid(context: RunContext) -> bool:
    """Accept only a selection artifact made without outer panel cases."""
    selected_path = _alpha_selected_features_path(context)
    manifest_path = _alpha_feature_dir(context) / "manifest.json"
    result_path = _alpha_feature_dir(context) / "result.json"
    exclusions_path = context.run_dir / "panels" / "hpo_exclude_cases.txt"
    if not all(path.is_file() for path in (selected_path, manifest_path, result_path)):
        return False
    try:
        selected = _selected_alpha_features(context)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        excluded = _read_case_file(exclusions_path)
        configured_exclusions = manifest["config"]["data"]["exclude_cases"]
        return (
            isinstance(configured_exclusions, list)
            and sorted(configured_exclusions) == sorted(excluded)
            and result.get("selected") == selected
        )
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
        return False


def _select_alpha_features_action(method: AlphaTrainingMethod):
    """Select one case-safe feature set before Conv1D HPO and panel training."""

    def action(context: RunContext) -> StageResult:
        output_dir = _alpha_feature_dir(context)
        context.run(
            _python_command(
                context,
                "-m",
                method.feature_selection.module,
                "--config-name",
                method.feature_selection.config_name,
                f"data.zarr_dir={_zarr_dir(context)}",
                "data.exclude_cases=[]",
                (f"data.exclude_cases_file={context.run_dir / 'panels' / 'hpo_exclude_cases.txt'}"),
                f"output.dir={output_dir}",
                label="select_alpha_features",
            )
        )
        selected = _selected_alpha_features(context)
        return StageResult(
            artifacts=[
                Artifact(
                    output_dir.relative_to(context.run_dir),
                    "PyCaret-selected Conv1D input columns",
                )
            ],
            details={"input_columns": selected},
        )

    return action


def _tune_alpha_action(method: AlphaTrainingMethod):
    def action(context: RunContext) -> StageResult:
        tuning = context.run_dir / "tuning"
        if not method.hpo.enabled:
            params: dict[str, Any] = {}
            _atomic_json(tuning / "best_params.json", params)
        else:
            (tuning / "hpo").mkdir(parents=True, exist_ok=True)
            context.run(
                _python_command(
                    context,
                    "-m",
                    method.runner_module,
                    "--config-name",
                    method.config_name,
                    f"data.zarr_dir={_zarr_dir(context)}",
                    f"data.include_acceleration_head={str(method.include_acceleration_head).lower()}",
                    f"data.input_columns_file={_alpha_selected_features_path(context)}",
                    f"data.exclude_cases_file={context.run_dir / 'panels' / 'hpo_exclude_cases.txt'}",
                    f"output.root_dir={tuning}",
                    "output.case_name=screening",
                    f"hpo.output_dir={tuning / 'hpo'}",
                    f"hpo.storage=sqlite:///{tuning / 'hpo' / 'study.db'}",
                    f"hpo.study_name={context.run_dir.name}_alpha_screening",
                    "hpo.retrain_best=false",
                    label="tune_alpha",
                )
            )
            params_path = tuning / "hpo" / "best_params.json"
            if not params_path.is_file():
                raise FileNotFoundError(f"HPO did not produce {params_path}")
            params = json.loads(params_path.read_text(encoding="utf-8"))
            _atomic_json(tuning / "best_params.json", params)
        (tuning / "best_overrides.txt").write_text(
            shlex.join(_hydra_override_argv(params)) + "\n", encoding="utf-8"
        )
        return StageResult(
            artifacts=[Artifact("tuning", f"frozen {method.method_id} parameters")],
            details={"method": method.manifest()},
        )

    return action


def _train_direct_action(panel: Panel):
    def action(context: RunContext) -> StageResult:
        artifact_root = _panel_dir(context, panel) / "artifacts"
        common = (
            f"data.zarr_dir={_zarr_dir(context)}",
            f"data.min_Dr={panel.reg_min_dr}",
            f"data.split.force_test_file={_case_file(context, panel)}",
            f"output.root_dir={artifact_root}",
            "output.case_name=direct",
        )
        context.run(
            _python_command(
                context,
                "-m",
                "cases.case_pressure_drop.run_case_pressure_drop",
                *common,
                label=f"train_direct_{panel.tag}",
            )
        )
        context.run(
            _python_command(
                context,
                "-m",
                "cases.case_pressure_drop.evaluate_case_pressure_drop",
                *common,
                "eval.save_plots=false",
                label=f"evaluate_direct_{panel.tag}",
            )
        )
        direct = artifact_root / "direct"
        return StageResult(
            artifacts=[Artifact(direct.relative_to(context.run_dir), "direct regression")]
        )

    return action


def _alpha_artifact_dir(context: RunContext, panel: Panel) -> Path:
    return _panel_dir(context, panel) / "artifacts" / "alpha"


def _require_alpha_training_contract(
    context: RunContext, panel: Panel, method: AlphaTrainingMethod
) -> None:
    alpha = _alpha_artifact_dir(context, panel)
    checkpoint = alpha / method.checkpoint
    run_meta_path = alpha / method.run_meta
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"{method.artifact_contract} requires a nonempty checkpoint: {checkpoint}")
    if not run_meta_path.is_file():
        raise ValueError(f"{method.artifact_contract} requires training metadata: {run_meta_path}")
    metadata = json.loads(run_meta_path.read_text(encoding="utf-8"))
    if metadata.get("training_run_meta_schema") != 3:
        raise ValueError(f"{method.artifact_contract} requires training_run_meta_schema 3")
    if metadata.get("adapter") != "profile":
        raise ValueError(f"{method.artifact_contract} requires adapter='profile'")
    if not isinstance(metadata.get("entrypoint"), str) or not metadata["entrypoint"]:
        raise ValueError(f"{method.artifact_contract} requires a model entrypoint")

    data = metadata.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"{method.artifact_contract} requires a data metadata object")
    if data.get("output_columns") != ["signed_log1p_alpha_D"]:
        raise ValueError(
            f"{method.artifact_contract} requires output_columns=['signed_log1p_alpha_D']"
        )
    input_columns = data.get("input_columns")
    if input_columns != _selected_alpha_features(context):
        raise ValueError(f"{method.artifact_contract} requires the frozen PyCaret feature contract")
    if not isinstance(data.get("effective"), dict):
        raise ValueError(f"{method.artifact_contract} requires effective dataset metadata")
    if data["effective"].get("include_acceleration_head") is not method.include_acceleration_head:
        raise ValueError(
            f"{method.artifact_contract} acceleration-head setting does not match the workflow"
        )
    if bool(data.get("normalize", False)):
        stats = data.get("norm_stats")
        if not isinstance(stats, dict):
            raise ValueError(f"{method.artifact_contract} requires normalization statistics")
        means = stats.get("x_mean")
        stds = stats.get("x_std")
        if (
            not isinstance(means, list)
            or not isinstance(stds, list)
            or len(means) != len(input_columns)
            or len(stds) != len(input_columns)
        ):
            raise ValueError(
                f"{method.artifact_contract} normalization statistics do not match input_columns"
            )
        if not all(math.isfinite(float(value)) for value in (*means, *stds)) or not all(
            float(value) > 0 for value in stds
        ):
            raise ValueError(
                f"{method.artifact_contract} normalization statistics must be finite "
                "with positive standard deviations"
            )


def _alpha_training_validator(panel: Panel, method: AlphaTrainingMethod):
    def validator(context: RunContext) -> bool:
        try:
            _require_alpha_training_contract(context, panel, method)
            return True
        except (KeyError, OSError, TypeError, ValueError):
            return False

    return validator


def _train_alpha_action(panel: Panel, method: AlphaTrainingMethod):
    def action(context: RunContext) -> StageResult:
        panel_dir = _panel_dir(context, panel)
        params = json.loads(
            (context.run_dir / "tuning" / "best_params.json").read_text(encoding="utf-8")
        )
        artifact_root = panel_dir / "artifacts"
        argv: list[str | Path] = [
            "-m",
            method.runner_module,
            "--config-name",
            method.config_name,
            f"data.zarr_dir={_zarr_dir(context)}",
            f"data.include_acceleration_head={str(method.include_acceleration_head).lower()}",
            f"data.input_columns_file={_alpha_selected_features_path(context)}",
            f"data.exclude_cases_file={_case_file(context, panel)}",
            f"output.root_dir={artifact_root}",
            "output.case_name=alpha",
            f"output.checkpoint={_alpha_artifact_dir(context, panel) / method.checkpoint}",
            f"output.run_meta={_alpha_artifact_dir(context, panel) / method.run_meta}",
            "hpo=null",
        ]
        argv.extend(_hydra_override_argv(params))
        context.run(_python_command(context, *argv, label=f"train_alpha_{panel.tag}"))
        alpha = _alpha_artifact_dir(context, panel)
        return StageResult(
            artifacts=[
                Artifact(
                    alpha.relative_to(context.run_dir),
                    f"alpha-D model ({method.method_id})",
                )
            ],
            details={"method": method.manifest()},
        )

    return action


def _require_export_case(
    output: Path, case: str, panel: Panel, method: AlphaTrainingMethod
) -> None:
    sidecar = output.with_suffix(".meta.json")
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError(f"{method.export.contract} requires a nonempty CSV for {panel.tag}/{case}")
    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["z", "F"]:
            raise ValueError(f"{method.export.contract} requires CSV columns ['z', 'F']")
        rows = list(reader)
    if not rows or not all(math.isfinite(float(row[axis])) for row in rows for axis in ("z", "F")):
        raise ValueError(
            f"{method.export.contract} requires finite profile rows for {panel.tag}/{case}"
        )
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("case_id") != case:
        raise ValueError(f"{method.export.contract} sidecar case mismatch for {panel.tag}/{case}")
    for key in ("delta_p_surrogate", "delta_p_truth"):
        if not math.isfinite(float(metadata[key])):
            raise ValueError(
                f"{method.export.contract} requires finite {key} for {panel.tag}/{case}"
            )


def _require_export_contract(
    context: RunContext, panel: Panel, method: AlphaTrainingMethod
) -> None:
    for case in _read_case_file(_case_file(context, panel, "report_cases.txt")):
        output = _panel_dir(context, panel) / "coupled" / case / "forchheimer_profile.csv"
        _require_export_case(output, case, panel, method)


def _export_validator(panel: Panel, method: AlphaTrainingMethod):
    def validator(context: RunContext) -> bool:
        try:
            _require_export_contract(context, panel, method)
            return True
        except (KeyError, OSError, TypeError, ValueError):
            return False

    return validator


def _export_action(panel: Panel, method: AlphaTrainingMethod):
    def action(context: RunContext) -> StageResult:
        panel_dir = _panel_dir(context, panel)
        alpha = _alpha_artifact_dir(context, panel)
        coupled = panel_dir / "coupled"
        for case in _read_case_file(_case_file(context, panel, "report_cases.txt")):
            output = coupled / case / "forchheimer_profile.csv"
            try:
                _require_export_case(output, case, panel, method)
                continue
            except (KeyError, OSError, TypeError, ValueError):
                pass
            context.run(
                _python_command(
                    context,
                    "-m",
                    method.export.module,
                    "--zarr",
                    _zarr_dir(context) / f"{case}.zarr",
                    "--checkpoint",
                    alpha / method.checkpoint,
                    "--run-meta",
                    alpha / method.run_meta,
                    "--output-csv",
                    output,
                    label=f"export_{panel.tag}_{case}",
                )
            )
        return StageResult(
            artifacts=[Artifact(coupled.relative_to(context.run_dir), "coupling profiles")],
            details={"method": method.manifest()},
        )

    return action


def _copy_command_records(result, destination: Path, attempt: str) -> tuple[str, str]:
    log_path = destination / f"moose_{attempt}.log"
    command_path = destination / f"moose_{attempt}.command.txt"
    shutil.copy2(result.log_path, log_path)
    shutil.copy2(result.command_path, command_path)
    return log_path.name, command_path.name


def _validate_moose_output(sidecar: Path, output_csv: Path) -> dict[str, Any]:
    from cases.alpha_d.verify_delta_p import compare, read_moose_inlet_pressure

    pressure = read_moose_inlet_pressure(output_csv)
    result = compare(sidecar_path=sidecar, delta_p_moose=pressure)
    if not math.isfinite(float(result["delta_p_moose"])) or float(result["delta_p_moose"]) <= 0:
        raise ValueError("MOOSE pressure output must be positive and finite")
    return result


def _moose_case_valid(case_dir: Path) -> bool:
    status_path = case_dir / "run_status.json"
    verification_path = case_dir / "verify_delta_p.json"
    if not status_path.is_file() or not verification_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        result = json.loads(verification_path.read_text(encoding="utf-8"))
        return (
            status.get("status") == "success"
            and result.get("verification_status") == "valid"
            and math.isfinite(float(result["delta_p_moose"]))
            and float(result["delta_p_moose"]) > 0
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _run_moose_case(context: RunContext, panel: Panel, case: str) -> bool:
    config = context.config.get("moose") or {}
    panel_dir = _panel_dir(context, panel)
    coupled = panel_dir / "coupled" / case
    destination = panel_dir / "moose" / case
    if _moose_case_valid(destination):
        return True
    destination.mkdir(parents=True, exist_ok=True)
    sidecar = coupled / "forchheimer_profile.meta.json"
    profile = coupled / "forchheimer_profile.csv"
    if not sidecar.is_file() or not profile.is_file():
        raise FileNotFoundError(f"Missing coupling profile for {panel.tag}/{case}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    template = _repo_path(context.repo_root, config["input_template"])
    shutil.copy2(template, destination / template.name)
    shutil.copy2(profile, destination / "forchheimer_profile.csv")

    executable = _repo_path(context.repo_root, config["executable"])
    mu = float(metadata["Dr"]) * float(metadata["D_big"]) / float(metadata["Re"])
    common = [
        str(executable),
        "-i",
        str(destination / template.name),
        f"mu={mu:.12g}",
        f"middle_radius={float(metadata['Dr']) * float(metadata['D_big']) / 2:.12g}",
        f"middle_length={float(metadata['throat_length_m']):.12g}",
        f"total_length={float(metadata['roi_length_m']):.12g}",
        f"delta_p_initial={float(metadata['delta_p_surrogate']):.12g}",
    ]
    attempts: list[dict[str, Any]] = []
    selected: str | None = None
    selected_result: dict[str, Any] | None = None
    definitions = [("primary", [])]
    if bool(config.get("retry", True)):
        definitions.append(
            (
                "retry",
                [
                    "Executioner/nl_max_its=200",
                    "Executioner/nl_rel_tol=1e-8",
                    "Executioner/nl_abs_tol=1e-10",
                ],
            )
        )
    for attempt, extra in definitions:
        output_base = destination / f"moose_{attempt}"
        for stale in (
            output_base.with_suffix(".csv"),
            output_base.with_suffix(".e"),
            destination / f"verify_{attempt}.json",
        ):
            stale.unlink(missing_ok=True)
        result = context.run(
            Command(
                argv=tuple(common + [f"Outputs/file_base={output_base}"] + extra),
                executor=str(config.get("executor", "moose")),
                cwd=destination,
                label=f"moose_{panel.tag}_{case}_{attempt}",
            ),
            check=False,
        )
        log_name, command_name = _copy_command_records(result, destination, attempt)
        record: dict[str, Any] = {
            "name": attempt,
            "solver_returncode": result.returncode,
            "log": log_name,
            "command": command_name,
            "output_csv": f"moose_{attempt}.csv",
        }
        if result.returncode == 0:
            try:
                verification = _validate_moose_output(sidecar, destination / f"moose_{attempt}.csv")
                _atomic_json(destination / f"verify_{attempt}.json", verification)
                record["verification_status"] = "valid"
                selected = attempt
                selected_result = verification
            except (KeyError, OSError, TypeError, ValueError) as exc:
                record["verification_status"] = "failed"
                record["verification_error"] = str(exc)
        else:
            record["verification_status"] = "not_run"
        attempts.append(record)
        if selected:
            break
    success = selected is not None and selected_result is not None
    if success:
        _atomic_json(destination / "verify_delta_p.json", selected_result)
    status = {
        "moose_case_status_schema": 2,
        "case": case,
        "panel": panel.tag,
        "status": "success" if success else "failed",
        "selected_attempt": selected,
        "profile_csv": "forchheimer_profile.csv",
        "attempts": attempts,
    }
    _atomic_json(destination / "run_status.json", status)
    return success


def _moose_matrix(context: RunContext) -> list[tuple[Panel, str]]:
    panel_map = {panel.tag: panel for panel in _panels(context.config)}
    config = context.config.get("moose") or {}
    primary = panel_map[str(config["primary_panel"])]
    matrix = [
        (primary, case)
        for case in _read_case_file(_case_file(context, primary, "report_cases.txt"))
    ]
    for tag in config.get("control_panels") or []:
        panel = panel_map[str(tag)]
        cases = _read_case_file(_case_file(context, panel, "report_cases.txt"))
        if not cases:
            raise ValueError(f"Control panel {tag!r} has no report cases")
        matrix.append((panel, cases[len(cases) // 2]))
    return list(dict.fromkeys(matrix))


def _solve_moose(context: RunContext) -> StageResult:
    config = context.config.get("moose") or {}
    executor_name = str(config.get("executor", "moose"))
    if executor_name not in context.executors:
        raise KeyError(f"MOOSE executor {executor_name!r} is not configured")
    executor = context.executors[executor_name]
    if hasattr(executor, "resolved_image"):
        image = executor.resolved_image()
        if not image.is_file():
            raise FileNotFoundError(f"MOOSE Apptainer image not found: {image}")
        if shutil.which("apptainer") is None:
            raise FileNotFoundError("apptainer executable not found on the host")
    executable = _repo_path(context.repo_root, config["executable"])
    if not executable.is_file():
        raise FileNotFoundError(f"MOOSE executable not found: {executable}")
    template = _repo_path(context.repo_root, config["input_template"])
    if not template.is_file():
        raise FileNotFoundError(f"MOOSE input template not found: {template}")

    failures: list[dict[str, str]] = []
    attempted = _moose_matrix(context)
    for panel, case in attempted:
        try:
            success = _run_moose_case(context, panel, case)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            destination = _panel_dir(context, panel) / "moose" / case
            _atomic_json(
                destination / "run_status.json",
                {
                    "moose_case_status_schema": 2,
                    "case": case,
                    "panel": panel.tag,
                    "status": "failed",
                    "selected_attempt": None,
                    "attempts": [],
                    "error": str(exc),
                },
            )
            success = False
        if not success:
            failures.append({"panel": panel.tag, "case": case})
    summary = {
        "moose_matrix_schema": 1,
        "attempted": len(attempted),
        "succeeded": len(attempted) - len(failures),
        "failed": len(failures),
        "failures": failures,
    }
    _atomic_json(context.run_dir / "moose_matrix.json", summary)
    artifacts = [Artifact("moose_matrix.json", "MOOSE coverage")]
    for panel in {panel for panel, _case in attempted}:
        artifacts.append(
            Artifact(
                (_panel_dir(context, panel) / "moose").relative_to(context.run_dir),
                f"{panel.tag} MOOSE records",
            )
        )
    return StageResult(
        status="partial" if failures else "succeeded",
        artifacts=artifacts,
        details=summary,
    )


def _validate_moose_matrix(context: RunContext) -> bool:
    path = context.run_dir / "moose_matrix.json"
    if not path.is_file():
        return False
    try:
        for panel, case in _moose_matrix(context):
            status_path = _panel_dir(context, panel) / "moose" / case / "run_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") not in {"success", "failed"}:
                return False
            if status["status"] == "success" and not _moose_case_valid(status_path.parent):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _summarize(context: RunContext) -> StageResult:
    report = context.run_dir / "report"
    context.run(
        _python_command(
            context,
            "-m",
            "cases.alpha_d.summarize_pressure_drop_comparison",
            "--study-root",
            context.run_dir / "panels",
            "--out-json",
            report / "pressure_drop_comparison.json",
            "--out-markdown",
            report / "pressure_drop_comparison.md",
            "--out-csv",
            report / "paired_case_errors.csv",
            "--out-moose-csv",
            report / "moose_paired_case_errors.csv",
            "--out-svg",
            report / "pressure_drop_comparison_errors.svg",
            label="summarize",
        )
    )
    return StageResult(artifacts=[Artifact("report", "study reports")])


def _published_manifest(context: RunContext) -> dict[str, Any]:
    workflow_manifest = json.loads(
        (context.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    if workflow_manifest["stages"]["summarize"].get("status") != "succeeded":
        raise ValueError("Refusing to publish before summarize succeeds")
    report = json.loads(
        (context.run_dir / "report" / "pressure_drop_comparison.json").read_text(encoding="utf-8")
    )
    figures = []
    for raw in (context.config.get("publish") or {}).get("files") or []:
        source = context.run_dir / str(raw["source"])
        figures.append(
            {
                "source": str(raw["source"]),
                "destination": str(raw["destination"]),
                "sha256": _sha256(source),
            }
        )
    matrix = json.loads((context.run_dir / "moose_matrix.json").read_text(encoding="utf-8"))
    workflow_contract = {
        "workflow_id": workflow_manifest["workflow_id"],
        "workflow_version": workflow_manifest["workflow_version"],
        "stages": {
            name: {
                "dependencies": stage.get("dependencies") or [],
                "description": stage.get("description") or "",
            }
            for name, stage in workflow_manifest["stages"].items()
        },
    }
    definition_hash = hashlib.sha256(
        json.dumps(workflow_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "published_results_schema": PUBLISHED_RESULTS_SCHEMA,
        "workflow_id": workflow_manifest["workflow_id"],
        "workflow_version": workflow_manifest["workflow_version"],
        "run_id": workflow_manifest["run_id"],
        "workflow_sha256": definition_hash,
        "config_sha256": workflow_manifest["binding"]["config_sha256"],
        "input_fingerprints": workflow_manifest["binding"]["inputs"],
        "code": workflow_manifest["binding"]["code"],
        "alpha_method": parse_alpha_training_method(context.config).manifest(),
        "result_summaries": report["summaries"],
        "conclusion": report["conclusion"],
        "moose_coverage": matrix,
        "figures": figures,
    }


def _publish(context: RunContext, check: bool) -> StageResult:
    publish = context.config.get("publish") or {}
    expected = _published_manifest(context)
    manifest_path = _repo_path(context.repo_root, publish["manifest"])
    drift: list[str] = []
    destinations: list[Path] = []
    for raw in publish.get("files") or []:
        source = context.run_dir / str(raw["source"])
        destination = _repo_path(context.repo_root, raw["destination"])
        destinations.append(destination)
        if check:
            if not destination.is_file() or _sha256(source) != _sha256(destination):
                drift.append(str(destination))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    if check:
        if (
            not manifest_path.is_file()
            or json.loads(manifest_path.read_text(encoding="utf-8")) != expected
        ):
            drift.append(str(manifest_path))
        if drift:
            raise ValueError("Published alpha-D artifacts have drifted: " + ", ".join(drift))
    else:
        _atomic_json(manifest_path, expected)
    return StageResult(
        artifacts=[Artifact(path, "published result") for path in (*destinations, manifest_path)],
        details={"checked": check, "files": len(destinations)},
    )


def build_workflow(config: dict[str, Any], repo_root: Path) -> WorkflowDefinition:
    """Build the case-owned stage DAG required by the generic runner."""
    del repo_root
    workflow = config.get("workflow") or {}
    panels = _panels(config)
    method = parse_alpha_training_method(config)
    stages: list[Stage] = [
        Stage(
            "prepare_data",
            _prepare_data,
            description="resolve or build processed Zarr data",
        ),
        Stage(
            "plan_panels",
            _plan_panels,
            dependencies=("prepare_data",),
            description="write canonical held-out and report case files",
        ),
    ]
    stages.append(
        Stage(
            "select_alpha_features",
            _select_alpha_features_action(method),
            dependencies=("plan_panels",),
            description="select one outer-case-safe PyCaret feature set for Conv1D",
            validator=_alpha_features_valid,
        )
    )
    stages.append(
        Stage(
            "tune_alpha",
            _tune_alpha_action(method),
            dependencies=("select_alpha_features",),
            description=f"tune {method.method_id} once and freeze its parameters",
        )
    )
    export_stages: list[str] = []
    for panel in panels:
        direct = f"panel.{panel.tag}.train_direct"
        alpha = f"panel.{panel.tag}.train_alpha"
        export = f"panel.{panel.tag}.export_closure"
        stages.extend(
            (
                Stage(
                    direct,
                    _train_direct_action(panel),
                    dependencies=("plan_panels",),
                    description="train and evaluate held-out direct regressors",
                ),
                Stage(
                    alpha,
                    _train_alpha_action(panel, method),
                    dependencies=("tune_alpha",),
                    description=(
                        f"train held-out alpha-D method {method.method_id} with frozen settings"
                    ),
                    validator=_alpha_training_validator(panel, method),
                ),
                Stage(
                    export,
                    _export_action(panel, method),
                    dependencies=(alpha,),
                    description=(f"export {method.method_id} through {method.export.contract}"),
                    validator=_export_validator(panel, method),
                ),
            )
        )
        export_stages.append(export)
    stages.extend(
        (
            Stage(
                "solve_moose",
                _solve_moose,
                dependencies=tuple(export_stages),
                description="run validated primary/retry MOOSE matrix",
                validator=_validate_moose_matrix,
            ),
            Stage(
                "summarize",
                _summarize,
                dependencies=tuple(
                    ["solve_moose"] + [f"panel.{panel.tag}.train_direct" for panel in panels]
                ),
                description="assemble distinct direct, integrated, and coupled evidence",
            ),
        )
    )
    return WorkflowDefinition(
        workflow_id=str(workflow["id"]),
        version=int(workflow.get("version", 1)),
        stages=tuple(stages),
        publisher=_publish,
        input_paths=workflow_input_paths,
    )
