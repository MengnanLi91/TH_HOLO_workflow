"""Export an alpha_D surrogate prediction as a MOOSE Forchheimer profile.

For a single target case, this script:
  1. Loads features from the case's existing ETL `.zarr` store.
  2. Loads a trained profile-model checkpoint.
  3. Runs forward inference to get the per-station alpha_D profile.
  4. Maps alpha_D(z) to MOOSE PINSFV Forchheimer coefficient C_F(z) using
     the full-ROI block-aware empirical equivalence (see spec Section 4.3
     and the constant-F=1 verification tests, 2026-05-28).
  5. Writes forchheimer_profile.csv (z, F) plus a sidecar metadata.json.

The CSV is consumed by MOOSE's PiecewiseLinear function backing the
ADGenericVectorFunctorMaterial applied on all blocks (full ROI).

Block-specific Forchheimer mapping (empirical, constant-F=1 verified):
  MOOSE PINSFV empirically computes -dP/dz = F / (2 · porosity²)
  (not F / (2 · porosity) as the kernel comment suggests; the extra 1/ε
  factor likely arises because the kernel uses interstitial² not
  interstitial·superficial velocity internally.)
  Equating to training α_D / (2 · D_h) gives F = α_D · porosity² / D_h:
  - Block 1 (upstream buffer): porosity = 1,   D_h = D_outer → F = α_D / D_outer
  - Block 2 (throat):          porosity = Dr², D_h = Dr·D_outer → F = α_D · Dr³ / D_outer
  - Block 3 (downstream buffer): porosity = 1, D_h = D_outer → F = α_D / D_outer
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cases.alpha_d.coupling_utils import (
    alpha_d_to_forchheimer,
    compute_baseline_encoded,
    decode_to_bulk_alpha_d,
    integrate_delta_p,
    restrict_to_throat,
    stepfence_porosity_boundaries,
    write_outputs,
)

__all__ = [
    "Args",
    "CaseData",
    "alpha_d_to_forchheimer",
    "build_model_input",
    "compute_baseline_encoded",
    "decode_to_bulk_alpha_d",
    "forward",
    "integrate_delta_p",
    "load_case_from_zarr",
    "main",
    "parse_args",
    "restrict_to_throat",
    "stepfence_porosity_boundaries",
    "write_outputs",
]

# Compatibility alias retained for existing callers and tests.
_stepfence_porosity_boundaries = stepfence_porosity_boundaries


@dataclass
class Args:
    zarr: Path
    checkpoint: Path
    run_meta: Path
    output_csv: Path


def parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr",
        type=Path,
        required=True,
        help="Path to the target case's .zarr store.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to the .mdlus checkpoint."
    )
    parser.add_argument(
        "--run-meta",
        type=Path,
        required=True,
        help="Path to the run_meta.json sibling of the checkpoint.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV path (Forchheimer profile).",
    )
    ns = parser.parse_args(argv)
    return Args(
        zarr=ns.zarr,
        checkpoint=ns.checkpoint,
        run_meta=ns.run_meta,
        output_csv=ns.output_csv,
    )


@dataclass
class CaseData:
    case_id: str
    features: np.ndarray  # (50, n_feat)
    targets: np.ndarray  # (50, n_tgt)
    feature_names: list[str]
    target_names: list[str]
    Re: float
    Dr: float
    Lr: float
    D_big: float  # outer pipe diameter, m
    outer_height_m: float
    buffer_diams: float
    rho: float
    V_bulk: float
    delta_p_truth: float = field(default=0.0)


def load_case_from_zarr(zarr_path: Path) -> CaseData:
    """Read features, targets, and geometry metadata for one case."""
    import zarr

    if not zarr_path.exists():
        raise FileNotFoundError(f"Case zarr not found: {zarr_path}")

    root = zarr.open(store=str(zarr_path), mode="r")
    meta = root["metadata"]

    feature_names = list(json.loads(meta.attrs["feature_names"]))
    target_names = list(json.loads(meta.attrs["target_names"]))

    return CaseData(
        case_id=str(meta.attrs["case_id"]),
        features=np.asarray(root["features"][:], dtype=np.float32),
        targets=np.asarray(root["targets"][:], dtype=np.float32),
        feature_names=feature_names,
        target_names=target_names,
        Re=float(meta.attrs["Re"]),
        Dr=float(meta.attrs["Dr"]),
        Lr=float(meta.attrs["Lr"]),
        D_big=float(meta.attrs.get("D_big", 0.2)),
        outer_height_m=float(meta.attrs.get("outer_height_m", 1.0)),
        buffer_diams=float(meta.attrs.get("buffer_diams", 1.0)),
        rho=float(meta.attrs.get("rho", 1.0)),
        V_bulk=float(meta.attrs.get("V_bulk", 1.0)),
        delta_p_truth=float(meta.attrs.get("delta_p_case", 0.0)),
    )


def build_model_input(case: CaseData, run_meta: dict) -> np.ndarray:
    """Project zarr features into the model's input_columns + normalize.

    Returns array of shape (1, n_features, n_stations) in profile NCL layout.
    Reuses `cases.alpha_d.feature_data.build_engineered_feature_map` for
    engineered columns to match training-time feature synthesis exactly.
    """
    from cases.alpha_d.feature_data import build_engineered_feature_map

    input_columns: list[str] = list(run_meta["data"]["input_columns"])
    raw_name_to_idx = {n: i for i, n in enumerate(case.feature_names)}
    engineered = build_engineered_feature_map(case.features, case.feature_names)

    # Engineered values win for names that appear in BOTH the zarr's raw
    # feature_names and the engineered set. This matches the training-time
    # TabularPairDataset (datasets_tabular.py:234), which builds
    # `feat_map = {n: i for i, n in enumerate(raw + engineered)}` — duplicate
    # keys make the later (engineered) entry win. `dist_to_throat_start` and
    # `dist_to_throat_end` live in both sets and differ by a fraction of an
    # axial bin (raw uses the analytic throat boundary; engineered uses
    # `min(z_hat[is_throat])`); without this preference the model sees
    # slightly different inputs at inference vs training.
    columns: list[np.ndarray] = []
    for name in input_columns:
        if name in engineered:
            col = engineered[name].astype(np.float32)
        elif name in raw_name_to_idx:
            col = case.features[:, raw_name_to_idx[name]].astype(np.float32)
        else:
            raise ValueError(
                f"input_column {name!r} not found in zarr feature_names "
                f"nor engineered set. zarr columns: {case.feature_names}; "
                f"engineered: {list(engineered.keys())}."
            )
        columns.append(col)

    raw_x = np.stack(columns, axis=0)  # (C, L)
    normalize = bool(
        run_meta["data"].get(
            "normalize", run_meta["data"].get("norm_stats") is not None
        )
    )
    if normalize:
        stats = run_meta["data"].get("norm_stats") or {}
        x_mean = np.asarray(stats.get("x_mean"), dtype=np.float32)
        x_std = np.asarray(stats.get("x_std"), dtype=np.float32)
        if len(x_mean) != len(input_columns) or len(x_std) != len(input_columns):
            raise ValueError(
                f"norm_stats length mismatch: {len(x_mean)} mean / "
                f"{len(x_std)} std vs {len(input_columns)} input_columns"
            )
        normed = (raw_x - x_mean[:, None]) / x_std[:, None]
    else:
        normed = raw_x
    return normed[None, :, :].astype(np.float32)  # (1, C, L)


def forward(checkpoint_path: Path, run_meta: dict, x_normed: np.ndarray) -> np.ndarray:
    """Load a profile-model checkpoint and run it on normalized input.

    Returns array of shape (n_stations,) — the encoded target prediction
    (signed_log1p_alpha_D in local-velocity basis if the model was trained
    with local_velocity_normalization=True).
    """
    import torch

    import physicsnemo  # delayed — heavy import; keep out of module scope

    entrypoint = str(run_meta.get("entrypoint") or "")
    if ":" not in entrypoint:
        raise ValueError("run_meta entrypoint must use 'module.path:callable' format")
    importlib.import_module(entrypoint.rsplit(":", 1)[0])

    model = physicsnemo.Module.from_checkpoint(str(checkpoint_path))
    model.eval()

    with torch.no_grad():
        x_t = torch.from_numpy(x_normed)
        y_t = model(x_t)  # expected (1, n_outputs, n_stations) profile NCL

    if y_t.dim() != 3 or y_t.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected forward output shape: {tuple(y_t.shape)}; "
            "expected (1, n_outputs, n_stations)."
        )

    output_columns = run_meta["data"].get("output_columns", [])
    if len(output_columns) != 1:
        raise RuntimeError(
            f"This exporter expects exactly one output column; got {output_columns}."
        )

    return y_t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)


def require_export_run_meta(run_meta: dict) -> bool:
    """Validate the metadata schema and return the persisted baseline flag."""
    schema = run_meta.get("training_run_meta_schema")
    if schema != 3:
        raise ValueError(
            f"Unsupported training_run_meta_schema={schema!r}; alpha-D export requires schema 3."
        )
    effective_data = run_meta.get("data", {}).get("effective")
    if (
        not isinstance(effective_data, dict)
        or "include_acceleration_head" not in effective_data
    ):
        raise ValueError(
            "run_meta data.effective.include_acceleration_head is required for alpha-D export."
        )
    include_acceleration_head = effective_data["include_acceleration_head"]
    if not isinstance(include_acceleration_head, bool):
        raise ValueError("Persisted include_acceleration_head must be a boolean.")
    return include_acceleration_head


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ---- Load + sanity-check inputs ----
    if not args.zarr.exists():
        raise FileNotFoundError(f"Case zarr not found: {args.zarr}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.run_meta.exists():
        raise FileNotFoundError(f"run_meta.json not found: {args.run_meta}")

    with args.run_meta.open() as fh:
        run_meta = json.load(fh)
    include_acceleration_head = require_export_run_meta(run_meta)
    effective_data = run_meta["data"]["effective"]
    case = load_case_from_zarr(args.zarr)

    # ---- Model inference ----
    x = build_model_input(case, run_meta)
    y_encoded = forward(args.checkpoint, run_meta, x)

    # ---- Add the residual baseline back, then decode to bulk-basis alpha_D ----
    feat_idx = case.feature_names.index("d_local_over_D")
    d_local_over_D = case.features[:, feat_idx].astype(np.float64)
    feat_idx_z = case.feature_names.index("z_hat")
    z_hat = case.features[:, feat_idx_z].astype(np.float64)
    output_columns = run_meta["data"]["output_columns"]
    local_norm = bool(run_meta["data"].get("local_velocity_normalization", False))

    # AlphaDProfileDataset auto-injects the residual transform via setdefault,
    # so the model predicts residuals from a closed-form baseline. Reproduce
    # the training-time add-back by computing the encoded baseline for this
    # case's geometry and adding it to the model output BEFORE decoding.
    has_target_baseline = bool(effective_data.get("has_target_baseline"))
    if has_target_baseline:
        baseline_encoded = compute_baseline_encoded(
            case,
            z_hat,
            d_local_over_D,
            local_velocity_normalization=local_norm,
            target_name=output_columns[0],
            include_acceleration_head=include_acceleration_head,
        )
        y_encoded_total = np.asarray(y_encoded, dtype=np.float64) + baseline_encoded
    else:
        y_encoded_total = np.asarray(y_encoded, dtype=np.float64)

    alpha_d_bulk = decode_to_bulk_alpha_d(
        y_encoded_total,
        d_local_over_D=d_local_over_D,
        local_velocity_normalization=local_norm,
        target_name=output_columns[0],
    )
    if not np.all(np.isfinite(alpha_d_bulk)):
        raise RuntimeError("Decoded alpha_D contains non-finite values.")
    if np.any(alpha_d_bulk <= 0.0):
        # Surrogate may legitimately predict negative alpha_D in recovery
        # regions; this is informational only.
        print(
            f"[warn] {np.sum(alpha_d_bulk <= 0)} of {len(alpha_d_bulk)} "
            "stations have non-positive alpha_D (recovery region or noise)."
        )

    # ---- ROI axial coordinates ----
    feat_idx_throat = case.feature_names.index("is_throat")
    is_throat = case.features[:, feat_idx_throat].astype(np.float64)

    throat_length_m = case.outer_height_m * case.Lr
    end_length_m = case.buffer_diams * case.D_big
    roi_length_m = throat_length_m + 2.0 * end_length_m
    z_phys = z_hat * roi_length_m  # ROI-local axial coordinate, in meters

    # Local hydraulic diameter at each ROI station: D_big outside throat,
    # Dr*D_big inside throat. Use d_local_over_D directly.
    D_h_roi = d_local_over_D * case.D_big

    delta_p_surrogate = integrate_delta_p(
        alpha_d_bulk,
        z_phys,
        D_h_roi,
        case.rho,
        case.V_bulk,
    )

    # ---- Block-aware full-ROI Forchheimer mapping ----
    # Per-station porosity: Dr² in throat, 1.0 in buffers
    porosity_per_station = np.where(is_throat > 0.5, case.Dr**2, 1.0)
    # D_h_roi already computed above (d_local_over_D * D_big)
    cf_all = alpha_d_to_forchheimer(
        alpha_d_bulk,
        porosity=porosity_per_station,
        D_h=D_h_roi,
    )

    # ---- Step-fence the porosity boundaries ----
    # MOOSE's PiecewiseLinear interpolates F across every adjacent CSV pair.
    # At the porosity steps (z = end_length and z = end_length + throat_length)
    # both ε and D_h jump, and the surrogate's α_D often spikes just upstream
    # of the contraction. Without explicit step-fencing, MOOSE's mesh cells
    # straddling z = 0.2 sample large interpolated F values that don't exist
    # in either block, over-counting the friction integral by ~3-5 Pa for the
    # target case (raising MOOSE_coupled to +24.8% vs truth).
    #
    # Inserting a duplicate-z fence (two rows separated by `step_eps`) at each
    # porosity boundary collapses the interpolation slope to a near-step and
    # restores MOOSE_coupled ≈ surrogate's full-ROI integral.
    step_eps = 1e-4  # well below mesh cell size (~0.01 m) and station spacing
    z_csv, cf_csv = _stepfence_porosity_boundaries(
        z_phys,
        cf_all,
        boundaries=(end_length_m, end_length_m + throat_length_m),
        step_eps=step_eps,
    )

    # ---- Write outputs ----
    # z is already in MOOSE mesh coordinates (ROI starts at inlet, x=0=inlet)
    # No pre-shift needed; PiecewiseLinear spans [0, roi_length].
    v_local_in_inputs = "V_local_over_V_bulk" in run_meta["data"].get(
        "input_columns", []
    )
    sidecar = {
        "coupling_export_schema": 1,
        "case_id": case.case_id,
        "checkpoint": str(args.checkpoint),
        "Re": case.Re,
        "Dr": case.Dr,
        "Lr": case.Lr,
        "D_big": case.D_big,
        "outer_height_m": case.outer_height_m,
        "buffer_diams": case.buffer_diams,
        "end_length_m": end_length_m,
        "throat_length_m": throat_length_m,
        "roi_length_m": roi_length_m,
        "rho": case.rho,
        "V_bulk": case.V_bulk,
        "forchheimer_multiplier_throat": (case.Dr**3) / case.D_big,
        "forchheimer_multiplier_buffer": 1.0 / case.D_big,
        "delta_p_truth": case.delta_p_truth,
        "delta_p_surrogate": delta_p_surrogate,
        "v_local_over_v_bulk_was_in_input_columns": v_local_in_inputs,
        "step_fence_eps": step_eps,
        "alpha_D_bulk_roi": alpha_d_bulk.tolist(),
        "z_phys_roi": z_phys.tolist(),
    }
    write_outputs(csv_path=args.output_csv, z=z_csv, cf=cf_csv, sidecar=sidecar)

    print(
        f"Wrote {args.output_csv} ({len(z_phys)} full-ROI stations); "
        f"delta_p_truth={case.delta_p_truth:.6g}, "
        f"delta_p_surrogate={delta_p_surrogate:.6g}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
