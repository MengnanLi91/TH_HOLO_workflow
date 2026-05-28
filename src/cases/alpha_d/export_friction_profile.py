"""Export an alpha_D surrogate prediction as a MOOSE Forchheimer profile.

For a single target case, this script:
  1. Loads features from the case's existing ETL `.zarr` store.
  2. Loads the trained Conv1D checkpoint.
  3. Runs forward inference to get the per-station alpha_D profile.
  4. Maps alpha_D(z) to MOOSE PINSFV Forchheimer coefficient C_F(z) using
     the throat-only 1D plug-flow equivalence.
  5. Writes forchheimer_profile.csv (z, F) plus a sidecar metadata.json.

The CSV is consumed by MOOSE's PiecewiseLinear function backing the
ADGenericVectorFunctorMaterial on block 2 (the throat).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import zarr


@dataclass
class Args:
    zarr: Path
    checkpoint: Path
    run_meta: Path
    output_csv: Path


def parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to the target case's .zarr store."
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
        "--output-csv", type=Path, required=True, help="Output CSV path (Forchheimer profile)."
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

    Returns array of shape (1, n_features, n_stations) — Conv1D NCL.
    Reuses `cases.alpha_d.feature_data.build_engineered_feature_map` for
    engineered columns to match training-time feature synthesis exactly.
    """
    from cases.alpha_d.feature_data import build_engineered_feature_map

    input_columns: list[str] = list(run_meta["data"]["input_columns"])
    x_mean = np.asarray(run_meta["data"]["norm_stats"]["x_mean"], dtype=np.float32)
    x_std = np.asarray(run_meta["data"]["norm_stats"]["x_std"], dtype=np.float32)
    if len(x_mean) != len(input_columns) or len(x_std) != len(input_columns):
        raise ValueError(
            f"norm_stats length mismatch: {len(x_mean)} mean / "
            f"{len(x_std)} std vs {len(input_columns)} input_columns"
        )

    raw_name_to_idx = {n: i for i, n in enumerate(case.feature_names)}
    engineered = build_engineered_feature_map(case.features, case.feature_names)

    columns: list[np.ndarray] = []
    for name in input_columns:
        if name in raw_name_to_idx:
            col = case.features[:, raw_name_to_idx[name]].astype(np.float32)
        elif name in engineered:
            col = engineered[name].astype(np.float32)
        else:
            raise ValueError(
                f"input_column {name!r} not found in zarr feature_names "
                f"nor engineered set. zarr columns: {case.feature_names}; "
                f"engineered: {list(engineered.keys())}."
            )
        columns.append(col)

    raw_x = np.stack(columns, axis=0)  # (C, L)
    normed = (raw_x - x_mean[:, None]) / x_std[:, None]
    return normed[None, :, :].astype(np.float32)  # (1, C, L)


def forward(checkpoint_path: Path, run_meta: dict, x_normed: np.ndarray) -> np.ndarray:
    """Load Conv1D checkpoint, run forward pass on normalized input.

    Returns array of shape (n_stations,) — the encoded target prediction
    (signed_log1p_alpha_D in local-velocity basis if the model was trained
    with local_velocity_normalization=True).
    """
    import physicsnemo  # delayed — heavy import; keep out of module scope

    # Ensure the Conv1DProfile class is defined (it registers itself only
    # when physicsnemo is importable) so from_checkpoint can find it.
    import training.models.conv1d_profile  # noqa: F401

    model = physicsnemo.Module.from_checkpoint(str(checkpoint_path))
    model.eval()

    with torch.no_grad():
        x_t = torch.from_numpy(x_normed)
        y_t = model(x_t)  # expected (1, n_outputs, n_stations) for Conv1D NCL

    if y_t.dim() != 3 or y_t.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected forward output shape: {tuple(y_t.shape)}; "
            "expected (1, n_outputs, n_stations)."
        )

    output_columns = run_meta["data"].get("output_columns", [])
    if len(output_columns) != 1:
        raise RuntimeError(f"Phase 1 expects exactly one output column; got {output_columns}.")

    return y_t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)


def alpha_d_to_forchheimer(alpha_d_bulk: np.ndarray, *, Dr: float, D_outer: float) -> np.ndarray:
    """Spec section 4.3 mapping: C_F = alpha_D / (2 * Dr^5 * D_outer).

    Throat-only 1D plug-flow equivalence with V_bulk = V_interstitial,
    D_h = D_throat = Dr * D_outer, porosity = Dr^2.
    """
    if Dr <= 0.0 or D_outer <= 0.0:
        raise ValueError(f"Dr and D_outer must be positive; got Dr={Dr}, D_outer={D_outer}.")
    denom = 2.0 * (Dr**5) * D_outer
    return np.asarray(alpha_d_bulk, dtype=np.float64) / denom


def restrict_to_throat(
    *,
    z_hat: np.ndarray,
    is_throat: np.ndarray,
    values: np.ndarray,
    throat_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only throat stations; remap z_hat to MOOSE throat coords.

    Returns (z_moose, values_at_throat). z_moose spans [0, throat_length].
    """
    mask = np.asarray(is_throat).astype(np.float64) > 0.5
    if not mask.any():
        raise ValueError("No throat stations found (is_throat all zero).")

    z_hat_throat = np.asarray(z_hat, dtype=np.float64)[mask]
    vals_throat = np.asarray(values, dtype=np.float64)[mask]

    order = np.argsort(z_hat_throat)
    z_hat_throat = z_hat_throat[order]
    vals_throat = vals_throat[order]

    z_start = z_hat_throat[0]
    z_end = z_hat_throat[-1]
    if z_end <= z_start:
        raise ValueError("Throat z_hat span is zero or negative; check inputs.")

    z_moose = (z_hat_throat - z_start) / (z_end - z_start) * float(throat_length)
    return z_moose, vals_throat


def decode_to_bulk_alpha_d(
    encoded: np.ndarray,
    *,
    d_local_over_D: np.ndarray,
    local_velocity_normalization: bool,
    target_name: str,
) -> np.ndarray:
    """Invert the encoder; convert local-velocity basis to bulk basis if needed.

    ``alpha_d_values_to_bulk`` handles both steps in one call:
    1. Decodes the encoded target (e.g. signed_log1p) to physical alpha_D.
    2. If ``local_velocity_normalization`` is True, converts from local-velocity
       basis to bulk-velocity basis by dividing by ``(d_local/D)^4``.
    """
    from cases.alpha_d.physics.targets import alpha_d_values_to_bulk

    encoded = np.asarray(encoded, dtype=np.float64)
    alpha = alpha_d_values_to_bulk(
        encoded,
        target_name=target_name,
        d_over_D=np.asarray(d_local_over_D, dtype=np.float64),
        local_velocity_normalization=local_velocity_normalization,
    )
    return np.asarray(alpha, dtype=np.float64)


def write_outputs(*, csv_path: Path, z: np.ndarray, cf: np.ndarray, sidecar: dict) -> None:
    """Write z/F CSV plus a sidecar JSON with metadata."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as fh:
        fh.write("z,F\n")
        for zi, ci in zip(z, cf):
            fh.write(f"{zi:.9e},{ci:.9e}\n")
    sidecar_path = csv_path.with_suffix(".meta.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))


def _integrate_delta_p_with_z_phys(
    alpha_d_bulk: np.ndarray,
    z_phys: np.ndarray,
    D_h: np.ndarray,
    rho: float,
    V_bulk: float,
) -> float:
    """Trapezoidal integral of -dP/dz over ROI.

    -dP/dz = alpha_D * rho * V_bulk**2 / (2 * D_h)
    delta_p = integral of (-dP/dz) dz over [z_start, z_end] (ROI)
    """
    integrand = alpha_d_bulk * rho * V_bulk**2 / (2.0 * D_h)
    return float(np.trapz(integrand, x=z_phys))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ---- Load + sanity-check inputs ----
    if not args.zarr.exists():
        raise FileNotFoundError(f"Case zarr not found: {args.zarr}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.run_meta.exists():
        raise FileNotFoundError(f"run_meta.json not found: {args.run_meta}")

    case = load_case_from_zarr(args.zarr)
    with args.run_meta.open() as fh:
        run_meta = json.load(fh)

    # ---- Model inference ----
    x = build_model_input(case, run_meta)
    y_encoded = forward(args.checkpoint, run_meta, x)

    # ---- Decode to bulk-basis alpha_D ----
    feat_idx = case.feature_names.index("d_local_over_D")
    d_local_over_D = case.features[:, feat_idx].astype(np.float64)
    output_columns = run_meta["data"]["output_columns"]
    local_norm = bool(run_meta["data"].get("local_velocity_normalization", False))

    alpha_d_bulk = decode_to_bulk_alpha_d(
        y_encoded,
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

    # ---- ROI -> throat axial mapping ----
    feat_idx_z = case.feature_names.index("z_hat")
    feat_idx_throat = case.feature_names.index("is_throat")
    z_hat = case.features[:, feat_idx_z].astype(np.float64)
    is_throat = case.features[:, feat_idx_throat].astype(np.float64)

    throat_length_m = case.outer_height_m * case.Lr
    roi_length_m = throat_length_m + 2.0 * case.buffer_diams * case.D_big
    z_phys = z_hat * roi_length_m  # ROI-local axial coordinate, in meters

    # Local hydraulic diameter at each ROI station: D_big outside throat,
    # Dr*D_big inside throat. Use d_local_over_D directly.
    D_h_roi = d_local_over_D * case.D_big

    delta_p_surrogate = _integrate_delta_p_with_z_phys(
        alpha_d_bulk=alpha_d_bulk,
        z_phys=z_phys,
        D_h=D_h_roi,
        rho=case.rho,
        V_bulk=case.V_bulk,
    )

    z_moose, alpha_throat = restrict_to_throat(
        z_hat=z_hat,
        is_throat=is_throat,
        values=alpha_d_bulk,
        throat_length=throat_length_m,
    )
    cf_throat = alpha_d_to_forchheimer(alpha_throat, Dr=case.Dr, D_outer=case.D_big)

    # ---- Write outputs ----
    v_local_in_inputs = "V_local_over_V_bulk" in run_meta["data"].get("input_columns", [])
    sidecar = {
        "case_id": case.case_id,
        "checkpoint": str(args.checkpoint),
        "Re": case.Re,
        "Dr": case.Dr,
        "Lr": case.Lr,
        "D_big": case.D_big,
        "outer_height_m": case.outer_height_m,
        "buffer_diams": case.buffer_diams,
        "throat_length_m": throat_length_m,
        "roi_length_m": roi_length_m,
        "rho": case.rho,
        "V_bulk": case.V_bulk,
        "denom_2_Dr5_Douter": 2.0 * (case.Dr**5) * case.D_big,
        "delta_p_truth": case.delta_p_truth,
        "delta_p_surrogate": delta_p_surrogate,
        "v_local_over_v_bulk_was_in_input_columns": v_local_in_inputs,
        "alpha_D_bulk_roi": alpha_d_bulk.tolist(),
        "z_phys_roi": z_phys.tolist(),
    }
    write_outputs(csv_path=args.output_csv, z=z_moose, cf=cf_throat, sidecar=sidecar)

    print(
        f"Wrote {args.output_csv} ({len(z_moose)} throat stations); "
        f"delta_p_truth={case.delta_p_truth:.6g}, "
        f"delta_p_surrogate={delta_p_surrogate:.6g}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
