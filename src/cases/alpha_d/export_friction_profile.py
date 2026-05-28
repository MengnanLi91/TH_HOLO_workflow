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


def main(argv: list[str] | None = None) -> int:
    _args = parse_args(argv)
    raise NotImplementedError(
        "export_friction_profile pipeline not yet implemented; "
        "see plan docs/superpowers/plans/2026-05-28-alpha-d-moose-coupling.md."
    )


if __name__ == "__main__":
    raise SystemExit(main())
