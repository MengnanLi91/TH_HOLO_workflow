"""AlphaDTransformation: extract contraction ROI, compute Darcy coefficient.

Implements the DataTransformation ABC from physicsnemo-curator.

For each CFD case the transformation:
  1. Computes element centroids from node coords + connectivity.
  2. Identifies the contraction region from geometry parameters.
  3. Bins elements into axial (z) stations.
  4. Computes cross-section-averaged pressure at each station.
  5. Derives the Darcy resistance coefficient alpha_D via the pressure
     gradient and local hydraulic diameter.
  6. Constructs the feature table that the MLP will consume.

Output dict contains ``features`` [N_stations, D_in], ``targets``
[N_stations, 1], and metadata.
"""

import logging
import math
from typing import Any, Optional

import numpy as np

from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig
from training.alpha_d_targets import encode_alpha_d_target

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _local_diameter(z: np.ndarray, z_throat_start: float, z_throat_end: float,
                    D_big: float, D_small: float) -> np.ndarray:
    """Compute local pipe diameter along the z-axis.

    Upstream/downstream of the contraction the diameter is D_big.
    In the contraction region between z_throat_start and z_throat_end,
    the diameter is D_small (abrupt contraction model).
    """
    d = np.full_like(z, D_big, dtype=np.float64)
    mask = (z >= z_throat_start) & (z <= z_throat_end)
    d[mask] = D_small
    return d


def _region_flags(z_hat: np.ndarray, z_norm_throat_start: float,
                  z_norm_throat_end: float) -> dict[str, np.ndarray]:
    """Compute region flags from normalized z coordinate.

    Returns ``is_upstream``, ``is_throat``, ``is_downstream`` as float32
    one-hot arrays.  The ``is_contraction`` and ``is_expansion`` flags are
    omitted because the abrupt geometry makes them always zero.
    """
    upstream = (z_hat < z_norm_throat_start).astype(np.float32)
    downstream = (z_hat > z_norm_throat_end).astype(np.float32)
    throat = ((z_hat >= z_norm_throat_start) & (z_hat <= z_norm_throat_end)).astype(np.float32)
    return {
        "is_upstream": upstream,
        "is_throat": throat,
        "is_downstream": downstream,
    }


def _sample_weights(region_flags: dict[str, np.ndarray]) -> np.ndarray:
    """Assign per-station sample weights for future weighted loss."""
    w = np.ones(len(region_flags["is_upstream"]), dtype=np.float32)
    w[region_flags["is_throat"] > 0.5] = 5.0
    return w


class AlphaDTransformation(DataTransformation):
    """Extract contraction-region Darcy resistance profiles.

    Args:
        cfg          : ProcessingConfig.
        n_stations   : Number of axial stations to bin into.
        buffer_diams : Number of pipe diameters of upstream/downstream buffer.
        rho          : Fluid density (kg/m^3), matching the CFD setup.
        min_elements : Minimum elements required per station; skip case if not met.
    """

    def __init__(
        self,
        cfg: ProcessingConfig,
        n_stations: int = 50,
        buffer_diams: float = 1.0,
        rho: float = 1.0,
        min_elements: int = 3,
    ):
        super().__init__(cfg)
        self.n_stations = n_stations
        self.buffer_diams = buffer_diams
        self.rho = rho
        self.min_elements = min_elements

    def transform(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Compute alpha_D axial profile for one CFD case."""
        case_name: str = data["case_name"]
        case_meta: dict = data["case_meta"]
        coords: np.ndarray = data["coords"]          # [N, 3] in meters
        connectivity: np.ndarray = data["connectivity"]
        field_names: list[str] = data["field_names"]
        fields: np.ndarray = data["fields"]           # [T, E, F]

        # --- Extract case parameters ---
        Re = float(case_meta.get("Re", 0))
        Dr = float(case_meta.get("diameter_ratio", 0))
        Lr = float(case_meta.get("length_ratio", 0))
        pipe_radius_m = float(case_meta.get("pipe_radius_m", 0.1))
        D_big = 2.0 * pipe_radius_m
        D_contraction_m = float(case_meta.get("D_contraction_m", D_big * Dr))

        if Re <= 0 or Dr <= 0 or Lr <= 0:
            logger.warning("Skipping %s: invalid case parameters (Re=%s Dr=%s Lr=%s)",
                           case_name, Re, Dr, Lr)
            return None

        # --- Geometry: locate the contraction region ---
        # Geometry is handled in meter units after source mesh scaling.
        # Lower pipe center at z=0, inner cylinder center at
        # z = outer_height/2 + inner_height/2
        outer_height_m = 1.0
        inner_height_m = outer_height_m * Lr
        z_throat_start = outer_height_m / 2.0  # 0.5 m
        z_throat_end = z_throat_start + inner_height_m

        # ROI: throat +/- buffer
        buffer = self.buffer_diams * D_big
        z_roi_start = z_throat_start - buffer
        z_roi_end = z_throat_end + buffer

        # --- Compute element centroids ---
        # coords: [N, 3], connectivity: [E, K]
        elem_centroids = coords[connectivity].mean(axis=1)  # [E, 3]
        elem_z = elem_centroids[:, 2]  # z-component

        # --- Identify pressure field ---
        if "pressure" not in field_names:
            logger.warning("Skipping %s: no 'pressure' field found.", case_name)
            return None
        p_idx = field_names.index("pressure")

        # Use last time step (converged steady state)
        pressure = fields[-1, :, p_idx]  # [E]

        # --- Select ROI elements ---
        roi_mask = (elem_z >= z_roi_start) & (elem_z <= z_roi_end)
        n_roi = roi_mask.sum()
        if n_roi < self.n_stations * self.min_elements:
            logger.warning(
                "Skipping %s: only %d ROI elements (need %d).",
                case_name, n_roi, self.n_stations * self.min_elements,
            )
            return None

        # --- Bin into axial stations ---
        station_edges = np.linspace(z_roi_start, z_roi_end, self.n_stations + 1)
        station_z = 0.5 * (station_edges[:-1] + station_edges[1:])

        # Compute element cross-sectional areas (projected onto z-plane)
        # For hex/tet elements, approximate as the area of the polygon
        # formed by node x,y coordinates projected onto the z-plane.
        # Simplified: use radial position to weight by annular area for
        # axisymmetric meshes.
        elem_r = np.sqrt(elem_centroids[:, 0]**2 + elem_centroids[:, 1]**2)

        p_avg = np.zeros(self.n_stations, dtype=np.float64)
        counts = np.zeros(self.n_stations, dtype=np.int32)

        for s in range(self.n_stations):
            mask = roi_mask & (elem_z >= station_edges[s]) & (elem_z < station_edges[s + 1])
            if mask.sum() == 0:
                continue
            # Area-weighted average: weight by r for axisymmetric
            r_weights = elem_r[mask]
            r_weights = np.maximum(r_weights, 1e-12)  # avoid zero
            p_avg[s] = np.average(pressure[mask], weights=r_weights)
            counts[s] = mask.sum()

        # Handle last bin edge
        if counts[-1] == 0 and counts[-2] > 0:
            p_avg[-1] = p_avg[-2]
            counts[-1] = 1

        # Skip if too many empty stations
        valid = counts > 0
        if valid.sum() < 5:
            logger.warning("Skipping %s: too few valid stations (%d).", case_name, valid.sum())
            return None

        # Interpolate missing stations
        if not valid.all():
            p_avg[~valid] = np.interp(station_z[~valid], station_z[valid], p_avg[valid])

        # --- Compute pressure gradient dP/dz ---
        dz = station_z[1] - station_z[0]
        dp_dz = np.gradient(p_avg, dz)

        # --- Compute Darcy resistance coefficient ---
        # alpha_D = -dP/dz / (0.5 * rho * V_bulk^2 / D_h)
        # V_bulk = inlet_u = 1.0 m/s (from simulation.i)
        V_bulk = 1.0
        D_local = _local_diameter(station_z, z_throat_start, z_throat_end, D_big, D_contraction_m)
        D_h = D_local  # hydraulic diameter = pipe diameter for circular cross-section

        # Darcy friction formulation: -dP/dz = alpha_D * (rho * V^2) / (2 * D_h)
        # => alpha_D = -dP/dz * 2 * D_h / (rho * V^2)
        alpha_D = -dp_dz * 2.0 * D_h / (self.rho * V_bulk**2)

        # --- Total pressure drop across ROI ---
        delta_p_case = float(p_avg[0] - p_avg[-1])

        # --- Build feature vectors for ALL stations ---
        z_hat = (station_z - z_roi_start) / (z_roi_end - z_roi_start)
        d_local_over_D = D_local / D_big
        A_local_over_A = (D_local / D_big) ** 2  # area ratio
        regions = _region_flags(
            z_hat,
            (z_throat_start - z_roi_start) / (z_roi_end - z_roi_start),
            (z_throat_end - z_roi_start) / (z_roi_end - z_roi_start),
        )

        # --- Encode targets ---
        # Keep the historical positive-only log target for backward
        # compatibility, and add a signed target that preserves favorable
        # pressure-gradient regions for physics-consistent training.
        alpha_D_floor = 1e-3
        log_alpha_D = encode_alpha_d_target(
            np.maximum(alpha_D, alpha_D_floor),
            target_name="log_alpha_D",
        ).astype(np.float32)
        signed_log1p_alpha_D = encode_alpha_d_target(
            alpha_D,
            target_name="signed_log1p_alpha_D",
        ).astype(np.float32)
        weights = _sample_weights(regions)

        feature_names = [
            "log10_Re", "Dr", "Lr", "z_hat", "d_local_over_D",
            "A_local_over_A", "is_upstream", "is_throat", "is_downstream",
        ]
        target_names = ["log_alpha_D", "signed_log1p_alpha_D"]

        n_out = self.n_stations
        features = np.column_stack([
            np.full(n_out, math.log10(Re), dtype=np.float32),
            np.full(n_out, Dr, dtype=np.float32),
            np.full(n_out, Lr, dtype=np.float32),
            z_hat.astype(np.float32),
            d_local_over_D.astype(np.float32),
            A_local_over_A.astype(np.float32),
            regions["is_upstream"],
            regions["is_throat"],
            regions["is_downstream"],
        ])  # [n_out, 9]

        targets = np.column_stack([log_alpha_D, signed_log1p_alpha_D]).astype(np.float32)

        self.logger.info(
            "  %s: %d stations, delta_p=%.4f Pa, "
            "log_alpha_D range=[%.2f, %.2f], "
            "signed_log1p_alpha_D range=[%.2f, %.2f]",
            case_name, n_out, delta_p_case,
            float(log_alpha_D.min()), float(log_alpha_D.max()),
            float(signed_log1p_alpha_D.min()), float(signed_log1p_alpha_D.max()),
        )

        return {
            "case_name": case_name,
            "features": features,
            "targets": targets,
            "feature_names": feature_names,
            "target_names": target_names,
            "sample_weight": weights,
            "delta_p_case": delta_p_case,
            "Re": Re,
            "Dr": Dr,
            "Lr": Lr,
        }
