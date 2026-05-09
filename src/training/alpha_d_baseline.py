"""Closed-form sudden contraction + throat-friction alpha_D baseline.

For an axisymmetric pipe with an abrupt contraction-then-expansion this
baseline keeps two terms whose CFD signature is genuinely localised:

  ΔP_baseline = K_c · q_throat + f_Darcy · (L_throat / D_throat) · q_throat

with q_throat = 0.5 ρ V_throat², and

  β²        = (D_throat / D_big)² = Dr²
  K_c       = 0.5 · (1 − β²)                 # sudden contraction (Idelchik)
  V_throat  = V_bulk / Dr²
  Re_throat = Re_bulk / Dr
  f_Darcy   = 0.316 · Re_throat^(-1/4)       # Blasius (turbulent)

For this dataset Re_throat ∈ [5.5e3, 7.5e5] — always turbulent, so the
laminar branch is omitted.

The Borda-Carnot expansion loss K_e = (1 − β²)² is **deliberately not
included**.  In CFD the expansion loss is dissipated gradually over
several diameters downstream, and the local α_D right at the expansion
plane is dominated by Bernoulli-like pressure recovery (often slightly
negative).  Adding K_e as a single positive spike at the outlet bin
forces the model to fight a phantom artifact (worth O(K_e · D_h / dz)
in α_D), which empirically destroyed the residual fit.  We let the
model learn the distributed downstream loss on its own.

The baseline is exposed as a *profile* α_D(z) so that integrating

  dp/dz = α_D · ρ V_bulk² / (2 D_h)          # bulk-basis convention

over the ROI reproduces ΔP_baseline above.  K_c is localised at the
contraction-inlet bin (single station of physical width dz_phys =
L_roi / n_stations); friction is uniform across the throat interior.

The closed form is intentionally simple — its job is to provide a
smooth shape prior so the MLP can fit the *residual* against CFD,
not to replace the CFD computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BaselineGeometry:
    """Per-case geometry constants required to evaluate the baseline."""
    Re: float
    Dr: float
    Lr: float
    D_big: float = 0.2
    outer_height_m: float = 1.0
    buffer_diams: float = 1.0
    rho: float = 1.0
    V_bulk: float = 1.0
    n_stations: int = 50

    @property
    def L_roi(self) -> float:
        return self.Lr * self.outer_height_m + 2.0 * self.buffer_diams * self.D_big

    @property
    def dz_phys(self) -> float:
        return self.L_roi / float(self.n_stations)

    @property
    def dz_norm(self) -> float:
        return 1.0 / float(self.n_stations)

    @property
    def z_throat_start_norm(self) -> float:
        # z_roi_start = 0.5*outer - buffer*D_big  →  throat start = 0.5*outer
        return self.buffer_diams * self.D_big / self.L_roi

    @property
    def z_throat_end_norm(self) -> float:
        return (self.Lr * self.outer_height_m + self.buffer_diams * self.D_big) / self.L_roi


def _blasius_friction_factor(re: float | np.ndarray) -> float | np.ndarray:
    """Darcy friction factor from Blasius (turbulent)."""
    return 0.316 * np.power(np.maximum(re, 1e-12), -0.25)


def alpha_d_baseline_profile(z_hat: np.ndarray, geom: BaselineGeometry) -> np.ndarray:
    """Compute the bulk-basis α_D baseline at each station.

    Parameters
    ----------
    z_hat
        Per-station normalised axial coordinate, shape ``[n_stations]``.
    geom
        Geometry + flow constants for the case.

    Returns
    -------
    alpha_D : ndarray
        Bulk-basis α_D = -dp/dz · 2 D_h / (ρ V_bulk²) at each station.
    """
    z = np.asarray(z_hat, dtype=np.float64)
    Dr = geom.Dr
    Dr2 = Dr * Dr
    Dr4 = Dr2 * Dr2

    K_c = 0.5 * (1.0 - Dr2)
    Re_throat = geom.Re / Dr
    f_blasius = _blasius_friction_factor(Re_throat)
    D_h_throat = geom.D_big * Dr

    in_throat = (z >= geom.z_throat_start_norm) & (z <= geom.z_throat_end_norm)
    if not np.any(in_throat):
        # Degenerate (Lr → 0); model has no signal here either.
        return np.zeros_like(z)

    is_inlet_bin = in_throat & ((z - geom.z_throat_start_norm) < geom.dz_norm)

    alpha = np.zeros_like(z)

    # Throat-interior friction (constant over the throat in bulk basis):
    #   α_D_friction_bulk = f / Dr⁴, since α_D_local = f and the
    #   bulk↔local rescaling is V_throat²/V_bulk² = 1/Dr⁴.
    alpha[in_throat] += f_blasius / Dr4

    # K_c localised in the inlet bin, scaled so it integrates to
    #   ΔP_c = K_c * 0.5 * ρ * V_throat²
    # over a span of dz_phys.
    alpha[is_inlet_bin] += K_c * D_h_throat / (Dr4 * geom.dz_phys)

    # K_e is intentionally omitted — see the module docstring for why.
    return alpha


def integrated_baseline_delta_p(geom: BaselineGeometry) -> float:
    """ΔP predicted by the closed-form profile (K_c + throat friction).

    K_e is intentionally excluded — see the module docstring.
    """
    Dr2 = geom.Dr ** 2
    K_c = 0.5 * (1.0 - Dr2)
    Re_throat = geom.Re / geom.Dr
    f = _blasius_friction_factor(Re_throat)
    V_throat = geom.V_bulk / Dr2
    q_throat = 0.5 * geom.rho * V_throat ** 2
    L_throat = geom.Lr * geom.outer_height_m
    D_throat = geom.D_big * geom.Dr
    return float((K_c + f * L_throat / D_throat) * q_throat)
