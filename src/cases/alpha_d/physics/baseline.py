"""Closed-form sudden contraction + throat-friction alpha_D baseline.

For an axisymmetric pipe with an abrupt contraction-then-expansion this
baseline keeps three terms whose CFD signature is genuinely localised:

  ΔP_baseline = ΔP_accel + K_c · q_throat + f_Darcy · (L_throat / D_throat) · q_throat

with q_throat = 0.5 ρ V_throat², q_bulk = 0.5 ρ V_bulk², and

  β²        = (D_throat / D_big)² = Dr²
  ΔP_accel  = q_throat − q_bulk = q_bulk · (1/Dr⁴ − 1)   # acceleration head
  K_c       = 0.5 · (1 − β²)                 # sudden contraction (Idelchik)
  V_throat  = V_bulk / Dr²
  Re_throat = Re_bulk / Dr
  f_Darcy   = 0.316 · Re_throat^(-1/4)       # Blasius (turbulent)

For this dataset Re_throat ∈ [5.5e3, 7.5e5] — always turbulent, so the
laminar branch is omitted.

**Acceleration head (ΔP_accel).**  The dominant term, ~half of the true
ΔP across the whole grid (see
``docs/superpowers/specs/2026-06-18-where-does-dp-live-findings.md``).
Because the contraction is *sudden* (d_local steps 1 → Dr in one cell),
the static pressure drops sharply at the contraction plane as the flow
accelerates to V_throat; the truncated 1-diameter downstream ROI never
recovers it, so it appears in ΔP_case.  Empirically (CFD α_D, 13 cases
over Dr∈[0.05,0.9], Lr∈[0.01,0.2]) this drop is a spike concentrated in
the last 1–2 stations *before* the throat entrance, of magnitude ≈
q_throat − q_bulk (matched within ~6–15%).  We deposit the analytic head
at the last station upstream of the throat — magnitude analytic (∝1/Dr⁴),
location geometric, no fitted constants.  This is a **shape prior matching
the measured static profile**, not a momentum-consistent reversible term:
the matching downstream recovery is absent from the ROI, so we do not add
it.  Putting the 1/Dr⁴ head in the baseline (rather than leaving the model
to invent it as a residual) is what lets the coupled prediction extrapolate
to small Dr.

The Borda-Carnot expansion loss K_e = (1 − β²)² is still **deliberately not
included**.  The downstream buffer carries ≈ 0 net ΔP in CFD (a sharp
Bernoulli recovery spike then flat), so the current baseline already matches
it; adding K_e as a positive spike there forces the model to fight a phantom
(worth O(K_e · D_h / dz) in α_D), which empirically destroyed the residual
fit.  The unrecovered expansion loss is already accounted for via ΔP_accel
above (for small Dr, K_e ≈ 1 ⇒ almost none of q_throat recovers, so the
acceleration head and the expansion loss are two views of the same head).

The baseline is exposed as a *profile* α_D(z) so that integrating

  dp/dz = α_D · ρ V_bulk² / (2 D_h)          # bulk-basis convention

over the ROI reproduces ΔP_baseline above.  ΔP_accel is localised at the
last upstream bin (D_h = D_big), K_c at the contraction-inlet bin
(D_h = Dr·D_big); both are single stations of physical width dz_phys =
L_roi / n_stations.  Friction is uniform across the throat interior.

The closed form is intentionally simple — its job is to provide a shape
prior so the model can fit the *residual* against CFD, not to replace the
CFD computation.  ``include_acceleration_head`` (default True) gates the
ΔP_accel term for A/B testing and reversibility; it must be the same at
training (``transforms.alpha_d_residual_transform``) and inference
(``export_friction_profile``) time, which the shared default guarantees.
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
        return (
            self.Lr * self.outer_height_m + self.buffer_diams * self.D_big
        ) / self.L_roi


def _blasius_friction_factor(re: float | np.ndarray) -> float | np.ndarray:
    """Darcy friction factor from Blasius (turbulent)."""
    return 0.316 * np.power(np.maximum(re, 1e-12), -0.25)


def alpha_d_baseline_profile(
    z_hat: np.ndarray,
    geom: BaselineGeometry,
    *,
    include_acceleration_head: bool = True,
) -> np.ndarray:
    """Compute the bulk-basis α_D baseline at each station.

    Parameters
    ----------
    z_hat
        Per-station normalised axial coordinate, shape ``[n_stations]``.
    geom
        Geometry + flow constants for the case.
    include_acceleration_head
        When True (default), deposit the sudden-contraction acceleration head
        ``ΔP_accel = q_throat − q_bulk`` at the last station upstream of the
        throat. See the module docstring. The flag exists for A/B / regression
        testing and reversibility; it must match between training and
        inference, which the shared default guarantees.

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

    # Acceleration head: the static drop as the flow accelerates into the
    # sudden contraction. CFD places it as a spike in the last 1–2 stations
    # *before* the throat entrance (d_local still = 1, so D_h = D_big). Deposit
    # the analytic head at the last upstream station, scaled so it integrates to
    #   ΔP_accel = q_throat − q_bulk = q_bulk · (1/Dr⁴ − 1)
    # over a span of dz_phys:  α = ΔP_accel · 2 D_big / (ρ V_bulk² · dz_phys).
    if include_acceleration_head:
        upstream = z < geom.z_throat_start_norm
        if np.any(upstream):
            accel_idx = np.flatnonzero(upstream)[int(np.argmax(z[upstream]))]
            alpha[accel_idx] += (1.0 / Dr4 - 1.0) * geom.D_big / geom.dz_phys

    # K_e is intentionally omitted — see the module docstring for why.
    return alpha


def integrated_baseline_delta_p(
    geom: BaselineGeometry,
    *,
    include_acceleration_head: bool = True,
) -> float:
    """ΔP predicted by the closed-form profile.

    ``ΔP_accel + K_c · q_throat + friction`` when ``include_acceleration_head``
    (default), else the legacy ``K_c · q_throat + friction``. By construction
    the per-station profile integrates trapezoidally to this value (each
    single-cell deposit contributes its full ΔP via two half-intervals); see
    ``tests/cases/alpha_d/test_baseline.py``. K_e is intentionally excluded —
    see the module docstring.
    """
    Dr2 = geom.Dr**2
    K_c = 0.5 * (1.0 - Dr2)
    Re_throat = geom.Re / geom.Dr
    f = _blasius_friction_factor(Re_throat)
    V_throat = geom.V_bulk / Dr2
    q_throat = 0.5 * geom.rho * V_throat**2
    q_bulk = 0.5 * geom.rho * geom.V_bulk**2
    L_throat = geom.Lr * geom.outer_height_m
    D_throat = geom.D_big * geom.Dr
    dp = (K_c + f * L_throat / D_throat) * q_throat
    if include_acceleration_head:
        # ΔP_accel = q_throat − q_bulk = q_bulk · (1/Dr⁴ − 1)
        dp += q_throat - q_bulk
    return float(dp)
