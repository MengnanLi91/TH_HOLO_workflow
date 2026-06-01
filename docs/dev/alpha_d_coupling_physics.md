# Alpha-D Surrogate → MOOSE PINSFV Coupling: Physics Reference

This document is the single source of truth for the physics, equations,
averaging conventions, and assumptions that underlie the alpha-D surrogate
and its coupling to MOOSE PINSFV.

It cross-references the implementation:

- Training-side ETL: `src/cases/alpha_d/etl/transform.py`
- Target decode/encode: `src/cases/alpha_d/physics/targets.py`
- Surrogate → MOOSE exporter: `src/cases/alpha_d/export_friction_profile.py`
- MOOSE input: `src/cases/alpha_d/moose/2d-porous-flow_alphaD.i`
- MOOSE PINSFV kernels: `moose/modules/navier_stokes/src/fvkernels/`

If equations here ever drift from the code, **the code is right**. Update
this document to match.

---

## 1. Notation

| Symbol | Meaning | Units |
|---|---|---|
| `ρ` | Fluid density | kg/m³ |
| `μ` | Dynamic viscosity | Pa·s |
| `V_bulk` | Case-level reference velocity (= inlet superficial velocity) | m/s |
| `V_local(z)` | Area-averaged streamwise velocity at axial station `z` | m/s |
| `v_super` | Superficial velocity (PINSFV's primary variable) | m/s |
| `v_inter` | Interstitial velocity, = `v_super / ε` | m/s |
| `ε(z)` | Porosity (PINSFV homogenized volume fraction) | — |
| `D_outer` | Pipe outer diameter | m |
| `d_local(z)` | Local pipe diameter (abrupt model: `D_outer` outside throat, `Dr·D_outer` inside) | m |
| `D_h(z)` | Hydraulic diameter, = `d_local(z)` for circular cross-section | m |
| `Re` | Throat Reynolds number, = `ρ V_bulk D_contraction / μ` | — |
| `Dr` | Diameter ratio, = `d_throat / D_outer` | — |
| `Lr` | Length ratio, = `inner_height / outer_height` | — |
| `α_D(z)` | Darcy-Weisbach friction factor (per-station) | — |
| `F(z)` | MOOSE Forchheimer coefficient (per-station) | 1/m |
| `p` | Static or Bernoulli pressure, depending on variable type | Pa |

Throughout the training data and the MOOSE input file, we use SI units with
`ρ = 1 kg/m³` and `V_bulk = 1 m/s` to keep dimensionless quantities
identifiable from their numerical values.

---

## 2. Training side: derivation of α_D(z)

### 2.1. Source data

A parametric study of 770 MOOSE 3-D k-ε RANS simulations of axial flow
through a circular pipe with an internal contraction-expansion. Each case
is parameterized by `(Re, Dr, Lr)`. Cases are stored under
`data/flow_contraction_expansion/parametric_study/<case_name>/simulation_out.e`
with `case_metadata.txt` recording the geometry and Reynolds number.

The CFD is **geometrically resolved** — no porous-medium homogenization. The
contraction is a sudden change in pipe inner diameter from `D_outer` to
`d_throat = Dr · D_outer`, held over a length `inner_height = Lr · outer_height`,
then a sudden expansion back to `D_outer`.

### 2.2. Region of interest (ROI)

Per `transform.py:127-137`, the ROI spans the throat plus one outer pipe
diameter of buffer upstream and downstream:

```
ROI = [z_throat_start − buffer_diams · D_outer,
       z_throat_end   + buffer_diams · D_outer]
```

with `buffer_diams = 1.0` by default. The ROI is binned into `n_stations = 50`
uniform axial slices; each slice corresponds to one row in the per-station
feature/target tables stored in the case `.zarr`.

### 2.3. Cross-section averaging at each station

Per `transform.py:181-195`, for each station bin:

1. Identify all elements whose centroid `z` falls within the bin.
2. Compute the area-weighted average of pressure and axial velocity:

   ```
   ⟨p⟩(z)   = ∑ pᵢ · rᵢ / ∑ rᵢ        (over elements in bin)
   ⟨v_z⟩(z) = ∑ v_zᵢ · rᵢ / ∑ rᵢ
   ```

   The radial weighting `r` makes this an area-weighted average for the
   axisymmetric mesh (annular elements at radius `r` carry area
   proportional to `r`).

3. Empty bins are linearly interpolated from neighbors.

### 2.4. Discrete pressure gradient

Per `transform.py:215-217`:

```
dP/dz(z) = np.gradient(⟨p⟩, dz)
```

`np.gradient` uses centered differences in the interior and one-sided
differences at the edges of the 50-station grid.

### 2.5. Darcy-Weisbach friction factor definition

Per `transform.py:219-228`:

$$\boxed{\alpha_D(z) \;=\; \frac{-\,dP/dz \cdot 2\,D_h(z)}{\rho \cdot V_{\text{bulk}}^2}}$$

Equivalently:

$$-\frac{dP}{dz} \;=\; \alpha_D(z) \cdot \frac{\rho\,V_{\text{bulk}}^2}{2\,D_h(z)}$$

This is the standard Darcy-Weisbach pipe-flow friction-factor definition
(see e.g. Bird, Stewart & Lightfoot §6.2 or Munson et al. §8.2).

**Crucial conventions:**

- **`V_bulk = inlet velocity` is a CASE CONSTANT,** hard-coded to 1.0 m/s in
  the ETL. It is *not* the local mean velocity `V_local(z)`. This is a
  deliberate non-dimensionalization choice that scales every case
  uniformly against its own inlet.
- **`D_h(z)` uses the abrupt-contraction model** — `D_outer` outside the
  throat and `Dr · D_outer` inside, with a step at the throat boundaries.
  `transform.py:_local_diameter`.

### 2.6. Per-station feature row

After the α_D definition, the ETL constructs a 13-column feature row for
each station (`transform.py:276-310`):

| Feature | Per-row? | Definition |
|---|---|---|
| `log10_Re` | case-level | log10 of throat-based Re |
| `Dr` | case-level | diameter ratio |
| `Lr` | case-level | length ratio |
| `z_hat` | local | (z − z_roi_start) / roi_length, ∈ [0, 1] |
| `d_local_over_D` | local | `d_local / D_outer` (abrupt-step) |
| `A_local_over_A` | local | `(d_local / D_outer)²` |
| `V_local_over_V_bulk` | local | `⟨v_z⟩(z) / V_bulk` (simulation-derived) |
| `is_upstream` | local | indicator (z < z_throat_start) |
| `is_throat` | local | indicator (z ∈ throat) |
| `is_downstream` | local | indicator (z > z_throat_end) |
| `dD_dz_local` | local | `np.gradient(d_local_over_D, z_hat)`; spikes at edges |
| `dist_to_throat_start` | local | `z_hat − z_throat_start_hat` (signed) |
| `dist_to_throat_end` | local | `z_throat_end_hat − z_hat` (signed) |

Engineered features (`feature_data.py::ENGINEERED_FEATURES`) like
`log10_Re_throat`, `inv_Dr`, `dist_to_nearest_step` are synthesized at
load time from these 13.

### 2.7. Target encoding

The ETL writes two target columns per station (`transform.py:265-273`):

```
log_alpha_D          = log(max(α_D, 1e-3))                       (positive-clipped, legacy)
signed_log1p_alpha_D = sign(α_D) · log1p(|α_D|)                  (sign-preserving)
```

The `signed_log1p` form preserves the sign of α_D so that **recovery regions
with favorable pressure gradient (`-dP/dz < 0` → `α_D < 0`) survive** the
encoder. This is important downstream of the throat where pressure recovers
across the sudden expansion.

The Conv1D model in `data/cases/train_conv1d/` uses `signed_log1p_alpha_D`
as its single output column.

---

## 3. Local-velocity normalization

### 3.1. Two bases for α_D

The Darcy-Weisbach definition (§2.5) can equivalently be written against
either reference velocity:

$$\alpha_{D,\text{bulk}}(z) \;=\; \frac{-dP/dz \cdot 2\,D_h}{\rho\,V_{\text{bulk}}^2}$$

$$\alpha_{D,\text{local}}(z) \;=\; \frac{-dP/dz \cdot 2\,D_h}{\rho\,V_{\text{local}}(z)^2}$$

Taking the ratio:

$$\boxed{\alpha_{D,\text{local}}(z) \;=\; \alpha_{D,\text{bulk}}(z) \cdot \left(\frac{V_{\text{bulk}}}{V_{\text{local}}(z)}\right)^2}$$

### 3.2. The round-pipe identity

For incompressible plug flow in a circular pipe of varying cross-section,
mass conservation gives:

$$V_{\text{local}} \cdot A_{\text{local}} = V_{\text{bulk}} \cdot A_{\text{ref}} \quad\Rightarrow\quad \frac{V_{\text{local}}}{V_{\text{bulk}}} = \frac{A_{\text{ref}}}{A_{\text{local}}} = \left(\frac{D_{\text{outer}}}{d_{\text{local}}}\right)^2$$

Substituting:

$$\alpha_{D,\text{local}} = \alpha_{D,\text{bulk}} \cdot \left(\frac{d_{\text{local}}}{D_{\text{outer}}}\right)^4$$

### 3.3. Training-time vs inference-time usage

| Step | Where | What is used |
|---|---|---|
| ETL | `transform.py:241` | `V_local_over_V_bulk = ⟨v_z⟩(z) / V_bulk` (simulation-derived, **not** round-pipe identity) |
| Training | `targets.py::alpha_d_bulk_to_values` | Encoder applies `α_D_local = α_D_bulk · (d_local/D)⁴` (round-pipe identity) on the way IN |
| Inference decode | `targets.py::alpha_d_values_to_bulk` line 113 | Decoder applies `α_D_bulk = α_D_local / (d_local/D)⁴` (round-pipe identity) on the way OUT |

**Key asymmetry:** the ETL records the simulation-derived
`V_local_over_V_bulk` as an *input feature* the model can read, but the
*target encoding/decoding* uses the round-pipe identity. This is a
deliberate choice — the encoder/decoder transformation must be invertible
without per-row knowledge of the simulated velocity, since at inference
time we may not have a simulation to read it from.

**Bias source:** the round-pipe identity holds exactly for ideal plug flow.
Near the contraction edge (vena contracta, separation, recirculation), the
actual area-averaged `⟨v_z⟩(z)` differs from `V_bulk · (D/d_local)²`. The
decoder's assumed identity introduces error at those stations. The error
is bounded by the magnitude of this deviation, and is typically dominant
in the throat-edge bins (z just before and just after the porosity step).

### 3.4. Run-time choice

The training config sets `data.local_velocity_normalization: true` for the
Conv1D model. This means the model output is in the local-velocity basis,
and the decoder must perform the `(d_local/D)⁴` correction at inference.
Set `false` to skip the basis change entirely and have the model predict
bulk-basis `α_D` directly.

---

## 4. Coupling side: MOOSE PINSFV

### 4.1. Governing equations (Whitaker volume-averaged form)

PINSFV solves the volume-averaged incompressible Navier-Stokes equations
in porous media (Whitaker 1986, 1996). For a steady incompressible flow
of a single Newtonian fluid:

$$\nabla \cdot \vec{v}_{\text{super}} \;=\; 0$$

$$\nabla \cdot \left( \rho\,\vec{v}_{\text{super}} \otimes \vec{v}_{\text{super}} / \varepsilon \right)
\;=\; -\varepsilon\,\nabla p \;+\; \varepsilon\,\nabla \cdot \mathbf{\tau} \;+\; \varepsilon\,\rho\,\vec{g}
\;+\; \vec{f}_{\text{drag}}$$

where:

- `v_super` is the superficial (Darcy) velocity, satisfying
  `v_super = ε · v_inter` everywhere.
- `ε` is the porosity (fluid volume fraction per unit total volume).
- `f_drag` is the per-total-volume drag force from the porous medium
  (Darcy + Forchheimer terms).
- The `ε` factor on `−∇p`, `∇·τ`, and `ρg` distinguishes PINSFV from a
  naive single-phase Navier-Stokes. It comes from volume-averaging the
  pressure gradient and viscous stress with the fluid-volume indicator.

This factor lives in MOOSE's `PINSFVMomentumPressure.C:41-44`:

```cpp
ADReal
PINSFVMomentumPressure::computeQpResidual()
{
  return _eps(makeElemArg(_current_elem), determineState()) *
         INSFVMomentumPressure::computeQpResidual();
}
```

with the class description (line 22): *"Introduces the coupled pressure
term `eps ∇P` into the Navier-Stokes porous media momentum equation."*

### 4.2. Forchheimer drag term

The Darcy-Forchheimer drag force per total volume in `f_drag` is, in MOOSE's
form (`PINSFVMomentumFriction.C:97-104` comment):

$$\vec{f}_{\text{drag}}^{\;\text{MOOSE}} \;=\; \mu\,\mathbf{D}\,\vec{v}_{\text{super}} \;+\; \frac{\rho}{2}\,\mathbf{F}\,|\vec{v}_{\text{super}}|\,\vec{v}_{\text{super}}$$

where `D` is the Darcy tensor and `F` is the Forchheimer tensor. The
factor of 1/2 in the Forchheimer term is a convention choice (the standard
Forchheimer coefficient `β` in Whitaker / Nield-Bejan satisfies `F = 2β`).

**Comment vs implementation:** the kernel comment cites
[holzmann-cfd.com/community/blog-and-tools/darcy-forchheimer](https://holzmann-cfd.com/community/blog-and-tools/darcy-forchheimer)
and the SimScale knowledge base. The implementation builds the residual
as (`PINSFVMomentumFriction.C:169` and `:188-199`):

```cpp
coefficient = (ρ/2) · F · speed
residual    = coefficient · v_super
```

where `speed` comes from `PINSFVSpeedFunctorMaterial.C:69-82`:

```cpp
interstitial_velocity = v_super / ε
speed = |interstitial_velocity| = |v_super| / ε
```

So `coefficient · v_super = (ρ/2) F (|v_super|/ε) v_super = (ρ/2) F |v_inter| v_super`.

**Reading the kernel comment in isolation is misleading.** The friction
kernel only contributes one term to the residual; the pressure-gradient
contribution comes from `PINSFVMomentumPressure` (which carries the
`ε∇p` factor described in §4.1). To get the actual relation between `F`
and `dP/dz`, you must combine both kernels.

### 4.3. Steady fully-developed plug flow simplification

For a 1-D steady fully-developed plug flow inside a single PINSFV block
(constant porosity, constant `v_super`), the convective and viscous terms
drop out of the momentum equation, and the residual contributions of
`PINSFVMomentumPressure` and `PINSFVMomentumFriction` must cancel:

$$-\varepsilon\,\frac{dp}{dz} \;=\; \frac{\rho}{2}\,F\,|v_{\text{inter}}|\,v_{\text{super}} \;=\; \frac{\rho\,F\,v_{\text{super}}^2}{2\,\varepsilon}$$

Solving for `dp/dz`:

$$\boxed{-\frac{dp}{dz} \;=\; \frac{\rho\,F\,v_{\text{super}}^2}{2\,\varepsilon^2}}$$

**This is the proper non-empirical MOOSE relation.** The `1/ε²` factor is
two `1/ε`'s combined: one from the pressure-gradient kernel's
multiplication by `ε`, and one from the speed material's division by `ε`.

For ρ=1, F=1, v_super=1 inside the throat (ε = Dr² = 0.272):

```
−dp/dz = 1 / (2 · 0.272²) = 6.76 Pa/m
ΔP    = 6.76 · 0.073 m   = 0.493 Pa  (over the throat length)
```

The constant-F=1 numerical verification (§7.3) measures 0.484 Pa, matching
the analytical result to 1.5% (finite-volume discretization error).

### 4.4. Mass conservation across porosity steps

For steady incompressible flow in PINSFV, mass conservation requires
`v_super = constant` along streamlines:

$$\nabla \cdot \vec{v}_{\text{super}} = 0 \quad\Rightarrow\quad v_{\text{super}}^{\text{block 1}} = v_{\text{super}}^{\text{block 2}} = v_{\text{super}}^{\text{block 3}} = V_{\text{bulk,inlet}}$$

The superficial velocity is conserved across the porosity step at the
throat entrance/exit; only the *interstitial* velocity jumps
(`v_inter = v_super / ε` becomes `1/Dr² ≈ 3.68 v_super` inside the
throat for the target case).

This is why our `F → α_D` mapping derivation (§5) can equate
`v_super = V_bulk = 1 m/s` everywhere — `v_super` is identically the case
inlet velocity throughout the mesh.

---

## 5. The α_D → F mapping

### 5.1. Equating definitions

Equate the training-side `−dP/dz` definition (§2.5) with the MOOSE-side
`−dP/dz` formula (§4.3):

$$\underbrace{\alpha_D(z) \cdot \frac{\rho\,V_{\text{bulk}}^2}{2\,D_h(z)}}_{\text{training Darcy-Weisbach}} \;=\; \underbrace{\frac{\rho\,F(z)\,v_{\text{super}}^2}{2\,\varepsilon(z)^2}}_{\text{MOOSE PINSFV}}$$

By §4.4, `v_super = V_bulk` everywhere. The `v_super²` and `V_bulk²` cancel,
along with the common `ρ/2`:

$$\boxed{F(z) \;=\; \alpha_D(z) \cdot \frac{\varepsilon(z)^2}{D_h(z)}}$$

This is the formula implemented in
`export_friction_profile.py::alpha_d_to_forchheimer` (lines 192-220).

### 5.2. Per-block specialization

For the 3-block mesh structure (`block 1 = upstream buffer`,
`block 2 = throat`, `block 3 = downstream buffer`):

| Block | ε | `D_h` | `F = α_D · ε² / D_h` |
|---|---|---|---|
| 1 (upstream buffer) | 1 | `D_outer` | `α_D / D_outer` |
| 2 (throat) | `Dr²` | `Dr · D_outer` | `α_D · Dr³ / D_outer` |
| 3 (downstream buffer) | 1 | `D_outer` | `α_D / D_outer` |

For the target case `Re_43938__Dr_0p522__Lr_0p073`
(`Dr = 0.522`, `D_outer = 0.2 m`):

- buffer multiplier: `1.0 / 0.2 = 5.00`
- throat multiplier: `0.522³ / 0.2 = 0.711`

### 5.3. Application in MOOSE

The Forchheimer profile CSV is consumed by:

```moose
[Functions]
  [forchheimer_profile_fn]
    type      = PiecewiseLinear
    data_file = forchheimer_profile.csv
    axis      = x
  []
[]

[FunctorMaterials]
  [forchheimer_all]
    type        = ADGenericVectorFunctorMaterial
    prop_names  = 'Forchheimer_coefficient'
    prop_values = 'forchheimer_profile_fn forchheimer_profile_fn forchheimer_profile_fn'
    # no `block =` restriction → acts on blocks 1, 2, 3
  []
[]

[FVKernels]
  [u_friction]
    type             = PINSFVMomentumFriction
    Forchheimer_name = 'Forchheimer_coefficient'
    speed            = speed
    rho              = ${rho}
    variable         = superficial_vel_x
    momentum_component = 'x'
  []
[]
```

`PiecewiseLinear` linearly interpolates F between CSV stations as MOOSE
samples cell centroids. The CSV's `z` column is in MOOSE mesh coordinates
(x = 0 at the inlet face), which means the upstream buffer corresponds
to z ∈ [0, end_length], throat to [end_length, end_length + middle_length],
downstream to [end_length + middle_length, total_length]. The exporter
pre-shifts the surrogate's ROI-frame coordinates so the CSV is already
in mesh frame:

```python
z_phys_csv = z_hat · roi_length     (no extra offset; ROI z=0 = inlet)
```

### 5.4. PiecewiseLinear smearing at the porosity step

The surrogate's α_D has a sharp peak at z ≈ 0.194 m (last upstream
station, F ≈ 893) followed by a small throat-region value at z ≈ 0.204
(first throat station, F ≈ 2.2). `PiecewiseLinear` interpolates linearly
between these stations, so MOOSE cells straddling the porosity step
sample large interpolated F values from both sides of z = 0.2.

This is **a discretization artifact, not a modeling decision**. It
contributes meaningfully (~6 Pa) to the integrated MOOSE ΔP for the
target case, partially compensating for the surrogate's under-prediction
of resolved-CFD truth. See §7.4 for the empirical numbers.

If a future version wants to avoid this smearing, alternatives include:

- Use a finer CSV grid near the porosity step (e.g., re-bin the ROI to
  put more stations within ±0.01 m of z = 0.2)
- Switch to a step-friendly interpolator (MOOSE `PiecewiseConstant`)
- Pre-process the surrogate to relocate the contraction-edge spike into
  the throat block before writing the CSV

---

## 6. Verification: comparing three ΔPs

### 6.1. delta_p_truth

From `case_metadata.delta_p_case` in the per-case `.zarr`. Computed by
the ETL (`transform.py:231`) as:

$$\Delta P_{\text{truth}} \;=\; \langle p \rangle(z_{\text{ROI start}}) \;-\; \langle p \rangle(z_{\text{ROI end}})$$

This is the difference of area-averaged station pressures at the ends of
the ROI. It is a *direct* CFD reading, not an integral over the α_D.

### 6.2. delta_p_surrogate

Computed by the exporter (`export_friction_profile.py:_integrate_delta_p_with_z_phys`):

$$\Delta P_{\text{surrogate}} \;=\; \int_{0}^{L_{\text{ROI}}} \alpha_D(z) \cdot \frac{\rho\,V_{\text{bulk}}^2}{2\,D_h(z)} \,dz$$

Numerically:

```python
integrand = alpha_d_bulk · ρ · V_bulk² / (2 · D_h)
delta_p_surrogate = np.trapz(integrand, x=z_phys)
```

where `α_D(z)` is the per-station bulk-basis output of the surrogate (§3.3
decode), `D_h(z) = d_local_over_D(z) · D_outer` jumps at the porosity step
to give `D_outer` in the buffer and `Dr·D_outer` in the throat.

The `np.trapz` linear interpolation bridges the D_h discontinuity at
z = end_length, contributing ~2 Pa of over-counting "between" the last
buffer station and the first throat station. Disjoint per-piece
integration gives a "cleaner" total but discards information about the
contraction-edge transition. The exporter writes the bridged version as
`delta_p_surrogate`.

### 6.3. delta_p_moose

From the MOOSE postprocessor CSV, computed by the verifier
(`verify_delta_p.py::read_moose_inlet_pressure`):

$$\Delta P_{\text{moose}} \;=\; \frac{\text{inlet-p}}{\pi \cdot R_{\text{outer}}^2}$$

The `inlet-p` `SideIntegralVariablePostprocessor` returns
`∫ pressure · dA` over the inlet boundary. In 2-D axisymmetric (RZ),
`dA = 2πr · dr`, so for a uniform inlet pressure:

$$\int p\,dA \;=\; p_{\text{inlet}} \cdot \pi \cdot R_{\text{outer}}^2 \;=\; p_{\text{inlet}} \cdot 0.0314 \,\text{m}^2 \,\text{(for } R = 0.1 \text{m)}$$

Dividing by `inlet_area = π · R²` recovers the actual pressure. **Skipping
this division** — i.e., reporting `inlet-p` itself as the pressure —
reads as a factor of ~32 high for `outer_radius = 0.1 m`, so any future
adaptation of the verifier to a different mesh radius needs to update
`INLET_AREA_M2` accordingly.

### 6.4. Two fidelity metrics

```
surrogate_fidelity_relerr = (delta_p_surrogate − delta_p_truth) / delta_p_truth
coupling_fidelity_relerr  = (delta_p_moose     − delta_p_surrogate) / delta_p_surrogate
```

The surrogate fidelity measures pure model quality (how well does the
trained Conv1D predict resolved-CFD ΔP integrated along the ROI). The
coupling fidelity measures how well our PINSFV simulation reproduces the
surrogate's own integral.

A third metric — `(delta_p_moose − delta_p_truth) / delta_p_truth` —
measures the entire coupling pipeline's predictive accuracy against the
resolved-CFD ground truth. For the target case this is +7.9% (§7.4).

---

## 7. Assumptions and limitations

### 7.1. Plug-flow / mass-conservation assumption

The mapping derivation (§5) uses `v_super = V_bulk` everywhere via mass
conservation across porosity steps. This is exactly true in incompressible
steady PINSFV (since `∇·v_super = 0`). On the training-data side, the same
`V_bulk = inlet velocity` convention is used. Both sides are self-consistent.

### 7.2. Round-pipe identity at inference time

The decoder (§3.3) uses `V_local/V_bulk = (D/d_local)²` from incompressible
plug-flow mass conservation. The training data was generated with the
simulation-derived `V_local`, which differs from the round-pipe identity
exactly where the surrogate's predictions are strongest (near the
contraction edge). This is a **documented approximation** that introduces
bias at the throat-edge stations.

Eliminating it would require either:

- Dropping `V_local_over_V_bulk` from the model's input features and
  retraining with `local_velocity_normalization=False` (so the decoder
  becomes a pure inverse of the `signed_log1p` encoder, with no basis
  conversion needed at inference time)
- Or providing a simulation-derived `V_local` at inference time
  (requires a coupled or pre-computed CFD run)

### 7.3. Empirical verification of the mapping formula

The α_D → F mapping was verified independently of the surrogate by
running MOOSE with a uniform `F = 1.0` over the throat block:

| Quantity | Predicted (§4.3) | Measured | Ratio |
|---|---|---|---|
| -dp/dz in throat | 6.76 Pa/m | 6.63 Pa/m* | 0.98× |
| ΔP across throat | 0.493 Pa | 0.484 Pa | 0.985× |
| Inlet-p postprocessor | 0.01550 Pa·m² | 0.01519 Pa·m² | 0.980× |

(* derived from postprocessor / area, assuming uniform inlet pressure.)

The remaining 1.5% gap is finite-volume discretization error. **The
mapping formula is correct to within the solver's spatial resolution.**

### 7.4. End-to-end numbers (target case)

For `Re_43938__Dr_0p522__Lr_0p073`:

| Quantity | Value | Relative error vs truth |
|---|---|---|
| `delta_p_truth` | 11.83 Pa | — |
| `delta_p_surrogate` | 8.19 Pa | **−30.8%** (model fit) |
| `delta_p_moose` | 12.77 Pa | **+7.9%** ← best |
| Constant F=1 MOOSE (no surrogate) | 0.48 Pa | −96% (degenerate baseline) |

The coupled MOOSE pressure is closer to training truth than either the
surrogate integral alone or vanilla MOOSE with a constant Forchheimer.
The +56% coupling fidelity gap (MOOSE vs surrogate integral) is dominated
by the PiecewiseLinear smearing at the porosity step (§5.4), which adds
~6 Pa of throat-edge contribution that the per-station surrogate integral
does not include.

```{figure} ../_static/alpha_d_coupling_delta_p.png
:alt: Four-panel comparison of the alpha_D → MOOSE coupling pipeline for
  Re_43938__Dr_0p522__Lr_0p073: ΔP bar chart, fidelity vs truth, surrogate
  α_D(z) profile with the vena-contracta peak annotated, and the
  Forchheimer profile fed to MOOSE split by block.
:width: 100%
:align: center

End-to-end pipeline visualization for `Re_43938__Dr_0p522__Lr_0p073`.
**Top row:** four ΔP values and their signed relative errors versus the
resolved-CFD truth. The constant-F=1 MOOSE baseline (−96%) underscores
how degenerate vanilla PINSFV is without a calibrated drag closure; the
surrogate Darcy-Weisbach integral (−30.8%) is the surrogate's own
prediction; the coupled MOOSE simulation (+7.9%) is closest to truth.
**Bottom-left:** the surrogate-predicted `α_D(z)` along the 50-station
ROI, with the throat block shaded orange. The vena-contracta peak
(α_D = 178.7) sits at z = 0.194 m — just *upstream* of the porosity
step — and is the dominant contributor to the integrated surrogate ΔP.
**Bottom-right:** the per-station Forchheimer profile `F(z)` written
into `forchheimer_profile.csv`, split by block. Blue bars (buffer,
ε = 1) carry the buffer multiplier 5.0; red bars (throat, ε = Dr²)
carry the throat multiplier ≈ 0.71. The peak F ≈ 893 in block 1
co-locates with the α_D peak.
```

The figure is generated by `src/cases/alpha_d/plot_delta_p_comparison.py`.
After re-running the exporter + MOOSE pipeline (see the Usage block in
`src/cases/alpha_d/moose/2d-porous-flow_alphaD.i`), regenerate the figure
with:

```bash
SIF=/data/lim2/projects/multifid-th/worktrees/refactor/multifid-th-cpu.sif
apptainer exec --bind /data/lim2/projects/multifid-th:/data/lim2/projects/multifid-th "$SIF" \
  python /data/lim2/projects/multifid-th/worktrees/integration/src/cases/alpha_d/plot_delta_p_comparison.py
```

The script reads numbers from the sidecar JSON, regenerates the
four-panel PNG, and writes it both to its source location (gitignored,
under `data/cases/`) and to `docs/_static/alpha_d_coupling_delta_p.png`
(tracked, embedded above).

### 7.5. Out-of-distribution risk

The original `2d-porous-flow.i` shipped with MOOSE uses `Re ≈ 2 × 10⁶`
and `Lr ≈ 0.333`, both outside the surrogate's training range
(`Re ∈ [5×10³, 2.5×10⁵]`, `Lr ∈ [0.01, 0.20]`). `2d-porous-flow_alphaD.i`
therefore re-parameterizes the input to `Re_43938__Dr_0p522__Lr_0p073`,
which sits comfortably in distribution.

For other target cases, verify that `(Re, Dr, Lr)` lie within the trained
range before drawing conclusions from `delta_p_moose`.

---

## 8. References

### 8.1. Porous-medium momentum equation

- **Whitaker, S. (1986).** "Flow in porous media I: A theoretical
  derivation of Darcy's law." *Transport in Porous Media* 1: 3–25.
  Volume-averaging derivation of the `ε∇p` form.
- **Whitaker, S. (1996).** "The Forchheimer equation: A theoretical
  development." *Transport in Porous Media* 25(1): 27–61.
  Inertial extension via the same volume-averaging framework.
- **Nield, D. A. & Bejan, A.** *Convection in Porous Media* (Springer,
  5th ed. 2017), §1.5. Textbook treatment of Darcy-Forchheimer-Brinkman
  with the porosity factor on the pressure gradient.
- **Vafai, K. & Tien, C. L. (1981).** "Boundary and inertia effects on
  flow and heat transfer in porous media." *Int. J. Heat Mass Transfer*
  24: 195–203. The PINSFV momentum form is closer to Vafai-Tien's
  notation than Whitaker's.

### 8.2. Darcy-Weisbach friction factor

- **Bird, R. B., Stewart, W. E. & Lightfoot, E. N.** *Transport Phenomena*
  (Wiley, 2nd ed.), §6.2. Standard derivation of the friction factor for
  pipe flow.
- **Munson, B. R. et al.** *Fundamentals of Fluid Mechanics* (Wiley,
  6th ed.), §8.2. Engineering pipe-flow Darcy-Weisbach treatment.

### 8.3. MOOSE PINSFV documentation

- **MOOSE Navier-Stokes module:**
  [mooseframework.inl.gov/modules/navier_stokes](https://mooseframework.inl.gov/modules/navier_stokes/)
- **Per-class docs:**
  - `PINSFVMomentumPressure`: `moose/modules/navier_stokes/doc/content/source/fvkernels/PINSFVMomentumPressure.md`
  - `PINSFVMomentumFriction`: `moose/modules/navier_stokes/doc/content/source/fvkernels/PINSFVMomentumFriction.md`
  - `PINSFVSpeedFunctorMaterial`: `moose/modules/navier_stokes/doc/content/source/functormaterials/PINSFVSpeedFunctorMaterial.md`

### 8.4. References cited inside MOOSE source

`PINSFVMomentumFriction.C:97-99`:

- [holzmann-cfd.com/community/blog-and-tools/darcy-forchheimer](https://holzmann-cfd.com/community/blog-and-tools/darcy-forchheimer) —
  practical CFD treatment of Darcy-Forchheimer perforated plates
- [simscale.com/knowledge-base/predict-darcy-and-forchheimer-coefficients-for-perforated-plates-using-analytical-approach](https://www.simscale.com/knowledge-base/predict-darcy-and-forchheimer-coefficients-for-perforated-plates-using-analytical-approach/) —
  analytical coefficient predictions for perforated plates

Neither reference explicitly discusses the `ε` factor on the pressure
gradient, since they consider non-volume-averaged closures. The MOOSE
implementation extends the Forchheimer treatment to the full PINSFV
volume-averaged form, which is where the extra `ε²` factor in our
mapping (§5) ultimately comes from.

### 8.5. Spec/plan documents

- Design spec (this project): `docs/superpowers/specs/2026-05-28-alpha-d-moose-coupling-design.md`
- Implementation plan (this project): `docs/superpowers/plans/2026-05-28-alpha-d-moose-coupling.md`

Both are gitignored under `docs/superpowers/` per user preference, but
live on disk for reference.

---

## 9. Code locations cheat-sheet

| Topic | File | Lines |
|---|---|---|
| α_D definition (training) | `src/cases/alpha_d/etl/transform.py` | 219–228 |
| `V_bulk = 1.0` convention | `src/cases/alpha_d/etl/transform.py` | 222 |
| ROI definition + station binning | `src/cases/alpha_d/etl/transform.py` | 125–195 |
| Encoder (signed_log1p) | `src/cases/alpha_d/physics/targets.py` | 91–96 |
| Decoder (signed_log1p) | `src/cases/alpha_d/physics/targets.py` | 100–113 |
| Local↔bulk basis conversion | `src/cases/alpha_d/physics/targets.py` | 100–129 |
| Exporter pipeline | `src/cases/alpha_d/export_friction_profile.py` | 307–415 |
| α_D → F mapping function | `src/cases/alpha_d/export_friction_profile.py` | 192–220 |
| ΔP_surrogate integral | `src/cases/alpha_d/export_friction_profile.py` | 291–304 |
| Verifier (postprocessor → pressure) | `src/cases/alpha_d/verify_delta_p.py` | 24–36 |
| MOOSE input (re-parameterized) | `src/cases/alpha_d/moose/2d-porous-flow_alphaD.i` | (whole file) |
| PINSFV pressure kernel (ε∇p) | `moose/.../fvkernels/PINSFVMomentumPressure.C` | 41–44 |
| PINSFV friction kernel | `moose/.../fvkernels/PINSFVMomentumFriction.C` | 118–200 |
| PINSFV speed material | `moose/.../functormaterials/PINSFVSpeedFunctorMaterial.C` | 50–98 |
