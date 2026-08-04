# Alpha-D Surrogate → MOOSE PINSFV Coupling Demo

This document is the single source of truth for the physics, equations,
averaging conventions, and assumptions that underlie the alpha-D surrogate
and its coupling to MOOSE PINSFV.

It cross-references the implementation:

- Training-side ETL: `src/cases/alpha_d/etl/transform.py`
- Target decode/encode: `src/cases/alpha_d/physics/targets.py`
- Reusable study definition: `src/cases/alpha_d/study_workflow.py`
- Tracked study matrix: `src/cases/alpha_d/configs/coupling_study.toml`
- Surrogate → MOOSE exporter: `src/cases/alpha_d/export_friction_profile.py`
- MOOSE input: `src/cases/alpha_d/moose/2d-porous-flow_alphaD.i`
- MOOSE PINSFV kernels: `moose/modules/navier_stokes/src/fvkernels/`

If equations here ever drift from the code, **the code is right**. Update
this document to match.

---

## Reproduce this study

Prerequisites are Python 3.11 or newer, `uv`, Apptainer, the processed alpha-D
Zarr dataset, a MULTIFID Python image containing the ML dependencies, and a
MOOSE image that can run `navier_stokes-opt`. From the repository root:

The tracked TOML explicitly selects the default method and its coupling
contracts:

```toml
[training.alpha]
id = "conv1d_profile"
runner_module = "cases.alpha_d.train"
config_name = "train_conv1d"
artifact_contract = "alpha_d_profile_v1"
checkpoint = "model.mdlus"
run_meta = "run_meta.json"
include_acceleration_head = true

[training.alpha.hpo]
enabled = true

[training.alpha.export]
module = "cases.alpha_d.export_friction_profile"
contract = "forchheimer_profile_v1"
```

To use another profile model, create its Hydra YAML and change `id` and
`config_name`. A custom runner or exporter can also be selected, but it must
honor the declared artifact and exporter CLI contracts. Run `plan` before
starting, then choose a new run ID because method selection is part of the
resolved-configuration hash. See
[Run a Reproducible Study Workflow](../user/running_workflows.md) for a CNN
example.

```bash
uv sync
export MULTIFID_PYTHON_IMAGE=/absolute/path/to/multifid-th.sif
export MULTIFID_MOOSE_IMAGE=/absolute/path/to/moose-dev.sif

uv run multifid-workflow plan \
  --config src/cases/alpha_d/configs/coupling_study.toml
uv run multifid-workflow run \
  --config src/cases/alpha_d/configs/coupling_study.toml \
  --run-id alpha-d-20260720
uv run multifid-workflow status \
  --run-dir data/workflows/alpha_d_coupling/alpha-d-20260720
```

The default `inputs.mode = "reuse"` fingerprints the existing processed Zarr
tree. To rebuild it from raw Exodus first, copy the tracked TOML, set
`inputs.mode = "raw_etl"`, and point `inputs.raw_dir` at the source campaign.
The ETL output then stays inside the run directory.

For an ETL-only run with its own provenance and resumability, use:

```bash
uv run multifid-workflow etl \
  --config src/cases/alpha_d/configs/etl_workflow.toml \
  --run-id alpha-d-etl-001 \
  --input-dir /absolute/path/to/parametric_study
```

It writes processed stores to
`data/workflows/alpha_d_etl/alpha-d-etl-001/data/processed/`. To use them in
this study, copy the coupling TOML, set `inputs.mode = "reuse"`, and set
`inputs.zarr_dir` to that directory.

Typical stage costs are:

| stage family | expected cost | resumable unit |
|---|---|---|
| `prepare_data`, `plan_panels` | seconds when reusing Zarr; ETL can take hours | whole stage |
| `tune_alpha` | the most expensive ML stage; one selected-method search | one reference-panel search |
| `panel.<tag>.train_direct` | minutes per panel | panel |
| `panel.<tag>.train_alpha` | tens of minutes to hours per panel | panel |
| `panel.<tag>.export_closure` | minutes per panel | panel |
| `solve_moose` | solver-dependent; all reported low-`Dr` cases plus controls | per-case primary/retry records |
| `summarize` | seconds | whole report |

Rerun the same command with the same run ID to resume. Successful stages are
not repeated unless an output checksum, an upstream artifact fingerprint, or a
semantic validator fails. Interrupted/failed stages rerun. MOOSE case failures
retain their commands and logs, are recorded as explicit failures rather than
zero pressure, and let `summarize` continue with observed coverage. Reusing a
run ID after changing the resolved config, Git worktree, or input data is
rejected; choose a new ID.

All generated artifacts are together under:

```text
data/workflows/alpha_d_coupling/<run-id>/
├── resolved_config.json
├── run_manifest.json
├── logs/
├── panels/<tag>/
│   ├── heldout_cases.txt
│   ├── report_cases.txt
│   ├── artifacts/{direct,alpha,alpha_feature_selection}/
│   ├── coupled/<case>/
│   └── moose/<case>/{commands,logs,status,verification}/
├── tuning/{best_params.json,best_overrides.txt}
├── moose_matrix.json
└── report/
```

The one canonical `heldout_cases.txt` per panel is passed to direct-regressor
testing, alpha-D feature selection, and alpha-D training. The summarizer checks
all three persisted contracts before producing evidence.

Method provenance appears in four places:

- `resolved_config.json` contains the full `[training.alpha]` selection;
- each `panel_manifest.json` records the method, profile, artifact paths, and
  contract versions;
- the selected training command writes the resolved model entrypoint, adapter,
  parameters, and effective data settings to its configured metadata file; and
- the published-results manifest records the selected method beside workflow,
  input, code, solver-coverage, and figure hashes.

The currently published figures remain the default Conv1D result. Selecting a
new method does not relabel or overwrite them until the complete workflow is
regenerated and explicitly published.

Only `publish` writes documentation assets:

```bash
uv run multifid-workflow publish \
  --run-dir data/workflows/alpha_d_coupling/alpha-d-20260720
uv run multifid-workflow publish \
  --run-dir data/workflows/alpha_d_coupling/alpha-d-20260720 \
  --check
```

| run artifact | published use |
|---|---|
| `report/pressure_drop_comparison.json` | result summaries, evidence classes, and coverage in the published-results manifest |
| `report/pressure_drop_comparison.md` | auditable generated pressure-drop comparison report |
| `report/paired_case_errors.csv` | direct scalar regression vs direct alpha-D integration rows |
| `report/moose_paired_case_errors.csv` | validated MOOSE-coupled alpha-D rows only |
| `report/pressure_drop_comparison_errors.svg` | `docs/_static/alpha_d_pressure_drop_comparison_errors.svg` |
| `moose_matrix.json` | attempted/succeeded/failed coverage in `docs/demo_cases/alpha_d_published_results.json` |

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
| `Dr` | {ref}`Throat-to-outer diameter ratio <geometry-interpretation>`, = `d_throat / D_outer` | — |
| `Lr` | {ref}`Throat axial-length ratio <geometry-interpretation>`, = `L_throat / L_ref` (`inner_height / outer_height` in the source metadata) | — |
| `α_D(z)` | Darcy-Weisbach friction factor (per-station) | — |
| `F(z)` | MOOSE Forchheimer coefficient (per-station) | 1/m |
| `p` | Static or Bernoulli pressure, depending on variable type | Pa |

Throughout the training data and the MOOSE input file, we use SI units with
`ρ = 1 kg/m³` and `V_bulk = 1 m/s` to keep dimensionless quantities
identifiable from their numerical values.

(geometry-interpretation)=
### 1.1. Geometry interpretation

The metadata names `inner_height` and `outer_height` come from the cylinder
mesh generator, where *height* means the cylinder's axial extrusion length.
They do not refer to a radial or vertical pipe dimension. In the notation used
throughout this report,

```
L_throat = inner_height
L_ref    = outer_height = 1.0 m
Lr       = L_throat / L_ref
```

Thus, `Lr = 0.073` represents a throat that is approximately `0.073 m` long.
`Dr` independently controls the throat diameter: `Dr = 0.522` means
`d_throat = 0.522 D_outer`.

```{figure} ../_static/high_fidelity.png
:alt: Resolved high-fidelity contraction-expansion mesh with a narrow central throat.
:width: 95%
:align: center

**Geometrically resolved high-fidelity model.** The axial direction is
horizontal. The narrow central section is the throat: `Dr` controls its
diameter and `Lr` controls its axial length relative to `L_ref`.
```

```{figure} ../_static/low_fidelity.png
:alt: Low-fidelity full-width PINSFV mesh divided into upstream, throat, and downstream blocks.
:width: 95%
:align: center

**MOOSE-coupled low-fidelity model.** The mesh remains full width. The middle
block occupies the throat's axial interval set by `Lr`; the effect of the
smaller diameter is represented by geometry-derived porosity and the
surrogate-derived Forchheimer closure rather than a geometrically narrowed
passage.
```

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
station, F ≈ 900 after the closure mapping) followed by a small
throat-region value at z ≈ 0.204 (first throat station, F ≈ 2).
Naively passed to `PiecewiseLinear`, that linear interpolation across
the ~0.0095 m gap leaks large F values into mesh cells on both sides of
the porosity step at z = 0.2 m. The same issue recurs at the throat
exit (z = 0.273 m). For the target case the two smearings together
contribute ~3.4 Pa of extra friction work, raising MOOSE_coupled to
+24.8% above truth and breaking the coupling fidelity.

**The exporter fixes this by step-fencing each porosity boundary.**
Just before writing the CSV, `_stepfence_porosity_boundaries` inserts
two near-duplicate rows at each boundary: `(z_boundary − step_eps,
F_just_below)` and `(z_boundary + step_eps, F_just_above)`. MOOSE's
`PiecewiseLinear` still bridges *within* each block but the
near-discontinuous fence collapses the bridge *across* the porosity
step to negligible width.

The default `step_eps = 1e-4 m` is well below the MOOSE mesh cell size
for this case (~0.0095 m) and well above floating-point spacing, so no
mesh cell ever straddles the fence and `PiecewiseLinear`'s
strictly-increasing requirement is satisfied. With this fence enabled:

- `delta_p_moose` drops from 14.76 Pa to 11.28 Pa
- coupling fidelity (MOOSE vs surrogate) goes from +30% to −0.7%
- MOOSE-vs-truth tracks the surrogate within 0.7%

Empirically the answer is insensitive to `step_eps` in
`[10⁻⁵, 10⁻³] m`: any value below mesh resolution gives the same
result. Above ~2·10⁻³ m the fence starts overlapping nearby stations
and reintroduces some bridging.

Alternatives that were considered and rejected:

- A **finer CSV grid near the step** would help but doesn't fully
  remove the leak — the discontinuity is intrinsic to the surrogate's
  per-station integrand, not a sampling artifact.
- A **`PiecewiseConstant`** function would solve the smearing
  fundamentally but requires Function-type changes in the MOOSE input
  and loses sub-block smoothness in the throat profile.
- **Relocating the contraction-edge spike** into the throat block by
  hand changes what the surrogate is actually predicting — moves the
  problem out of the exporter and into a different modeling
  assumption.

Step-fencing is the smallest-surface-area fix that closes the gap.

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
resolved-CFD ground truth. For the target case this is +1.35% (§7.5).

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

### 7.4. Surrogate-side validation (training output)

Before looking at how MOOSE coupling propagates the surrogate's
predictions, it's worth verifying that the surrogate predicts well on
its own held-out test set. The runner saves four diagnostic plots into
`data/cases/train_conv1d/plots/` at the end of every evaluation;
the four below are copied verbatim from the latest run and are
ground-truth references for the section below.

These plots describe the default in-distribution training run. The technical
comparison in §7.7 retrains per held-out panel with geometry-only features,
so its archived metrics, rather than these plots, control the comparison of
surrogate approaches.

**Per-case Δp parity.** Each marker is one test case (153 in total);
points colored by diameter ratio `Dr`. The cluster sits tightly along
`y = x`: median 3.6%, mean 4.2%, p90 8.2%, max 15.5% (see
`eval_metrics.json.extended.delta_p`).

```{figure} ../_static/alpha_d_train_delta_p_parity.png
:alt: Per-case Δp parity plot for the held-out test set.
:width: 75%
:align: center

Held-out per-case Δp parity. ±10% reference lines drawn for context;
the surrogate keeps every case inside the bounds the parity plot
advertises.
```

**Per-station α_D parity.** A finer-grained check: every test-set
station from every test case plotted as `α_D_pred` vs `α_D_truth`. The
parity points spread further than the Δp version (the per-station
target is noisier, especially at low magnitudes) but the centred
distribution shows the model isn't systematically biased.

```{figure} ../_static/alpha_d_train_parity_alpha_D.png
:alt: Per-station α_D parity plot across the held-out test set.
:width: 75%
:align: center

Per-station α_D parity for every test station. Sub-station-level
noise tolerated; per-case Δp integral averages it out.
```

**Best-fit per-station case.** Top-ranked case by per-station fit
RMSE: `Re_104807__Dr_0p9__Lr_0p052` (RMSE on `signed_log1p_alpha_D` =
0.031). The prediction tracks the ground-truth profile across the
entire ROI; baseline (analytical closure) shown as the dotted line
for context.

```{figure} ../_static/alpha_d_train_best_profile.png
:alt: Best-fit per-station case profile.
:width: 80%
:align: center

Best-fit α_D profile (`Re_104807__Dr_0p9__Lr_0p052`). The Conv1D
model output tracks resolved-CFD α_D across every station inside the
ROI.
```

**Worst-fit per-station case.** Top of the failure list:
`Re_7722__Dr_0p333__Lr_0p158` (RMSE 0.477). The model misses the
high-α_D feature inside the throat — typical of the low-Dr × low-Re
corner the user guide flags as the historically hardest region.

```{figure} ../_static/alpha_d_train_worst_profile.png
:alt: Worst-fit per-station case profile.
:width: 80%
:align: center

Worst-fit α_D profile (`Re_7722__Dr_0p333__Lr_0p158`). The Conv1D
model under-predicts the throat-interior α_D. The user guide flags
the low-Dr × low-Re corner as the historically hardest part of the
parameter space and this is its representative.
```

### 7.5. Representative in-distribution spot check

The July 14, 2026 technical comparison includes
`Re_43938__Dr_0p522__Lr_0p073` in a deterministic 36-case interior panel.
The case is excluded from feature selection and training for both methods.

| Quantity | Value | Relative error vs CFD truth |
|---|---:|---:|
| Resolved-CFD truth | 11.8318 Pa | -- |
| Direct random-forest ΔP regressor | 11.7502 Pa | **-0.69%** |
| Direct ΔP from α_D pressure-gradient integration | 11.9725 Pa | +1.19% |
| MOOSE-coupled α_D closure solve | 11.9914 Pa | **+1.35%** |

The first α_D result is not a coupled flow solve. It converts the surrogate's
α_D profile to `-dp/dz` with the Darcy-Weisbach relation and integrates that
gradient over the ROI to obtain one scalar ΔP. The second result inserts the
surrogate-derived Forchheimer correlation into MOOSE and solves the PINSFV
mass and momentum equations for pressure and velocity.

For this case, the MOOSE-coupled result differs from the direct gradient
integration by only +0.16%, showing that the step-fenced closure is
transferred into PINSFV without material additional scalar error. This is a
coupling-fidelity check for one case, not the basis for the extrapolation
assessment. The panel-level and low-`Dr` results are reported in §7.7.

The case-level regressor and the MOOSE-coupled route remain independent
estimators: the regressor maps `(Re, Dr, Lr)` directly to ΔP, while MOOSE
consumes the predicted α_D profile. Both are compared against the same
resolved-CFD truth.

### 7.6. Why a broader comparison is needed

The representative case in §7.5 verifies that the surrogate-derived closure
can be transferred into MOOSE without adding material scalar error for one
in-distribution condition. One case cannot determine which surrogate-model
approach performs better overall, and it says nothing about extrapolation.

The following technical study compares direct regression and direct α_D
pressure-gradient integration across a 36-case in-distribution panel and
predefined out-of-distribution (OOD) shells. MOOSE-coupled closure solves then
measure how the surrogate closure behaves after it is inserted back into the
flow solver. Full MOOSE coverage is available for the primary low-`Dr` panel;
the in-distribution, `Re` OOD, and `Lr` OOD panels have selected MOOSE spot
checks.

The three evaluated approaches are:

- direct case-level ΔP regression;
- direct ΔP from α_D pressure-gradient integration; and
- the MOOSE-coupled α_D closure solve.

The in-distribution panel measures ordinary interpolation performance. The
held-out `Dr`, `Re`, and `Lr` shells show how each approach behaves when one
physical parameter moves outside its training support.

---

### 7.7. Technical comparison of surrogate-model approaches

This section compares surrogate-model approaches for predicting ROI ΔP under
both in-distribution and out-of-distribution conditions. The comparison
evaluates direct case-level regression against two uses of the predicted α_D
profile. The two α_D results are kept separate throughout:

- **Direct α_D pressure-gradient integration** reconstructs `-dp/dz` from
  α_D and integrates it. This is a surrogate-only scalar diagnostic, not a
  coupled MOOSE solve.
- **MOOSE-coupled α_D closure solve** inserts the surrogate-derived
  Forchheimer correlation into MOOSE and solves the PINSFV equations.

#### 7.7.1. Experimental controls

`cases.alpha_d.extrapolation.build_split()` defines every split. Each saved
held-out list is reused for direct-regressor `force_test`, direct feature
selection, α_D feature selection, and selected α_D method training. The archived
metadata verifies that these sets match and that no held-out case enters
training or feature selection.

The comparison also removes the CFD-velocity leak described earlier:
feature selection uses a geometry-only candidate pool and selects
`Dr, z_hat, log10_Re_throat, z_hat_times_Dr, z_hat_times_Lr,
dist_to_throat_end, dist_to_nearest_step`. Hyperparameters are optimized once
and then frozen across the study panels.

The evaluation panels are:

- **In-distribution:** 36 deterministic interior cases with `Dr ≥ 0.333`,
  excluding the outer two levels of `Re`, `Dr`, and `Lr`.
- **Dr OOD:** the complete two-level low-`Dr` shell is held out, but
  reporting is restricted to cases with `Re` and `Lr` on interior levels.
  This leaves 50 reported cases while preventing mixed-axis extrapolation.
- **Re OOD:** the low- and high-`Re` shells evaluate extrapolation in Reynolds
  number while retaining the common `Dr ≥ 0.333` population.
- **Lr OOD:** the low- and high-`Lr` shells evaluate extrapolation in length
  ratio while retaining the common `Dr ≥ 0.333` population.
- **MOOSE-coupled closure validation:** PINSFV is attempted for every reported
  low-side `Dr` OOD case, with separate verifier output and failure records.

#### 7.7.2. Scalar ΔP results

Each model column reports the mean absolute relative error (MARE) against
resolved-CFD truth over the `n` cases in that row:
`100 × mean(|(ΔP_pred - ΔP_CFD) / ΔP_CFD|)`. For a `Dr` OOD row, the held-out
`Dr` value is fixed while `Re` and `Lr` vary across the reported cases. The
in-distribution panel spans multiple interior `Dr` values. The final column
is not an average error; it is the percentage of paired cases in which direct
α_D integration has a smaller absolute relative error than RF.

The direct-integration result reconstructs and integrates `-dp/dz` from the
surrogate α_D profile; no MOOSE solve is performed for this column.

| evaluation set | held-out Dr value | cases (n) | direct α_D integration MARE | RF MARE | MLP MARE | linear MARE | cases where α_D integration has lower error than RF |
|---|---:|---:|---:|---:|---:|---:|---:|
| In-distribution | multiple interior values | 36 | 4.74% | **0.75%** | 5.30% | 24.39% | 13.89% |
| **Dr OOD** | 0.144 | 20 | **17.69%** | 96.47% | 80.70% | 90.23% | **100%** |
| **Dr OOD** | 0.239 | 30 | **13.74%** | 73.79% | 39.18% | 62.21% | **100%** |
| Dr OOD | 0.806 | 36 | **16.82%** | 171.94% | 43.89% | 53.10% | **100%** |

The paired bootstrap interval for `direct α_D integration error - RF error`
excludes zero in each primary panel. It is +2.92 to +5.07 percentage points
in-distribution, where RF is better; -83.11 to -74.65 points at `Dr=0.144`;
and -60.98 to -59.16 points at `Dr=0.239`, where direct α_D integration is
better.

The `Dr=0.9` stress level is retained in the archived results but omitted
from the accuracy table. Its absolute errors are too large for either method
to count as useful prediction (direct α_D integration 98.41%, RF 1108.72%).
Its relative ranking therefore does not demonstrate useful accuracy.

#### 7.7.3. Re OOD and Lr OOD

These rows use the same MARE and paired lower-error percentage definitions as
§7.7.2. The held-out value is fixed within each row while the other physical
parameters vary across the reported cases.

| evaluation set | held-out value | cases (n) | direct α_D integration MARE | RF MARE | cases where α_D integration has lower error than RF |
|---|---:|---:|---:|---:|---:|
| Re OOD (low) | 5 000 | 66 | 15.53% | **7.04%** | 12.12% |
| Re OOD (low) | 7 722 | 70 | 14.58% | **4.33%** | 5.71% |
| Re OOD (high) | 161 870 | 51 | 14.00% | **1.48%** | 9.80% |
| Re OOD (high) | 250 000 | 45 | 17.84% | **2.62%** | 13.33% |
| Lr OOD (low) | 0.010 | 61 | 24.92% | **12.98%** | 39.34% |
| Lr OOD (low) | 0.031 | 65 | 33.56% | **5.71%** | 0.00% |
| Lr OOD (high) | 0.179 | 63 | 18.18% | **1.13%** | 9.52% |
| Lr OOD (high) | 0.200 | 64 | 35.18% | **1.62%** | 4.69% |

These OOD results show that the α_D route is not uniformly better for
extrapolation. RF's bounded nearest-training-region behavior is effective
where ΔP is comparatively flat in the extrapolated variable. Low `Dr` is
different: shrinking the throat drives a strong nonlinear pressure-drop
increase, and the α_D pressure-gradient reconstruction retains the explicit
hydraulic-diameter dependence needed to follow that trend.

#### 7.7.4. MOOSE-coupled surrogate-closure validation

MOOSE-coupled PINSFV solves validate 39 of the 50 reported low-side `Dr` OOD
cases. Errors below are computed on the same converged subset for a paired
comparison.

| Dr | reported | validated MOOSE | coverage | MOOSE-coupled closure | direct α_D integration | random forest | cases where MOOSE has lower error than RF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.144 | 20 | 10 | 50.0% | **11.96%** | 13.51% | 96.48% | **100%** |
| 0.239 | 30 | 29 | 96.7% | **10.25%** | 13.94% | 73.78% | **100%** |
| **All** | **50** | **39** | **78.0%** | **10.68%** | **13.83%** | **79.60%** | **100%** |

The MOOSE-coupled solve improves on the direct-integration error in 29 of 39
validated cases. The mean absolute MOOSE-minus-integration difference is
7.86%, so direct pressure-gradient integration is useful for the complete
50-case scalar sweep but is not an exact substitute for a coupled flow solve.

The missing MOOSE results are not random. Ten of the eleven failures occur
at the more extreme `Dr=0.144` level. Across the failed cases, eight stop on
nonlinear line-search divergence, two reach the nonlinear iteration limit,
and one encounters a PETSc matrix-preallocation error. Consequently, direct
α_D pressure-gradient integration supports the complete low-`Dr` scalar
panel, while the MOOSE-coupled result supports only the 39 converged and
validated cases. The 50% MOOSE coverage at `Dr=0.144` must be reported
alongside its error.

The additional OOD spot checks reinforce the same scope. The in-distribution
MOOSE case has 1.35% error, but the checked `Re`-high and `Lr`-low cases have
35.51% and 39.59% error, respectively. Running MOOSE does not repair an
inaccurate extrapolated α_D profile.

#### 7.7.5. Overall assessment and remaining limit

The technical comparison supports the following assessment:

> Across predefined held-out panels, direct random forest provides the best
> scalar pressure-drop accuracy in-distribution and for `Re` OOD and `Lr` OOD.
> For low-side `Dr` OOD, the α_D surrogate route is substantially
> more robust: direct pressure-gradient integration has lower error than RF in
> all 50 paired cases, and validated MOOSE-coupled α_D closure solves have
> lower error than RF in all 39 converged cases. MOOSE-coupled validation at
> `Dr=0.144` remains
> partial because only 10 of 20 cases converged.

This study validates scalar ΔP. The coupled solve also produces pressure and
velocity fields, but field-level extrapolation accuracy has not yet been
quantified against resolved CFD and is not assessed here.

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

### 8.5. Reproducible workflow definition

The executable study definition is the tracked
`src/cases/alpha_d/configs/coupling_study.toml`, interpreted by
`src/cases/alpha_d/study_workflow.py` through the case-independent workflow
kernel. The [workflow extension guide](../dev/workflows.md) documents how to
apply the same architecture to another case.

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
| Workflow and panel matrix | `src/cases/alpha_d/study_workflow.py` | case-owned stage builder |
| Tracked study configuration | `src/cases/alpha_d/configs/coupling_study.toml` | whole file |
| Exporter pipeline | `src/cases/alpha_d/export_friction_profile.py` | `main()` |
| α_D → F mapping function | `src/cases/alpha_d/coupling_utils.py` | `alpha_d_to_forchheimer()` |
| ΔP surrogate integral | `src/cases/alpha_d/coupling_utils.py` | `integrate_delta_p()` |
| Verifier (postprocessor → pressure) | `src/cases/alpha_d/verify_delta_p.py` | 24–36 |
| MOOSE input (baseline coupled case) | `src/cases/alpha_d/moose/2d-porous-flow_alphaD.i` | (whole file) |
| PINSFV pressure kernel (ε∇p) | `moose/.../fvkernels/PINSFVMomentumPressure.C` | 41–44 |
| PINSFV friction kernel | `moose/.../fvkernels/PINSFVMomentumFriction.C` | 118–200 |
| PINSFV speed material | `moose/.../functormaterials/PINSFVSpeedFunctorMaterial.C` | 50–98 |
