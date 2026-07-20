# Alpha-D Surrogate: Axial-Profile MLP Tutorial

This guide walks through extracting Darcy resistance coefficient profiles from
the flow contraction-expansion parametric study and training an MLP surrogate
model.

## Problem overview

The parametric study runs 1000 MOOSE k-epsilon RANS simulations of flow through
a pipe with an internal contraction-expansion.  Three parameters are varied:

| Parameter | Symbol | Range | Spacing |
|-----------|--------|-------|---------|
| Reynolds number | Re | 5 000 -- 250 000 | log-spaced (10 values) |
| Diameter ratio | Dr | 0.05 -- 0.9 | linear (10 values) |
| Length ratio | Lr | 0.01 -- 0.2 | linear (10 values) |

The surrogate predicts the Darcy resistance coefficient `alpha_D(z)` along
the contraction region as a function of case parameters and axial position:

```
[log10(Re), Dr, Lr, z_hat, d_local/D_big, region_flags]  -->  log(alpha_D)
```

The model is PhysicsNeMo's `FullyConnected` MLP.

## Prerequisites

- `moose-physicsnemo` conda environment (or the `etl` Docker service)
- If using Docker Compose, use `etl` or `etl-ngc` for training/evaluation.
  `etl-dev` is ETL-only and does not include PyTorch.
- Completed parametric study cases under
  `data/flow_contraction_expansion/parametric_study/` (each case directory
  must contain `simulation_out.e` and `case_metadata.txt`)

Verify that you have completed cases:

```bash
ls data/flow_contraction_expansion/parametric_study/Re_5000__Dr_0p144__Lr_0p01/simulation_out.e
```

## Step 1: Extract alpha_D profiles (ETL)

The alpha_D ETL pipeline reads each case's `simulation_out.e`, extracts the
contraction region, computes the Darcy resistance coefficient at axial
stations, and writes per-case `.zarr` stores.

### Run the extraction

```bash
cd src && python cases/alpha_d/run_etl.py \
    etl.source.input_dir=../data/flow_contraction_expansion/parametric_study \
    etl.sink.output_dir=../data/flow_contraction_expansion/parametric_study/processed
```

This processes all cases with `simulation_out.e` files.  Use
`etl.processing.num_processes=8` to speed up with more workers.

### What the ETL does

For each case:

1. Opens `simulation_out.e` via netCDF4 and reads node coordinates, element
   connectivity, and the pressure field at the final (converged) time step.
2. Applies mesh scaling (`1.0` by default).
3. Computes element centroids and identifies the contraction region of
   interest (ROI), defined as the throat plus a buffer of 1 pipe diameter
   upstream and downstream.
4. Bins ROI elements into 50 axial stations and computes area-weighted
   (radially weighted for axisymmetric geometry) average pressure at each.
5. Derives the Darcy coefficient via the pressure gradient:

   ```
   alpha_D(z) = -dP/dz * 2 * D_h(z) / (rho * V_bulk^2)
   ```

6. Constructs feature vectors: `log10(Re)`, `Dr`, `Lr`, normalized axial
   position `z_hat`, local diameter ratio `d_local/D_big`, and region flags.
7. Stores `log(alpha_D)` as the target (log transform for positivity and
   better regression over wide ranges).

### Output Zarr layout

```
{case_name}.zarr/
    features      float32 [50, 13]   -- 50 stations x 13 input features
    targets       float32 [50, 2]    -- log_alpha_D and signed_log1p_alpha_D
    sample_weight float32 [50]       -- per-station weights (higher in throat)
    metadata/
        attrs: case_id, feature_names, target_names, Re, Dr, Lr, delta_p_case
```

### Feature reference

The per-station feature row written by `AlphaDTransformation` and the
engineered features synthesized at load time by
`cases/alpha_d/feature_data.py::build_engineered_feature_map`.

Notation:
- `Re_global = ρ V_bulk D / μ` is the case-level Reynolds number
  (`V_bulk = ρ = 1` in the simulation, so `μ = D / Re_global`).
- `D` is the outer pipe diameter; `d_local(z)` is the local diameter,
  equal to `D` upstream/downstream and `Dr × D` inside the throat.
- `D_h_local = d_local` for the axisymmetric circular cross-section.
- `V_local(z)` is the area-mean axial velocity (`vel_z`) at axial
  station `z`, area-weighted with `r` over the cross-section to match
  the pressure averaging.

#### Base features (written by ETL into each case's Zarr)

| Feature | Expression / definition | Per-row varying? | Source |
|---|---|---|---|
| `log10_Re` | `log10(Re_global)` | case-level | `case_metadata.txt` |
| `Dr` | `d_throat / D` | case-level | `case_metadata.txt` |
| `Lr` | `inner_height / outer_height` | case-level | `case_metadata.txt` |
| `z_hat` | `(z − z_roi_start) / (z_roi_end − z_roi_start)` | local | Geometry, ∈ [0, 1] across ROI |
| `d_local_over_D` | `d_local(z) / D` | local | Geometry (abrupt model: `D` outside throat, `Dr·D` inside) |
| `A_local_over_A` | `(d_local / D)²` | local | Identity for circular cross-section |
| `V_local_over_V_bulk` | `<vel_z(z)>_A / V_bulk` | local | MOOSE `vel_z` element field, station-binned with `r`-weighted average |
| `is_upstream` | `1` if `z < z_throat_start` else `0` | local | One-hot region flag |
| `is_throat` | `1` if `z_throat_start ≤ z ≤ z_throat_end` else `0` | local | One-hot region flag |
| `is_downstream` | `1` if `z > z_throat_end` else `0` | local | One-hot region flag |
| `dD_dz_local` | `np.gradient(d_local_over_D, z_hat)` | local | Discrete axial gradient of `d_local/D`; spikes at contraction edges, ~0 elsewhere |
| `dist_to_throat_start` | `z_hat − z_throat_start_hat` | local | Signed: positive past the entry (inside throat / downstream), negative upstream |
| `dist_to_throat_end` | `z_throat_end_hat − z_hat` | local | Signed: positive upstream of the exit (inside throat / upstream), negative past it. Note the sign convention is opposite of `dist_to_throat_start` so both are "positive toward throat centre" |

#### Engineered features (synthesized at load time)

These are included in the default `selected_from_allowlist: null`
candidate set and can be filtered down explicitly by listing the base
names. They are pure functions of base features so `TabularPairDataset`
can reproduce them at training time without any extra storage.

| Feature | Expression | Per-row varying? | Notes |
|---|---|---|---|
| `log10_Re_throat` | `log10_Re − log10(Dr)` | case-level | Re scaled by throat diameter |
| `log10_Re_local` | `log10_Re + log10(V_local/V_bulk) + log10(d_local/D)` | local | Local Re using local V, local D, constant ν per case |
| `inv_Dr` | `1 / Dr` | case-level | |
| `Dr_times_Lr` | `Dr × Lr` | case-level | |
| `z_hat_times_Dr` | `z_hat × Dr` | local | |
| `z_hat_times_Lr` | `z_hat × Lr` | local | |
| `dist_to_nearest_step` | `min(\|dist_to_throat_start\|, \|dist_to_throat_end\|)` | local | Unsigned |

Historical note: `dist_to_throat_start` and `dist_to_throat_end` are still
listed in `ENGINEERED_FEATURES` (and the `build_engineered_feature_map`
fallback computation), but the ETL writes them into Zarr directly, and
`load_feature_matrix` prefers the Zarr column when a name is in both
sets. They are documented under **Base features** above.

In ideal incompressible flow `V_local/V_bulk` equals `1/A_local_over_A`
exactly, so `log10_Re_local` would algebraically reduce to
`log10_Re − log10(d_local/D)`. The simulation-derived `V_local` lets
this term carry any deviation between the as-computed velocity and the
round-pipe identity (e.g., near separation regions).

`delta_p_case` (total ROI pressure drop) is stored in `metadata/` for
reference but is intentionally **excluded** from the allowlist — it is
target-adjacent and would leak.

### Verify extraction

```bash
# Check that Zarr stores were created
ls data/flow_contraction_expansion/parametric_study/processed/*.zarr | head -5

# Inspect one store (requires Python with zarr)
cd src && python -c "
import zarr, json
root = zarr.open('../data/flow_contraction_expansion/parametric_study/processed/Re_5000__Dr_0p144__Lr_0p01.zarr', mode='r')
print('features shape:', root['features'].shape)
print('targets shape:', root['targets'].shape)
meta = root['metadata']
print('case_id:', meta.attrs['case_id'])
print('Re:', meta.attrs['Re'])
print('feature_names:', json.loads(meta.attrs['feature_names']))
print('target_names:', json.loads(meta.attrs['target_names']))
"
```

Expected output:
```
features shape: (50, 13)
targets shape: (50, 2)
case_id: Re_5000__Dr_0p144__Lr_0p01
Re: 5000.0
feature_names: ['log10_Re', 'Dr', 'Lr', 'z_hat', 'd_local_over_D', 'A_local_over_A', 'V_local_over_V_bulk', 'is_upstream', 'is_throat', 'is_downstream', 'dD_dz_local', 'dist_to_throat_start', 'dist_to_throat_end']
target_names: ['log_alpha_D', 'signed_log1p_alpha_D']
```

### ETL configuration reference

Config file: `src/cases/alpha_d/configs/etl.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `etl.processing.num_processes` | `4` | Parallel workers |
| `etl.source.input_dir` | *(required)* | Parametric study root directory |
| `etl.source.mesh_scale` | `1.0` | Mesh coordinate scale factor |
| `etl.source.exodus_filename` | `simulation_out.e` | Exodus filename in each case dir |
| `etl.transformations.alpha_d_transform.n_stations` | `50` | Axial stations per case |
| `etl.transformations.alpha_d_transform.buffer_diams` | `1.0` | Upstream/downstream buffer (pipe diameters) |
| `etl.transformations.alpha_d_transform.rho` | `1.0` | Fluid density (kg/m^3) |
| `etl.sink.output_dir` | *(required)* | Directory for output Zarr stores |

## Step 2: PyCaret feature selection

The MLP training config selects its inputs from a `selected_features.txt`
file written by PyCaret. Run this step after ETL and before training:

```bash
cd src && python cases/alpha_d/run_feature_selection_pycaret.py
```

The script reads the per-case Zarr stores from
`../data/flow_contraction_expansion/parametric_study/processed/` (override
with `data.zarr_dir=…`), runs PyCaret regression with `GroupKFold` by
`case_id` over the ALLOWLIST-constrained candidate set, and writes
`selected_features.txt` to
`../data/feature_analysis/pycaret/default/` (override with
`output.dir=…`).

That output path is exactly where `train_mlp.yaml`'s
`data.input_columns_file` points, so the MLP training step picks it up
with no extra flags.

### When you can skip this step

- **Conv1D** (`train_conv1d`) hard-codes its input columns in
  `train_conv1d.yaml`, so the Conv1D path does not need PyCaret output.
- For the MLP, you can also bypass PyCaret by passing an explicit list
  on the CLI:
  ```bash
  python train.py --config-path cases/alpha_d/configs --config-name train_mlp \
    data.input_columns_file=null \
    'data.input_columns=[log10_Re,Dr,Lr,z_hat,d_local_over_D,V_local_over_V_bulk]'
  ```

### Common knobs

```bash
# Keep more features than the default 6
python cases/alpha_d/run_feature_selection_pycaret.py \
  pycaret.setup.n_features_to_select=8

# Write output somewhere else
python cases/alpha_d/run_feature_selection_pycaret.py \
  output.dir=../data/feature_analysis/pycaret/run_1
```

If you change `output.dir`, also override
`data.input_columns_file=…/selected_features.txt` in Step 3 so training
reads from the same place.

## Step 3: Train the MLP

The training config `src/cases/alpha_d/configs/train_mlp.yaml` is
pre-configured for the axial-profile MLP.

### Run training

```bash
docker compose run --rm etl bash -lc 'cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp'
```

This trains a 6-layer FullyConnected MLP (`layer_size=128`,
`activation=silu`, `skip_connections=true`) on `log(alpha_D)` using MSE loss.

### Override defaults from the CLI

```bash
# More epochs, different learning rate
docker compose run --rm etl bash -lc 'cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp \
  training.epochs=500 training.lr=1e-4'

# Larger model
docker compose run --rm etl bash -lc 'cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp \
  model.params.layer_size=256 model.params.num_layers=8'

# Different data directory (if your Zarr stores are elsewhere)
docker compose run --rm etl bash -lc 'cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp \
  data.zarr_dir=../data/my_processed_cases'
```

### Training output

- Checkpoint: `data/cases/train_mlp/model.mdlus` (PhysicsNeMo format; path = `${output.root_dir}/${hydra:job.config_name}/model.mdlus`)
- Run metadata: `data/cases/train_mlp/run_meta.json` (records split, model params, final loss)

### Lock a split for fair model comparisons

If you want future experiments to use exactly the same train/test cases,
export the split from an existing `run_meta.json` once and then switch
training to the file-based split strategy.

```bash
cd src && python export_split.py \
  --run-meta ../data/models/run_meta.json \
  --output-dir ../data/splits/alpha_d_locked_v1
```

This writes:

- `../data/splits/alpha_d_locked_v1/train.txt`
- `../data/splits/alpha_d_locked_v1/test.txt`

Then train future models against that same split:

```bash
docker compose run --rm etl bash -lc 'cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp \
  data.split.strategy=file \
  data.split.train_file=../data/splits/alpha_d_locked_v1/train.txt \
  data.split.test_file=../data/splits/alpha_d_locked_v1/test.txt'
```

This is the easiest way to make evaluation metrics comparable across
feature sets, HPO versions, and model architectures.

### Verify training

Check that the loss decreases:

```
Training model='mlp' adapter='pointwise' on 612 train case(s), 153 test case(s), device=cpu.
epoch 1/200: loss=2.345678e+00
epoch 10/200: loss=4.567890e-01
epoch 50/200: loss=1.234567e-02
...
```

### Training config reference

Config file: `src/cases/alpha_d/configs/train_mlp.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `model.name` | `mlp` | Model from registry (PhysicsNeMo FullyConnected) |
| `model.params.layer_size` | `128` | Hidden layer width |
| `model.params.num_layers` | `6` | Number of hidden layers |
| `model.params.activation_fn` | `silu` | Activation function |
| `model.params.skip_connections` | `true` | Skip connections every 2 layers |
| `data.zarr_dir` | `../data/.../processed` | Directory of alpha_D Zarr stores |
| `data.input_columns` | `null` | Explicit column list; `null` means defer to `input_columns_file` |
| `data.input_columns_file` | `…/pycaret/default/selected_features.txt` | PyCaret-written file (Step 2). Set `null` if you populate `input_columns` directly |
| `data.output_columns` | `[signed_log1p_alpha_D]` | Target column |
| `data.split.strategy` | `random` | Split strategy (`sequential`, `random`, `file`) |
| `data.split.train_ratio` | `0.8` | Fraction of cases for training |
| `training.epochs` | `200` | Training epochs |
| `training.batch_size` | `2048` | Batch size (rows, not cases) |
| `training.lr` | `3e-4` | Learning rate |
| `training.loss` | `mse` | Loss function |

## Step 4: Evaluate

```bash
docker compose run --rm etl bash -lc 'cd src && python evaluate.py --config-path cases/alpha_d/configs --config-name train_mlp'
```

`eval.checkpoint` defaults to `${output.checkpoint}` (the path written during
training), so no override is required for the standard flow.

The evaluator:

1. Loads the checkpoint and `run_meta.json`.
2. Reconstructs the dataset and train/test split from metadata.
3. Computes per-field MSE and RMSE on the held-out test cases.

### Expected output

```
Evaluated adapter='pointwise' on 153 test case(s), overall mse=X.XXXXXXe-XX, rmse=X.XXXXXXe-XX.
log_alpha_D: mse=X.XXXXXXe-XX, rmse=X.XXXXXXe-XX
```

A good baseline RMSE on `log(alpha_D)` is below 0.5 (meaning the predicted
alpha_D is within a factor of ~1.6 of the CFD value on average).  Lower values
indicate better fits.

### Verify with metrics JSON

```bash
docker compose run --rm etl bash -lc 'cd src && python evaluate.py --config-path cases/alpha_d/configs --config-name train_mlp \
  output.metrics_out=../data/cases/train_mlp/metrics.json'
```

### Plots

Pass `output.plot_dir=<path>` to the evaluator to generate the full
diagnostic set:

```bash
python evaluate.py --config-path cases/alpha_d/configs --config-name train_conv1d \
    output.plot_dir=../data/cases/train_conv1d/plots
```

The plotter writes (for the pointwise and profile adapters):

| File | What it shows |
|---|---|
| `best_<case>_alpha_D_profile.png` | Best-fit case: α_D(z) — three curves: ground truth (solid + markers), prediction (dashed), and the closed-form analytical baseline (dotted gray). |
| `worst_<case>_alpha_D_profile.png` | Same but for the worst-fit case by pointwise RMSE on the test split. |
| `parity_alpha_D.png` | Per-station scatter of predicted vs. ground-truth α_D across the entire test set, with `y = x` and ±10% reference lines. Points coloured by region (`is_upstream` / `is_throat` / `is_downstream`) when those flags appear in the selected feature set. |
| `delta_p_parity.png` | Per-case scatter of predicted ΔP (trapezoidal integral of predicted α_D over the ROI) vs. ground-truth `delta_p_case`. Same `y = x` + ±10% references, log-log axes when both sides are positive. Points coloured by `Dr` on a viridis colorbar — useful for spotting the low-Dr × high-Re corner where the surrogate has historically had the largest tail. |

`output.plot_dir` defaults to `null`, so no plots are produced unless
you opt in. Baseline / extended metrics (per-region MSE, R², Δp
relative-error breakdown by Dr) are written to
`output.metrics_out` regardless.

Then inspect:

```bash
python -c "import json; m=json.load(open('../data/cases/train_mlp/metrics.json')); print(json.dumps(m, indent=2))"
```

## Quick-reference command summary

```bash
# 1. Extract alpha_D profiles from CFD
docker compose run --rm etl bash -lc 'cd src && python cases/alpha_d/run_etl.py'

# 2. PyCaret feature selection (writes selected_features.txt)
docker compose run --rm etl bash -lc 'cd src && python cases/alpha_d/run_feature_selection_pycaret.py'

# 3. Train (HPO + retrain best, all in one command)
docker compose run --rm etl bash -lc 'cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp'

# 3b. Train directly without HPO (power user)
docker compose run --rm etl bash -lc 'cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp hpo=null'

# 4. Evaluate on held-out cases
docker compose run --rm etl bash -lc 'cd src && python evaluate.py --config-path cases/alpha_d/configs --config-name train_mlp'
```

If you only want the Conv1D model, skip step 2 and substitute
`--config-name train_conv1d` in step 3.

### Apptainer (HPC) equivalent

On systems without Docker, use the pre-built `multifid-th-gpu.sif` at the repo root
(or build one yourself per
[Getting Started → Build a SIF image](getting_started.md#step-2-build-a-sif-image)).
Add `--nv` for GPU; drop it for CPU-only. Replace `/path/to/project` with the
absolute path to your repo checkout (or set `APPTAINER_BIND` once and omit
`--bind`).

```bash
# 1. Extract alpha_D profiles (CPU is fine for ETL — drop --nv)
apptainer exec --bind /path/to/project:/path/to/project multifid-th-gpu.sif \
  bash -lc 'cd /path/to/project/src && python cases/alpha_d/run_etl.py'

# 2. PyCaret feature selection (CPU is fine — drop --nv)
apptainer exec --bind /path/to/project:/path/to/project multifid-th-gpu.sif \
  bash -lc 'cd /path/to/project/src && python cases/alpha_d/run_feature_selection_pycaret.py'

# 3. Train (HPO + retrain best)
apptainer exec --nv --bind /path/to/project:/path/to/project multifid-th-gpu.sif \
  bash -lc 'cd /path/to/project/src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp'

# 3b. Train directly without HPO
apptainer exec --nv --bind /path/to/project:/path/to/project multifid-th-gpu.sif \
  bash -lc 'cd /path/to/project/src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp hpo=null'

# 4. Evaluate
apptainer exec --nv --bind /path/to/project:/path/to/project multifid-th-gpu.sif \
  bash -lc 'cd /path/to/project/src && python evaluate.py --config-path cases/alpha_d/configs --config-name train_mlp'
```

`multifid-th-gpu.sif` (built from `docker/gpu.def`) ships PyTorch CUDA 12.4 wheels +
PhysicsNeMo and matches the `etl-gpu` Docker service; it runs CPU-only without
`--nv`. For a smaller CPU-only image, build `multifid-th-cpu.sif` from
`docker/physicsnemo-cpu.def`.

## Coupling the surrogate back into MOOSE

Once a model is trained, its per-station `α_D(z)` predictions can drive
a porous-medium Forchheimer closure inside a PINSFV simulation. The
end-to-end pipeline (exporter → MOOSE input → verifier) lives under
`src/cases/alpha_d/` and is described in
[Alpha-D Coupling Demo](../demo_cases/alpha_d_coupling_physics.md), which
covers the volume-averaged momentum equation, the α_D → C_F mapping
`F = α_D · ε² / D_h`, and a quantitative comparison of the coupled
MOOSE pressure drop against the resolved-CFD ground truth.

## Architecture notes

The alpha_D pipeline integrates with the existing generic training framework:

- **Model**: `mlp` in the model registry ({py:mod}`training.models.mlp`),
  wrapping PhysicsNeMo's `FullyConnected`.
- **Adapter**: {py:class}`training.adapters.PointwiseAdapter`, handling
  tabular `(B, D_in) -> (B, D_out)` data.
- **Dataset**: {py:class}`training.datasets_tabular.TabularPairDataset`,
  reads per-case `.zarr` stores and concatenates row-wise.  Splitting is
  done at the case level (never mixing rows from the same CFD case across
  train and test).
- **ETL**: `AlphaDSource` / `AlphaDTransformation` / `AlphaDZarrSink` in
  {py:mod}`cases.alpha_d.etl.source`, {py:mod}`cases.alpha_d.etl.transform`,
  and {py:mod}`cases.alpha_d.etl.sink`, following the PhysicsNeMo Curator
  pattern.
- **Experiment hooks**: {py:class}`cases.alpha_d.experiment.AlphaDExperiment`
  adds throat-weighted loss and the plotting hooks
  ({py:meth}`~cases.alpha_d.experiment.AlphaDExperiment.decode_for_plotting`,
  {py:meth}`~cases.alpha_d.experiment.AlphaDExperiment.baseline_for_plotting`)
  the runner uses for per-case profile and parity plots.
- **Feature data**: {py:mod}`cases.alpha_d.feature_data` owns the ALLOWLIST,
  engineered-feature synthesis, and the `FeatureAnalysisData` loader.

## Note: HPO is built into training

The `train_mlp.yaml` config includes an `hpo` section with a search
space. Both entry points — the top-level `train.py` and the case
wrapper `cases/alpha_d/train.py` — call the same
`training.runner.train_or_hpo` dispatcher, which runs Optuna HPO first
and then retrains with the best hyperparameters whenever the config
contains a non-empty `hpo.search_space`.

To skip HPO and train directly with the parameters in the config, set
`hpo=null` on the CLI:

```bash
cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp hpo=null
# or, equivalently, via the case wrapper:
cd src && python cases/alpha_d/train.py hpo=null
```

See [Hyperparameter Optimization Guide](hyperparameter_optimization.md)
for details on search-space format, study settings, and output artifacts.

Before training, use the
[Case Distribution Analysis](case_distribution_analysis.md) tool to
preview how much data you have in each ``Dr`` / ``Re`` / ``Lr`` bin --
especially after applying ``min_Dr`` or ``exclude_cases`` filters.

After running multiple HPO versions, review training progress and
compare evaluation metrics across versions using whichever notebook
or script you've adopted for cross-run comparison.

## Status

Landed since the original tutorial draft:

- **Region-weighted loss** — `data.throat_weight` / `data.downstream_weight`
  in `TabularPairDataset` (Conv1D + MLP).
- **Stratified case split** — `data.split.strategy: stratified` with
  `n_bins` covering Re × Dr × Lr.
- **Parity and per-case profile plots** — see [Plots](#plots) above.
- **Per-case Δp integral evaluation** — `extended.delta_p.per_case` in
  `eval_metrics.json` reports trapezoidal-integrated ΔP_pred vs.
  ground-truth `delta_p_case` for every test case.

Explicitly **not** done (tried, regressed, removed):

- A pressure-drop-consistency training loss (`delta_p_weight` in the
  experiment). Two attempts (`delta_p_weight = 0.25` direct;
  `delta_p_weight = 0.05` via HPO selection) degraded the test Δp by
  30–130% versus the pure relative_l2 baseline, even when val loss
  looked comparable. The log-space Δp loss has ~unit magnitude, so any
  non-trivial weight starves the per-station signal that the model
  actually fits well. The term and its hooks were removed from
  `AlphaDExperiment`.
- MOOSE offline coupling: export predictions for closed-loop verification
