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
cd src && python run_alpha_d_etl.py \
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
    features      float32 [50, 10]   -- 50 stations x 10 input features
    targets       float32 [50, 1]    -- 50 stations x 1 target (log_alpha_D)
    sample_weight float32 [50]       -- per-station weights (higher in throat)
    metadata/
        attrs: case_id, feature_names, target_names, Re, Dr, Lr, delta_p_case
```

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
features shape: (50, 10)
targets shape: (50, 1)
case_id: Re_5000__Dr_0p144__Lr_0p01
Re: 5000.0
feature_names: ['log10_Re', 'Dr', 'Lr', 'z_hat', 'd_local_over_D', 'is_upstream', 'is_contraction', 'is_throat', 'is_expansion', 'is_downstream']
target_names: ['log_alpha_D']
```

### ETL configuration reference

Config file: `src/alpha_d_etl/config/alpha_d_etl.yaml`

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

## Step 2: Train the MLP

The training config `src/config/alpha_d_mlp.yaml` is pre-configured for
the axial-profile MLP.

### Run training

```bash
docker compose run --rm etl bash -lc 'cd src && python train.py --config-name alpha_d_mlp'
```

This trains a 6-layer FullyConnected MLP (`layer_size=128`,
`activation=silu`, `skip_connections=true`) on `log(alpha_D)` using MSE loss.

### Override defaults from the CLI

```bash
# More epochs, different learning rate
docker compose run --rm etl bash -lc 'cd src && python train.py --config-name alpha_d_mlp \
  training.epochs=500 training.lr=1e-4'

# Larger model
docker compose run --rm etl bash -lc 'cd src && python train.py --config-name alpha_d_mlp \
  model.params.layer_size=256 model.params.num_layers=8'

# Different data directory (if your Zarr stores are elsewhere)
docker compose run --rm etl bash -lc 'cd src && python train.py --config-name alpha_d_mlp \
  data.zarr_dir=../data/my_processed_cases'
```

### Training output

- Checkpoint: `data/models/alpha_d_mlp.mdlus` (PhysicsNeMo format)
- Run metadata: `data/models/run_meta.json` (records split, model params, final loss)

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

Config file: `src/config/alpha_d_mlp.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `model.name` | `mlp` | Model from registry (PhysicsNeMo FullyConnected) |
| `model.params.layer_size` | `128` | Hidden layer width |
| `model.params.num_layers` | `6` | Number of hidden layers |
| `model.params.activation_fn` | `silu` | Activation function |
| `model.params.skip_connections` | `true` | Skip connections every 2 layers |
| `data.zarr_dir` | `../data/.../processed` | Directory of alpha_D Zarr stores |
| `data.input_columns` | 10 features | See config for full list |
| `data.output_columns` | `[log_alpha_D]` | Target column |
| `data.split.strategy` | `random` | Split strategy (`sequential`, `random`, `file`) |
| `data.split.train_ratio` | `0.8` | Fraction of cases for training |
| `training.epochs` | `200` | Training epochs |
| `training.batch_size` | `2048` | Batch size (rows, not cases) |
| `training.lr` | `3e-4` | Learning rate |
| `training.loss` | `mse` | Loss function |

## Step 3: Evaluate

```bash
docker compose run --rm etl bash -lc 'cd src && python evaluate.py --config-name alpha_d_mlp \
  eval.checkpoint=../data/models/alpha_d_mlp.mdlus'
```

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
docker compose run --rm etl bash -lc 'cd src && python evaluate.py --config-name alpha_d_mlp \
  eval.checkpoint=../data/models/alpha_d_mlp.mdlus \
  output.metrics_out=../data/models/alpha_d_mlp_metrics.json'
```

Then inspect:

```bash
python -c "import json; m=json.load(open('../data/models/alpha_d_mlp_metrics.json')); print(json.dumps(m, indent=2))"
```

## Quick-reference command summary

```bash
# 1. Extract alpha_D profiles from CFD
docker compose run --rm etl bash -lc 'cd src && python run_alpha_d_etl.py \
  etl.source.input_dir=../data/flow_contraction_expansion/parametric_study \
  etl.sink.output_dir=../data/flow_contraction_expansion/parametric_study/processed'

# 2. Train MLP surrogate
docker compose run --rm etl bash -lc 'cd src && python train.py --config-name alpha_d_mlp'

# 3. Evaluate on held-out cases
docker compose run --rm etl bash -lc 'cd src && python evaluate.py --config-name alpha_d_mlp \
  eval.checkpoint=../data/models/alpha_d_mlp.mdlus \
  output.metrics_out=../data/models/alpha_d_mlp_metrics.json'
```

## Architecture notes

The alpha_D pipeline integrates with the existing generic training framework:

- **Model**: `mlp` in the model registry (`src/training/models/mlp.py`),
  wrapping PhysicsNeMo's `FullyConnected`.
- **Adapter**: `PointwiseAdapter` in `src/training/adapters.py`, handling
  tabular `(B, D_in) -> (B, D_out)` data.
- **Dataset**: `TabularPairDataset` in `src/training/datasets_tabular.py`,
  reads per-case `.zarr` stores and concatenates row-wise.  Splitting is
  done at the case level (never mixing rows from the same CFD case across
  train and test).
- **ETL**: `AlphaDSource` / `AlphaDTransformation` / `AlphaDZarrSink` in
  `src/alpha_d_etl/`, following the PhysicsNeMo Curator pattern.

## Next steps (planned)

- Weighted MSE loss with higher weight in the throat/expansion region
- Stratified 70/15/15 split over Re, Dr, Lr
- Pressure-drop consistency loss (case-level integrated delta-P)
- Parity plots and error-by-parameter diagnostics
- MOOSE offline coupling: export predictions for closed-loop verification
