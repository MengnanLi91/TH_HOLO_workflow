# Alpha-D surrogate case

This directory packages everything specific to the **α_D surrogate** — a
per-station Darcy resistance coefficient predicted along a pipe
contraction-expansion as a function of (Re, Dr, Lr, z). Two model
variants share the same ETL + feature pipeline:

- **MLP (`train_mlp`)** — pointwise FullyConnected predicting one row at
  a time. HPO over ~10 hyperparameters is enabled by default.
- **Conv1D profile (`train_conv1d`)** — 1D convolutional surrogate that
  consumes the full 50-station profile per case. No HPO by default.

## Layout

```
cases/alpha_d/
├── configs/           # Hydra YAMLs (train_mlp, train_conv1d, etl, pycaret)
├── datasets/          # AlphaDProfileDataset + build_dataset entry point
├── etl/               # PhysicsNeMo Curator pipeline (source, transform, sink)
├── physics/           # baseline, targets — alpha_D encoding + analytical baseline
├── experiment.py      # AlphaDExperiment — throat-weighted loss, Δp loss, decode hooks
├── feature_data.py    # ALLOWLIST, GROUPED_FEATURES, engineered_features_spec
├── metrics.py         # extended metrics (per-region MSE/RMSE, Δp evaluation)
├── transforms.py      # alpha_d_residual_transform (target = signed-log1p residual)
├── run_etl.py         # ETL entry point
├── train.py           # discoverability wrapper around the shared trainer
└── README.md          # this file
```

## End-to-end (from `src/`)

### 1. ETL: MOOSE → per-case Zarr

```bash
python cases/alpha_d/run_etl.py \
  etl.source.input_dir=../data/flow_contraction_expansion/parametric_study \
  etl.sink.output_dir=../data/flow_contraction_expansion/parametric_study/processed
```

Writes one `{case_name}.zarr` per simulation, each with a 50-station
feature/target matrix plus per-case metadata. See
[`docs/user/alpha_d_surrogate.md`](../../../docs/user/alpha_d_surrogate.md)
for the Zarr layout and feature reference.

### 2. PyCaret feature selection — required for MLP, skip for Conv1D

```bash
python cases/alpha_d/run_feature_selection_pycaret.py
```

Reads the Zarr stores, runs PyCaret regression with the
ALLOWLIST-constrained candidate set, writes `selected_features.txt`.

- **MLP** (`train_mlp.yaml`) pulls its input columns from
  `data.input_columns_file: …/selected_features.txt`, so this step
  must run first (or you must override `data.input_columns=[…]` and
  set `data.input_columns_file=null` from the CLI).
- **Conv1D** (`train_conv1d.yaml`) hard-codes its `input_columns` list
  in the YAML and does not read `input_columns_file`, so the Conv1D
  path skips this step entirely.

### 3. Train

```bash
# MLP with HPO + retrain best  (needs Step 2 output)
python train.py --config-path cases/alpha_d/configs --config-name train_mlp

# MLP, skip HPO  (needs Step 2 output)
python train.py --config-path cases/alpha_d/configs --config-name train_mlp hpo=null

# Conv1D profile model (no HPO; does not need Step 2)
python train.py --config-path cases/alpha_d/configs --config-name train_conv1d
```

A discoverability wrapper exists for the MLP path. It defaults to
`--config-name train_mlp` for this case but is otherwise equivalent to
the top-level `train.py` — both honour an `hpo` block in the config:

```bash
python cases/alpha_d/train.py                       # MLP with HPO (default config)
python cases/alpha_d/train.py hpo=null              # MLP, skip HPO
python cases/alpha_d/train.py --config-name train_conv1d   # Conv1D
```

### 4. Evaluate

```bash
python evaluate.py --config-path cases/alpha_d/configs --config-name train_mlp
```

`run_meta.json` written alongside the checkpoint reconstructs the exact
dataset, split, and `target_transform`, so the eval reproduces the
training conditions.

## Further reading

- [`docs/user/alpha_d_surrogate.md`](../../../docs/user/alpha_d_surrogate.md) — full tutorial with feature reference and config knobs.
- [`docs/user/hyperparameter_optimization.md`](../../../docs/user/hyperparameter_optimization.md) — HPO study layout and CLI overrides.
- [`docs/user/case_distribution_analysis.md`](../../../docs/user/case_distribution_analysis.md) — pre-training data audit.
