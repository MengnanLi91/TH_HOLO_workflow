# Hyperparameter Optimization

This guide covers using the Optuna-based HPO module to automatically tune
model and training hyperparameters.

## How it works

The HPO module wraps the existing training pipeline.  For each trial Optuna:

1. Samples hyperparameters from a YAML-defined search space.
2. Builds a fresh model with those hyperparameters.
3. Trains on an inner training split and evaluates on a held-out
   validation split.
4. Reports the validation loss to Optuna, which may prune unpromising
   trials early.

The held-out **test set** is never used during HPO.  After optimization
you retrain on the full training pool with the best hyperparameters and
evaluate on the test set as usual.

```
Full dataset
  +-- Test set (untouched during HPO)
  +-- Training pool
       +-- Inner train (gradient updates during HPO)
       +-- Validation (HPO objective / pruning signal)
```

## Quick start

HPO is built into `train.py`.  When a config contains an `hpo` section
with a non-empty `search_space`, `train.py` automatically runs Optuna
HPO first, then retrains the best configuration.

```bash
# HPO + retrain best (default for alpha_d_mlp)
cd src && python train.py --config-name alpha_d_mlp

# Skip HPO, train directly (power user)
cd src && python train.py --config-name alpha_d_mlp hpo=null

# Quick HPO test
cd src && python train.py --config-name alpha_d_mlp hpo.n_trials=3 training.epochs=2
```

All commands run from `src/` inside the container.  From the host:

```bash
docker compose run --rm etl bash -lc 'cd src && python train.py --config-name alpha_d_mlp'
```

## Add HPO to a training config

Add an `hpo` section directly to the model's training config.  When the
section is present, `train.py` runs HPO automatically.  Set `hpo: null`
on the CLI to bypass it.

### Example: `hpo` section in `src/config/alpha_d_mlp.yaml`

```yaml
defaults:
  - default
  - _self_

model: ...
data: ...
training: ...
output: ...

hpo:
  study_name: alpha_d_mlp_hpo
  direction: minimize
  n_trials: 100
  show_progress_bar: true
  seed: 42
  storage: sqlite:///hpo_results/${hpo.study_name}.db
  load_if_exists: true
  retrain_best: false

  sampler:
    name: TPESampler
    params:
      seed: ${hpo.seed}

  pruner:
    name: MedianPruner
    params:
      n_startup_trials: 5
      n_warmup_steps: 10

  validation:
    split_ratio: 0.2
    seed: ${hpo.seed}

  output_dir: hpo_results

  search_space:
    training.lr:
      type: float
      low: 1.0e-5
      high: 1.0e-2
      log: true

    training.batch_size:
      type: categorical
      choices: [512, 1024, 2048, 4096]

    model.params.layer_size:
      type: categorical
      choices: [64, 128, 256, 512]

    model.params.num_layers:
      type: int
      low: 2
      high: 10

    model.params.activation_fn:
      type: categorical
      choices: [silu, relu, tanh, gelu]

    model.params.skip_connections:
      type: categorical
      choices: [true, false]

    training.loss:
      type: categorical
      choices: [mse, l1, relative_l2]
```

## Search space format

Each entry in `hpo.search_space` maps a **dot-path config key** to a
parameter specification.  The dot-path must point to an existing key
in the base training config.

### Supported types

| Type | Keys | Example |
|------|------|---------|
| `float` | `low`, `high`, `log` (optional, default false) | `training.lr: {type: float, low: 1e-5, high: 1e-2, log: true}` |
| `int` | `low`, `high`, `log` (optional) | `model.params.num_layers: {type: int, low: 2, high: 10}` |
| `categorical` | `choices` (list) | `model.params.activation_fn: {type: categorical, choices: [silu, relu]}` |

### Allowed parameter paths

Only `training.*` and `model.params.*` paths are allowed.  The following
prefixes are **rejected** because they change the dataset or model identity
rather than tuning hyperparameters:

- `data.*` (dataset path, columns, split)
- `model.name` (model type)
- `model.entrypoint` (custom model class)
- `model.adapter` (adapter family)

Typos in dot-paths (e.g. `training.lrr`) are caught before any trial
runs.

## Study settings reference

| Key | Default | Description |
|-----|---------|-------------|
| `hpo.study_name` | `hpo_study` | Name for the Optuna study (used in storage) |
| `hpo.direction` | `minimize` | `minimize` or `maximize` |
| `hpo.n_trials` | `50` | Number of optimization trials |
| `hpo.timeout` | `null` | Optional time limit in seconds |
| `hpo.show_progress_bar` | `true` | Show Optuna's study-level progress bar during optimization |
| `hpo.seed` | `42` | Seed for sampler and validation split |
| `hpo.storage` | `sqlite:///...` | Optuna storage URL (enables resume) |
| `hpo.load_if_exists` | `true` | Resume a previous study if storage exists |
| `hpo.retrain_best` | `false` | Automatically retrain the best config after HPO |
| `hpo.output_dir` | `hpo_results` | Directory for artifacts |

### Sampler

| Key | Default | Description |
|-----|---------|-------------|
| `hpo.sampler.name` | `TPESampler` | Optuna sampler class name |
| `hpo.sampler.params` | `{seed: 42}` | Passed to sampler constructor |

Available samplers: `TPESampler` (default, best for mixed types),
`CmaEsSampler` (continuous-only), `RandomSampler`, `GridSampler`.

### Pruner

| Key | Default | Description |
|-----|---------|-------------|
| `hpo.pruner.name` | `MedianPruner` | Optuna pruner class name |
| `hpo.pruner.params.n_startup_trials` | `5` | Complete this many trials before pruning |
| `hpo.pruner.params.n_warmup_steps` | `10` | Epochs before pruning within a trial |

### Validation split

| Key | Default | Description |
|-----|---------|-------------|
| `hpo.validation.split_ratio` | `0.2` | Fraction of training cases used for validation |
| `hpo.validation.seed` | `42` | Seed for the inner train/val split |

The same validation split is used across all trials for fair comparison.

## Usage examples

### Default: HPO + retrain best

```bash
cd src && python train.py --config-name alpha_d_mlp
```

### Skip HPO (power user, direct training)

```bash
cd src && python train.py --config-name alpha_d_mlp hpo=null
```

### Quick test run

```bash
cd src && python train.py --config-name alpha_d_mlp \
    hpo.n_trials=3 training.epochs=2
```

### HPO only, no retrain

```bash
cd src && python train.py --config-name alpha_d_mlp hpo.retrain_best=false
```

### Resume a previous study

Studies are automatically resumed when `hpo.load_if_exists` is true
(default) and the storage file exists.  Simply re-run the same command:

```bash
cd src && python train.py --config-name alpha_d_mlp
```

New trials are added to the existing study.

### Manual retrain with the saved best config

HPO saves a train-ready config (no `hpo` section) to the output
directory:

```bash
cd src && python train.py \
    --config-path ../data/hpo --config-name best_config
```

## Output artifacts

After HPO completes, the output directory (default `hpo_results/`)
contains:

| File | Description |
|------|-------------|
| `best_config.yaml` | Full training config with best hyperparameters applied (train-ready, no `hpo` section) |
| `best_params.json` | Just the best trial's parameter values |
| `split_metadata.json` | Outer train/test sims and inner train/val sims for reproducibility |
| `optimization_history.png` | Objective value over trials |
| `param_importances.png` | Which parameters matter most |
| `parallel_coordinate.png` | Multi-dimensional parameter visualization |
| `slice_plot.png` | Per-parameter effect on objective |
| `trials.csv` | Full trial history as CSV |
| `{study_name}.db` | SQLite database (Optuna storage) |

Visualization files are generated on a best-effort basis.  If
`matplotlib` is not installed they are silently skipped.

## Add HPO for a new model

No code changes are needed.  Add an `hpo` section to the model's
training config:

```yaml
# src/config/fno.yaml  (add this at the end)
hpo:
  study_name: fno_hpo
  n_trials: 50
  retrain_best: true
  storage: sqlite:///../data/hpo/${hpo.study_name}.db
  output_dir: ../data/hpo
  search_space:
    training.lr:
      type: float
      low: 1.0e-5
      high: 1.0e-2
      log: true
    model.params.num_fno_layers:
      type: int
      low: 2
      high: 8
    model.params.latent_channels:
      type: categorical
      choices: [16, 32, 64, 128]
```

Then run:

```bash
cd src && python train.py --config-name fno          # HPO + retrain
cd src && python train.py --config-name fno hpo=null # direct training
```

## Custom experiment classes

If your training config uses `training.experiment` to specify a custom
`Experiment` subclass, HPO will use it with full parity.  The custom
class's `validation_step()` method (if defined) is called during
validation.  If not defined, the default `eval_step()` + `loss_fn`
path is used.

## Tips

- Start with a small number of trials (`hpo.n_trials=10`) and few epochs
  (`training.epochs=5`) to verify the search space is correct before
  launching a full sweep.
- Use `log: true` for learning rate and other parameters that span
  orders of magnitude.
- The `MedianPruner` with `n_warmup_steps=10` avoids pruning trials
  that just need more epochs to converge.
- If you change the search space definition, start a fresh study by
  choosing a new `hpo.study_name` or deleting the old `.db` file.
- After finishing an HPO run, use the
  [version comparison tool](version_comparison.md) to review progress
  and check for regressions across versions.
- Before tuning ``data.min_Dr`` or ``data.exclude_cases``, preview the
  resulting distribution with
  [`analyze_case_distribution.py`](case_distribution_analysis.md) so
  you don't accidentally drop bins below ``⚠ low`` support.
