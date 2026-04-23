# Version Comparison

This guide covers using the version comparison tool to review HPO
training progress across versions and compare evaluation metrics.

## How it works

The comparison tool reads Optuna SQLite databases from `data/hpo/` and
evaluation metrics from `data/models*/`.  It produces:

1. **HPO summary table** -- best validation loss, trial counts, and
   version-over-version improvement for each HPO run.
2. **Hyperparameter diff** -- best parameters side by side so you can
   see what changed between versions.
3. **Evaluation comparison** -- test-set MSE, RMSE, R², per-region
   breakdown, and worst/best cases between a baseline and the latest
   model.
4. **Comparison plots** (optional) -- bar charts, box plots,
   convergence curves saved as a PNG.

```
data/hpo/
  alpha_d_mlp_hpo_v3.db   <-- discovered automatically
  alpha_d_mlp_hpo_v7.db
  ...
data/models/
  eval_metrics.json        <-- latest model evaluation
data/models_v1/
  eval_metrics.json        <-- baseline evaluation
```

Databases follow the naming convention `alpha_d_mlp_hpo_v*.db`.  The
tool discovers them automatically, so adding a v8 database requires no
configuration.

## Quick start

From inside the container:

```bash
cd src && python -m evaluation.compare_hpo_versions
```

From the host with Apptainer:

```bash
apptainer exec th-holo-gpu.sif bash -c \
    'cd src && python -m evaluation.compare_hpo_versions'
```

This prints the full comparison to stdout and saves a plot to
`data/hpo/version_comparison.png`.

## Usage examples

### Compare all discovered versions

```bash
cd src && python -m evaluation.compare_hpo_versions
```

### Compare specific versions only

```bash
cd src && python -m evaluation.compare_hpo_versions --versions v6 v7
```

### Text-only output (no matplotlib required)

```bash
cd src && python -m evaluation.compare_hpo_versions --no-plot
```

This works outside the container where matplotlib may not be installed.

### Save plot to a custom path

```bash
cd src && python -m evaluation.compare_hpo_versions --save ../data/hpo/report_v7.png
```

### Compare custom evaluation metrics files

By default the tool compares `data/models_v1/eval_metrics.json`
(baseline) against `data/models/eval_metrics.json` (latest).  Override
with:

```bash
cd src && python -m evaluation.compare_hpo_versions \
    --eval-a ../data/models_v1/eval_metrics.json \
    --eval-b ../data/models/eval_metrics.json
```

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--versions` | all discovered | Space-separated version names (e.g. `v6 v7`) |
| `--save` | `data/hpo/version_comparison.png` | Path to save the comparison plot |
| `--no-plot` | `false` | Skip plot generation |
| `--eval-a` | `data/models_v1/eval_metrics.json` | Baseline evaluation metrics JSON |
| `--eval-b` | `data/models/eval_metrics.json` | Latest evaluation metrics JSON |

## Output sections

### HPO Version Comparison

Shows each version's study name, number of completed and pruned trials,
and best validation loss.  A summary line reports the percentage change
between the latest version and the previous one, and between the latest
and the first.

```
Version  Study                        Complete  Pruned  Best Val Loss
v1       alpha_d_mlp_hpo                    56      44       0.390069
v7       alpha_d_mlp_hpo_v7                 20      30       0.267040

  Latest vs previous: -14.8% (v6 0.313535 -> v7 0.267040)
  Latest vs first:    -31.5% (v1 0.390069 -> v7 0.267040)
```

### Best Hyperparameters

Displays the best trial's hyperparameters for each version in columns.
Parameters not present in a version's search space show `---`.  This
makes it easy to spot when new parameters were introduced (e.g.
`training.delta_p_weight` in v7) or removed.

### Evaluation Metrics

Side-by-side comparison of test-set metrics between two model
checkpoints:

- **Overall**: MSE, RMSE, test case/sample counts.
- **Extended**: R², MAE, physical-space relative errors (median, mean,
  p90) -- available when the evaluation was run with the pointwise
  adapter.
- **Per-region**: R², RMSE, and median relative error for upstream,
  throat, and downstream regions.
- **Worst/best cases**: The 5 worst and 5 best test cases by RMSE.

Each row includes a percentage change column so regressions are
immediately visible.

### Comparison plots

When matplotlib is available the tool generates a four-panel figure:

| Panel | Description |
|-------|-------------|
| Best Validation Loss | Bar chart of best loss per version |
| Trial Value Distribution | Box plot of all completed trial losses |
| Trial Outcomes | Completed vs pruned counts per version |
| Convergence per Version | Running-best loss over completed trials |

## Typical workflow

After finishing a new HPO run (e.g. v8):

1. The HPO pipeline saves `alpha_d_mlp_hpo_v8.db` to `data/hpo/`.
2. Retrain the best config and evaluate, producing
   `data/models/eval_metrics.json`.
3. Run the comparison tool:

```bash
cd src && python -m evaluation.compare_hpo_versions --versions v7 v8
```

4. Review the output to confirm the new version improves over the
   previous one and check for regressions in specific regions.

## Adding a new baseline

To preserve the current model as a named baseline before starting a new
round of experiments:

```bash
mkdir -p data/models_v2
cp data/models/eval_metrics.json data/models_v2/
cp data/models/run_meta.json     data/models_v2/
cp data/models/alpha_d_mlp.mdlus data/models_v2/
```

Then compare against it:

```bash
cd src && python -m evaluation.compare_hpo_versions \
    --eval-a ../data/models_v2/eval_metrics.json \
    --eval-b ../data/models/eval_metrics.json
```

## Related guides

- [Hyperparameter Optimization](hyperparameter_optimization.md) --
  configuring and running HPO studies.
- [Alpha-D Surrogate Tutorial](alpha_d_surrogate.md) -- end-to-end
  training and evaluation workflow.
