# Hyperparameter Optimization

The training runner uses a two-phase Optuna workflow: inexpensive screening on
one explicit case fold, followed by confirmation of the strongest sampled
candidates and scientific controls on every fold. The outer test cases are
never available to either phase.

## Required configuration

Every non-null `hpo` block uses the complete contract below. Obsolete
top-level `hpo.n_trials` and random `validation.split_ratio` configurations
are rejected.

```yaml
hpo:
  study_name: example_hpo_v1
  direction: minimize
  storage: sqlite:///${hpo.output_dir}/${hpo.study_name}.db
  load_if_exists: true
  retrain_best: true
  output_dir: ${output.case_dir}/hpo
  show_progress_bar: true

  screening:
    n_trials: 40
    max_epochs: 200
    early_stopping:
      patience: 20
      min_delta: 0.0

  validation:
    splitter_entrypoint: training.hpo.splits:random_case_folds
    n_folds: 3
    screening_fold: 0
    seed: 42

  objective:
    weights:
      profile_val_loss: 1.0

  confirmation:
    top_k: 5
    max_epochs: 500
    early_stopping:
      patience: 30
      min_delta: 0.0
    aggregate_std_weight: 0.5
    guard_metric: null
    guard_reference: null

  enqueue_trials: []

  sampler:
    name: TPESampler
    params:
      seed: ${hpo.validation.seed}
  pruner:
    name: MedianPruner
    params:
      n_startup_trials: 5
      n_warmup_steps: 10

  search_space:
    training.lr:
      type: float
      low: 1.0e-5
      high: 1.0e-2
      log: true
    training.batch_size:
      type: categorical
      choices: [128, 256, 512]
```

The search-space keys are existing configuration dot paths. Float, integer,
and categorical distributions are supported. Dataset identity, selected
input/output columns, model identity, adapter, and outer split paths cannot be
tuned. Case-owned scalar settings such as regional loss weights may be tuned
when the adapter can rebuild the dataset for each candidate.

## Fold builders and normalization

A splitter entry point implements:

```python
build_folds(
    sim_names: list[str],
    candidate_indices: list[int],
    n_folds: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]
```

`candidate_indices` is the outer training pool. Each candidate must appear in
validation exactly once; outer test cases must appear in no fold. Templates
use `training.hpo.splits:random_case_folds`. Alpha-D uses
`cases.alpha_d.hpo:balanced_parameter_folds`, which balances marginal Re, Dr,
and Lr levels.

When `data.normalize: true`, pointwise and profile adapters rebuild their
dataset using only the current fold's training indices to fit normalization
statistics. Normalization with an adapter that lacks this capability is an
error.

## Objective and checkpoints

Every epoch reports profile validation loss to Optuna for pruning. Screening
and confirmation both track the best profile-validation checkpoint, apply the
configured early stopping rule, restore the best weights, and only then
calculate objective metrics.

The runner always provides `profile_val_loss`. A custom experiment supplies
the other named components:

```python
class MyExperiment(Experiment):
    def compute_hpo_metrics(self, validation_dataset) -> dict[str, float]:
        return {"case_metric": compute_case_metric(self.model, validation_dataset)}
```

Every metric referenced by `objective.weights` must be present and finite.
The screening score is the weighted sum. Trials record best epoch, epochs
trained, component metrics, and composite score.

## Controls and confirmation

Controls are named, fixed parameter dictionaries and count toward the
screening budget:

```yaml
enqueue_trials:
  - name: published_control
    params:
      training.lr: 0.0003
      training.batch_size: 32
```

Confirmation runs the best `top_k` sampled candidates plus all controls on
every configured fold. It ranks candidates by:

```text
mean composite objective + aggregate_std_weight * population standard deviation
```

Set `guard_metric` and `guard_reference: best_control` to reject candidates
whose worst absolute fold value exceeds the best-ranked control's value. Set
both fields to `null` when the case has no scientific guard.

`best_params.json` and `best_config.yaml` always represent the confirmed
winner, never the single-fold screening winner.

## Artifacts

The HPO output directory contains:

| File | Contents |
|---|---|
| `screening.csv`, `screening.json` | Trial state, parameters, metrics, best epoch, and composite score |
| `confirmation.csv`, `confirmation.json` | Every candidate/fold/seed run, aggregates, guard state, and winner |
| `best_params.json` | Confirmed winner parameters |
| `best_config.yaml` | Train-ready confirmed configuration without `hpo` |
| `split_metadata.json` | Outer split and all named case folds |
| `trials.csv` and plots | Optuna screening history and diagnostics |
| study database | Resumable screening state |

Worker-backed loaders use persistent workers in screening, confirmation, and
final training.

## Running

From a prepared environment:

```bash
uv run python src/train.py \
  --config-path cases/alpha_d/configs \
  --config-name train_mlp
```

Set `hpo=null` through Hydra only when intentionally training once with the
base parameters. Changing a search space or scientific target requires a new
study name and database; databases from older HPO contracts are not consumed.
