# Case Template

Copy this package when starting a new tabular or profile-surrogate case. It
demonstrates the repository's supported extension seams without adding a fake
case to the installed `cases` package.

The template includes:

- a case-owned workflow builder;
- a PyCaret feature-selection wrapper;
- a generic training/HPO wrapper;
- tracked study, PyCaret, and model YAML/TOML configuration; and
- one explicit case-specific TODO: constructing leakage-safe
  `FeatureAnalysisData`.

Its default DAG is:

```text
prepare_data -> plan_cases -> select_features -> train_model -> summarize
```

`plan_cases` writes one deterministic train/held-out split. Feature selection
never sees the held-out cases, while HPO uses the same file as its outer test
split and divides only the training cases with the configured explicit fold
builder. Screening restores its best validation checkpoint; confirmation ranks
the top sampled candidates across every fold before retraining the winner.

## Start a case

1. Copy `templates/case/src/cases/template_case` to
   `src/cases/<your_case>`.
2. Replace every `template_case` occurrence with your importable case name.
3. Implement `load_feature_matrix()` in `feature_data.py` and replace the
   example feature allowlist and target.
4. Update `configs/train_model.yaml` for the model, adapter, data fields, loss,
   and HPO search space.
5. Update `configs/pycaret.yaml` and `configs/study.toml` for your data and
   workflow ID.
6. Run `rg -n "template_case|TODO\(case\)" src/cases/<your_case>`; no template
   identifiers should remain and every case TODO should be resolved.

For the assumed Zarr schema, `feature_data.py` is the only mandatory Python
implementation. Change `study_workflow.py` when the case needs different
stages, an ETL step, physics validation, solver coupling, or publication.
Changing the ML architecture or HPO search space normally requires only
`configs/train_model.yaml`; selecting a different training entrypoint requires
only `[training.method]` in `configs/study.toml`.

The template assumes each case is a `.zarr` store containing `features` and
`targets` arrays plus metadata attributes `case_id`, `feature_names`, and
`target_names`. If your data has a different sample contract, add a
case-owned dataset entrypoint or adapter as described in the workflow
developer guide.

## Validate before training

```bash
uv run multifid-workflow plan \
  --config src/cases/<your_case>/configs/study.toml
```

Then run with a new ID:

```bash
uv run multifid-workflow run \
  --config src/cases/<your_case>/configs/study.toml \
  --run-id <your-case>-001
```

The workflow creates one canonical held-out case file, passes it to PyCaret
and training, runs Optuna because `train_model.yaml` has a nonempty
`hpo.search_space`, retrains the best trial, and records the checkpoint and
metadata under the workflow run directory. Set `hpo: null` in the training
YAML when the case should train once without HPO.

## Template files

| path | customize for |
|---|---|
| `feature_data.py` | leakage-safe PyCaret rows, targets, groups, and allowlist |
| `configs/pycaret.yaml` | selection target, preprocessing, ranker, and feature count |
| `configs/train_model.yaml` | model/adapter, data contract, loss, schedule, and HPO space |
| `configs/study.toml` | workflow identity, input data, executor, and selected method |
| `study_workflow.py` | case-specific stages, validators, physics, solver, or reports |
| `train.py` | normally unchanged; dispatches training versus HPO |
