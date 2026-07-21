# Reusable Study Workflows

The workflow layer composes existing ETL, training, evaluation, solver, and
report commands without moving case physics into the orchestration core. A
tracked TOML selects a case builder. The builder returns a
`WorkflowDefinition`; the generic runner owns DAG ordering, execution, logs,
provenance, checksums, and resume decisions.

## Public workflow objects

| object | responsibility |
|---|---|
| `WorkflowDefinition` | workflow ID/version, stages, input fingerprint hook, publisher |
| `Stage` | one named action, dependencies, description, and optional validator |
| `RunContext` | repository/run paths, resolved configuration, executors, command logging |
| `Command` | immutable argument array, executor, working directory, environment, label |
| `Artifact` | durable file or directory whose content is fingerprinted |
| `StageResult` | succeeded/partial state, artifacts, and structured details |
| `WorkflowRunner` | manifest transitions, dependency invalidation, resume, publication |

A validator runs immediately after its action and again before a succeeded or
partial stage is reused. It should return `False` when an artifact exists but
is semantically invalid. The action should also raise a detailed error when it
can identify the failed contract, so the first-run log is actionable.

## Generic core and case boundary

`src/workflows/` never imports `cases.*`. It knows how to run and resume a DAG,
but not how to build a panel, interpret pressure, choose a target, or validate
a solver result. Those decisions belong in `src/cases/<case>/`.

A minimal case integration has two files:

1. `configs/<study>.toml` with `[workflow]`, inputs, executors, and case-owned
   configuration.
2. `study_workflow.py` exporting
   `build_workflow(config, repo_root) -> WorkflowDefinition`.

```python
from workflows import Artifact, Stage, StageResult, WorkflowDefinition


def prepare(context):
    output = context.run_dir / "data/prepared.json"
    # Case-owned preparation writes output here.
    return StageResult(artifacts=[Artifact("data/prepared.json")])


def build_workflow(config, repo_root):
    del repo_root
    return WorkflowDefinition(
        workflow_id=config["workflow"]["id"],
        version=int(config["workflow"]["version"]),
        stages=(Stage("prepare", prepare),),
    )
```

Stage actions may create generated files only beneath `context.run_dir`. Run
subprocesses through `context.run(Command(argv=(...)))`; never use a shell
command string. Return every durable output as an `Artifact`.

## Configuration ownership

Keep the three extension surfaces separate:

- Study TOML: orchestration, inputs, panels, executors, selected training
  method, artifact contracts, and publication.
- Hydra YAML: model architecture, dataset/adapter, split, loss, optimizer,
  training schedule, and HPO search space.
- Case builder: stages, dependencies, physics, command overrides, semantic
  validators, solver behavior, and reporting.

One run selects one method. Because the resolved TOML is part of the run
binding, changing the selected method requires a new run ID.

## Alpha-D training-method contract

The alpha-D builder parses this required case-owned method specification:

```toml
[training.alpha]
id = "conv1d_profile"
runner_module = "cases.alpha_d.train"
config_name = "train_conv1d"
artifact_contract = "alpha_d_profile_v1"
checkpoint = "model.mdlus"
run_meta = "run_meta.json"

[training.alpha.hpo]
enabled = true
reference_panel = "indist_panel"

[training.alpha.export]
module = "cases.alpha_d.export_friction_profile"
contract = "forchheimer_profile_v1"
```

`runner_module` is invoked with `python -m`, `--config-name`, and Hydra
overrides for the canonical held-out file, selected features, artifact root,
checkpoint, metadata, and HPO. A replacement runner must honor that CLI and
produce the configured artifacts.

`alpha_d_profile_v1` requires:

- a nonempty checkpoint;
- `training_run_meta_schema: 2`;
- a recorded model entrypoint and `profile` adapter;
- nonempty input columns and effective dataset metadata;
- valid normalization statistics when normalization is enabled; and
- exactly one `signed_log1p_alpha_D` output.

`forchheimer_profile_v1` requires the exporter CLI flags `--zarr`,
`--checkpoint`, `--run-meta`, and `--output-csv`. For every report case it
must write a nonempty `z,F` CSV and adjacent `.meta.json` with matching case ID
and finite truth/surrogate pressure values.

These contracts are case-specific; they do not belong in `workflows`.

## Add or replace an ML method

Use the smallest matching seam:

### Registered model

Implement `build(model_cfg, dataset_info)`, then register it with an existing
adapter:

```python
register_model("my_cnn", build_fn=build, adapter="profile")
```

### Custom model entrypoint

No registry edit is required when the YAML supplies:

```yaml
model:
  entrypoint: user_models.cnn:build
  adapter: profile
```

The build function must accept `(model_cfg, dataset_info)`. The generic runner
records the resolved entrypoint, adapter, parameters, and effective dataset
contract in `run_meta.json`.

### New adapter or dataset

Reuse `grid`, `graph`, `pointwise`, or `profile` when the sample and output
shapes match. Add an adapter only for a genuinely new batch/forward/metric
contract. A case-owned dataset can be selected with
`data.dataset_entrypoint` without importing the case from the generic adapter.

### Custom experiment

Set `training.experiment` only when the method needs custom loss composition,
metrics, preparation, or train/eval steps. Ordinary supervised methods use the
base experiment.

### Custom exporter

Set `training.alpha.export.module` to another module implementing the standard
exporter CLI and `forchheimer_profile_v1`. If the new model cannot satisfy that
output contract, it needs a different case workflow rather than silently
skipping coupling.

## Commands and executors

Commands are argument arrays, so neither the local nor Apptainer executor uses
shell evaluation. Each invocation writes:

- `logs/<stage>.<index>.<label>.command.txt`, containing the replayable command;
- `logs/<stage>.<index>.<label>.log`, containing combined stdout/stderr.

The host planner requires only Python 3.11 standard-library support. Heavy ML
and solver imports stay inside the configured subprocess environment.
Apptainer images may be configured directly or through an environment
variable such as `MULTIFID_PYTHON_IMAGE`.

## Fingerprints, manifests, and resume

Declare external input datasets through `WorkflowDefinition.input_paths`.
Their full SHA-256 tree fingerprints bind the run ID; unchanged file digests
are cached after the first scan. The binding also includes the resolved TOML,
Git SHA, dirty status, and dirty-content digest.

`run_manifest.json` transitions stages through `pending`, `running`,
`succeeded`, `partial`, and `failed`. A stage is reusable only when:

1. its previous state is `succeeded` or `partial`;
2. every artifact checksum still matches;
3. dependency fingerprints still match; and
4. its semantic validator passes.

Failed or interrupted stages rerun. Changed upstream outputs invalidate their
dependents. `partial` is terminal and is intended for matrices that retain
explicit per-case failures while still allowing reports to describe coverage.

## Publication

A case publisher is the only workflow operation allowed to copy declared
figures into `docs/_static`. It must write a published-results manifest with
workflow/config hashes, code and input provenance, selected method/contracts,
summaries, coverage, and figure checksums. `publish --check` performs the same
comparison without writing.

## Checklists

### Add a new case

- Start by copying `templates/case/src/cases/template_case` to
  `src/cases/<your_case>`. Its README maps every template file to its intended
  customization seam and includes working PyCaret and Optuna configuration.
- Implement the case's leakage-safe `FeatureAnalysisData` loader; keep rows
  from the same simulation in one group and enforce a reviewed feature
  allowlist.
- Add tracked study TOML and a `build_workflow` entrypoint.
- Keep all generated artifacts under the run directory.
- Declare external input trees.
- Use argument-array commands and named executors.
- Add semantic validators for model, solver, and report contracts.
- Add publisher drift checking only for declared documentation outputs.
- Test DAG errors, resume, partial failures, and one fake end-to-end run.

### Add only a new ML method

- Implement or register the model builder.
- Select an existing adapter or add the minimum new adapter contract.
- Add a Hydra YAML with data, model, training, and optional HPO settings.
- Select that YAML and method ID in the study TOML.
- Confirm the method satisfies the case artifact/export contracts.
- Run `multifid-workflow plan` and start with a new run ID.
