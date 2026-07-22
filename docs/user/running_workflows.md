# Run a Reproducible Study Workflow

Use `multifid-workflow` when you want a complete study—data preparation,
training, evaluation, solver coupling, reporting, and publication—to share one
configuration and one resumable artifact directory.

## Which file do I change?

Three files have separate jobs:

| file | owned by | change it when |
|---|---|---|
| Study TOML | workflow user | choosing inputs, panels, executors, or an ML method |
| Hydra training YAML | ML user | changing architecture, adapter, loss, optimizer, or HPO |
| `study_workflow.py` | case developer | changing stages, physics, solver behavior, or reports |

For an ordinary model change, edit or add YAML and select it from TOML. You do
not need to modify `study_workflow.py`.

## Start a new case from the template

Copy `templates/case/src/cases/template_case` to
`src/cases/<your_case>`. The template provides a case-owned workflow builder,
PyCaret wrapper, generic training/HPO wrapper, and tracked TOML/YAML examples.
Its default workflow produces a canonical case-level split, excludes the
held-out cases from feature selection, uses them as the training runner's
outer test split, runs Optuna, and retrains the best trial.

For the template's tabular/profile Zarr convention, the one mandatory Python
customization is the leakage-safe `FeatureAnalysisData` loader. Model and HPO
changes belong in YAML. Edit `study_workflow.py` only when the new case needs
different stages, physics, solver coupling, or reporting. See
`templates/case/README.md` for the copy-and-rename checklist.

## Plan and run alpha-D

From the repository root:

```bash
uv sync
export MULTIFID_PYTHON_IMAGE=/absolute/path/to/multifid-th.sif
export MULTIFID_MOOSE_IMAGE=/absolute/path/to/moose-dev.sif

uv run multifid-workflow plan \
  --config src/cases/alpha_d/configs/coupling_study.toml

uv run multifid-workflow run \
  --config src/cases/alpha_d/configs/coupling_study.toml \
  --run-id alpha-d-conv1d-001
```

`plan` validates the TOML and stage DAG without training. To stop after one
stage and its dependencies, use `--target`:

```bash
uv run multifid-workflow run \
  --config src/cases/alpha_d/configs/coupling_study.toml \
  --run-id alpha-d-conv1d-001 \
  --target panel.indist_panel.train_alpha
```

Inspect progress at any time:

```bash
uv run multifid-workflow status \
  --run-dir data/workflows/alpha_d_coupling/alpha-d-conv1d-001
```

Rerun the same `run` command to resume. A succeeded stage is reused only while
its output checksums, upstream fingerprints, and semantic validator still
pass. Interrupted and failed stages rerun. A `partial` solver stage retains
failed cases and coverage instead of converting them into zero-valued results.

## Build alpha-D Zarr data from raw Exodus cases

The repository does not ship the large alpha-D campaign; `data/` is ignored.
Use the ETL-only workflow when you have a raw campaign whose case directories
contain `simulation_out.e` and `case_metadata.txt` (or whose root has
`cases_manifest.csv`). It needs only the Python image, not the MOOSE image:

```bash
export MULTIFID_PYTHON_IMAGE=/absolute/path/to/multifid-th.sif

uv run multifid-workflow plan \
  --config src/cases/alpha_d/configs/etl_workflow.toml

uv run multifid-workflow etl \
  --config src/cases/alpha_d/configs/etl_workflow.toml \
  --run-id alpha-d-etl-001 \
  --input-dir /absolute/path/to/parametric_study
```

`--input-dir` replaces `inputs.raw_dir` and automatically binds that directory
into the Python Apptainer invocation when it is outside the repository. The
generated stores and `data/etl_summary.json` are kept together under:

```text
data/workflows/alpha_d_etl/alpha-d-etl-001/data/processed/
```

To train or run the coupled study from those stores, copy the study TOML, set
`inputs.mode = "reuse"`, set `inputs.zarr_dir` to that generated directory,
then start the study with a different run ID. The ETL summary records any raw
cases that did not yield a Zarr store; that condition is reported as `partial`
rather than being silently treated as complete.

## Choose another ML method

The alpha-D TOML selects one method for the entire run:

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

To use a custom CNN:

1. Copy `src/cases/alpha_d/configs/train_conv1d.yaml` to
   `train_cnn.yaml`.
2. Set either a registered `model.name` or a custom builder:

   ```yaml
   model:
     entrypoint: user_models.cnn:build
     adapter: profile
     params:
       channels: 64
       depth: 4
   ```

3. Change the TOML method selection:

   ```toml
   [training.alpha]
   id = "custom_cnn"
   runner_module = "cases.alpha_d.train"
   config_name = "train_cnn"
   artifact_contract = "alpha_d_profile_v1"
   checkpoint = "model.mdlus"
   run_meta = "run_meta.json"
   ```

4. Run `plan`, then choose a new run ID. Model parameters and the HPO search
   space remain in `train_cnn.yaml`; do not duplicate them in TOML.

The coupling workflow requires `alpha_d_profile_v1`: metadata schema 2, the
`profile` adapter, reconstructible inputs, and one
`signed_log1p_alpha_D` profile. The exporter must produce
`forchheimer_profile_v1`. A method that does not satisfy those contracts fails
explicitly; coupling stages are never silently skipped.

## Run identity and artifacts

A run ID is bound to the resolved TOML, Git SHA and dirty-tree digest, and
fingerprints of declared input trees. If any of those change, use a new run ID.
This prevents checkpoints from one method or dataset being resumed as another.

```text
data/workflows/<workflow-id>/<run-id>/
├── resolved_config.json
├── run_manifest.json
├── fingerprint_cache.json
├── logs/
├── data/
├── panels/<tag>/
│   ├── panel_manifest.json
│   ├── heldout_cases.txt
│   ├── artifacts/alpha/{checkpoint,run_meta}
│   ├── coupled/<case>/
│   └── moose/<case>/
├── tuning/
└── report/
```

- `resolved_config.json` records the selected method.
- `run_manifest.json` records stage states, commands, checksums, and errors.
- `logs/` contains stdout/stderr and the exact argument array for each command.
- `panel_manifest.json` records method and contract identities plus artifact
  locations.
- `report/` contains generated evidence and figures.

## Publish results

Only `publish` may copy declared figures into `docs/_static`:

```bash
uv run multifid-workflow publish \
  --run-dir data/workflows/alpha_d_coupling/alpha-d-conv1d-001

uv run multifid-workflow publish \
  --run-dir data/workflows/alpha_d_coupling/alpha-d-conv1d-001 \
  --check
```

`--check` is read-only and reports drift between the run and published files.
The published-results manifest includes the method, contracts, configuration
hash, input fingerprints, solver coverage, summaries, and figure checksums.

## Troubleshooting

| error | meaning | action |
|---|---|---|
| missing `[training.alpha]` | the study has no selected method | add the complete TOML method block |
| unsupported contract | the builder cannot validate this artifact format | use the documented contract or implement a matching case exporter |
| checkpoint or metadata missing | the trainer did not honor its artifact contract | inspect the stage command and log under `logs/` |
| adapter must be `profile` | the selected YAML produces an incompatible sample/output shape | use a profile model or a different case workflow |
| output target mismatch | the model does not produce `signed_log1p_alpha_D` | correct `data.output_columns` in the YAML |
| Apptainer image not found | an executor image environment variable is missing or stale | set the absolute image path and resume |
| exporter sidecar/CSV invalid | coupling output is missing, non-finite, or for the wrong case | inspect the export log and rerun the stage |
| run directory bound to different inputs | TOML, code, or input data changed | choose a new `--run-id` |
