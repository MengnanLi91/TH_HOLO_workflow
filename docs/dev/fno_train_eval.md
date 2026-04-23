# FNO Training and Evaluation

This page describes the FNO workflow using the generic training framework.

## Entry points and configs

- Training entry point: `src/train.py`
- Evaluation entry point: `src/evaluate.py`
- Base config: `src/config/default.yaml`
- FNO example config: `src/config/fno.yaml`

`fno.yaml` inherits `default.yaml` and sets FNO-specific model/data/training/eval/output values.

## Train an FNO model

From `src/`:

```bash
python train.py --config-name fno
```

With overrides:

```bash
python train.py --config-name fno training.epochs=50 model.params.num_fno_layers=6
```

Training writes:

- checkpoint: `output.checkpoint` (for example `../data/models/lid_driven_fno.mdlus`)
- run metadata: `run_meta.json` next to the checkpoint

`run_meta.json` stores resolved data fields, split names, adapter, model entrypoint, and model params.

## Evaluate an FNO checkpoint

From `src/`:

```bash
python evaluate.py --config-name fno
```

You can also pass checkpoint explicitly:

```bash
python evaluate.py --config-name fno eval.checkpoint=../data/models/lid_driven_fno.mdlus
```

Evaluation reads `run_meta.json` from the checkpoint directory (or `eval.run_meta` if set),
reconstructs the dataset/split, and computes element-weighted metrics.

## Optional plots during evaluation

```bash
python evaluate.py --config-name fno output.plot_dir=../data/models/lid_driven_fno_plots
```

Plot controls:

- `output.plot_max_cases`
- `output.plot_case_indices`
- `output.plot_velocity_x_field`
- `output.plot_velocity_y_field`
- `output.plot_quiver_step`
- `output.plot_cmap`
- `output.plot_dpi`

## Other built-in models

The generic training framework supports multiple model families through the
adapter pattern.  All use the same `train.py` / `evaluate.py` entry points.

| Model | Config name | Adapter | Use case |
|-------|-------------|---------|----------|
| FNO | `fno` | grid | Regular-grid operator learning |
| AFNO | *(custom)* | grid | Adaptive Fourier neural operator |
| Pix2Pix | *(custom)* | grid | Image-to-image translation |
| MeshGraphNet | *(custom)* | graph | Unstructured mesh GNN |
| MLP (FullyConnected) | `alpha_d_mlp` | pointwise | Tabular/axial-profile surrogate |

The MLP model uses the `pointwise` adapter, which reads per-case `.zarr`
stores containing `features/` and `targets/` arrays (tabular data).  See
[Alpha-D Surrogate Tutorial](../user/alpha_d_surrogate.md) for the full
workflow.

## Hyperparameter optimization

All models support Optuna-based hyperparameter optimization via
`train.py`. Add an `hpo` section to the training config that defines a
search space over `training.*` and `model.params.*` paths.

```bash
cd src && python train.py --config-name alpha_d_mlp
```

See [Hyperparameter Optimization Guide](../user/hyperparameter_optimization.md)
for search-space format, study settings, output artifacts, and how to add
HPO for new models.

## Notes

- The legacy wrappers `train_fno.py` / `eval_fno.py` are removed.
- Use `train.py` / `evaluate.py` for all supervised one-step models, including FNO.
