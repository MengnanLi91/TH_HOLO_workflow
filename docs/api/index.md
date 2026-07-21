# API Reference

Auto-generated documentation for the Python packages that make up
MULTIFID-TH. Signatures, docstrings, and "[source]" links are
extracted directly from the code in `src/`.

The packages mirror the layout described on the
[Architecture](../architecture.md) page:

- **`training`** — the generic training framework (registry, adapters,
  runner, experiment hooks, HPO).
- **`cases.<name>`** — per-case ETL, datasets, physics, and experiments.
- **`feature_selection`** — shared feature-analysis helpers.
- **`dataset`** — the public `MOOSEDataset` consumed by downstream user
  code.
- **`workflows`** — resumable DAG contracts, executors, and manifest runner.

```{toctree}
:maxdepth: 1

training
workflows
cases
feature_selection
dataset
```
