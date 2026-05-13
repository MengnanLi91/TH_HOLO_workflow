# Cases

Each `src/cases/<name>/` folder bundles a single research case: its ETL
pipeline, Hydra configs, datasets, and any case-specific physics or
experiment hooks. The generic training core (`src/training/`) does not
import from `cases/`; coupling flows in one direction only.

```{toctree}
:maxdepth: 1

moose_grid
alpha_d
case_pressure_drop
```
