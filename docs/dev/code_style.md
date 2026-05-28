# Code style and pre-commit

The project uses [ruff](https://docs.astral.sh/ruff/) for both linting and
formatting. The full configuration lives in `pyproject.toml` under
`[tool.ruff]`; this page documents the day-to-day workflow.

## One-time setup

Pre-commit and ruff live in a dedicated Conda env so they stay isolated
from the ETL / training / MOOSE envs you already have. (The multifid-th
Apptainer image is also a poor fit — it's read-only at runtime, so
`pip install`s into it don't persist.)

```bash
conda create -n multifid-th-dev python=3.11 -y
conda activate multifid-th-dev
pip install pre-commit ruff==0.14.5
pre-commit install                  # run from the repo root
```

Pin `ruff==0.14.5` to match the version in `.pre-commit-config.yaml` —
the CLI binary and the hook should produce identical output.

`pre-commit install` writes `.git/hooks/pre-commit`, which then runs
the hooks listed in `.pre-commit-config.yaml` on every `git commit` in
that clone. You'll need to re-run it after a fresh `git clone` or a
fresh worktree.

If you'd rather skip pre-commit and run ruff manually, you can — see
[Running ruff by hand](#running-ruff-by-hand) below.

## What runs on commit

Two hooks, both from
[`astral-sh/ruff-pre-commit`](https://github.com/astral-sh/ruff-pre-commit),
pinned in `.pre-commit-config.yaml`:

| Hook | Command | Purpose |
|---|---|---|
| `ruff` | `ruff check --fix` | Lint staged files; auto-fix safe issues (mostly import order + unused imports). |
| `ruff-format` | `ruff format` | Reformat staged files in place. |

When a hook makes changes, the commit is **rejected** and the modified
files are left in your working tree. Re-stage and commit again. This is
intentional — it forces you to see (and accept) the auto-fixes before
they land in history.

To run the hooks against the whole repo on demand:

```bash
pre-commit run --all-files
```

Use this after a config bump or when cleaning up an older branch.

## Running ruff by hand

The same checks the hooks run, available as plain commands from the repo
root:

```bash
# Lint only (CI-style: non-zero exit on any issue, no writes)
ruff check src/ tests/

# Lint + auto-fix safe issues
ruff check --fix src/ tests/

# Format in place
ruff format src/ tests/

# Format check only (CI-style)
ruff format --check src/ tests/
```

A combined "is this commit-clean?" one-liner:

```bash
ruff check src/ tests/ && ruff format --check src/ tests/
```

## Configuration

Everything lives in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
extend-exclude = ["moose", "physicsnemo", "physicsnemo-curator", ...]

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
ignore = ["E501", "E731", "E402"]

[tool.ruff.format]
quote-style = "double"
```

Notable choices:

- **Line length 100** — gives scientific identifiers (`local_velocity_normalization`, `compute_pointwise_extended_metrics`) and four-space-indented context room to breathe. Black's default 88 was a noticeable squeeze in this codebase.
- **`E501` ignored at the lint level** — the formatter (`ruff format`) owns line wrapping; the linter would otherwise flag long comments and docstrings the formatter can't auto-wrap.
- **`E402` ignored globally** — the codebase uses two patterns the rule misreads: `pytest.importorskip(...)` followed by project imports in test files, and conditional `try: import torch` blocks at the top of training modules.
- **Test files have `F401` / `F811` relaxed** — pytest fixtures often look "unused" or "re-defined" to a static linter.

To upgrade ruff, bump **both** the `rev:` in `.pre-commit-config.yaml`
and the version in any developer environment. Mismatched versions can
silently produce different output.

## Why not black + isort + flake8?

Ruff replaces all three with a single binary, one config block, and ~20×
faster execution. The output of `ruff format` is ~99% black-compatible;
the remaining edge cases are documented in the
[ruff/black compatibility page](https://docs.astral.sh/ruff/formatter/black/).
Picking one tool over three is a maintenance choice, not a correctness
one — both produce code that looks the same in 99 of 100 lines.
