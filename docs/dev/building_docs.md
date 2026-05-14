# Building the documentation

The MULTIFID-TH site is built with Sphinx + MyST + Furo. The build is
driven by `docs/Makefile`; the `[docs]` extras-require in
`pyproject.toml` pins the Sphinx stack.

## Quick start

From the repository root:

```bash
pip install -e ".[docs]"
make -C docs html
```

Open `docs/_build/html/index.html` in a browser.

If you don't want to install into your active environment, build inside
the project container instead:

```bash
apptainer exec --bind "$PWD:$PWD" --pwd "$PWD" multifid-th-cpu.sif \
  python -m sphinx -b html docs docs/_build/html
```

The `multifid-th-cpu.sif` image already has the full Sphinx stack
installed (see `docker/physicsnemo-cpu.def`), so no extra install step
is needed.

## Live reload while editing

For iterative writing, run `sphinx-autobuild` via the `livehtml`
target. It watches the source tree and rebuilds on every save:

```bash
make -C docs livehtml
# Browse to http://localhost:8000
```

Override the port with `PORT=...`:

```bash
make -C docs livehtml PORT=9001
```

## Catch broken cross-references

The `strict` target builds with `-W -n` (warnings fatal, nitpicky
cross-reference checking). Run it before pushing doc changes:

```bash
make -C docs strict
```

`-W` is the same flag the CI build uses; if `strict` is green, the
build will be green in CI.

`-n` (nitpicky) flags every unresolved `{py:class}` / `{py:mod}` /
similar cross-reference. The project's default build does **not** use
nitpicky mode because some autodoc-driven references reach into
`physicsnemo` / `physicsnemo_curator` (which don't publish intersphinx
inventories) — those would otherwise produce a flood of false
positives. Use `strict` for an occasional sweep, not for every build.

## Other targets

| Target | What it does |
|---|---|
| `make -C docs html` | Default build to `_build/html` |
| `make -C docs strict` | `html` plus `-W -n` (warnings fatal, nitpicky) |
| `make -C docs livehtml` | Auto-rebuild and serve at `http://localhost:8000` |
| `make -C docs serve` | Serve an already-built `_build/html` at `http://localhost:8000` |
| `make -C docs linkcheck` | Validate external links |
| `make -C docs clean` | Remove `_build/` |

`SPHINXOPTS` is forwarded to `sphinx-build`, so you can pass extra
flags ad hoc:

```bash
make -C docs html SPHINXOPTS="-W --keep-going"
```

## Where the source lives

| Path | Purpose |
|---|---|
| `docs/conf.py` | Sphinx configuration (extensions, theme, autodoc options) |
| `docs/index.md` | Site landing page + top-level toctree |
| `docs/user/` | User-facing tutorials and how-tos |
| `docs/dev/` | Developer-oriented docs (this page lives here) |
| `docs/cases/` | One page per surrogate case |
| `docs/api/` | Auto-generated API reference (`.. automodule::` blocks) |
| `docs/archive/` | Historical planning / refactor records |
| `docs/superpowers/` | Internal planning; **excluded from the rendered site** via `exclude_patterns` in `conf.py` |

`docs/_build/` is git-ignored. Don't commit anything under it.
