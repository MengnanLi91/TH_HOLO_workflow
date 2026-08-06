# Building the documentation

The MULTIFID-TH site is built with Sphinx + MyST + Furo. The build is
driven by `docs/Makefile`; the `[docs]` extras-require in
`pyproject.toml` declares the Sphinx stack.

## Quick start

From the repository root:

```bash
uv sync --extra docs
uv run --no-sync make -C docs html
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
uv run --no-sync make -C docs livehtml
# Browse to http://localhost:8000
```

Override the port with `PORT=...`:

```bash
uv run --no-sync make -C docs livehtml PORT=9001
```

## Catch broken cross-references

The `strict` target builds with `-W -n` (warnings fatal, nitpicky
cross-reference checking). Run it before pushing doc changes:

```bash
uv run --no-sync make -C docs strict
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
| `uv run --no-sync make -C docs html` | Default build to `_build/html` |
| `uv run --no-sync make -C docs strict` | `html` plus `-W -n` (warnings fatal, nitpicky) |
| `uv run --no-sync make -C docs livehtml` | Auto-rebuild and serve at `http://localhost:8000` |
| `uv run --no-sync make -C docs serve` | Serve an already-built `_build/html` at `http://localhost:8000` |
| `uv run --no-sync make -C docs linkcheck` | Validate external links |
| `uv run --no-sync make -C docs clean` | Remove `_build/` |

`SPHINXOPTS` is forwarded to `sphinx-build`, so you can pass extra
flags ad hoc:

```bash
uv run --no-sync make -C docs html SPHINXOPTS="-W --keep-going"
```

## Where the source lives

| Path | Purpose |
|---|---|
| `docs/conf.py` | Sphinx configuration (extensions, theme, autodoc options) |
| `docs/index.md` | Site landing page + top-level toctree |
| `docs/user/` | User-facing tutorials and how-tos |
| `docs/dev/` | Developer-oriented docs (this page lives here) |
| `docs/cases/` | One page per surrogate case |
| `docs/demo_cases/` | Reproducible end-to-end demonstrations for specific cases |
| `docs/api/` | Auto-generated API reference (`.. automodule::` blocks) |
| `docs/superpowers/` | Historical/internal planning; **excluded from the rendered site** via `exclude_patterns` in `conf.py` |

`docs/_build/` is git-ignored. Don't commit anything under it.

## Hosted site (GitHub Pages)

The rendered site is auto-deployed from `main` to GitHub Pages at:

<https://mengnanli91.github.io/multifid-th/>

The deploy job lives in `.github/workflows/docs.yml`:

- **Every PR** builds the docs and uploads the HTML as a `docs-preview`
  artifact (downloadable from the PR's Actions tab for 14 days). The PR
  build is a status check — broken docs fail the PR.
- **Every push to `main`** rebuilds and deploys the result to Pages via
  `actions/deploy-pages@v4`. Deploys are queued via the `pages`
  concurrency group so two pushes in quick succession don't race.

The build runs in a stock Python 3.11 GitHub-hosted runner with just
`pip install -e ".[docs]"` — no torch, no physicsnemo. Autodoc imports
work because `docs/conf.py` lists those heavy runtime deps under
`autodoc_mock_imports`. If you add a new module that imports a runtime
dep not yet mocked, the CI build will fail with an `ImportError`; add
the dep to the `autodoc_mock_imports` list in `conf.py`.

The `actions/configure-pages` step prepares an already-enabled Pages site for
deployment. Enable Pages once in the repository settings with GitHub Actions
as its source before the first deployment. No `gh-pages` branch is involved.

If you ever want to confirm or override the Pages source by hand, it's
at: Repo Settings → **Pages** → Build and deployment → Source.
