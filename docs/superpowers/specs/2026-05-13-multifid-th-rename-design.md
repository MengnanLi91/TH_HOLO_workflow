# Project rename: TH_HOLO_workflow → MULTIFID-TH

**Date:** 2026-05-13
**Branch:** `rename/multifid-th` (to be cut from `refactor/repo-layout` tip `9d9af62`)

## Why rename

`TH_HOLO_workflow` fails three readability tests:

1. `HOLO` is an opaque acronym that is never expanded anywhere in the
   README, docs, or CLAUDE.md. New readers cannot guess what it stands
   for.
2. The mixed style — `ALL_CAPS` plus `snake_case` plus the generic
   `workflow` suffix — is awkward for a Python distribution name.
3. The name does not describe what the project does (multifidelity
   surrogate modeling for thermal-hydraulics).

What `HOLO` originally meant: **HI**gh-fidelity → **LO**w-fidelity via
surrogate models — i.e. use high-fidelity MOOSE CFD simulations to
inform low-fidelity ML surrogates. That concept is worth preserving;
the opaque acronym is not.

## Chosen name

| Surface | Form |
|---|---|
| Canonical display (docs title, README H1, `html_title`, `html_short_title`) | **MULTIFID-TH** |
| PyPI distribution (`pyproject.toml [project] name`) | **`multifid-th`** |
| GitHub repo slug | **`multifid-th`** (full URL: `https://github.com/MengnanLi91/multifid-th`) |
| Filesystem path | **`/data/lim2/projects/multifid-th/`** |
| Docker image tag prefix | **`multifid-th-physicsnemo:{dev,cpu,gpu,ngc}`** |
| Apptainer SIF filenames | **`multifid-th-{gpu,cpu,dev,ngc}.sif`** |
| Tagline (README intro + docs announcement) | **"Multifidelity surrogates for thermal-hydraulics"** |

PyPI availability verified on 2026-05-13: `multifid-th`, `multifid_th`,
`multifidth`, `multifid`, `multifid-thermal`, `multifid-moose` all
return HTTP 404 from `pypi.org/pypi/<name>/json` (free).

GitHub availability verified on 2026-05-13: `MengnanLi91/multifid-th`
returns HTTP 404 from the repos API (free).

## Out of scope

- **No rename of `src/` sub-packages.** The codebase has no top-level
  `th_holo_workflow` Python package today; sub-packages (`training`,
  `cases`, `dataset`, `feature_selection`, `read_exdous`) keep their
  current names. The distribution name on PyPI is just what
  `pip install` resolves; it does not force an import-path change.
- **No version bump.** Stay at `0.1.0`. The project has never been
  published on PyPI, so no downstream consumer has the old name pinned.
- **No deprecation shim.** Nothing to redirect from.
- **No edit to historical commit messages.** Pre-rename commits will
  continue to say "TH_HOLO_workflow"; rewriting history would be worse.
- **No edits to `moose/`, `physicsnemo/`, or `physicsnemo-curator/`
  submodules.** Per the repo's editing convention, submodules are not
  touched as part of a feature change. The "HOLO" hits in those trees
  (porous_flow `tidal.md`, healthcare bloodflow examples) are unrelated.

## Execution plan

The rename lands on a fresh branch `rename/multifid-th` cut from the
current `refactor/repo-layout` tip (`9d9af62`), in five batches.

### Batch 1 — In-repo string sweep (one commit)

Committed files that mention the old name:

```
pyproject.toml
README.md
docs/conf.py
docs/index.md
docs/api/index.md
docs/architecture.md
docs/archive/repo_layout_refactor_plan.md
docs/reference/conv1d_checkpoint_migration.md
docs/user/alpha_d_surrogate.md
docs/user/case_distribution_analysis.md
docs/user/getting_started.md
docs/_static/custom.css
docker-compose.yml
docker/gpu.def
```

Mechanical substitutions:

| Old | New |
|---|---|
| `TH_HOLO_workflow` | `MULTIFID-TH` |
| `TH_HOLO` (standalone) | `MULTIFID-TH` |
| `th-holo-workflow` | `multifid-th` |
| `th-holo-physicsnemo` | `multifid-th-physicsnemo` |
| `th-holo-gpu.sif` (and `cpu` / `dev` / `ngc` variants) | `multifid-th-gpu.sif` (etc.) |
| `MengnanLi91/TH_HOLO_workflow` | `MengnanLi91/multifid-th` |

Plus a one-time addition: insert the tagline *"Multifidelity surrogates
for thermal-hydraulics"* in the README intro line and in the docs
announcement banner (`docs/conf.py` `html_theme_options.announcement`)
so the meaning is finally explicit.

**Care points:**

- `docs/archive/repo_layout_refactor_plan.md` is *historical* documentation
  of the per-case refactor. Only update self-referential prose (e.g.
  "Update `CLAUDE.md` (top-level + `TH_HOLO_workflow/CLAUDE.md`)" becomes
  "(top-level + `multifid-th/CLAUDE.md`)"). Leave any text that
  reproduces old commit messages or shows historical paths alone.
- `docs/_static/custom.css` only has a one-line comment header
  mentioning the old name — change for consistency.

### Batch 2 — Build and verify locally (no commit)

After Batch 1 is staged:

1. Build docs: `apptainer exec --pwd /data/lim2/projects/TH_HOLO_workflow th-holo-gpu.sif python -m sphinx -W --keep-going -b html docs docs/_build/html`
2. Spot-check rendered title, announcement banner, README badge URL.
3. `git grep -nEi 'TH[_ ]?HOLO|th-holo' -- ':!moose' ':!physicsnemo' ':!physicsnemo-curator' ':!docs/archive/repo_layout_refactor_plan.md'`
   should return zero hits.
4. `pytest` — should be a no-op; rename does not touch test files. 110
   passed, 3 deselected (per
   `~/.claude/projects/-data-lim2-projects-TH-HOLO-workflow/memory/apptainer-runtime.md`).

Then commit Batch 1.

### Batch 3 — Rebuild SIF (separate commit)

The repo-root `th-holo-gpu.sif` is a 1.6+ GB Apptainer image. **Rebuild**
(not just rename), per user preference:

```bash
apptainer build multifid-th-gpu.sif docker/gpu.def
rm th-holo-gpu.sif
```

Add `multifid-th-*.sif` to `.gitignore` (likely already there as
`*.sif`; verify). The rebuild takes ~30 minutes but produces a clean
artifact whose internal `%help` messages match the new name (Batch 1
edits the `.def` file already).

Docker images: anyone who already pulled `th-holo-physicsnemo:*` tags
needs to rebuild against the new tags. One line in release notes.

### Batch 4 — External: GitHub repo and filesystem rename

Order matters here. Each step is irreversible without manual cleanup.

1. **Push the Batch 1 + 3 commits to the old GitHub URL** first, so the
   rename happens on top of merged work.
2. **GitHub repo rename:** `gh repo rename multifid-th -R MengnanLi91/TH_HOLO_workflow`.
   GitHub keeps `MengnanLi91/TH_HOLO_workflow/*` URLs as redirects, so
   existing clones continue to fetch via the old URL, but the canonical
   URL changes.
3. **Update local remote:** `git remote set-url origin git@github.com:MengnanLi91/multifid-th.git`.
4. **Stop the running HTTP server on port 8765** (its cwd is inside the
   old path). Background task ID `b4xxc198h`.
5. **Filesystem path rename:** `mv /data/lim2/projects/TH_HOLO_workflow /data/lim2/projects/multifid-th`.
6. **Reload VS Code Remote SSH workspace** at the new path.

### Batch 5 — Memory entries

After Batch 4 lands, the auto-memory directory key changes (it is
derived from the project path). Migration steps:

1. **Rename the memory directory:**
   `mv ~/.claude/projects/-data-lim2-projects-TH-HOLO-workflow ~/.claude/projects/-data-lim2-projects-multifid-th`
2. **Update existing memory bodies** that reference the old path or
   name:
   - `MEMORY.md` (top line: tip after rename commits)
   - `apptainer-runtime.md` (path strings in apptainer commands)
   - `repo-layout-refactor.md` (no string change needed; memory is
     about the per-case refactor, not the project name)
   - `commit-style.md` (no change; memory is about commit style on the
     branch, not the project name)
   - `git-mv-then-edit.md` (no change)
3. **Write a new memory** `project-rename.md` capturing: TH_HOLO_workflow
   was renamed to MULTIFID-TH on 2026-05-13; HOLO originally meant
   HI→LO fidelity via surrogate; the rename lives on branch
   `rename/multifid-th`, with the rename's final SHA recorded once the
   branch is merged. This keeps the genealogy so future sessions can
   answer "what was this called before?".

## Verification gates

| Gate | After | Check |
|---|---|---|
| Strings clean | Batch 1 | `git grep -nEi 'TH[_ ]?HOLO|th-holo' -- ':!moose' ':!physicsnemo' ':!physicsnemo-curator' ':!docs/archive/repo_layout_refactor_plan.md'` returns zero. |
| Docs build green | Batch 2 | `python -m sphinx -W --keep-going -b html docs docs/_build/html` exits 0; rendered `<title>` reads "MULTIFID-TH". |
| Tests still pass | Batch 2 | `pytest` reports 110 passed / 3 deselected. |
| SIF works | Batch 3 | `apptainer exec --pwd $(pwd) multifid-th-gpu.sif python -c 'import physicsnemo, torch; print(torch.__version__)'` succeeds. |
| Remote rewired | Batch 4 | `git remote -v` shows `multifid-th`; `git push` succeeds from the renamed working tree at `/data/lim2/projects/multifid-th/`. |
| Memory accessible | Batch 5 | A fresh Claude session in the new working dir loads memory from `~/.claude/projects/-data-lim2-projects-multifid-th/`. |

## Risks and mitigations

- **Risk:** Other shells/scripts/cron jobs hard-coded
  `/data/lim2/projects/TH_HOLO_workflow/`. **Mitigation:** create a
  one-time symlink `ln -s multifid-th TH_HOLO_workflow` in
  `/data/lim2/projects/` after the rename if needed, but do *not* commit
  this; it's strictly for personal cron-style backstops.
- **Risk:** Collaborators have local clones pointing at the old GitHub
  URL. **Mitigation:** GitHub auto-redirects fetch/push URLs for the
  old repo slug; clones keep working without manual intervention.
- **Risk:** Anyone who built the docker images locally needs to rebuild.
  **Mitigation:** call this out in the rename PR description and the
  README's changelog (if one exists; otherwise add a single CHANGELOG
  entry).
- **Risk:** SIF rebuild fails (network, NGC auth, disk). **Mitigation:**
  fall back to plain `mv th-holo-gpu.sif multifid-th-gpu.sif` and defer
  the rebuild to a follow-up. The rename still completes.

## Acceptance criteria

The rename is done when all of:

1. `git grep -nEi 'TH[_ ]?HOLO|th-holo'` outside the submodule and archive
   exclusions returns zero in the working tree.
2. `pyproject.toml [project] name` is `multifid-th`.
3. Docs build clean with `-W`; rendered title is "MULTIFID-TH" and the
   announcement banner carries the tagline "Multifidelity surrogates
   for thermal-hydraulics".
4. GitHub repo URL resolves at `MengnanLi91/multifid-th`; the local
   `origin` remote uses the new URL.
5. Filesystem working tree is at `/data/lim2/projects/multifid-th/`.
6. `multifid-th-gpu.sif` exists at the repo root and successfully
   imports `physicsnemo` and `torch`.
7. Auto-memory directory is at `~/.claude/projects/-data-lim2-projects-multifid-th/`
   and contains a `project-rename.md` entry pointing back at this spec.
