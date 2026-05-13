# MULTIFID-TH Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the project from `TH_HOLO_workflow` to `MULTIFID-TH` across every committed surface, the GitHub repo, the local filesystem path, the Apptainer SIF artifact, and the auto-memory directory — and add an explicit tagline that finally unpacks the project's purpose ("Multifidelity surrogates for thermal-hydraulics").

**Architecture:** Five logical batches landing as a sequence of small commits on a fresh branch `rename/multifid-th` cut from `refactor/repo-layout` tip. Batches 1-4 are an in-repo string sweep (mechanical substitutions + one tagline insertion) verified by docs build + grep gate + pytest. Batch 5 rebuilds the SIF. Batches 6-7 are the external rename (GitHub repo + filesystem path) and require user action because they break the active session's `pwd`. Batches 8-9 happen in a fresh Claude session at the new path: editable install re-run + memory migration. No version bump, no `src/` sub-package rename, no submodule edits.

**Tech Stack:** Python 3.11 (setuptools src-layout, `pyproject.toml`), Sphinx 9 + Furo + sphinx-design + sphinxcontrib-mermaid for docs, Hydra configs, Docker Compose + Apptainer (`docker/*.def`), `gh` CLI for GitHub repo ops, pytest.

**Spec:** [`docs/superpowers/specs/2026-05-13-multifid-th-rename-design.md`](../specs/2026-05-13-multifid-th-rename-design.md)

---

## File structure

This is purely a rename — no new modules, no new files except a memory entry. Touchpoints (committed files):

| File | What changes |
|---|---|
| `pyproject.toml` | `[project] name` (line 6); `description` (line 8, optional tagline-style update). |
| `README.md` | Title (line 1), badge URLs (line 3), prose intro (line 5), section header (line 8), `## Quick Start` SIF filenames if any. |
| `docs/conf.py` | Module docstring (line 1), `project` / `author` / `copyright` (lines 20–22), `html_title` (132), `html_short_title` (133), `announcement` (149–151), `source_repository` (155). |
| `docs/index.md` | H1 (line 1), prose intro (line 3), `th-holo-cpu.sif` in apptainer example (lines 110, 112). |
| `docs/api/index.md` | Prose mention (line 4). |
| `docs/architecture.md` | Prose mention (line 3). |
| `docs/reference/conv1d_checkpoint_migration.md` | Apptainer `--pwd` path + SIF filename (lines 19–20). |
| `docs/user/alpha_d_surrogate.md` | `th-holo-{gpu,cpu}.sif` in code blocks (lines 352, 361, 365, 369, 373, 377, 379). |
| `docs/user/case_distribution_analysis.md` | `th-holo-gpu.sif` (line 56). |
| `docs/user/getting_started.md` | `apptainer build th-holo-{dev,cpu,gpu,ngc}.sif` + several `apptainer exec` calls (lines 86, 89, 92, 95, 106, 111, 123, 129, 136, 151). |
| `docs/_static/custom.css` | One-line comment header (line 1). |
| `docker-compose.yml` | Image tags `th-holo-physicsnemo:{dev,cpu,gpu,ngc}` (lines 16, 40, 66, 99). |
| `docker/gpu.def` | `%help` block mentioning the SIF filename (lines 15, 18, 19, 22, 23, 26). |

Not touched (deliberately):

- `docs/archive/repo_layout_refactor_plan.md` — frozen historical record; rewriting falsifies what the plan said at the time.
- `moose/`, `physicsnemo/`, `physicsnemo-curator/` submodules — per the repo's editing convention. The `HOLO` hits inside them are unrelated upstream code (porous-flow tidal docs, healthcare bloodflow examples).
- Any pre-rename commit messages — the git log will continue to say `TH_HOLO_workflow` for commits prior to the rename. Correct behavior; rewriting history would be worse.

New file (post-rename, in the renamed memory directory):

- `~/.claude/projects/-data-lim2-projects-multifid-th/memory/project-rename.md` — captures the genealogy so a future session can answer "what was this called before?".

---

## Task 0: Branch prerequisites

**Files:**
- Modify: `docs/index.md` (commit the leftover "Where do I start?" reorder before cutting the rename branch)

- [ ] **Step 1: Confirm working-tree state**

  Run:
  ```bash
  git status --short
  ```

  Expected output (the `?` line for `moose` is the submodule's working-tree drift and is fine):
  ```
   M docs/index.md
   ? moose
  ```

  If anything else is modified, stop and reconcile before proceeding.

- [ ] **Step 2: Inspect the dangling index.md change**

  Run:
  ```bash
  git diff --stat docs/index.md
  git diff docs/index.md | head -40
  ```

  Expected: the diff moves the "Where do I start?" `::::{grid}` block from after "Train the Alpha-D MLP surrogate" to before "Quick start". No other content changes.

- [ ] **Step 3: Commit the docs reorder as a standalone change**

  Run:
  ```bash
  git add docs/index.md
  git commit -m "Move \"Where do I start?\" landing section above Quick start"
  ```

  Per [[commit-style]] memory: terse single-line imperative, no body, no Co-Authored-By footer.

- [ ] **Step 4: Cut the rename branch**

  Run:
  ```bash
  git checkout -b rename/multifid-th
  git log --oneline -3
  ```

  Expected: the new commit from Step 3 is HEAD, followed by `f26f5d8 Add explicit pip install -e . step to the rename spec` and `cbbefb2 Add design doc for MULTIFID-TH rename`.

---

## Task 1: Rename in-repo strings — pyproject.toml + README.md + docs/conf.py

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `docs/conf.py`

These three files carry the most user-visible identity. Group them in one commit so the rename's headline change is atomic.

- [ ] **Step 1: Update `pyproject.toml`**

  Replace at line 6:
  ```toml
  name = "th-holo-workflow"
  ```
  with:
  ```toml
  name = "multifid-th"
  ```

  Replace at line 8:
  ```toml
  description = "Thermal-hydraulic surrogate workflow on MOOSE outputs"
  ```
  with:
  ```toml
  description = "Multifidelity surrogates for thermal-hydraulics on MOOSE outputs"
  ```

- [ ] **Step 2: Update `README.md`**

  Replace at line 1:
  ```markdown
  # TH_HOLO_workflow
  ```
  with:
  ```markdown
  # MULTIFID-TH
  ```

  Replace at line 3 (the pytest badge — two URL occurrences on the same line):
  ```markdown
  [![pytest](https://github.com/MengnanLi91/TH_HOLO_workflow/actions/workflows/pytest.yml/badge.svg)](https://github.com/MengnanLi91/TH_HOLO_workflow/actions/workflows/pytest.yml)
  ```
  with:
  ```markdown
  [![pytest](https://github.com/MengnanLi91/multifid-th/actions/workflows/pytest.yml/badge.svg)](https://github.com/MengnanLi91/multifid-th/actions/workflows/pytest.yml)
  ```

  Replace at line 5:
  ```markdown
  TH_HOLO_workflow is a PhysicsNeMo-based ETL pipeline that converts MOOSE
  ```
  with:
  ```markdown
  MULTIFID-TH is a PhysicsNeMo-based ETL pipeline that converts MOOSE
  ```

  Replace at line 8:
  ```markdown
  ## TH_HOLO_workflow Plot
  ```
  with:
  ```markdown
  ## Pipeline overview
  ```

  (Renaming the section makes the heading describe the diagram instead of repeating the project name.)

- [ ] **Step 3: Update `docs/conf.py`**

  Replace at line 1:
  ```python
  """Sphinx configuration for the TH_HOLO_workflow documentation site."""
  ```
  with:
  ```python
  """Sphinx configuration for the MULTIFID-TH documentation site."""
  ```

  Replace at lines 20–22:
  ```python
  project = "TH_HOLO_workflow"
  author = "TH_HOLO_workflow contributors"
  copyright = "2026, TH_HOLO_workflow contributors"  # noqa: A001
  ```
  with:
  ```python
  project = "MULTIFID-TH"
  author = "MULTIFID-TH contributors"
  copyright = "2026, MULTIFID-TH contributors"  # noqa: A001
  ```

  Replace at lines 132–133:
  ```python
  html_title = "TH_HOLO_workflow"
  html_short_title = "TH_HOLO"
  ```
  with:
  ```python
  html_title = "MULTIFID-TH"
  html_short_title = "MULTIFID-TH"
  ```

  Replace at line 155:
  ```python
      "source_repository": "https://github.com/MengnanLi91/TH_HOLO_workflow/",
  ```
  with:
  ```python
      "source_repository": "https://github.com/MengnanLi91/multifid-th/",
  ```

  (Leave lines 148–151 — the `announcement` text — for Task 5, which rewrites it to incorporate the tagline.)

- [ ] **Step 4: Verify the substitutions landed and nothing else broke**

  Run:
  ```bash
  grep -n "TH_HOLO\|th-holo" pyproject.toml README.md docs/conf.py
  ```

  Expected: only the announcement line 149 in `docs/conf.py` (`"TH_HOLO_workflow: PhysicsNeMo-based ETL and surrogate-modeling "`) remains — it's intentionally deferred to Task 5. Everything else should be clean.

- [ ] **Step 5: Commit**

  Run:
  ```bash
  git add pyproject.toml README.md docs/conf.py
  git commit -m "Rename headline project identifiers to MULTIFID-TH"
  ```

---

## Task 2: Rename in-repo strings — docs prose pages

**Files:**
- Modify: `docs/index.md`
- Modify: `docs/api/index.md`
- Modify: `docs/architecture.md`
- Modify: `docs/reference/conv1d_checkpoint_migration.md`

- [ ] **Step 1: Update `docs/index.md`**

  Replace at line 1:
  ```markdown
  # TH_HOLO_workflow
  ```
  with:
  ```markdown
  # MULTIFID-TH
  ```

  Replace at line 3 (only the project-name mention; the rest of the sentence is unchanged):
  ```markdown
  TH_HOLO_workflow is a PhysicsNeMo-based pipeline that turns
  ```
  with:
  ```markdown
  MULTIFID-TH is a PhysicsNeMo-based pipeline that turns
  ```

  Replace at line 110 (inside the apptainer-build code block):
  ```markdown
  apptainer build th-holo-cpu.sif docker/physicsnemo-cpu.def
  ```
  with:
  ```markdown
  apptainer build multifid-th-cpu.sif docker/physicsnemo-cpu.def
  ```

  Replace at line 112 (same code block, the `apptainer exec` invocation):
  ```markdown
    th-holo-cpu.sif bash -lc 'cd /path/to/project/src && python run_etl.py'
  ```
  with:
  ```markdown
    multifid-th-cpu.sif bash -lc 'cd /path/to/project/src && python run_etl.py'
  ```

- [ ] **Step 2: Update `docs/api/index.md`**

  Replace at line 4:
  ```
  TH_HOLO_workflow. Signatures, docstrings, and "[source]" links are
  ```
  with:
  ```
  MULTIFID-TH. Signatures, docstrings, and "[source]" links are
  ```

- [ ] **Step 3: Update `docs/architecture.md`**

  Replace at line 3:
  ```
  TH_HOLO_workflow has two complementary halves: a pair of **ETL
  ```
  with:
  ```
  MULTIFID-TH has two complementary halves: a pair of **ETL
  ```

- [ ] **Step 4: Update `docs/reference/conv1d_checkpoint_migration.md`**

  Replace at lines 19–20 (the absolute path inside an `apptainer exec --pwd` invocation):
  ```
  apptainer exec --pwd /data/lim2/projects/TH_HOLO_workflow \
    /data/lim2/projects/TH_HOLO_workflow/th-holo-gpu.sif \
  ```
  with:
  ```
  apptainer exec --pwd /data/lim2/projects/multifid-th \
    /data/lim2/projects/multifid-th/multifid-th-gpu.sif \
  ```

- [ ] **Step 5: Verify and commit**

  Run:
  ```bash
  grep -n "TH_HOLO\|th-holo" docs/index.md docs/api/index.md docs/architecture.md docs/reference/conv1d_checkpoint_migration.md
  ```

  Expected: no hits.

  ```bash
  git add docs/index.md docs/api/index.md docs/architecture.md docs/reference/conv1d_checkpoint_migration.md
  git commit -m "Rename project mentions in docs prose pages"
  ```

---

## Task 3: Rename SIF filenames in user-facing docs

**Files:**
- Modify: `docs/user/alpha_d_surrogate.md`
- Modify: `docs/user/case_distribution_analysis.md`
- Modify: `docs/user/getting_started.md`

These three files only reference `th-holo-*.sif` filenames inside code blocks; no prose mentions of `TH_HOLO_workflow`. A single `sed -i` per file would be reliable here, but using `Edit` with `replace_all=true` keeps the workflow uniform.

- [ ] **Step 1: Update `docs/user/alpha_d_surrogate.md`**

  Replace **every** occurrence of `th-holo-gpu.sif` with `multifid-th-gpu.sif` (7 hits across lines 352, 361, 365, 369, 373, 377). Replace **every** occurrence of `th-holo-cpu.sif` with `multifid-th-cpu.sif` (1 hit at line 379).

  Both substitutions are textual whole-token replacements; use `replace_all=true` per token.

- [ ] **Step 2: Update `docs/user/case_distribution_analysis.md`**

  Replace at line 56:
  ```
  apptainer exec th-holo-gpu.sif bash -c \
  ```
  with:
  ```
  apptainer exec multifid-th-gpu.sif bash -c \
  ```

- [ ] **Step 3: Update `docs/user/getting_started.md`**

  Replace **every** occurrence of:
  - `th-holo-dev.sif` → `multifid-th-dev.sif`
  - `th-holo-cpu.sif` → `multifid-th-cpu.sif`
  - `th-holo-gpu.sif` → `multifid-th-gpu.sif`
  - `th-holo-ngc.sif` → `multifid-th-ngc.sif`

  All four substitutions are whole-token; use `replace_all=true` per token.

- [ ] **Step 4: Verify and commit**

  Run:
  ```bash
  grep -n "TH_HOLO\|th-holo" docs/user/*.md
  ```

  Expected: no hits.

  ```bash
  git add docs/user/alpha_d_surrogate.md docs/user/case_distribution_analysis.md docs/user/getting_started.md
  git commit -m "Rename SIF filename references in user docs"
  ```

---

## Task 4: Rename Docker / Apptainer artifact names

**Files:**
- Modify: `docker-compose.yml`
- Modify: `docker/gpu.def`
- Modify: `docs/_static/custom.css`

- [ ] **Step 1: Update `docker-compose.yml`**

  Replace **every** occurrence of `th-holo-physicsnemo` with `multifid-th-physicsnemo` (4 hits at lines 16, 40, 66, 99). Use `replace_all=true`.

- [ ] **Step 2: Update `docker/gpu.def`**

  Replace **every** occurrence of `th-holo-gpu.sif` with `multifid-th-gpu.sif` (6 hits at lines 15, 18, 19, 22, 23, 26 — all inside the `%help` block). Use `replace_all=true`.

- [ ] **Step 3: Update `docs/_static/custom.css`**

  Replace at line 1:
  ```css
  /* TH_HOLO_workflow docs polish. Light-touch overrides on top of Furo. */
  ```
  with:
  ```css
  /* MULTIFID-TH docs polish. Light-touch overrides on top of Furo. */
  ```

- [ ] **Step 4: Verify and commit**

  Run:
  ```bash
  grep -n "TH_HOLO\|th-holo" docker-compose.yml docker/gpu.def docs/_static/custom.css
  ```

  Expected: no hits.

  ```bash
  git add docker-compose.yml docker/gpu.def docs/_static/custom.css
  git commit -m "Rename Docker image tags, SIF filenames, and CSS header"
  ```

---

## Task 5: Add the multifidelity tagline

**Files:**
- Modify: `README.md`
- Modify: `docs/conf.py`

Up to this point the substitutions were purely mechanical. This task adds *new* prose that finally makes HOLO's original meaning explicit, on the most-visible surfaces (README intro + docs banner).

- [ ] **Step 1: Update README.md intro**

  Replace at line 5–6:
  ```markdown
  MULTIFID-TH is a PhysicsNeMo-based ETL pipeline that converts MOOSE
  thermal-hydraulics outputs (Exodus + CSV probes) into ML-ready Zarr datasets.
  ```
  with:
  ```markdown
  MULTIFID-TH — **multifidelity surrogates for thermal-hydraulics** — is a
  PhysicsNeMo-based ETL pipeline that converts MOOSE thermal-hydraulics
  outputs (Exodus + CSV probes) into ML-ready Zarr datasets and trains
  low-fidelity ML surrogates against the high-fidelity simulations.
  ```

  The em-dash apposition unpacks the acronym on first mention; the
  trailing clause makes the "high-fidelity → low-fidelity via surrogate"
  workflow explicit.

- [ ] **Step 2: Update the Sphinx announcement banner**

  Replace at lines 148–151 of `docs/conf.py`:
  ```python
  html_theme_options = {
      "announcement": (
          "TH_HOLO_workflow: PhysicsNeMo-based ETL and surrogate-modeling "
          "pipeline for MOOSE thermal-hydraulics simulations."
      ),
  ```
  with:
  ```python
  html_theme_options = {
      "announcement": (
          "MULTIFID-TH — multifidelity surrogates for thermal-hydraulics. "
          "High-fidelity MOOSE simulations inform low-fidelity ML "
          "surrogates through a single generic training core."
      ),
  ```

- [ ] **Step 3: Verify the announcement is the last remaining hit**

  Run:
  ```bash
  git grep -nEi 'TH[_ ]?HOLO|th-holo' -- ':!moose' ':!physicsnemo' ':!physicsnemo-curator' ':!docs/archive/repo_layout_refactor_plan.md' ':!docs/superpowers/specs/' ':!docs/superpowers/plans/'
  ```

  Expected: **zero hits**. The spec and this plan file are excluded because they contain the rename history.

- [ ] **Step 4: Commit**

  ```bash
  git add README.md docs/conf.py
  git commit -m "Add multifidelity tagline to README intro and Sphinx banner"
  ```

---

## Task 6: Verify the in-repo rename

**Files:** none modified — this task is verification only.

- [ ] **Step 1: Grep gate — no `TH_HOLO` or `th-holo` outside the deliberate exclusions**

  Run:
  ```bash
  git grep -nEi 'TH[_ ]?HOLO|th-holo' -- ':!moose' ':!physicsnemo' ':!physicsnemo-curator' ':!docs/archive/repo_layout_refactor_plan.md' ':!docs/superpowers/specs/' ':!docs/superpowers/plans/'
  ```

  Expected: zero hits. If anything appears, stop and trace it back to which task missed it.

- [ ] **Step 2: Docs build with warnings-as-errors**

  Run:
  ```bash
  rm -rf docs/_build && apptainer exec --pwd /data/lim2/projects/TH_HOLO_workflow th-holo-gpu.sif python -m sphinx -W --keep-going -b html docs docs/_build/html 2>&1 | tail -10
  ```

  Expected: ends with `build succeeded.`. The apptainer `--pwd` and SIF filename in this command are still the *old* values — the host filesystem path and SIF file haven't been renamed yet (that happens in Tasks 7 + 9). This command must remain on the old paths until those tasks complete.

- [ ] **Step 3: Visually verify the rendered title and banner**

  Run:
  ```bash
  grep -o '<title>[^<]*</title>' docs/_build/html/index.html
  grep -o 'class="announcement[^"]*"[^<]*<[^>]*>[^<]*' docs/_build/html/index.html | head -3
  ```

  Expected: title contains `MULTIFID-TH`; announcement contains the multifidelity tagline.

- [ ] **Step 4: Run pytest — should be a no-op for a rename**

  Run:
  ```bash
  apptainer exec --pwd /data/lim2/projects/TH_HOLO_workflow th-holo-gpu.sif python -m pytest -q --deselect tests/training/test_hpo.py::TestObjective::test_make_objective_returns_float --deselect tests/case_pressure_drop/test_workflow.py::test_training_and_evaluation_smoke --deselect tests/case_pressure_drop/test_workflow.py::test_feature_selection_uses_only_train_cases 2>&1 | tail -5
  ```

  Expected: `110 passed, 3 deselected`. (The three deselected are pre-existing env-drift failures on this host — see [[apptainer-runtime]] memory.)

  No commit on this task — verification only.

---

## Task 7: Rebuild the Apptainer SIF

**Files:**
- Build artifact: `multifid-th-gpu.sif` (new) at repo root
- Removed: `th-holo-gpu.sif` (old) at repo root
- Possibly modify: `.gitignore` if SIF filenames aren't already pattern-matched

The Batch 1–5 commits already updated `docker/gpu.def`'s `%help` block to reference the new SIF filename, so the rebuild bakes the correct help text in.

- [ ] **Step 1: Confirm `.gitignore` excludes both old and new SIF names**

  Run:
  ```bash
  grep -nE '\.sif|th-holo|multifid' .gitignore
  ```

  Expected: a pattern like `*.sif` or two explicit entries. If `th-holo-gpu.sif` is the only SIF entry, add `multifid-th-*.sif`.

  If a change is needed, edit `.gitignore` and stage it (commit happens in Step 4 below).

- [ ] **Step 2: Rebuild the SIF**

  Run:
  ```bash
  apptainer build multifid-th-gpu.sif docker/gpu.def
  ```

  Expected: completes in ~20–30 minutes. New file `multifid-th-gpu.sif` appears at repo root.

- [ ] **Step 3: Verify the new SIF works**

  Run:
  ```bash
  apptainer exec --pwd /data/lim2/projects/TH_HOLO_workflow multifid-th-gpu.sif python -c 'import physicsnemo, torch; print(torch.__version__, physicsnemo.__name__)'
  ```

  Expected: prints a torch version + `physicsnemo`, no `ImportError`.

  Also verify `%help` text:
  ```bash
  apptainer run-help multifid-th-gpu.sif | head -30
  ```

  Expected: lines reference `multifid-th-gpu.sif`, not `th-holo-gpu.sif`.

- [ ] **Step 4: Remove the old SIF and commit any .gitignore tweak**

  Run:
  ```bash
  rm th-holo-gpu.sif
  ls -la *.sif
  ```

  Expected: only `multifid-th-gpu.sif` (and any other pre-existing SIFs like `th-holo-ngc.sif` if present — leave those for a future rename pass since they're not in scope for this branch).

  If `.gitignore` was modified in Step 1:
  ```bash
  git add .gitignore
  git commit -m "Allow multifid-th-*.sif build artifacts"
  ```

---

## Task 8: Push the rename branch to the old GitHub URL

**Files:** none modified.

The rename commits (Tasks 0–5 + optional Task 7) need to land on GitHub before the repo rename. Push to the *old* URL — GitHub will auto-redirect after the rename in Task 9.

- [ ] **Step 1: Confirm the remote is the old URL**

  Run:
  ```bash
  git remote -v
  ```

  Expected: `origin` points at `git@github.com:MengnanLi91/TH_HOLO_workflow.git` (or `https://github.com/MengnanLi91/TH_HOLO_workflow.git`).

- [ ] **Step 2: Push the branch**

  Run:
  ```bash
  git push -u origin rename/multifid-th
  ```

  Expected: success; remote prints a URL to open a PR.

  At this point you can optionally open a PR on GitHub against `main` (or `refactor/repo-layout`, depending on the user's merge strategy). The PR review can happen in parallel with Tasks 9–11.

---

## Task 9: GitHub repo rename + local remote update

**Files:**
- Modify: `.git/config` (via `git remote set-url`)

- [ ] **Step 1: Rename the repo on GitHub**

  Run:
  ```bash
  gh repo rename multifid-th -R MengnanLi91/TH_HOLO_workflow
  ```

  Expected: confirmation that the repo is now at `MengnanLi91/multifid-th`. The old URL keeps redirecting fetches/pushes.

- [ ] **Step 2: Update the local remote**

  Run:
  ```bash
  git remote set-url origin git@github.com:MengnanLi91/multifid-th.git
  git remote -v
  ```

  Expected: `origin` now shows `git@github.com:MengnanLi91/multifid-th.git`.

- [ ] **Step 3: Verify a push still works under the new URL**

  Run:
  ```bash
  git push --dry-run origin rename/multifid-th
  ```

  Expected: `Everything up-to-date` (since Task 8 already pushed).

---

## Task 10 (USER ACTION): Filesystem path + auto-memory dir rename

> **This task cannot be executed by Claude in the current session.**
> The `mv` of `/data/lim2/projects/TH_HOLO_workflow` invalidates this
> session's cwd. The user runs these commands from a shell **outside**
> the project directory (e.g., a fresh terminal in `~`).

- [ ] **Step 1: Stop in the current Claude session and the running HTTP server**

  Background task `b4xxc198h` (Python `http.server` on port 8765) has its cwd inside the old path. It would survive the `mv` at the inode level (see spec's Risks section), but cleanest is to stop it first. Have Claude run:
  ```bash
  # Inside the Claude session, before the user runs the mv:
  kill $(pgrep -f 'http.server 8765')
  ```

  Or let it ride — it'll keep serving the old _build/html inode.

- [ ] **Step 2 (USER, outside the project dir): Rename the filesystem path**

  From a fresh terminal at `~`:
  ```bash
  cd ~
  mv /data/lim2/projects/TH_HOLO_workflow /data/lim2/projects/multifid-th
  ls -ld /data/lim2/projects/multifid-th
  ```

  Expected: directory exists at the new path.

- [ ] **Step 3 (USER): Rename the auto-memory directory**

  ```bash
  mv ~/.claude/projects/-data-lim2-projects-TH-HOLO-workflow ~/.claude/projects/-data-lim2-projects-multifid-th
  ls -ld ~/.claude/projects/-data-lim2-projects-multifid-th
  ```

  Expected: directory exists at the new path. The memory file contents are untouched — only the directory key changes (Claude's memory system derives the key from the project path).

- [ ] **Step 4 (USER): Reload the VS Code Remote SSH workspace**

  *File → Open Folder* → `/data/lim2/projects/multifid-th/`, or use `code -r /data/lim2/projects/multifid-th` from a remote terminal.

- [ ] **Step 5 (USER): Start a fresh Claude session**

  In the new VS Code workspace, start Claude. The system-reminder block at session start should now show:
  ```
  Primary working directory: /data/lim2/projects/multifid-th
  ```
  …and the auto-memory section should load entries from the renamed directory.

  From here on, Tasks 11 and 12 happen in the **new** Claude session.

---

## Task 11 (new session): Re-run the editable install at the new path

**Files:** none modified — only system state (Python's editable-install index).

- [ ] **Step 1: Verify the new path**

  Run:
  ```bash
  pwd
  git rev-parse --abbrev-ref HEAD
  ```

  Expected: `/data/lim2/projects/multifid-th` and `rename/multifid-th`.

- [ ] **Step 2: Confirm the existing editable install is broken**

  Run:
  ```bash
  apptainer exec --pwd /data/lim2/projects/multifid-th multifid-th-gpu.sif python -c 'import training; print(training.__file__)'
  ```

  Expected: either a `ModuleNotFoundError`, or the import succeeds but `training.__file__` shows a path under `/data/lim2/projects/TH_HOLO_workflow/src/training/` (the old path, no longer existing). Either way means the editable-install pointer is stale.

- [ ] **Step 3: Re-run the editable install**

  Run:
  ```bash
  apptainer exec --pwd /data/lim2/projects/multifid-th multifid-th-gpu.sif \
    pip install --user --no-deps -e .
  ```

  Expected: completes successfully; mentions writing to `~/.local/lib/python3.11/site-packages/`.

- [ ] **Step 4: Verify imports resolve to the new path**

  Run:
  ```bash
  apptainer exec --pwd /data/lim2/projects/multifid-th multifid-th-gpu.sif \
    python -c 'import training, cases, dataset; print(training.__file__)'
  ```

  Expected: prints a path starting with `/data/lim2/projects/multifid-th/src/training/`.

- [ ] **Step 5: Run pytest to confirm the install works end-to-end**

  Run:
  ```bash
  apptainer exec --pwd /data/lim2/projects/multifid-th multifid-th-gpu.sif \
    python -m pytest -q --deselect tests/training/test_hpo.py::TestObjective::test_make_objective_returns_float --deselect tests/case_pressure_drop/test_workflow.py::test_training_and_evaluation_smoke --deselect tests/case_pressure_drop/test_workflow.py::test_feature_selection_uses_only_train_cases 2>&1 | tail -5
  ```

  Expected: `110 passed, 3 deselected`.

  No commit on this task — only system state changes.

---

## Task 12 (new session): Memory migration

**Files:**
- Create: `~/.claude/projects/-data-lim2-projects-multifid-th/memory/project-rename.md`
- Modify: `~/.claude/projects/-data-lim2-projects-multifid-th/memory/MEMORY.md`
- Modify: `~/.claude/projects/-data-lim2-projects-multifid-th/memory/apptainer-runtime.md`

The auto-memory directory rename in Task 10 carried the file *contents* across, so existing memory bodies that mention path strings need updating. New session can read the renamed dir directly.

- [ ] **Step 1: Write the new `project-rename.md` memory entry**

  Create `~/.claude/projects/-data-lim2-projects-multifid-th/memory/project-rename.md` with the following content:

  ```markdown
  ---
  name: project-rename
  description: Project was renamed from TH_HOLO_workflow to MULTIFID-TH on 2026-05-13. HOLO originally stood for HI-fidelity → LO-fidelity via surrogate models — i.e. multifidelity surrogate modeling for thermal-hydraulics. The rename landed on branch rename/multifid-th merged into refactor/repo-layout.
  metadata:
    type: project
  ---

  TH_HOLO_workflow was renamed to MULTIFID-TH on 2026-05-13. The genealogy:

  - **What HOLO meant:** HI-fidelity → LO-fidelity via surrogate models —
    use high-fidelity MOOSE CFD to train low-fidelity ML surrogates. The
    acronym was never expanded anywhere in the docs, which is partly
    why the rename happened.
  - **Why renamed:** three pain points — opaque acronym, ugly style
    (caps + underscore + generic "workflow" suffix), and the name
    didn't describe what the project does. See spec at
    `docs/superpowers/specs/2026-05-13-multifid-th-rename-design.md`
    and plan at `docs/superpowers/plans/2026-05-13-multifid-th-rename.md`.
  - **New surfaces:** display `MULTIFID-TH` (caps); PyPI / GitHub slug /
    repo path / Docker tags / SIF filenames all `multifid-th` (lowercase
    kebab-case); tagline "Multifidelity surrogates for thermal-hydraulics"
    in README intro + Sphinx announcement banner.
  - **What did NOT change:** `src/` sub-packages (`training`, `cases`,
    `dataset`, `feature_selection`, `read_exdous`) kept their current
    names; `[project] version` stayed at `0.1.0`; submodules
    (`moose/`, `physicsnemo/`, `physicsnemo-curator/`) untouched.
  - **External moves done:** GitHub repo renamed to
    `MengnanLi91/multifid-th` (old URL auto-redirects); filesystem path
    moved to `/data/lim2/projects/multifid-th/`; Apptainer SIF rebuilt
    as `multifid-th-gpu.sif`.

  **How to apply:** when a user mentions "TH_HOLO_workflow" in future
  sessions, treat it as the historical name for this project. The
  current name is **MULTIFID-TH**. Commit messages prior to 2026-05-13
  use the old name in their text; that's intentional (no history
  rewrite).
  ```

- [ ] **Step 2: Add a pointer to `MEMORY.md`**

  Read `~/.claude/projects/-data-lim2-projects-multifid-th/memory/MEMORY.md`. Insert a new line at the top of the bullet list:

  ```markdown
  - [Project rename TH_HOLO_workflow → MULTIFID-TH](project-rename.md) — 2026-05-13; HOLO meant HI→LO fidelity; new name unpacks the concept; old name preserved in commit messages
  ```

  Also update the existing `repo-layout-refactor` entry's hook to reflect the merge into the rename branch (replace `9d9af62` with the current `rename/multifid-th` HEAD SHA if appropriate, otherwise leave alone).

- [ ] **Step 3: Update `apptainer-runtime.md` path references**

  Read `~/.claude/projects/-data-lim2-projects-multifid-th/memory/apptainer-runtime.md`. Replace **every** occurrence of:
  - `TH_HOLO_workflow` → `multifid-th` (in path strings)
  - `th-holo-gpu.sif` → `multifid-th-gpu.sif`

  Use `replace_all=true` per token. Specifically the line:

  ```
  - `apptainer exec --pwd /data/lim2/projects/TH_HOLO_workflow th-holo-gpu.sif python -m pytest -q` runs the suite.
  ```

  becomes:

  ```
  - `apptainer exec --pwd /data/lim2/projects/multifid-th multifid-th-gpu.sif python -m pytest -q` runs the suite.
  ```

  …and similar updates throughout the file.

- [ ] **Step 4: Spot-check the other memory files**

  Read each of:
  - `~/.claude/projects/-data-lim2-projects-multifid-th/memory/repo-layout-refactor.md`
  - `~/.claude/projects/-data-lim2-projects-multifid-th/memory/commit-style.md`
  - `~/.claude/projects/-data-lim2-projects-multifid-th/memory/git-mv-then-edit.md`

  For each, decide:
  - If the file mentions `TH_HOLO_workflow` or `th-holo-*.sif` in a way that's now stale, update it.
  - If the file describes a historical event (e.g. the per-case refactor that happened *on* the TH_HOLO_workflow branch), leave the historical name alone — it was the name at that time.

  In practice: `commit-style.md` and `git-mv-then-edit.md` are about general patterns and likely need no edits. `repo-layout-refactor.md` is historical and probably needs no edits either, but its body line `tip 9e25bb7` is already stale and was updated in a prior session — re-verify it matches the post-rename branch tip.

- [ ] **Step 5: Verify**

  Run:
  ```bash
  ls ~/.claude/projects/-data-lim2-projects-multifid-th/memory/
  cat ~/.claude/projects/-data-lim2-projects-multifid-th/memory/MEMORY.md
  ```

  Expected: `project-rename.md` exists; `MEMORY.md` lists it as the first entry.

  No git commit on this task — memory files live outside the repo.

---

## Task 13: Final acceptance check

**Files:** none modified — verification only.

Run each acceptance criterion from the spec and confirm.

- [ ] **Step 1: Grep clean (acceptance #1)**

  ```bash
  cd /data/lim2/projects/multifid-th
  git grep -nEi 'TH[_ ]?HOLO|th-holo' -- ':!moose' ':!physicsnemo' ':!physicsnemo-curator' ':!docs/archive/repo_layout_refactor_plan.md' ':!docs/superpowers/specs/' ':!docs/superpowers/plans/'
  ```

  Expected: zero hits.

- [ ] **Step 2: pyproject.toml name (acceptance #2)**

  ```bash
  grep '^name' pyproject.toml
  ```

  Expected: `name = "multifid-th"`.

- [ ] **Step 3: Docs build + title (acceptance #3)**

  ```bash
  rm -rf docs/_build && apptainer exec --pwd /data/lim2/projects/multifid-th multifid-th-gpu.sif python -m sphinx -W --keep-going -b html docs docs/_build/html 2>&1 | tail -5
  grep -o '<title>[^<]*</title>' docs/_build/html/index.html
  grep -o 'announcement[^<]*<[^>]*>[^<]*Multifidelity[^<]*' docs/_build/html/index.html | head -1
  ```

  Expected: `build succeeded.`; title contains `MULTIFID-TH`; announcement contains `Multifidelity`.

- [ ] **Step 4: GitHub URL (acceptance #4)**

  ```bash
  git remote -v
  curl -s -o /dev/null -w '%{http_code}\n' https://github.com/MengnanLi91/multifid-th
  ```

  Expected: remote shows `multifid-th`; HTTP 200 on the new repo URL.

- [ ] **Step 5: Filesystem path (acceptance #5)**

  ```bash
  pwd
  ```

  Expected: `/data/lim2/projects/multifid-th`.

- [ ] **Step 6: SIF works (acceptance #6)**

  ```bash
  apptainer exec --pwd /data/lim2/projects/multifid-th multifid-th-gpu.sif python -c 'import physicsnemo, torch; print(torch.__version__, physicsnemo.__name__)'
  ```

  Expected: prints versions, no import error.

- [ ] **Step 7: Editable install (acceptance #7)**

  ```bash
  apptainer exec --pwd /data/lim2/projects/multifid-th multifid-th-gpu.sif python -c 'import training; print(training.__file__)'
  ```

  Expected: path starts with `/data/lim2/projects/multifid-th/src/training/`.

- [ ] **Step 8: Auto-memory (acceptance #8)**

  ```bash
  ls ~/.claude/projects/-data-lim2-projects-multifid-th/memory/project-rename.md
  ```

  Expected: file exists.

- [ ] **Step 9: pytest one final time**

  ```bash
  apptainer exec --pwd /data/lim2/projects/multifid-th multifid-th-gpu.sif python -m pytest -q --deselect tests/training/test_hpo.py::TestObjective::test_make_objective_returns_float --deselect tests/case_pressure_drop/test_workflow.py::test_training_and_evaluation_smoke --deselect tests/case_pressure_drop/test_workflow.py::test_feature_selection_uses_only_train_cases 2>&1 | tail -3
  ```

  Expected: `110 passed, 3 deselected`.

- [ ] **Step 10: Open PR (optional but recommended)**

  Run:
  ```bash
  gh pr create --base main --head rename/multifid-th --title "Rename project to MULTIFID-TH" --body "$(cat <<'EOF'
  ## Summary
  - Rename TH_HOLO_workflow → MULTIFID-TH across PyPI distribution, docs, Docker tags, Apptainer SIF, GitHub repo, and host filesystem
  - Add tagline "Multifidelity surrogates for thermal-hydraulics" to README intro + Sphinx banner so HOLO's original concept is finally explicit
  - Spec: docs/superpowers/specs/2026-05-13-multifid-th-rename-design.md
  - Plan: docs/superpowers/plans/2026-05-13-multifid-th-rename.md

  ## Test plan
  - [x] git grep returns zero non-archive hits for the old name
  - [x] Sphinx build clean with -W
  - [x] pytest: 110 passed, 3 deselected (pre-existing env drift)
  - [x] SIF rebuilt + imports physicsnemo/torch
  - [x] Editable install repointed to new path
  - [x] GitHub repo rename + local remote update
  EOF
  )"
  ```

  Expected: a PR URL is returned.

---

## Open questions for the executor

If anything in this plan looks ambiguous at execution time, prefer these defaults:

1. **Pre-existing pytest failures:** If pytest reports more or fewer than 3 deselected failures, check [[apptainer-runtime]] memory before treating it as a real regression — the named tests are pre-existing env drift on this host.
2. **`gh repo rename` requires confirmation:** If `gh` prompts interactively, supply `--yes` or run it from a terminal that can answer the prompt.
3. **Branch base:** The plan assumes you base off `refactor/repo-layout` tip. If `main` has advanced and the user wants to base off `main` instead, redo Step 4 of Task 0 with `git checkout -b rename/multifid-th main` after merging or rebasing first.
4. **PR target branch:** Step 10 of Task 13 targets `main`. If the user wants to keep the layered structure (rename merges into `refactor/repo-layout` first, then both merge into `main`), change `--base main` to `--base refactor/repo-layout`.

If the executor hits any "this doesn't match the plan" situation that isn't covered above, **stop and surface it** rather than improvising.
