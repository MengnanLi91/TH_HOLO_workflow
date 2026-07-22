"""End-to-end regression test for the alpha_D -> MOOSE coupling.

Runs the exporter -> MOOSE -> verifier pipeline on
Re_43938__Dr_0p522__Lr_0p073 and asserts the coupling-fidelity error
stays within tolerance.

Reference values for `Re_43938__Dr_0p522__Lr_0p073`:
  delta_p_truth     = 11.832 Pa  (training CFD direct)
  delta_p_surrogate = 11.357 Pa  (alpha_D ROI trapz integral, baseline included)
  delta_p_moose     = 11.282 Pa  (PINSFV pressure = postprocessor / inlet_area)
  surrogate_fidelity_relerr = -0.040  (surrogate vs truth — matches parity plot ≤10%)
  coupling_fidelity_relerr  = -0.007  (MOOSE vs surrogate — essentially exact)

Both the standalone surrogate (-4.0% vs truth) and the coupled MOOSE
pressure (-4.7% vs truth) sit within the parity plot's "all cases within
~10%" claim. The coupling adds value rather than distortion: MOOSE now
reproduces the surrogate's own integral to within 0.7%, where it
previously over-predicted by +30% because of PiecewiseLinear bridging
across the porosity steps (the exporter now step-fences both block
boundaries — see physics reference §5.4).

Forchheimer mapping (corrected derivation against PINSFVMomentumFriction.C
empirical behavior):
  F = α_D · porosity² / D_h
  Block 1/3 (buffer, porosity=1):  F = α_D / D_outer
  Block 2 (throat, porosity=Dr²):  F = α_D · Dr³ / D_outer

The kernel comment in PINSFVMomentumFriction.C:102-104 states
``∇p = (ρ/2) F |v| v`` with v=superficial, but constant-F=1 throat
verification gives ΔP = F · throat_length / (2 · porosity²) — an extra
1/porosity factor relative to that comment. The mapping above matches the
empirical behavior to within 1.5%.

The verifier reports ``delta_p_moose`` as ``inlet-p / inlet_area`` where
``inlet_area = π · outer_radius²`` for the 2-D axisymmetric (RZ) mesh.
SideIntegralVariablePostprocessor returns ``∫ pressure · 2πr dr``, not the
pressure itself.

Execution environment:
  The test drives apptainer subprocesses for the Python-SIF (exporter)
  and MOOSE-SIF steps, so it must run from a host shell where
  ``apptainer`` is on PATH — **not** from inside the Python SIF, because
  the SIF does not bundle apptainer. The test gates on
  ``shutil.which("apptainer")`` and skips cleanly when launched from
  within the SIF instead of crashing.

  To run manually:
    PYTHONPATH=src pytest tests/case_pressure_drop/test_alpha_d_delta_p.py -v -s

Skips silently when checkpoint, target zarr, the MOOSE executable,
the container SIFs, or the ``apptainer`` binary itself isn't available.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Resolve the repo root from this file's location:
#   tests/case_pressure_drop/<this>.py → parents[2] is the repo root,
# so the test works in any worktree without hardcoded user paths.
REPO = Path(__file__).resolve().parents[2]

# Apptainer bind path: by default, bind the directory that contains all
# multifid-th worktrees (assumed to be REPO's grandparent — i.e., the
# `<...>/multifid-th/` shared root above `worktrees/<branch>/`). Override
# with MULTIFID_BIND when the SIFs live outside that subtree.
BIND_PATH = Path(os.environ.get("MULTIFID_BIND", str(REPO.parents[1])))
BIND = f"{BIND_PATH}:{BIND_PATH}"

TARGET_ZARR = (
    REPO
    / "data/flow_contraction_expansion/parametric_study/processed/Re_43938__Dr_0p522__Lr_0p073.zarr"
)
CKPT = REPO / "data/cases/train_conv1d/model.mdlus"
META = REPO / "data/cases/train_conv1d/run_meta.json"
MOOSE_DIR = REPO / "src/cases/alpha_d/moose"
NS_EXE = REPO / "moose/modules/navier_stokes/navier_stokes-opt"

# Container locations: defaults match the sibling `worktrees/refactor`
# layout this repo ships with; override via MULTIFID_MOOSE_SIF /
# MULTIFID_PY_SIF when those .sif files live elsewhere on a given host.
MOOSE_SIF = Path(
    os.environ.get(
        "MULTIFID_MOOSE_SIF",
        # The MOOSE SIF is typically distributed centrally, not in-repo.
        # No portable default; an empty path makes the skipif gate trigger.
        "",
    )
)
PY_SIF = Path(
    os.environ.get(
        "MULTIFID_PY_SIF",
        str(REPO.parent / "refactor" / "multifid-th-cpu.sif"),
    )
)

# Set well above the observed |coupling_fidelity_relerr| of 0.007 so that
# routine numerical drift doesn't trip the test, while keeping enough margin
# to catch real regressions (e.g., a future change that drops the
# porosity-boundary step-fence and re-introduces PiecewiseLinear smearing
# would push this back above 0.30). See module docstring for context.
COUPLING_TOL = 0.05


@pytest.mark.skipif(
    not (
        shutil.which("apptainer") is not None
        and TARGET_ZARR.exists()
        and CKPT.exists()
        and META.exists()
        and NS_EXE.exists()
        and MOOSE_SIF.exists()
        and PY_SIF.exists()
    ),
    reason=(
        "Required artifacts (checkpoint, zarr, MOOSE executable, SIFs) not present, "
        "or apptainer not on PATH (test must run from host, not inside a SIF)."
    ),
)
def test_end_to_end_coupling_within_tolerance(tmp_path):
    out_csv = tmp_path / "forchheimer_profile.csv"

    # 1) Export — runs inside the Python SIF (needs torch, physicsnemo, zarr)
    subprocess.run(
        [
            "apptainer",
            "exec",
            "--bind",
            BIND,
            str(PY_SIF),
            "bash",
            "-lc",
            f"cd {REPO}/src && PYTHONPATH=. python -m cases.alpha_d.export_friction_profile "
            f"--zarr {TARGET_ZARR} --checkpoint {CKPT} --run-meta {META} "
            f"--output-csv {out_csv}",
        ],
        check=True,
    )
    sidecar = out_csv.with_suffix(".meta.json")
    assert sidecar.exists(), "Exporter did not produce sidecar JSON"

    # 2) Isolate this case's MOOSE input and profile. MOOSE resolves data_file
    # relative to the .i file's directory.
    moose_work_dir = tmp_path / "moose"
    moose_work_dir.mkdir()
    staged_input = moose_work_dir / "2d-porous-flow_alphaD.i"
    staged_csv = moose_work_dir / "forchheimer_profile.csv"
    shutil.copyfile(MOOSE_DIR / "2d-porous-flow_alphaD.i", staged_input)
    shutil.copyfile(out_csv, staged_csv)
    delta_p_initial = json.loads(sidecar.read_text(encoding="utf-8"))["delta_p_surrogate"]

    # 3) Run MOOSE — runs inside the MOOSE SIF (libwasphit, libvtk)
    pp_csv = tmp_path / "2d-porous-flow_alphaD_out.csv"
    subprocess.run(
        [
            "apptainer",
            "exec",
            "--bind",
            BIND,
            str(MOOSE_SIF),
            "bash",
            "-lc",
            f"cd {moose_work_dir} && {NS_EXE} "
            f"-i {staged_input} "
            f"delta_p_initial={delta_p_initial} "
            f"Outputs/file_base={pp_csv.with_suffix('').as_posix()}",
        ],
        check=True,
    )
    assert pp_csv.exists(), f"MOOSE did not produce postprocessor CSV at {pp_csv}"

    # 4) Verify
    from cases.alpha_d.verify_delta_p import compare, read_moose_inlet_pressure

    delta_p_moose = read_moose_inlet_pressure(pp_csv)
    result = compare(sidecar_path=sidecar, delta_p_moose=delta_p_moose)

    print("\ndelta_p_truth     :", result["delta_p_truth"])
    print("delta_p_surrogate :", result["delta_p_surrogate"])
    print("delta_p_moose     :", result["delta_p_moose"])
    print("surrogate_relerr  :", result["surrogate_fidelity_relerr"])
    print("coupling_relerr   :", result["coupling_fidelity_relerr"])

    assert abs(result["coupling_fidelity_relerr"]) < COUPLING_TOL, (
        f"coupling fidelity {result['coupling_fidelity_relerr']:+.3f} "
        f"exceeds tolerance {COUPLING_TOL:.3f}"
    )
