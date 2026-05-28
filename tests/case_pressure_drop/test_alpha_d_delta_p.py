"""End-to-end regression test for the alpha_D -> MOOSE coupling.

Runs the exporter -> MOOSE -> verifier pipeline on
Re_43938__Dr_0p522__Lr_0p073 and asserts the coupling-fidelity error
stays within tolerance.

Observed when this test was added (2026-05-28):
  delta_p_truth     = 11.832  Pa  (training CFD direct)
  delta_p_surrogate =  8.186  Pa  (alpha_D ROI integral)
  delta_p_moose     =  2.106  Pa  (PINSFV with surrogate-driven C_F)
  surrogate_fidelity_relerr = -0.308  (surrogate vs truth)
  coupling_fidelity_relerr  = -0.743  (MOOSE vs surrogate)

The large coupling gap reflects the physical-model mismatch the design
spec already documented in Section 2 (homogenized PINSFV with a Forchheimer
closure is a different PDE from resolved CFD). The tolerance here just
guards the *baseline coupling number from regressing further*; it is not
a physics-quality bound.

Execution environment note (2026-05-28):
  This test drives apptainer subprocesses for the Python-SIF (exporter)
  and MOOSE-SIF steps. It must be run from a host shell where ``apptainer``
  is on PATH — **not** from inside the Python SIF, because the SIF does not
  bundle apptainer. The test gates on ``shutil.which("apptainer")`` so it
  skips cleanly when launched from within the SIF instead of crashing.

  To run manually:
    PYTHONPATH=src pytest tests/case_pressure_drop/test_alpha_d_delta_p.py -v -s

Skips silently when checkpoint, target zarr, the MOOSE executable,
the container SIFs, or the ``apptainer`` binary itself isn't available.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path("/data/lim2/projects/multifid-th/worktrees/integration")
TARGET_ZARR = (
    REPO
    / "data/flow_contraction_expansion/parametric_study/processed/Re_43938__Dr_0p522__Lr_0p073.zarr"
)
CKPT = REPO / "data/cases/train_conv1d/model.mdlus"
META = REPO / "data/cases/train_conv1d/run_meta.json"
MOOSE_DIR = REPO / "src/cases/alpha_d/moose"
NS_EXE = REPO / "moose/modules/navier_stokes/navier_stokes-opt"
MOOSE_SIF = Path("/data/lim2/containers/moose-dev-openmpi-x86_64_latest.sif")
PY_SIF = Path("/data/lim2/projects/multifid-th/worktrees/refactor/multifid-th-cpu.sif")

# Calibrated against the observed value at landing time. See module
# docstring for the reasoning behind the wide bound.
COUPLING_TOL = 0.80


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
            "/data/lim2/projects/multifid-th:/data/lim2/projects/multifid-th",
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

    # 2) Stage CSV in the MOOSE input directory (MOOSE resolves data_file
    # relative to the .i file's directory)
    staged_csv = MOOSE_DIR / "forchheimer_profile.csv"
    shutil.copyfile(out_csv, staged_csv)

    # 3) Run MOOSE — runs inside the MOOSE SIF (libwasphit, libvtk)
    pp_csv = tmp_path / "2d-porous-flow_alphaD_out.csv"
    subprocess.run(
        [
            "apptainer",
            "exec",
            "--bind",
            "/data/lim2/projects/multifid-th:/data/lim2/projects/multifid-th",
            str(MOOSE_SIF),
            "bash",
            "-lc",
            f"cd {MOOSE_DIR} && {NS_EXE} "
            f"-i 2d-porous-flow_alphaD.i "
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
