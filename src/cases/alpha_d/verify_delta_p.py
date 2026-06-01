"""Compare the three relevant pressure drops after running MOOSE.

  delta_p_truth     = case_metadata.delta_p_case (from the ETL zarr)
  delta_p_surrogate = trapezoidal alpha_D integral (written by exporter)
  delta_p_moose     = MOOSE inlet-p postprocessor (since outlet=0)

surrogate_fidelity_relerr = (delta_p_surrogate - delta_p_truth) / delta_p_truth
coupling_fidelity_relerr  = (delta_p_moose - delta_p_surrogate) / delta_p_surrogate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INLET_AREA_M2 = 0.0314159265  # = pi * (outer_radius=0.1)^2, RZ axisymmetric
# Hardcoded for this case. The right thing for the long term is to read
# outer_radius from the .i file or pass it as an arg; left as a TODO.


def read_moose_inlet_pressure(csv_path: Path) -> float:
    """Read inlet-p (pressure × inlet_area integral) and divide by area.

    The MOOSE SideIntegralVariablePostprocessor returns ∫ pressure dA;
    we want the actual pressure for comparison with the surrogate's α_D
    integral, which has units of Pa.
    """
    data = np.genfromtxt(str(csv_path), delimiter=",", names=True)
    if data.ndim == 0:
        integral = float(data["inletp"])
    else:
        integral = float(np.atleast_1d(data["inletp"])[-1])
    return integral / INLET_AREA_M2


def compare(*, sidecar_path: Path, delta_p_moose: float) -> dict:
    sidecar = json.loads(sidecar_path.read_text())
    truth = float(sidecar["delta_p_truth"])
    surro = float(sidecar["delta_p_surrogate"])
    return {
        "delta_p_truth": truth,
        "delta_p_surrogate": surro,
        "delta_p_moose": float(delta_p_moose),
        "surrogate_fidelity_relerr": (surro - truth) / truth,
        "coupling_fidelity_relerr": (delta_p_moose - surro) / surro,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--moose-csv", type=Path, required=True)
    args = parser.parse_args(argv)

    p = read_moose_inlet_pressure(args.moose_csv)
    result = compare(sidecar_path=args.sidecar, delta_p_moose=p)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
