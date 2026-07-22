"""Compare the three relevant pressure drops after running MOOSE.

  delta_p_truth     = case_metadata.delta_p_case (from the ETL zarr)
  delta_p_surrogate = trapezoidal alpha_D integral (written by exporter)
  delta_p_moose     = MOOSE inlet-p postprocessor (since outlet=0)

surrogate_fidelity_relerr = (delta_p_surrogate - delta_p_truth) / delta_p_truth
coupling_fidelity_relerr  = (delta_p_moose - delta_p_surrogate) / delta_p_surrogate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Inlet area of the 2-D RZ axisymmetric mesh used by 2d-porous-flow_alphaD.i,
# = pi * outer_radius^2 with outer_radius = 0.1 m. Override here when adapting
# this verifier to a different mesh radius.
INLET_AREA_M2 = 0.0314159265


def _require_positive_finite(value: float, *, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return value


def read_moose_inlet_pressure(csv_path: Path, *, inlet_area_m2: float = INLET_AREA_M2) -> float:
    """Read inlet-p (pressure × inlet_area integral) and divide by area.

    The MOOSE SideIntegralVariablePostprocessor returns ∫ pressure dA;
    we want the actual pressure for comparison with the surrogate's α_D
    integral, which has units of Pa.
    """
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        raise ValueError(f"MOOSE CSV is missing or empty: {csv_path}")
    inlet_area_m2 = _require_positive_finite(inlet_area_m2, label="inlet area")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        inlet_name = next((name for name in ("inlet-p", "inletp") if name in names), None)
        if inlet_name is None:
            raise ValueError(f"MOOSE CSV does not contain the inlet-p postprocessor: {csv_path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"MOOSE CSV contains no postprocessor rows: {csv_path}")
    raw_integral = rows[-1].get(inlet_name)
    if raw_integral in (None, ""):
        raise ValueError(f"MOOSE CSV does not contain the inlet-p postprocessor: {csv_path}")
    integral = _require_positive_finite(raw_integral, label="MOOSE inlet-p integral")
    return _require_positive_finite(integral / inlet_area_m2, label="MOOSE pressure drop")


def compare(*, sidecar_path: Path, delta_p_moose: float) -> dict:
    sidecar = json.loads(sidecar_path.read_text())
    truth = _require_positive_finite(sidecar["delta_p_truth"], label="truth pressure drop")
    surro = _require_positive_finite(sidecar["delta_p_surrogate"], label="surrogate pressure drop")
    delta_p_moose = _require_positive_finite(delta_p_moose, label="MOOSE pressure drop")
    return {
        "verification_schema": 1,
        "verification_status": "valid",
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
    parser.add_argument("--inlet-area-m2", type=float, default=INLET_AREA_M2)
    args = parser.parse_args(argv)

    try:
        p = read_moose_inlet_pressure(args.moose_csv, inlet_area_m2=args.inlet_area_m2)
        result = compare(sidecar_path=args.sidecar, delta_p_moose=p)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"MOOSE pressure-drop verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
