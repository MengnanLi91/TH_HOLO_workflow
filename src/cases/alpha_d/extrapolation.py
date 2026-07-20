"""Grid enumeration and interior/shell splitting for the extrapolation study.

A "shell" is the set of cases at the outer-k values of one axis (Re/Dr/Lr).
The same shell list drives the regressor's ``data.split.force_test`` and the
alpha_D model's ``data.exclude_cases`` — this module is the single source of
truth that keeps those two symmetric.

Floor handling:
  - ``exclude_below_dr`` drops the sparse Dr=0.05 column from everything.
  - ``dr_floor`` is the common interior floor (alpha_D trains at Dr>=0.333).
    The *interior* never contains Dr<dr_floor. For axis != "Dr" the shell is
    also restricted to Dr>=dr_floor so both models evaluate the same cases.
    For axis == "Dr" + side == "low" the shell deliberately reaches below the
    floor — that IS the extrapolation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

AXES = ("Re", "Dr", "Lr")


@dataclass(frozen=True)
class CaseParams:
    name: str
    Re: float
    Dr: float
    Lr: float


@dataclass(frozen=True)
class SplitResult:
    interior: list[CaseParams]
    shell: list[CaseParams]
    shell_axis_values: list[float]
    report: list[CaseParams] = field(default_factory=list)


def parse_case_params(name: str) -> CaseParams:
    parts = dict(p.split("_", 1) for p in name.split("__"))

    def f(x: str) -> float:
        return float(x.replace("p", "."))

    return CaseParams(
        name=name, Re=f(parts["Re"]), Dr=f(parts["Dr"]), Lr=f(parts["Lr"])
    )


def enumerate_cases(zarr_dir: str | Path) -> list[CaseParams]:
    paths = sorted(Path(zarr_dir).glob("*.zarr"))
    if not paths:
        raise FileNotFoundError(f"No .zarr stores under {zarr_dir}")
    return [parse_case_params(p.stem) for p in paths]


def _distinct_sorted(cases: list[CaseParams], axis: str) -> list[float]:
    return sorted({getattr(c, axis) for c in cases})


def _inner_axis_values(
    cases: list[CaseParams],
    axis: str,
    *,
    guard_k: int,
    dr_floor: float,
    exclude_below_dr: float,
) -> list[float]:
    if guard_k < 0:
        raise ValueError(f"guard_k must be >= 0, got {guard_k}")
    pool = [c for c in cases if c.Dr >= exclude_below_dr]
    if axis == "Dr":
        pool = [c for c in pool if c.Dr >= dr_floor]
    vals = _distinct_sorted(pool, axis)
    if guard_k == 0:
        return vals
    if 2 * guard_k >= len(vals):
        raise ValueError(
            f"guard_k={guard_k} leaves no interior {axis} values "
            f"from {len(vals)} distinct levels"
        )
    return vals[guard_k:-guard_k]


def _parse_axes(raw: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        axes = tuple(axis.strip() for axis in raw.split(",") if axis.strip())
    else:
        axes = tuple(raw)
    unknown = [axis for axis in axes if axis not in AXES]
    if unknown:
        raise ValueError(f"guard axes must be from {AXES}, got {unknown}")
    return axes


def build_split(
    cases: list[CaseParams],
    *,
    axis: str,
    side: str,
    k: int,
    dr_floor: float = 0.333,
    exclude_below_dr: float = 0.1,
    report_guard_axes: str | tuple[str, ...] | list[str] | None = None,
    report_guard_k: int = 0,
) -> SplitResult:
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got {axis!r}")
    if side not in ("low", "high"):
        raise ValueError(f"side must be 'low' or 'high', got {side!r}")

    usable = [c for c in cases if c.Dr >= exclude_below_dr]

    # For non-Dr axes both interior and shell stay at/above the floor so the
    # two models are compared on identical cases. For the Dr axis the shell is
    # the experiment, so we split over the full usable Dr range.
    pool = usable if axis == "Dr" else [c for c in usable if c.Dr >= dr_floor]

    vals = _distinct_sorted(pool, axis)
    if k < 1 or k >= len(vals):
        raise ValueError(f"k={k} out of range for {len(vals)} distinct {axis} values")
    shell_vals = vals[:k] if side == "low" else vals[-k:]
    shell_set = set(shell_vals)

    shell = [c for c in pool if getattr(c, axis) in shell_set]
    interior = [
        c for c in usable if c.Dr >= dr_floor and getattr(c, axis) not in shell_set
    ]
    guard_axes = tuple(ax for ax in _parse_axes(report_guard_axes) if ax != axis)
    report = list(shell)
    if guard_axes and report_guard_k > 0:
        allowed = {
            guard_axis: set(
                _inner_axis_values(
                    cases,
                    guard_axis,
                    guard_k=report_guard_k,
                    dr_floor=dr_floor,
                    exclude_below_dr=exclude_below_dr,
                )
            )
            for guard_axis in guard_axes
        }
        report = [
            c
            for c in shell
            if all(
                getattr(c, guard_axis) in values
                for guard_axis, values in allowed.items()
            )
        ]
    return SplitResult(
        interior=interior,
        shell=shell,
        shell_axis_values=list(shell_vals),
        report=report,
    )


def build_indist_panel(
    cases: list[CaseParams],
    *,
    count: int = 36,
    guard_k: int = 2,
    dr_floor: float = 0.333,
    exclude_below_dr: float = 0.1,
) -> list[CaseParams]:
    """Return a deterministic interior panel for in-distribution evidence."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    usable = [c for c in cases if c.Dr >= max(dr_floor, exclude_below_dr)]
    allowed = {
        axis: set(
            _inner_axis_values(
                usable,
                axis,
                guard_k=guard_k,
                dr_floor=dr_floor,
                exclude_below_dr=exclude_below_dr,
            )
        )
        for axis in AXES
    }
    candidates = sorted(
        (
            c
            for c in usable
            if all(getattr(c, axis) in values for axis, values in allowed.items())
        ),
        key=lambda c: (c.Re, c.Dr, c.Lr, c.name),
    )
    if not candidates:
        raise ValueError("No in-distribution panel candidates after guard filtering.")
    if count >= len(candidates):
        return candidates

    if count == 1:
        return [candidates[len(candidates) // 2]]

    selected = {round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)}
    if len(selected) < count:
        for idx in range(len(candidates)):
            selected.add(idx)
            if len(selected) == count:
                break
    return [candidates[idx] for idx in sorted(selected)]


def _hydra_list(names: list[str]) -> str:
    return "[" + ",".join(names) + "]"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr-dir", required=True)
    p.add_argument("--axis", choices=AXES)
    p.add_argument("--side", choices=("low", "high"))
    p.add_argument("--k", type=int)
    p.add_argument("--dr-floor", type=float, default=0.333)
    p.add_argument("--exclude-below-dr", type=float, default=0.1)
    p.add_argument("--report-guard-axes", default="")
    p.add_argument("--report-guard-k", type=int, default=2)
    p.add_argument("--count", type=int, default=36)
    p.add_argument(
        "--emit",
        choices=(
            "shell-hydra",
            "interior-hydra",
            "shell-names",
            "heldout-hydra",
            "heldout-names",
            "report-hydra",
            "report-names",
            "indist-panel-hydra",
            "indist-panel-names",
            "summary",
        ),
        default="summary",
    )
    ns = p.parse_args(argv)

    cases = enumerate_cases(ns.zarr_dir)
    if ns.emit in ("indist-panel-hydra", "indist-panel-names"):
        panel = build_indist_panel(
            cases,
            count=ns.count,
            guard_k=ns.report_guard_k,
            dr_floor=ns.dr_floor,
            exclude_below_dr=ns.exclude_below_dr,
        )
        if ns.emit == "indist-panel-hydra":
            print(_hydra_list([c.name for c in panel]))
        else:
            print("\n".join(c.name for c in panel))
        return 0

    if ns.axis is None or ns.side is None or ns.k is None:
        p.error(
            "--axis, --side, and --k are required unless emitting an in-distribution panel"
        )

    res = build_split(
        cases,
        axis=ns.axis,
        side=ns.side,
        k=ns.k,
        dr_floor=ns.dr_floor,
        exclude_below_dr=ns.exclude_below_dr,
        report_guard_axes=ns.report_guard_axes,
        report_guard_k=ns.report_guard_k,
    )
    if ns.emit in ("shell-hydra", "heldout-hydra"):
        print(_hydra_list([c.name for c in res.shell]))
    elif ns.emit == "interior-hydra":
        print(_hydra_list([c.name for c in res.interior]))
    elif ns.emit in ("shell-names", "heldout-names"):
        print("\n".join(c.name for c in res.shell))
    elif ns.emit == "report-hydra":
        print(_hydra_list([c.name for c in res.report]))
    elif ns.emit == "report-names":
        print("\n".join(c.name for c in res.report))
    else:
        print(
            f"axis={ns.axis} side={ns.side} k={ns.k} | "
            f"shell_values={res.shell_axis_values} | "
            f"n_shell={len(res.shell)} n_report={len(res.report)} "
            f"n_interior={len(res.interior)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
