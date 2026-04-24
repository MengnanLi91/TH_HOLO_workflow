"""Data-distribution preprocessing analysis for the case pressure-drop dataset.

Shows training / test / total case counts per parameter bin (Re, Dr, Lr) so
that under-supported regions are obvious before training.  Optional train/test
split information can be provided via a ``run_meta.json`` file.
"""

from __future__ import annotations

import re as _re
from collections import Counter
from pathlib import Path
from typing import Any


AXES: tuple[str, ...] = ("Dr", "Re", "Lr")


def parse_case_params(name: str) -> tuple[float, float, float]:
    """Extract ``(Re, Dr, Lr)`` from a case name like ``Re_*__Dr_XpXXX__Lr_XpXXX``.

    Values with a ``p`` decimal separator (e.g. ``0p333``) are parsed as
    floats.  Missing keys default to ``0.0``.
    """

    def _get(key: str) -> float:
        m = _re.search(rf"{key}_([0-9p]+)", name)
        if not m:
            return 0.0
        try:
            return float(m.group(1).replace("p", "."))
        except ValueError:
            return 0.0

    return _get("Re"), _get("Dr"), _get("Lr")


def bin_by(sim_names: list[str], axis: str) -> dict[float, int]:
    """Count cases by a given axis.  ``axis`` must be ``Re``, ``Dr``, or ``Lr``."""
    idx = {"Re": 0, "Dr": 1, "Lr": 2}[axis]
    counter: Counter[float] = Counter()
    for name in sim_names:
        counter[round(parse_case_params(name)[idx], 3)] += 1
    return dict(counter)


def support_level(n_train: int) -> tuple[str, str]:
    """Classify training-set support for a bin.  Returns ``(marker, rich_style)``."""
    if n_train == 0:
        return "✗ none", "bold red"
    if n_train < 3:
        return "⚠ very low", "bold red"
    if n_train < 10:
        return "⚠ low", "yellow"
    if n_train < 30:
        return "◦ ok", "cyan"
    return "✓ good", "green"


def load_sim_names_from_zarr(
    zarr_dir: str | Path,
    *,
    exclude_cases: list[str] | None = None,
    min_Dr: float | None = None,
) -> list[str]:
    """Discover case names in a zarr directory, with optional filters.

    Mirrors the filtering performed by ``TabularPairDataset`` so that the
    reported distribution matches what training will actually see.
    """
    zarr_dir = Path(zarr_dir).expanduser().resolve()
    sim_paths = sorted(zarr_dir.glob("*.zarr"))
    if not sim_paths:
        raise FileNotFoundError(f"No .zarr stores found in {zarr_dir}")

    exclude_set = set(exclude_cases or [])
    names: list[str] = []
    for sp in sim_paths:
        stem = sp.stem
        if stem in exclude_set:
            continue
        if min_Dr is not None and parse_case_params(stem)[1] < min_Dr:
            continue
        names.append(stem)
    return names


def load_split_from_run_meta(run_meta_path: str | Path) -> tuple[list[str], list[str]]:
    """Read ``(train_sims, test_sims)`` from a training ``run_meta.json``."""
    import json

    run_meta_path = Path(run_meta_path).expanduser().resolve()
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    split = run_meta.get("split") or {}
    return list(split.get("train_sims", [])), list(split.get("test_sims", []))


def _fmt_axis_value(axis: str, value: float) -> str:
    if axis == "Re":
        return f"{int(value)}"
    return f"{value:.3f}"


def print_distribution_rich(
    *,
    all_sims: list[str] | None = None,
    train_sims: list[str] | None = None,
    test_sims: list[str] | None = None,
    zarr_dir: str | Path | None = None,
    axes: tuple[str, ...] = AXES,
) -> None:
    """Render distribution tables using ``rich`` with a plain-text fallback."""
    try:
        _print_distribution_rich(
            all_sims=all_sims,
            train_sims=train_sims,
            test_sims=test_sims,
            zarr_dir=zarr_dir,
            axes=axes,
        )
    except ImportError:
        _print_distribution_plain(
            all_sims=all_sims,
            train_sims=train_sims,
            test_sims=test_sims,
            zarr_dir=zarr_dir,
            axes=axes,
        )


def _resolve_sims(
    *,
    all_sims: list[str] | None,
    train_sims: list[str] | None,
    test_sims: list[str] | None,
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(all, train, test)`` lists filling in gaps when possible."""
    train = list(train_sims or [])
    test = list(test_sims or [])
    if all_sims is not None:
        full = list(all_sims)
    else:
        full = sorted(set(train) | set(test))
    return full, train, test


def _print_distribution_rich(
    *,
    all_sims: list[str] | None,
    train_sims: list[str] | None,
    test_sims: list[str] | None,
    zarr_dir: str | Path | None,
    axes: tuple[str, ...],
) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console(width=120)

    full, train, test = _resolve_sims(
        all_sims=all_sims, train_sims=train_sims, test_sims=test_sims,
    )
    show_split = bool(train or test)

    header_lines = [f"[bold]Total cases:[/bold] {len(full)}"]
    if zarr_dir is not None:
        header_lines.append(f"[bold]Zarr directory:[/bold] [cyan]{zarr_dir}[/cyan]")
    if show_split:
        header_lines.append(
            f"[bold]Train / Test:[/bold] {len(train)} / {len(test)}"
        )
    console.print(
        Panel(
            "\n".join(header_lines),
            title="[bold bright_blue]Case Dataset Distribution[/bold bright_blue]",
            border_style="bright_blue",
            expand=False,
        )
    )

    console.print(
        "[dim]Support thresholds: "
        "[bold red]⚠ very low[/bold red] (<3), "
        "[yellow]⚠ low[/yellow] (<10), "
        "[cyan]◦ ok[/cyan] (<30), "
        "[green]✓ good[/green] (≥30).  "
        "When train/test is provided, Support is based on Train count.[/dim]"
    )

    for axis in axes:
        full_counts = bin_by(full, axis)
        train_counts = bin_by(train, axis) if show_split else {}
        test_counts = bin_by(test, axis) if show_split else {}

        all_values = sorted(
            set(full_counts) | set(train_counts) | set(test_counts)
        )
        if not all_values:
            continue

        table = Table(
            title=f"[bold]Distribution by {axis}[/bold]",
            header_style="bold magenta",
            show_lines=False,
            padding=(0, 1),
        )
        table.add_column(axis, style="cyan", justify="right", no_wrap=True)
        if show_split:
            table.add_column("Train", justify="right")
            table.add_column("Test", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Support", justify="left")

        for v in all_values:
            n_train = train_counts.get(v, 0)
            n_test = test_counts.get(v, 0)
            n_total = full_counts.get(v, n_train + n_test)

            # Use train count for support when a split is present; otherwise
            # classify by total sample count.
            marker, style = support_level(n_train if show_split else n_total)
            support_text = Text(marker)
            support_text.stylize(style)

            row = [_fmt_axis_value(axis, v)]
            if show_split:
                row.extend([str(n_train), str(n_test)])
            row.extend([str(n_total), support_text])
            table.add_row(*row)

        console.print(table)


def _print_distribution_plain(
    *,
    all_sims: list[str] | None,
    train_sims: list[str] | None,
    test_sims: list[str] | None,
    zarr_dir: str | Path | None,
    axes: tuple[str, ...],
) -> None:
    full, train, test = _resolve_sims(
        all_sims=all_sims, train_sims=train_sims, test_sims=test_sims,
    )
    show_split = bool(train or test)

    print("=" * 60)
    print("Case Dataset Distribution")
    print("=" * 60)
    print(f"Total cases: {len(full)}")
    if zarr_dir is not None:
        print(f"Zarr directory: {zarr_dir}")
    if show_split:
        print(f"Train / Test: {len(train)} / {len(test)}")

    for axis in axes:
        full_counts = bin_by(full, axis)
        train_counts = bin_by(train, axis) if show_split else {}
        test_counts = bin_by(test, axis) if show_split else {}

        all_values = sorted(
            set(full_counts) | set(train_counts) | set(test_counts)
        )
        if not all_values:
            continue

        print(f"\nDistribution by {axis}:")
        header = f"  {axis:>10}"
        if show_split:
            header += f"  {'Train':>6}  {'Test':>6}"
        header += f"  {'Total':>6}  Support"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for v in all_values:
            n_train = train_counts.get(v, 0)
            n_test = test_counts.get(v, 0)
            n_total = full_counts.get(v, n_train + n_test)
            marker, _ = support_level(n_train if show_split else n_total)
            line = f"  {_fmt_axis_value(axis, v):>10}"
            if show_split:
                line += f"  {n_train:>6}  {n_test:>6}"
            line += f"  {n_total:>6}  {marker}"
            print(line)
