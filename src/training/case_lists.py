"""Shared loading and validation for newline-delimited case selections."""

from __future__ import annotations

from pathlib import Path


def load_case_selection(
    inline: list[str] | tuple[str, ...] | None,
    file_path: str | Path | None,
    *,
    label: str,
) -> list[str]:
    """Resolve an inline or file-backed case list, rejecting ambiguity."""
    inline_values = [
        str(value).strip() for value in (inline or []) if str(value).strip()
    ]
    if file_path and inline_values:
        raise ValueError(f"Set only one of {label} or {label}_file, not both.")
    if not file_path:
        return inline_values
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label}_file not found: {path}")
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"{label}_file is empty: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"{label}_file contains duplicate case names: {path}")
    return values
