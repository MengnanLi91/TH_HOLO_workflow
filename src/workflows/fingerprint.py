"""Content fingerprints for workflow inputs and produced artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _file_digest(
    path: Path, cached: dict[str, Any] | None
) -> tuple[str, dict[str, Any]]:
    stat = path.stat()
    key = str(path.resolve())
    if cached:
        entry = cached.get(key)
        if (
            entry
            and entry.get("size") == stat.st_size
            and entry.get("mtime_ns") == stat.st_mtime_ns
        ):
            return str(entry["sha256"]), entry
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    entry = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }
    return digest.hexdigest(), entry


def fingerprint_path(
    path: str | Path,
    *,
    cache: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a deterministic SHA-256 fingerprint and refreshed file cache."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Fingerprint input does not exist: {root}")
    files = (
        [root]
        if root.is_file()
        else sorted(item for item in root.rglob("*") if item.is_file())
    )
    combined = hashlib.sha256()
    refreshed: dict[str, Any] = {}
    total_size = 0
    for item in files:
        relative = item.name if root.is_file() else item.relative_to(root).as_posix()
        digest, entry = _file_digest(item, cache)
        refreshed[str(item.resolve())] = entry
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        total_size += int(entry["size"])
    return (
        {
            "path": str(root),
            "kind": "file" if root.is_file() else "tree",
            "sha256": combined.hexdigest(),
            "file_count": len(files),
            "size_bytes": total_size,
        },
        refreshed,
    )


def load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
