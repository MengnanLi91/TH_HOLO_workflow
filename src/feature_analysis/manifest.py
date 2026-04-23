"""Run manifest for feature-analysis reproducibility.

Captures config, dataset identity (zarr paths + mtimes hash), sklearn /
numpy versions, git SHA (best-effort), and seeds. Written next to the
report as ``manifest.json``.
"""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _zarr_dir_hash(zarr_dir: Path) -> dict[str, Any]:
    """Hash of sorted zarr store paths + their mtimes + count.

    Catches most "data changed since last run" cases without reading the
    stores. Not cryptographic — intended for reproducibility tracking.
    """
    sim_paths = sorted(zarr_dir.glob("*.zarr"))
    entries = [(sp.name, sp.stat().st_mtime_ns) for sp in sim_paths]
    h = hashlib.sha256()
    for name, mtime in entries:
        h.update(name.encode())
        h.update(str(mtime).encode())
    return {
        "zarr_dir": str(zarr_dir.resolve()),
        "n_stores": len(entries),
        "sha256": h.hexdigest(),
    }


def _git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _lib_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for mod_name in ("numpy", "sklearn", "scipy", "zarr", "pandas", "matplotlib"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            continue
    return versions


def build_manifest(
    *,
    config: dict[str, Any],
    zarr_dir: str | Path,
    feature_names: list[str],
    target_name: str,
    n_rows: int,
    n_cases: int,
    seeds: dict[str, int],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble a manifest dict describing the current run."""
    zarr_dir = Path(zarr_dir)
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root)

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "dataset": {
            **_zarr_dir_hash(zarr_dir),
            "feature_names": list(feature_names),
            "target_name": target_name,
            "n_rows": int(n_rows),
            "n_cases": int(n_cases),
        },
        "seeds": dict(seeds),
        "versions": _lib_versions(),
        "git_sha": _git_sha(repo_root),
        "platform": platform.platform(),
    }


def write_manifest(manifest: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path
