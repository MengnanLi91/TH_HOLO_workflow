"""Rewrite the embedded ``__name__`` in a Conv1DProfile ``.mdlus`` checkpoint.

PhysicsNeMo stores the class name and module path inside ``args.json``
of the saved archive; ``from_checkpoint`` re-imports via
``getattr(module, name)``. Renaming the class (AlphaDConv1D →
Conv1DProfile) makes old checkpoints depend on the backward-compat
subclass alias. Running this script on an existing checkpoint rewrites
the embedded name so the alias is no longer needed.

PhysicsNeMo's save format has been both ``tar`` (legacy) and ``zip``
(current) over the project's history; this utility auto-detects and
preserves the original format when rewriting.

Usage::

    python -m training.models._migrate_conv1d_checkpoint path/to/model.mdlus
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

OLD_NAME = "AlphaDConv1D"
NEW_NAME = "Conv1DProfile"


def _migrate_zip(src: Path) -> bool:
    """Returns True when the archive was modified."""
    with zipfile.ZipFile(src, "r") as z:
        names = z.namelist()
        if "args.json" not in names:
            raise RuntimeError(f"args.json not found inside {src}")
        payloads = {n: z.read(n) for n in names}

    args = json.loads(payloads["args.json"])
    if args.get("__name__") == NEW_NAME:
        return False
    if args.get("__name__") != OLD_NAME:
        raise RuntimeError(
            f"unexpected __name__={args.get('__name__')!r} in {src}; "
            "this script only migrates AlphaDConv1D → Conv1DProfile."
        )
    args["__name__"] = NEW_NAME
    payloads["args.json"] = json.dumps(args).encode("utf-8")

    out_tmp = src.with_suffix(src.suffix + ".migrating")
    with zipfile.ZipFile(out_tmp, "w", compression=zipfile.ZIP_STORED) as z:
        for name in names:
            z.writestr(name, payloads[name])
    shutil.move(out_tmp, src)
    return True


def _migrate_tar(src: Path) -> bool:
    with tempfile.TemporaryDirectory() as scratch_dir:
        scratch = Path(scratch_dir)
        with tarfile.open(src, "r") as tar:
            tar.extractall(scratch)
            members = list(tar.getmembers())

        args_path = scratch / "args.json"
        if not args_path.is_file():
            raise RuntimeError(f"args.json not found inside {src}")

        args = json.loads(args_path.read_text())
        if args.get("__name__") == NEW_NAME:
            return False
        if args.get("__name__") != OLD_NAME:
            raise RuntimeError(
                f"unexpected __name__={args.get('__name__')!r} in {src}; "
                "this script only migrates AlphaDConv1D → Conv1DProfile."
            )
        args["__name__"] = NEW_NAME
        args_path.write_text(json.dumps(args))

        out_tmp = src.with_suffix(src.suffix + ".migrating")
        with tarfile.open(out_tmp, "w") as tar:
            for member in members:
                if not member.isfile():
                    continue
                tar.add(scratch / member.name, arcname=member.name)
        shutil.move(out_tmp, src)
        return True


def migrate(path: str | Path) -> bool:
    """Rewrite ``__name__`` in the checkpoint's ``args.json`` in place.

    Returns ``True`` if the file was modified, ``False`` if it already
    carried the new name.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"checkpoint not found: {src}")
    if zipfile.is_zipfile(src):
        return _migrate_zip(src)
    if tarfile.is_tarfile(src):
        return _migrate_tar(src)
    raise RuntimeError(
        f"{src} is neither a zip nor a tar archive; "
        "physicsnemo .mdlus uses one of those two formats."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help=".mdlus files to migrate in place")
    args = ap.parse_args(argv)

    rc = 0
    for raw in args.paths:
        try:
            changed = migrate(raw)
        except Exception as exc:  # noqa: BLE001
            print(f"{raw}: ERROR {exc}", file=sys.stderr)
            rc = 1
            continue
        print(f"{raw}: {'migrated' if changed else 'already up to date'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
