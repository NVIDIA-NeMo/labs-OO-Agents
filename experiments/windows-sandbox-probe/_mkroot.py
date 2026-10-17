# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a private runtime root: interpreter + stdlib + DLLs + site-packages.

The base interpreter directory denies DACL writes (Errno 5), so the container
cannot be granted access to it in place. A user-owned copy is the only way to
make the interpreter reachable from inside the AppContainer.

Two stages, deliberately separable:

``--full`` (default)
    Copy the interpreter, stdlib and site-packages. ~274 s, 443 MB. One-time.

``--sync-nooa``
    Refresh the ``nooa`` package inside an existing root and delete the
    editable hooks. Sub-second. Runs after any source edit.

Why the package is copied rather than installed or granted in place: the
shipped ``_editable_impl_nooa.pth`` contains ``E:\\labs-OO-Agents\\src``, so
``import nooa`` in the container resolves to the checkout — a path the
container must never reach. Granting the checkout instead would put a
persistent ACE on a directory the user edits, and the M0 probe showed that
byte-exact DACL rollback is impossible on an inheriting directory. Copying
the package keeps every ACL write inside the one directory the runtime root
already requires.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# The uv-managed CPython install. ``sys.base_prefix`` points at it even when
# this script runs from the project venv, so derive it instead of hardcoding.
SRC_BASE = Path(sys.base_prefix)
SRC_SP = REPO / ".venv" / "Lib" / "site-packages"
SRC_NOOA = REPO / "src" / "nooa"

DST = HERE / ".rt"
DST_SP = DST / "Lib" / "site-packages"
SYNC_MARKER = DST_SP / ".nooa-sync.json"

# Directories copied wholesale from the base install; everything else at the
# top level is a file (python.exe, python313.dll, ...).
_COPY_DIRS = ("Lib", "DLLs", "libs", "include")
# stdlib subtrees no worker needs; each is tens of MB.
_SKIP_STDLIB = ("__pycache__", "test", "tests", "idlelib", "tkinter", "turtledemo")


# --- stage 2: the nooa package ---------------------------------------------
def _tree_signature(root: Path) -> dict[str, int]:
    """Cheap content fingerprint: file count, byte total, newest mtime.

    Used to skip the copy when nothing changed. mtime_ns is what actually
    catches an edit; the count and byte total catch additions and deletions
    cheaply enough that a full re-hash is unnecessary.
    """
    files = 0
    total = 0
    newest = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for name in filenames:
            if name.endswith(".pyc"):
                continue
            try:
                st = os.stat(os.path.join(dirpath, name))
            except OSError:
                continue
            files += 1
            total += st.st_size
            newest = max(newest, st.st_mtime_ns)
    return {"files": files, "bytes": total, "newest_ns": newest}


def sync_nooa(*, force: bool = False) -> str:
    """Materialise ``src/nooa`` into the runtime root; drop the editable hooks.

    Returns a one-line summary. Deletes the target before copying so a stale
    ``__pycache__`` from a previous source revision cannot shadow the new code.
    """
    if not SRC_NOOA.is_dir():
        raise SystemExit(f"source package missing: {SRC_NOOA}")
    if not DST_SP.is_dir():
        raise SystemExit(f"runtime root missing: {DST}\nrun without --sync-nooa first")

    removed = []
    for hook in DST_SP.glob("_editable_impl_*.pth"):
        # Every one of these points into the checkout. None may survive: a hook
        # that silently resolves an import to an unreadable path is worse than
        # no hook at all.
        hook.unlink()
        removed.append(hook.name)

    signature = _tree_signature(SRC_NOOA)
    if not force and SYNC_MARKER.exists():
        try:
            previous = json.loads(SYNC_MARKER.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        if previous.get("signature") == signature:
            return f"unchanged ({signature['files']} files, {signature['bytes'] / 1e6:.1f} MB)"

    target = DST_SP / "nooa"
    t0 = time.perf_counter()
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(
        SRC_NOOA,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    elapsed = time.perf_counter() - t0
    SYNC_MARKER.write_text(
        json.dumps(
            {"signature": signature, "source": str(SRC_NOOA), "copied_at": time.time()},
            indent=2,
        ),
        encoding="utf-8",
    )
    hooks = f", removed {len(removed)} editable hook(s)" if removed else ""
    return (
        f"copied {signature['files']} files / {signature['bytes'] / 1e6:.1f} MB "
        f"in {elapsed:.2f}s{hooks}"
    )


# --- stage 1: the interpreter ----------------------------------------------
def build_root() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    t0 = time.perf_counter()
    DST.mkdir(parents=True, exist_ok=True)

    for name in os.listdir(SRC_BASE):
        src = SRC_BASE / name
        dst = DST / name
        if src.is_dir():
            if name in _COPY_DIRS:
                shutil.copytree(
                    src,
                    dst,
                    symlinks=True,
                    ignore=shutil.ignore_patterns(*_SKIP_STDLIB),
                )
        else:
            shutil.copy2(src, dst)

    # site-packages: only what an import probe needs; skip caches and huge trees.
    DST_SP.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        SRC_SP,
        DST_SP,
        symlinks=True,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    elapsed = time.perf_counter() - t0
    total = _tree_bytes(DST)
    files = sum(len(f) for _, _, f in os.walk(DST))
    print(f"root={DST}")
    print(f"files={files} bytes={total / 1e6:.1f}MB build_s={elapsed:.2f}")
    print("python.exe:", os.path.exists(DST / "python.exe"))


def _tree_bytes(root: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sync-nooa",
        action="store_true",
        help="refresh the nooa package in the existing root (sub-second) and exit",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="with --sync-nooa, copy even when the source signature is unchanged",
    )
    args = ap.parse_args()

    if args.sync_nooa:
        summary = sync_nooa(force=args.force)
        print(f"nooa package: {summary}")
        marker = DST_SP / "nooa" / "__init__.py"
        print("package present:", marker.exists())
        return 0

    build_root()
    summary = sync_nooa(force=True)
    print(f"nooa package: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
