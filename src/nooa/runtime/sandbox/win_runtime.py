# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate and maintain the private runtime root the container runs from.

The container cannot read the system interpreter (its directory denies DACL
writes to a non-admin user), so a user-owned copy of the interpreter, stdlib and
site-packages is mandatory rather than an optimisation. This module owns three
things about that copy:

* **where it lives** — explicit override, ``NOOA_SANDBOX_RUNTIME``, then a
  per-user cache;
* **what must be inside it** — ``python.exe`` plus the ``nooa`` package, with
  no editable ``.pth`` hook left behind that would resolve imports back to the
  checkout;
* **the ACL journal** — which grants were written, so a crashed parent's ACEs
  can be revoked on the next start. The journal is for recovery only; live
  teardown always uses ``REVOKE_ACCESS``.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "AclJournal",
    "RuntimeRoot",
    "default_runtime_root",
    "prepare_root",
    "resolve_root",
    "sync_package",
]

DEFAULT_CONTAINER_NAME = "nooa-sandbox"
_RUNTIME_ENV = "NOOA_SANDBOX_RUNTIME"
_SUBDIR = "sandbox-runtime"


@dataclass(frozen=True)
class RuntimeRoot:
    """A validated runtime root, with the paths the launcher needs."""

    root: Path

    @property
    def python(self) -> Path:
        return self.root / "python.exe"

    @property
    def site_packages(self) -> Path:
        return self.root / "Lib" / "site-packages"

    @property
    def package(self) -> Path:
        return self.site_packages / "nooa"

    @property
    def journal(self) -> Path:
        return self.root / "acl-journal.jsonl"

    def check(self) -> None:
        """Raise ``SandboxUnavailable``-shaped errors for an unusable root."""
        from nooa.runtime.sandbox.errors import SandboxUnavailable

        if os.name != "nt":
            raise SandboxUnavailable("the AppContainer backend requires Windows")
        if not self.python.exists():
            raise SandboxUnavailable(
                f"sandbox runtime root {self.root} has no python.exe. "
                "Build it once with `python experiments/windows-sandbox-probe/_mkroot.py`, "
                f"or point {_RUNTIME_ENV} at an existing root."
            )
        if not (self.package / "__init__.py").exists():
            raise SandboxUnavailable(
                f"sandbox runtime root {self.root} does not contain the nooa package. "
                "Run `python experiments/windows-sandbox-probe/_mkroot.py --sync-nooa`."
            )
        editable = list(self.site_packages.glob("_editable_impl_*.pth"))
        if editable:
            names = ", ".join(sorted(p.name for p in editable))
            raise SandboxUnavailable(
                f"sandbox runtime root {self.root} still carries editable install hooks "
                f"({names}); those resolve imports to the project checkout, which the "
                "container is not granted. Remove them (see `_mkroot.py --sync-nooa`)."
            )


def default_runtime_root() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / "nooa" / _SUBDIR


def resolve_root(explicit: str | os.PathLike[str] | None = None) -> RuntimeRoot:
    """Pick the runtime root: explicit argument, environment, then user cache."""
    chosen = explicit or os.environ.get(_RUNTIME_ENV) or default_runtime_root()
    return RuntimeRoot(Path(chosen).expanduser().resolve())


def _installed_package_dir() -> Path:
    """The ``nooa`` package this process imported, wherever it lives."""
    import nooa

    return Path(nooa.__file__).resolve().parent


def _tree_signature(root: Path) -> dict[str, int]:
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
                stat = os.stat(os.path.join(dirpath, name))
            except OSError:
                continue
            files += 1
            total += stat.st_size
            newest = max(newest, stat.st_mtime_ns)
    return {"files": files, "bytes": total, "newest_ns": newest}


def sync_package(root: RuntimeRoot, *, force: bool = False) -> str:
    """Mirror the running ``nooa`` package into the root and drop editable hooks.

    Copying is deliberately chosen over installing: it is sub-second for a
    ~4 MB package, needs no build backend or dependency resolution, and the
    source location is whatever this process imported — so an editable checkout
    and an installed wheel both work without special-casing.
    """
    from nooa.runtime.sandbox.errors import SandboxUnavailable

    source = _installed_package_dir()
    if not (source / "__init__.py").exists():
        raise SandboxUnavailable(f"cannot find the running nooa package at {source}")

    root.site_packages.mkdir(parents=True, exist_ok=True)
    for hook in root.site_packages.glob("_editable_impl_*.pth"):
        hook.unlink()

    marker = root.site_packages / ".nooa-sync.json"
    signature = _tree_signature(source)
    if not force and marker.exists():
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        if previous.get("signature") == signature:
            return f"unchanged ({signature['files']} files)"

    started = time.perf_counter()
    shutil.rmtree(root.package, ignore_errors=True)
    shutil.copytree(source, root.package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    elapsed = time.perf_counter() - started
    marker.write_text(
        json.dumps({"signature": signature, "source": str(source), "copied_at": time.time()}),
        encoding="utf-8",
    )
    return f"copied {signature['files']} files in {elapsed:.2f}s"


@dataclass(frozen=True)
class _Grant:
    path: str
    sid: str


class AclJournal:
    """Append-only record of grants, for recovery after a parent crash.

    A killed parent leaves its ACEs behind: DACLs are persistent objects, not
    process resources. The next start revokes every journaled entry before
    adding its own. Live teardown still uses ``REVOKE_ACCESS`` directly — the
    journal exists only for the paths teardown never got to run.
    """

    def __init__(self, path: Path):
        self._path = path
        self._entries: list[_Grant] = []

    def record(self, path: str, sid: str) -> None:
        grant = _Grant(str(path), sid)
        if grant in self._entries:
            return
        self._entries.append(grant)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"path": grant.path, "sid": grant.sid}) + "\n")
        except OSError:
            # The journal is a recovery aid; losing it must not break a session.
            pass

    def recover(self) -> list[str]:
        """Revoke everything a previous process left behind; return path list."""
        from nooa.runtime.sandbox import win_api

        if not self._path.exists():
            return []
        import ctypes

        revoked: list[str] = []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            try:
                entry = json.loads(line)
                path = entry["path"]
                sid_text = entry["sid"]
            except (ValueError, KeyError, TypeError):
                continue
            sid = ctypes.c_void_p()
            if not win_api._adv.ConvertStringSidToSidW(sid_text, ctypes.byref(sid)):
                continue
            try:
                if os.path.exists(path):
                    win_api.revoke(path, sid)
                    revoked.append(path)
            except OSError:
                # Path gone or ACL already clean; nothing to recover.
                pass
            finally:
                win_api._k32.LocalFree(sid)
        self.clear()
        return revoked

    def clear(self) -> None:
        self._entries.clear()
        try:
            self._path.unlink()
        except OSError:
            pass

    def pending(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        out = []
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            return []
        return out


def prepare_root(
    root: RuntimeRoot,
    *,
    sync: bool = True,
) -> str:
    """Validate the root, refresh the package if asked, recover stale ACEs.

    The path is already resolved by :func:`resolve_root`; accepting it again
    here would allow the two to disagree.
    """
    root.check()
    summary = sync_package(root) if sync else "skipped"
    recovered = AclJournal(root.journal).recover()
    if recovered:
        summary += f"; revoked {len(recovered)} stale grant(s)"
    return summary
