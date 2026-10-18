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

    @property
    def tokenizer_cache(self) -> Path:
        """Writable cache for tokenizer data (``TIKTOKEN_CACHE_DIR``).

        The container has no network and its interpreter tree is read-only, so
        anything an import chain fetches at runtime must be pre-populated here
        by the parent. ``tiktoken`` is the concrete case: importing the LLM
        stack pulls it in, and it otherwise downloads ``cl100k_base`` on first
        use — which fails inside the sandbox. See :func:`warm_tokenizer_cache`.
        """
        return self.root / "tokenizer-cache"

    def check(self, *, require_package: bool = True) -> None:
        """Raise ``SandboxUnavailable``-shaped errors for an unusable root.

        ``require_package=False`` checks only the interpreter. The package check
        must be skippable: ``--sync`` exists to *install* the package, so
        demanding it first would make the command refuse to fix exactly the
        state it was written for.
        """
        from nooa.runtime.sandbox.errors import SandboxUnavailable

        if os.name != "nt":
            raise SandboxUnavailable("the AppContainer backend requires Windows")
        if not self.python.exists():
            raise SandboxUnavailable(
                f"sandbox runtime root {self.root} has no python.exe. "
                "Build it once with "
                f"`python -m nooa.runtime.sandbox.win_runtime --build --root {self.root}`, "
                f"or point {_RUNTIME_ENV} at an existing root."
            )
        if not require_package:
            return
        if not (self.package / "__init__.py").exists():
            raise SandboxUnavailable(
                f"sandbox runtime root {self.root} does not contain the nooa package. "
                "Run `python -m nooa.runtime.sandbox.win_runtime --sync "
                f"--root {self.root}`."
            )
        editable = list(self.site_packages.glob("_editable_impl_*.pth"))
        if editable:
            names = ", ".join(sorted(p.name for p in editable))
            raise SandboxUnavailable(
                f"sandbox runtime root {self.root} still carries editable install hooks "
                f"({names}); those resolve imports to the project checkout, which the "
                "container is not granted. Remove them by refreshing the package: "
                "`python -m nooa.runtime.sandbox.win_runtime --sync "
                f"--root {self.root}`."
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


def _source_layout() -> tuple[Path, Path]:
    """``(interpreter prefix, site-packages)`` of the *current* environment.

    Derived from the running interpreter rather than hardcoded: ``sys.base_prefix``
    is the CPython install even inside a venv, and ``sysconfig`` resolves the
    venv's site-packages. A checkout-relative ``.venv`` path would be wrong for
    an installed package and for any other environment layout.
    """
    import site
    import sys
    import sysconfig

    base = Path(sys.base_prefix)
    candidates: list[Path] = []
    for key in ("purelib", "platlib"):
        try:
            candidates.append(Path(sysconfig.get_path(key)))
        except (KeyError, OSError):
            pass
    try:
        candidates.extend(Path(p) for p in site.getsitepackages())
    except Exception:  # noqa: BLE001 - site may be unavailable in odd embedders
        pass
    for candidate in candidates:
        if (candidate / "nooa").is_dir() or list(candidate.glob("*.dist-info")):
            return base, candidate
    if not candidates:
        raise RuntimeError("could not locate a site-packages directory")
    return base, candidates[0]


# Directories copied wholesale from the interpreter prefix; everything else at
# the top level is a file (python.exe, python313.dll, ...).
_COPY_DIRS = ("Lib", "DLLs", "libs", "include")
# stdlib subtrees no worker needs; each is tens of MB.
_SKIP_STDLIB = ("__pycache__", "test", "tests", "idlelib", "tkinter", "turtledemo")


def build_root(
    root: RuntimeRoot,
    *,
    interpreter_prefix: Path | None = None,
    site_packages: Path | None = None,
) -> str:
    """Copy interpreter + stdlib + site-packages into ``root``.

    This is stage one; :func:`sync_package` is stage two and must run after it.
    An existing root is removed first: a partial merge would produce an
    interpreter whose DLLs and stdlib come from different builds.
    """
    source_prefix, source_sp = (
        (Path(interpreter_prefix), Path(site_packages))
        if interpreter_prefix and site_packages
        else _source_layout()
    )
    if not (source_prefix / "python.exe").exists() and not (source_prefix / "python").exists():
        raise RuntimeError(f"{source_prefix} does not look like an interpreter prefix")
    if not source_sp.is_dir():
        raise RuntimeError(f"{source_sp} is not a directory")

    started = time.perf_counter()
    shutil.rmtree(root.root, ignore_errors=True)
    root.root.mkdir(parents=True, exist_ok=True)

    for name in os.listdir(source_prefix):
        source = source_prefix / name
        target = root.root / name
        if source.is_dir():
            if name in _COPY_DIRS:
                shutil.copytree(
                    source,
                    target,
                    symlinks=True,
                    ignore=shutil.ignore_patterns(*_SKIP_STDLIB),
                )
        else:
            shutil.copy2(source, target)

    root.site_packages.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_sp,
        root.site_packages,
        symlinks=True,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    elapsed = time.perf_counter() - started
    files = 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root.root):
        for filename in filenames:
            files += 1
            try:
                total += os.path.getsize(os.path.join(dirpath, filename))
            except OSError:
                pass
    return f"{root.root}: {files} files, {total / 1e6:.1f} MB, {elapsed:.1f}s"


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


def warm_tokenizer_cache(root: RuntimeRoot) -> str:
    """Populate the tokenizer cache the container will read.

    ``tiktoken`` fetches ``cl100k_base.tiktoken`` over HTTPS on first use and
    caches it under ``TIKTOKEN_CACHE_DIR`` (or the temp directory). The
    container has no network and cannot read the temp directory, so the parent
    downloads it once into a granted location. Without this, importing an agent
    module that pulls in the LLM stack fails inside the sandbox with a DNS
    error — which is what CI caught.

    Best-effort: a backend that never imports ``tiktoken`` must not fail to
    start because a tokenizer could not be fetched. The directory is created
    either way so the child's environment points somewhere valid.

    ``tiktoken.get_encoding`` is deliberately NOT used. It goes through a
    process-wide registry, so if the encoding was already constructed (importing
    the LLM stack does exactly that) it returns the cached object without
    touching the cache directory — the warm-up then reports success having
    written nothing, and the child still fails. Calling the plugin's
    constructor directly forces ``read_file_cached`` to run.
    """
    cache = root.tokenizer_cache
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"tokenizer cache unavailable ({exc})"

    previous = os.environ.get("TIKTOKEN_CACHE_DIR")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
    try:
        from tiktoken_ext import openai_public

        openai_public.cl100k_base()
    except ImportError:
        return "no tiktoken installed; nothing to warm"
    except Exception as exc:  # noqa: BLE001 - offline host, proxy, etc.
        return f"tokenizer warm-up failed ({type(exc).__name__}: {exc})"
    finally:
        if previous is None:
            os.environ.pop("TIKTOKEN_CACHE_DIR", None)
        else:
            os.environ["TIKTOKEN_CACHE_DIR"] = previous

    # Verify rather than trust: the whole point is that the child finds a file.
    written = list(cache.iterdir())
    if not written:
        return (
            f"tokenizer warm-up wrote nothing to {cache}; the container will "
            "fail to import the LLM stack"
        )
    return f"tokenizer cache ready at {cache} ({len(written)} file(s))"


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
    # The child has no network, so anything its import chain would fetch has to
    # exist before it starts. This is cheap when already warm.
    summary += f"; {warm_tokenizer_cache(root)}"
    return summary


def main(argv: list[str] | None = None) -> int:
    """Command line: build the root, refresh the package, or report on it.

    ``python -m nooa.runtime.sandbox.win_runtime --build``
        Create the runtime root from the running interpreter (one-time, ~4.5 min).
    ``--sync``
        Refresh only the ``nooa`` package inside an existing root (sub-second).
    ``--status``
        Say where the root is and whether it is usable.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m nooa.runtime.sandbox.win_runtime")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--build", action="store_true", help="create the runtime root")
    group.add_argument("--sync", action="store_true", help="refresh the nooa package only")
    group.add_argument("--status", action="store_true", help="report on the current root")
    parser.add_argument("--root", default=None, help="override the runtime root path")
    parser.add_argument("--force", action="store_true", help="with --sync, copy unconditionally")
    args = parser.parse_args(argv)

    root = resolve_root(args.root)

    if args.status:
        print(f"root: {root.root}")
        try:
            root.check()
        except Exception as exc:  # noqa: BLE001 - status reports, never raises
            print(f"usable: no ({exc})")
            return 1
        print("usable: yes")
        return 0

    if args.sync:
        # Only the interpreter must exist: installing the package is this
        # command's job, so requiring it here would be circular.
        root.check(require_package=False)
        print(f"nooa package: {sync_package(root, force=args.force)}")
        return 0

    if args.build:
        print(f"building {root.root} ...")
        print(build_root(root))
        print(f"nooa package: {sync_package(root, force=True)}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
