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
* **the ACL journal** — one per session, which grants that session wrote, so a
  crashed parent's ACEs can be revoked on the next start. The journal is for
  recovery only; live teardown always uses ``REVOKE_ACCESS``.

The journal is also the session's liveness marker. Its owning process holds an
exclusive lock on it for as long as it runs; the kernel drops the lock when the
process dies. A start that finds an unlocked journal therefore knows the session
is gone and may revoke its grants — a start that finds a locked one must leave
it alone. Sharing a journal (or a container name) across sessions is what made
two concurrent sessions revoke each other's grants.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nooa._file_lock import lock_file

__all__ = [
    "AclJournal",
    "RuntimeRoot",
    "default_container_name",
    "default_runtime_root",
    "compile_bytecode",
    "prepare_root",
    "recover_stale_grants",
    "resolve_root",
    "sync_package",
]

# Retained for callers that name the profile explicitly; a session that does not
# choose a name gets a unique one from :func:`default_container_name`.
DEFAULT_CONTAINER_NAME = "nooa-sandbox"
_RUNTIME_ENV = "NOOA_SANDBOX_RUNTIME"
_SUBDIR = "sandbox-runtime"
_JOURNAL_STEM = "acl-journal"
_JOURNAL_SUFFIX = ".jsonl"


def default_container_name() -> str:
    """A container profile name unique to one session.

    AppContainer profiles are host-global: the SID is derived from the name, so
    two sessions sharing a name share an identity, and every ACE written for one
    is usable by the other. A per-session name is what makes the grants a
    session writes mean "this session" rather than "whoever ran last".
    """
    return f"{DEFAULT_CONTAINER_NAME}-{uuid.uuid4().hex[:12]}"


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
        """The pre-session journal path, kept only so recovery can reclaim it."""
        return self.root / f"{_JOURNAL_STEM}{_JOURNAL_SUFFIX}"

    def journal_for(self, container_name: str) -> Path:
        """The journal one session owns: ``acl-journal-<container>.jsonl``.

        Keyed by container name rather than by PID: the name is what identifies
        the SID whose ACEs the journal describes, so recovery can revoke with
        the same identity that wrote them even after the writer is gone.
        """
        return self.root / f"{_JOURNAL_STEM}-{container_name}{_JOURNAL_SUFFIX}"

    def journals(self) -> list[Path]:
        """Every journal in the root, current per-session ones and the legacy one."""
        return sorted(self.root.glob(f"{_JOURNAL_STEM}*{_JOURNAL_SUFFIX}"))

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
    compiled = compile_bytecode(root)
    return f"{root.root}: {files} files, {total / 1e6:.1f} MB, {elapsed:.1f}s; {compiled}"


def compile_bytecode(
    root: RuntimeRoot,
    *,
    only: tuple[Path, ...] | None = None,
) -> str:
    """Pre-compile the packages a worker imports, with the root's interpreter.

    The copy above deliberately drops ``__pycache__``, so without this every
    container start recompiles several thousand modules before the first cell
    can run. Doing it once at build time moves that cost off the startup path.

    The root's own interpreter does the work, not the caller's: a ``.pyc`` is
    only reusable by the interpreter whose magic number wrote it, and the root
    is a copy of a possibly different CPython.

    ``only`` limits the work to specific directories — :func:`sync_package`
    uses it to recompile just the packages it re-copied, rather than the whole
    tree on every start.

    Best-effort — a compile failure costs startup time, never correctness,
    because the interpreter compiles on demand anyway.
    """
    import subprocess

    if not root.python.exists():
        return "bytecode not compiled: no interpreter in the root"

    targets = list(only) if only else [root.site_packages, root.root / "Lib"]
    # ``-j 0`` uses every core. stdin is DEVNULL for the same reason as the
    # tokenizer warm-up: an ACP host holds this process's stdin open, and a
    # child that inherits it blocks in process creation.
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [
                str(root.python),
                "-I",
                "-m",
                "compileall",
                "-q",
                "-j",
                "0",
                *(str(path) for path in targets),
            ],
            capture_output=True,
            text=True,
            timeout=900,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"bytecode compile failed ({type(exc).__name__}: {exc})"

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit {completed.returncode}"
        return f"bytecode compile failed ({tail})"
    return f"bytecode compiled in {time.perf_counter() - started:.1f}s"


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


def _workspace_packages() -> dict[str, Path]:
    """The workspace distributions this process imported, by import name.

    A spawned worker rebuilds the agent class by importing its module, so the
    container needs whatever module the agent lives in — not only ``nooa``. An
    agent defined in ``nooa_cli`` (the coding agent) or in a host package fails
    with ModuleNotFoundError otherwise.

    "Workspace" is the operative word. ``build_root`` already copies the whole
    of the source environment's site-packages and stdlib into the root, so a
    third-party package or a stdlib module found there is *already* present;
    mirroring it again copies thousands of files for nothing. What the root
    cannot have is an editable install, because its source lives in the
    checkout rather than in site-packages. Those are exactly the packages
    whose path falls outside every interpreter directory.

    An earlier version returned every imported top-level package. In a plain
    executor that is two packages; in an ACP process, which imports litellm,
    openai, pandas and numpy, it was sixty — and every container start copied
    all of them.
    """
    import site
    import sys
    import sysconfig

    excluded: list[Path] = []
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        try:
            excluded.append(Path(sysconfig.get_path(key)).resolve())
        except (KeyError, OSError):
            pass
    try:
        excluded.extend(Path(p).resolve() for p in site.getsitepackages())
    except Exception:  # noqa: BLE001 - site may be unavailable in odd embedders
        pass
    excluded.append(Path(sys.base_prefix).resolve())
    excluded.append(Path(sys.prefix).resolve())

    def inside_interpreter(path: Path) -> bool:
        return any(path == base or base in path.parents for base in excluded)

    found: dict[str, Path] = {}
    for name, module in list(sys.modules.items()):
        if "." in name or module is None:
            continue
        location = getattr(module, "__file__", None)
        if not location:
            continue
        path = Path(location).resolve()
        if path.name != "__init__.py" or not path.parent.is_dir():
            continue
        if inside_interpreter(path.parent):
            continue
        found[name] = path.parent
    return found


def sync_package(
    root: RuntimeRoot,
    *,
    force: bool = False,
    packages: dict[str, Path] | None = None,
) -> str:
    """Mirror the running nooa package into the root and drop editable hooks.

    Copying is deliberately chosen over installing: it is sub-second for a
    ~4 MB package, needs no build backend or dependency resolution, and the
    source location is whatever this process imported — so an editable checkout
    and an installed wheel both work without special-casing.

    ``packages`` names additional importable distributions to mirror alongside
    ``nooa`` (see :func:`_workspace_packages`). Each gets its own signature, so
    a change to one does not force a re-copy of the other.
    """
    from nooa.runtime.sandbox.errors import SandboxUnavailable

    source = _installed_package_dir()
    if not (source / "__init__.py").exists():
        raise SandboxUnavailable(f"cannot find the running nooa package at {source}")

    root.site_packages.mkdir(parents=True, exist_ok=True)
    for hook in root.site_packages.glob("_editable_impl_*.pth"):
        hook.unlink()

    wanted: dict[str, Path] = {"nooa": source}
    for name, path in (packages or {}).items():
        if name in wanted or not (path / "__init__.py").exists():
            continue
        wanted[name] = path

    marker = root.site_packages / ".nooa-sync.json"
    signatures = {name: _tree_signature(path) for name, path in wanted.items()}
    if not force and marker.exists():
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        if previous.get("signatures") == signatures:
            files = sum(sig["files"] for sig in signatures.values())
            return f"unchanged ({files} files in {len(signatures)} package(s))"

    started = time.perf_counter()
    for name, path in wanted.items():
        target = root.site_packages / name
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(path, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # Re-copying drops the bytecode build_root compiled, and these packages
        # are the ones the worker imports first. Compile just what changed.
        compile_bytecode(root, only=(target,))
    elapsed = time.perf_counter() - started
    files = sum(sig["files"] for sig in signatures.values())
    marker.write_text(
        json.dumps(
            {
                "signatures": signatures,
                "sources": {name: str(path) for name, path in wanted.items()},
                "copied_at": time.time(),
            }
        ),
        encoding="utf-8",
    )
    return f"copied {files} files in {len(wanted)} package(s) in {elapsed:.2f}s"


@dataclass(frozen=True)
class _Grant:
    path: str
    sid: str


class AclJournal:
    """Append-only record of one session's grants, for recovery after a crash.

    A killed parent leaves its ACEs behind: DACLs are persistent objects, not
    process resources. Live teardown uses ``REVOKE_ACCESS`` directly — the
    journal exists only for the paths teardown never got to run.

    Each journal belongs to exactly one session, and its lock is that session's
    liveness marker. :meth:`open` takes the lock and fails if another live
    process holds it; :func:`recover_stale_grants` only reclaims journals whose
    lock can be taken, which is exactly the ones whose owner died.
    """

    def __init__(self, path: Path, sid: str | None = None):
        self._path = path
        self._sid = sid
        self._entries: list[_Grant] = []
        self._lock_fd: int | None = None

    @property
    def path(self) -> Path:
        return self._path

    @classmethod
    def open(cls, path: Path, sid: str) -> AclJournal:
        """Create the journal and hold its lock until :meth:`clear`.

        ``sid`` is the container SID this session's grants name; it is written
        with every entry so recovery never has to re-derive it (a profile
        deleted between crash and recovery would otherwise strand the ACEs).

        A journal that is already locked belongs to a live session. That is a
        caller bug — two sessions were given the same identity — and refusing is
        the only safe answer: continuing would let this session revoke or
        overwrite another's recovery record.
        """
        journal = cls(path, sid)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        except OSError:
            # The journal is a recovery aid; losing it must not break a session.
            return journal
        try:
            lock_file(fd, blocking=False)
        except BlockingIOError:
            os.close(fd)
            raise
        except OSError:
            # Locking unsupported here: keep the journal, skip the liveness
            # marker. Recovery will treat it as stale, which is the old
            # behaviour, not a regression.
            journal._lock_fd = fd
            return journal
        journal._lock_fd = fd
        return journal

    def record(self, path: str, sid: str | None = None) -> None:
        grant = _Grant(str(path), sid or self._sid or "")
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

    def clear(self, *, keep: bool = False) -> None:
        """Release the lock, and delete this session's journal unless ``keep``.

        The handle is closed first: Windows refuses to delete a file another
        handle has open, so unlinking while holding the lock leaves the file
        behind. Teardown has already revoked the ACEs by this point, so the
        brief window without the lock exposes nothing.

        ``keep=True`` is for a teardown that could not revoke everything: the
        lock must still be released (or the file would look live forever), but
        the entries are the only record of ACEs that outlived the session, and
        the next start reclaims them.
        """
        self._entries.clear()
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None
        if keep:
            return
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


def _read_entries(path: Path) -> list[dict[str, Any]]:
    """Parse a journal, skipping lines a crash truncated mid-write."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries: list[dict[str, Any]] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _revoke_entries(entries: list[dict[str, Any]]) -> tuple[list[str], int]:
    """Revoke every ACE a journal lists. Assumes the journal's owner is gone.

    Returns ``(revoked paths, failures)``. A failure means the ACE may still be
    there, so the caller keeps the journal rather than losing the only record
    of a grant nothing would ever revoke again.

    Each entry is checked before it is revoked. ``REVOKE_ACCESS`` on an
    inheriting directory rewrites the DACL of every child, which is ~5 s for a
    runtime root of 22k files even when the SID is already absent; reading the
    DACL first is milliseconds. Without the check, a journal left behind by an
    earlier crash costs that much on every single start.
    """
    import ctypes

    from nooa.runtime.sandbox import win_api

    revoked: list[str] = []
    failed = 0
    for entry in entries:
        try:
            target = entry["path"]
            sid_text = entry["sid"]
        except (KeyError, TypeError):
            continue
        sid = ctypes.c_void_p()
        if not win_api._adv.ConvertStringSidToSidW(sid_text, ctypes.byref(sid)):
            failed += 1
            continue
        try:
            if not os.path.exists(target):
                continue
            try:
                if sid_text not in win_api.read_dacl(target):
                    continue
            except OSError:
                # The read is an optimisation, so an unreadable DACL falls
                # through to revoke and lets it report. A path that vanished
                # between the check and the read is not that case: nothing is
                # left to revoke, and counting it as a failure would keep the
                # journal alive forever.
                if not os.path.exists(target):
                    continue
            win_api.revoke(target, sid)
            revoked.append(target)
        except OSError:
            failed += 1
        finally:
            win_api._k32.LocalFree(sid)
    return revoked, failed


def recover_stale_grants(root: RuntimeRoot) -> list[str]:
    """Revoke the ACEs of every journal whose owning session is dead.

    Liveness is the journal's lock, not its existence: a live session holds it
    for as long as it runs, and the kernel drops it when the process exits
    however it exits. A journal that can be locked has no live owner, so its
    grants are leftovers and safe to revoke.

    A journal that cannot be locked is left completely alone — including its
    file — so one session's startup can never destroy another's recovery record.
    """
    revoked: list[str] = []
    for path in root.journals():
        try:
            fd = os.open(path, os.O_RDWR)
        except OSError:
            continue
        try:
            try:
                lock_file(fd, blocking=False)
            except (BlockingIOError, OSError):
                # Locked by a live session, or locking unsupported. Either way
                # this journal is not ours to touch.
                continue
            paths, failed = _revoke_entries(_read_entries(path))
            revoked.extend(paths)
            # Keep the journal until every grant it lists is really gone: a
            # failed revoke would otherwise lose the only record of a permanent
            # ACE, with nothing left that could ever remove it.
            if failed:
                continue
        finally:
            # Close before unlinking. Windows refuses to delete a file that any
            # handle still has open — including this one — so an unlink while
            # holding the lock fails with WinError 32 and leaves the journal
            # behind to be recovered again on every start. The lock is dropped
            # here; nothing can race in, because the file is deleted next.
            os.close(fd)
        try:
            path.unlink()
        except OSError:
            pass
    return revoked


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

    The download is performed by the **runtime root's own interpreter**, not by
    the process calling this function. Content is validated against an
    ``expected_hash`` compiled into ``tiktoken_ext``, and the filename is only
    a hash of the blob URL — so a parent and a root carrying different
    ``tiktoken`` versions produce a cache entry that the child reads, fails to
    validate, deletes, and re-downloads (which it cannot, having no network).
    Warming with the reader's interpreter makes that mismatch impossible by
    construction.
    """
    cache = root.tokenizer_cache
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"tokenizer cache unavailable ({exc})"

    if not root.python.exists():
        return f"tokenizer cache not warmed: {root.python} is missing"

    import subprocess

    env = {**os.environ, "TIKTOKEN_CACHE_DIR": str(cache)}
    try:
        completed = subprocess.run(
            [
                str(root.python),
                "-I",
                "-c",
                "from tiktoken_ext import openai_public; openai_public.cl100k_base()",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            # The child must not inherit the parent's stdin. An ACP host (Zed,
            # an editor) gives this process a pipe as stdin and keeps it open
            # for the life of the session; the first child launched with that
            # handle inherited blocks in process creation until the 180 s
            # timeout expires. Measured: the same launch takes 0.07 s with
            # DEVNULL. The warm-up never reads stdin, so nothing is lost.
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"tokenizer warm-up failed ({type(exc).__name__}: {exc})"

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit {completed.returncode}"
        return f"tokenizer warm-up failed ({tail})"

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
    packages: dict[str, Path] | None = None,
) -> str:
    """Validate the root, refresh the package if asked, recover stale ACEs.

    The path is already resolved by :func:`resolve_root`; accepting it again
    here would allow the two to disagree.
    """
    root.check()
    summary = sync_package(root, packages=packages) if sync else "skipped"
    # Only dead sessions' grants: a live session's journal is locked, so its
    # ACEs are untouched no matter how many sessions start concurrently.
    recovered = recover_stale_grants(root)
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
