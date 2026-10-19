# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start the CodeAct worker inside an AppContainer.

Why this cannot reuse ``multiprocessing.Process``: on Windows, ``Popen`` calls
``_winapi.CreateProcess`` with no attribute list, and ``SECURITY_CAPABILITIES``
is a creation-time attribute. There is no way to enter an AppContainer after
the process exists, so the launcher has to create the process itself.

Everything *around* process creation is reused unchanged. The broker still
speaks ``multiprocessing.connection``: the parent creates a duplex pipe and
hands the child a duplicated, inheritable handle. ``DuplicateHandle`` does not
consult the pipe's DACL, so the container can use it without any grant — this
was measured (probe scenario 8), not assumed.

Guards map onto the two Windows mechanisms:

===============  ===================================================
guardrail        mechanism
===============  ===================================================
filesystem       AppContainer DACL grants written before creation
network          omission of capability SIDs (TCP and UDP)
memory           Job Object ``ProcessMemoryLimit``
cpu              Job Object ``PerJobUserTimeLimit``
timeout / kill   ``TerminateJobObject``
environment      allowlist (``CONTAINER_ENV`` + ``env_passthrough``)
===============  ===================================================

Memory keeps the Linux meaning: ``max_memory_mb`` is *headroom above the
worker's baseline*, not an absolute ceiling. The baseline is measured from the
job's peak usage after the worker reports ready, then the limit is set to
``baseline + headroom``.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any

from nooa.runtime.sandbox import wire
from nooa.runtime.sandbox.config import ResolvedSpec
from nooa.runtime.sandbox.errors import SandboxUnavailable
from nooa.runtime.sandbox.win_api import (
    FILE_ALL_ACCESS,
    HANDLE_FLAG_INHERIT,
    READ_EXECUTE,
    STILL_ACTIVE,
    WAIT_TIMEOUT,
    Child,
    Job,
    _k32,
    container_sid,
    grant,
    revoke,
    sid_from_string,
)
from nooa.runtime.sandbox.win_runtime import AclJournal, RuntimeRoot, default_container_name

__all__ = ["ContainerProcess", "start_container_worker"]

# 0xC000013A (STATUS_CONTROL_C_EXIT) is what a Job Object returns when it
# terminates a process for exceeding its time limit. The M0 probe measured it.
EXIT_JOB_TIME_LIMIT = 0xC000013A

_READY_TIMEOUT_S = 180.0
# The startup handshake is worker -> parent control traffic, not agent data:
# decode it with the default codec, never the agent-typed one.
_READY_CODEC = wire.Codec()

# What the container inherits from the parent's environment. Copying
# ``os.environ`` made every parent variable readable by cell code, credentials
# included (measured: the probe's sentinel crossed). The list is Windows' own
# session variables; anything a user or installer adds, which is where tokens
# live, stays in the parent unless named in ``env_passthrough``.
CONTAINER_ENV: tuple[str, ...] = (
    # Process basics. Without SYSTEMROOT, Winsock cannot initialise and
    # ``socket.socket()`` fails with WinError 10106 (measured).
    "SYSTEMROOT",
    "WINDIR",
    "SYSTEMDRIVE",
    "COMSPEC",
    "PATH",
    "PATHEXT",
    "OS",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER",
    "PROCESSOR_LEVEL",
    "PROCESSOR_REVISION",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "COMMONPROGRAMFILES",
    "COMMONPROGRAMFILES(X86)",
    "COMMONPROGRAMW6432",
    "ALLUSERSPROFILE",
    "PUBLIC",
    # Identity. Without USERPROFILE ``Path.home()`` raises RuntimeError (nooa's
    # ``get_user_dir`` uses it); without USERNAME ``getpass.getuser()`` raises
    # OSError. Both measured. These are names, not secrets.
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "USERNAME",
    "USERDOMAIN",
    "COMPUTERNAME",
    "APPDATA",
    # Inside the container these point at its own folder
    # (``%LOCALAPPDATA%\Packages\<name>\AC``), not at the parent's values, and
    # that is where ``tempfile`` lands. Measured with and without the allowlist.
    "LOCALAPPDATA",
    "TEMP",
    "TMP",
)


def _container_environment(passthrough: tuple[str, ...] = ()) -> dict[str, str]:
    """The container's starting environment: an allowlist, never a copy.

    Names match case-insensitively, as Windows does. A requested name the
    parent does not have is skipped rather than set empty.
    """
    env: dict[str, str] = {}
    for name in (*CONTAINER_ENV, *passthrough):
        value = os.environ.get(name)
        if value is not None:
            env[name.upper()] = value
    return env


def _agent_import_roots(init: dict[str, Any]) -> list[str]:
    """Directories that must be importable for the agent class to resolve.

    Linux ``spawn`` imports the agent module *before* installing guards, so the
    agent's directory only needs to be on the parent's ``sys.path``. An
    AppContainer is confined from process creation, which means the import
    happens inside the sandbox and every directory on that import chain needs a
    read grant.

    The grants stay as narrow as the blueprint allows: the package directory (or
    the single module's directory) for the agent's module, never its ancestors.
    """
    blueprint = init.get("blueprint")
    module_name = getattr(blueprint, "agent_module", None)
    if not module_name:
        return []
    import importlib.util

    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return []
    if spec is None:
        return []
    roots: list[str] = []
    if spec.submodule_search_locations:
        # A package: import it from the parent of its directory.
        for location in spec.submodule_search_locations:
            roots.append(os.path.dirname(os.path.abspath(location)))
    elif spec.origin and spec.origin not in ("built-in", "frozen"):
        roots.append(os.path.dirname(os.path.abspath(spec.origin)))
    return roots


def _crash_log_path(workspace: str) -> str:
    """Where the container worker appends a startup traceback.

    Must live under a granted path (the runtime root is read-only), and a fresh
    name per launch is not needed: the launcher truncates it before spawning, so
    a stale traceback cannot be mistaken for the current failure.
    """
    return os.path.join(workspace, "nooa-worker-crash.txt")


def _wait(handle: int, timeout_ms: int) -> int:
    return _k32.WaitForSingleObject(handle, timeout_ms)


class ContainerProcess:
    """The subset of ``multiprocessing.Process`` the executor actually uses.

    Deliberately not a ``BaseProcess`` subclass: inheritance would drag in
    ``mp``'s own creation path, which cannot enter a container.
    """

    def __init__(
        self,
        child: Child,
        job: Job,
        journal: AclJournal,
        granted: list[str],
        container_name: str,
        sid_text: str,
    ):
        self._child = child
        self._job = job
        self._journal = journal
        self._granted = granted
        # Revoking with the wrong SID silently leaves the grants in place, so
        # the identity this process was created under has to be carried, not
        # assumed. The text form is the durable one: it still revokes after the
        # profile is gone, which re-deriving the SID from the name cannot.
        self._container_name = container_name
        self._sid_text = sid_text
        self.pid = child.pid
        self._released = False
        # The child's own report of what it sees before building the namespace.
        # Empty until the child sends it; quoted in startup failures.
        self.startup_report: dict[str, Any] = {}

    @property
    def container_name(self) -> str:
        """The profile this worker runs under; unique to its session."""
        return self._container_name

    # --- liveness ----------------------------------------------------------
    def is_alive(self) -> bool:
        if self._released:
            return False
        return _wait(self._child.pi.hProcess, 0) == WAIT_TIMEOUT

    @property
    def exitcode(self) -> int | None:
        if self._child.pi.hProcess is None:
            return None
        code = ctypes.c_uint32()
        if not _k32.GetExitCodeProcess(self._child.pi.hProcess, ctypes.byref(code)):
            return None
        return None if code.value == STILL_ACTIVE else int(code.value)

    # --- termination -------------------------------------------------------
    def terminate(self) -> None:
        """Stop the worker process. Descendants survive this one."""
        if self._child.pi.hProcess is not None:
            _k32.TerminateProcess(self._child.pi.hProcess, 1)

    def kill(self) -> None:
        """Kill the whole job, including any subprocess the cell started."""
        self._job.terminate(1)

    def join(self, timeout: float | None = None) -> None:
        ms = 0xFFFFFFFF if timeout is None else int(max(0.0, timeout) * 1000)
        _wait(self._child.pi.hProcess, ms)

    # --- teardown ----------------------------------------------------------
    def release(self) -> None:
        """Revoke the ACLs and close every handle this launcher owns.

        ``REVOKE_ACCESS`` is the teardown path, not SDDL replay: a second
        ``GRANT_ACCESS`` merges into the existing ACE for the same SID, and an
        inheriting directory recomputes its ``AI`` bit on every write.
        """
        if self._released:
            return
        self._released = True
        failed = 0
        try:
            # Recorded when the grants were written, not re-derived from the
            # profile name: a profile that disappeared before teardown would
            # otherwise leave every ACE in place, silently and with nothing
            # left that knows they exist.
            sid = sid_from_string(self._sid_text)
            if sid is None:
                # Unparseable SID text. The journal still holds it verbatim, so
                # keep the record and let recovery try rather than dropping it.
                failed = len(self._granted)
            else:
                for path in reversed(self._granted):
                    try:
                        revoke(path, sid)
                    except OSError:
                        failed += 1
                _k32.LocalFree(sid)
        finally:
            self._granted = []
            # A revoke that failed leaves a persistent ACE behind. Keep the
            # journal then — it is the only thing that knows the path and SID —
            # and the next start reclaims it. Otherwise the record dies with
            # the session and nothing can ever remove the grant.
            self._journal.clear(keep=bool(failed))
            self._job.close()
            self._child.close()
            # Only when nothing was left behind: the SID is what a later
            # recovery would need, and deleting the profile strands any ACE
            # that still names it.
            if not failed:
                _delete_profile_quiet(self._container_name)


def _delete_profile_quiet(name: str) -> None:
    """Remove a session's AppContainer profile. Never raises."""
    from nooa.runtime.sandbox.win_api import delete_profile

    try:
        delete_profile(name)
    except OSError:
        # Leaking a profile is survivable; failing teardown is not. Recovery
        # cannot clean it either, since it is not part of a DACL.
        pass


@dataclass
class ContainerHandles:
    proc: ContainerProcess
    conn: Any


def start_container_worker(
    init: dict[str, Any],
    spec: ResolvedSpec,
    root: RuntimeRoot,
    *,
    container_name: str | None = None,
    workspace: str | None = None,
    env_passthrough: tuple[str, ...] = (),
    ready_timeout_s: float = _READY_TIMEOUT_S,
) -> ContainerHandles:
    """Create the container, grant, launch suspended, assign, resume, handshake.

    ``init`` is sent as the first pipe message rather than a file: the pipe is
    the only channel the container is guaranteed to have, and the payload is
    already picklable by construction.
    """
    from nooa.runtime.sandbox.win_api import create_profile

    # One profile per session: the name determines the SID, and a shared SID
    # means one session's teardown revokes the other's grants.
    name = container_name or default_container_name()
    root.check()
    create_profile(name)
    app_sid = container_sid(name)
    sid_text = _sid_text(app_sid)

    # Opened before the first grant, so every recorded ACE has an owner that can
    # be proved dead before it is reclaimed. Refuses if another live session
    # already holds this journal — that would mean two sessions share a name.
    try:
        journal = AclJournal.open(root.journal_for(name), sid_text)
    except BlockingIOError:
        # The profile belongs to the session that holds the journal: it is still
        # running and still needs its SID, so it must not be deleted here.
        _k32.LocalFree(app_sid)
        raise SandboxUnavailable(
            f"container profile {name!r} is already in use by a live session: "
            "SandboxConfig.container_name must be unique among concurrent sessions"
        ) from None
    granted: list[str] = []

    def grant_path(path: str, access: int) -> None:
        if not os.path.exists(path):
            return
        grant(path, app_sid, access)
        granted.append(path)
        journal.record(path, sid_text)

    try:
        # 1. The runtime root: read+execute, never write. The worker may import
        #    from it but must not be able to rewrite its own interpreter's code.
        grant_path(str(root.root), READ_EXECUTE)
        # 2. User-requested paths. The interpreter/venv paths in the spec are
        #    deliberately ignored: on Windows "system paths" means the private
        #    runtime root granted above, not the parent's interpreter.
        for rule in spec.path_grants:
            if not rule.required:
                continue
            if not os.path.exists(rule.path):
                raise SandboxUnavailable(f"sandbox allow path does not exist: {rule.path}")
            grant_path(rule.path, FILE_ALL_ACCESS if rule.write else READ_EXECUTE)
        # 3. Tokenizer data. The container has no network; importing the LLM
        #    stack makes tiktoken look for cl100k_base and download it when the
        #    cache is missing. The parent warmed this directory in prepare_root,
        #    which means the files already exist when the grant is written.
        #
        #    Granting the directory alone is not enough: an inheritable ACE is
        #    not reliably pushed onto pre-existing children, and tiktoken checks
        #    ``os.path.exists`` — which returns *False* rather than raising when
        #    stat is denied. The failure is therefore silent and looks like a
        #    missing cache. Grant the files themselves.
        tokenizer_cache = root.tokenizer_cache
        grant_path(str(tokenizer_cache), READ_EXECUTE)
        if tokenizer_cache.is_dir():
            for entry in tokenizer_cache.iterdir():
                if entry.is_file():
                    grant_path(str(entry), READ_EXECUTE)
        # 4. The agent's own module. The worker imports it by name inside the
        #    container, which is confined from creation — unlike Linux spawn,
        #    where that import happens before the guards go on. Without this the
        #    blueprint fails with "cannot resolve X in 'mymodule'".
        import_roots = _agent_import_roots(init)
        for import_root in import_roots:
            grant_path(import_root, READ_EXECUTE)
        # 5. The workspace last, so it is revoked first on teardown.
        if workspace:
            grant_path(str(workspace), FILE_ALL_ACCESS)

        # The child runs with ``-I``, which keeps cwd off sys.path and ignores
        # PYTHONPATH, so the granted roots have to be named explicitly.
        if import_roots:
            init = {**init, "import_roots": import_roots}
        # The tokenizer path travels on argv, not here: it is set before any
        # import, and a value that must be correct before the first import
        # cannot depend on a channel whose delivery is only observable after
        # the fact.
        child_env = _container_environment(env_passthrough)
        # nooa's config discovery otherwise walks up from the package to the
        # nearest pyproject.toml, and one the container cannot read makes
        # ``Path.exists()`` raise: with the root inside a checkout, every agent
        # built on nooa's helpers fails to import. Point discovery at a
        # location under the read-only root that does not exist, so reads find
        # no settings and writes fail like any other ungranted path (measured:
        # importing the helpers writes nothing). A passed-through value wins.
        absent = root.root / "no-nooa-config"
        child_env.setdefault("NEMO_OO_PROJECT_DIR", str(absent / "project"))
        child_env.setdefault("NEMO_OO_USER_DIR", str(absent / "user"))

        parent_conn, child_conn = mp.Pipe(duplex=True)
        handle = child_conn.fileno()
        # HANDLE_LIST accepts inheritable handles only; a non-inheritable one
        # makes CreateProcessW fail with ERROR_INVALID_PARAMETER (87).
        if not _k32.SetHandleInformation(
            ctypes.c_void_p(handle), HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT
        ):
            raise SandboxUnavailable("could not mark the broker pipe inheritable")

        argv = [
            str(root.python),
            "-I",  # isolated: no cwd on sys.path, no PYTHON* env, no user site
            "-m",
            "nooa.runtime.sandbox.win_worker",
            str(handle),
            name,
            # Positional from here on: the child indexes them directly, so every
            # slot must be present even when empty.
            str(_crash_log_path(workspace)) if workspace else "",
            str(tokenizer_cache),
        ]
        job = Job()
        try:
            if workspace:
                try:
                    os.remove(_crash_log_path(workspace))
                except OSError:
                    pass
            child = _spawn(
                argv,
                name,
                handle,
                workspace or str(root.root),
                env=child_env,
                # network=False is the absence of capability SIDs; network=True
                # has to grant one, or the container still has no network.
                capabilities=() if spec.block_network else ("internetClient",),
            )
        except BaseException:
            parent_conn.close()
            child_conn.close()
            job.close()
            raise
        job.assign(child.pi.hProcess)
        child.resume()
        child_conn.close()

        proc = ContainerProcess(child, job, journal, granted, name, sid_text)
        try:
            _handshake(parent_conn, init, proc, spec, ready_timeout_s)
        except BaseException:
            proc.terminate()
            proc.join(timeout=2.0)
            proc.release()
            parent_conn.close()
            raise
        return ContainerHandles(proc=proc, conn=parent_conn)
    except BaseException:
        # Any failure before the process is handed to the caller must not leak
        # ACEs: they are persistent, and a retry would otherwise stack them.
        failed = 0
        for path in reversed(granted):
            try:
                revoke(path, app_sid)
            except OSError:
                failed += 1
        # Same rule as ``ContainerProcess.release``: a grant that could not be
        # revoked keeps its journal entry and its SID, so the next start can
        # finish the job. Everything else is removed now.
        journal.clear(keep=bool(failed))
        if not failed:
            _delete_profile_quiet(name)
        raise
    finally:
        # ``container_sid`` returns a LocalAlloc'd SID; both the success and the
        # failure path must free it. ``ContainerProcess.release`` re-derives its
        # own copy, so nothing downstream depends on this handle staying alive.
        _k32.LocalFree(app_sid)


def _spawn(
    argv: list[str],
    container_name: str,
    handle: int,
    cwd: str,
    env: dict[str, str] | None = None,
    capabilities: tuple[str, ...] = (),
) -> Child:
    from nooa.runtime.sandbox.win_api import spawn_in_container

    return spawn_in_container(
        argv,
        container_name=container_name,
        capabilities=capabilities,
        inherit_handles=(handle,),
        cwd=cwd,
        env=env,
    )


def _handshake(
    conn: Any,
    init: dict[str, Any],
    proc: ContainerProcess,
    spec: ResolvedSpec,
    timeout_s: float,
) -> None:
    """Send init, wait for ``ready``, then apply the memory limit.

    The memory ceiling must be set *after* the baseline is known, otherwise a
    cap expressed as headroom would be read as an absolute ceiling and kill a
    worker that has not even imported the agent module yet.
    """
    conn.send(init)
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SandboxUnavailable(f"sandbox worker did not report ready within {timeout_s:.0f}s")
        if not conn.poll(min(remaining, 1.0)):
            if not proc.is_alive():
                raise SandboxUnavailable(
                    f"sandbox worker exited during startup (exit code {proc.exitcode})"
                )
            continue
        try:
            # Never ``conn.recv()``: that unpickles worker bytes. The ready
            # message carries only a string and an int, so the default codec
            # (no agent types) is the right one here.
            message = _READY_CODEC.loads(conn.recv_bytes())
            if not isinstance(message, dict):
                raise SandboxUnavailable("sandbox worker sent a malformed ready message")
        except (EOFError, OSError) as exc:
            raise SandboxUnavailable(f"sandbox worker pipe closed during startup: {exc}") from exc
        kind = message.get("type")
        if kind == "startup":
            # The child's view of its own preconditions. Kept on the process so
            # a later failure can quote it; the child cannot put it in a file the
            # parent is guaranteed to find (tests use a tmp_path workspace).
            proc.startup_report = message
            continue
        if kind == "ready":
            baseline = int(message.get("baseline_bytes") or 0)
            memory_mb = 0
            if spec.max_memory_mb:
                overhead = int(baseline / (1024 * 1024))
                memory_mb = overhead + int(spec.max_memory_mb)
            proc._job.set_limits(memory_mb=memory_mb, cpu_seconds=spec.max_cpu_seconds)
            return
        if kind == "fatal":
            raise SandboxUnavailable(
                f"sandbox worker failed to start: {message.get('error')} "
                f"[child saw: {proc.startup_report}]"
            )


def _sid_text(sid: ctypes.c_void_p) -> str:
    from nooa.runtime.sandbox.win_api import sid_string

    return sid_string(sid)
