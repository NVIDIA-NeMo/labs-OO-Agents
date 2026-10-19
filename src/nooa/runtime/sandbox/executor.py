# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parent-side process backend for sandboxed cell execution.

:class:`SandboxedExecutor` owns one locked-down worker process, runs cells in it
with a parent-enforced hard timeout, brokers ``self.*`` calls back to the live
agent, and terminates/restarts the worker on timeout, CPU kill, or crash. It is
created per CodeAct session and reused across the session's cells so the worker's
REPL namespace persists.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as futures
import logging
import multiprocessing as mp
import os
import signal
import time
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

from nooa.errors.formatting import _hard_bound_text
from nooa.events import ExecutionResult
from nooa.runtime.sandbox import wire
from nooa.runtime.sandbox.config import ResolvedSpec, SandboxConfig, resolve_spec
from nooa.runtime.sandbox.errors import (
    CellMemoryError,
    CellSerializationError,
    CellTimeoutError,
    SandboxUnavailable,
    WorkerDiedError,
)
from nooa.runtime.sandbox.guards import Capabilities, probe_capabilities
from nooa.runtime.sandbox.serialization import (
    ResultDTO,
    dto_from_wire,
    dto_to_result,
    effective_error_limit,
    is_picklable,
)
from nooa.runtime.sandbox.worker import worker_main

logger = logging.getLogger(__name__)


# Broker responses cross a pickle pipe. Keep all parent-generated diagnostics
# primitive and bounded before sending them to an untrusted worker. The other
# direction is the security boundary: worker bytes are decoded as msgpack by
# ``wire.Codec``, never unpickled.


def _bounded_text(value: object, fallback: str, *, limit: int) -> str:
    try:
        text = str(value)
    except BaseException:
        text = fallback
    return _hard_bound_text(text, limit)


# Keyed by ``SandboxConfig.runtime_root``: on Windows the answer depends on
# which runtime root a config names, so one cached answer per host would let
# a config that names a working root inherit another config's failure (or the
# reverse). ``None`` is the root resolved from the environment.
_CAPS_CACHE: dict[str | None, Capabilities] = {}


def _is_container_proc(proc: Any) -> bool:
    """True for a ``ContainerProcess`` (checked by duck type, not import)."""
    return hasattr(proc, "release")


def _release_worker(proc: Any) -> None:
    """Revoke a container worker's ACLs and close its job/handles.

    A no-op for ``multiprocessing`` processes, which own no OS resources the
    parent must undo. Kept untyped so the fork/spawn paths stay importable on
    platforms without the Win32 launcher.
    """
    release = getattr(proc, "release", None)
    if release is not None:
        try:
            release()
        except Exception:  # noqa: BLE001 - teardown must not raise
            logger.debug("container worker release failed", exc_info=True)


def _capabilities(runtime_root: str | None = None) -> Capabilities:
    if runtime_root not in _CAPS_CACHE:
        _CAPS_CACHE[runtime_root] = probe_capabilities(runtime_root)
    return _CAPS_CACHE[runtime_root]


def check_enforceable(config: SandboxConfig, caps: Capabilities | None = None) -> list[str]:
    """Return the list of requested-but-unenforceable guardrails on this host.

    Each guardrail is reported independently: naming only the mechanism this
    host lacks makes the diagnosis actionable instead of a blanket "not Linux".
    A config that requests nothing returns an empty list even if the host could
    enforce nothing.
    """
    caps = caps or _capabilities(config.runtime_root)
    missing: list[str] = []
    # The probe records why a mechanism is unavailable (missing runtime root,
    # unsupported kernel, ...). That reason is the actionable part; without it
    # the user only learns that *something* is missing.
    reason = caps.detail.get("reason") if caps.detail else None

    if config.backend == "landlock" and caps.filesystem != "landlock":
        missing.append("filesystem (landlock requested but unavailable)")
    if config.backend == "appcontainer":
        if caps.filesystem != "appcontainer":
            missing.append("filesystem (appcontainer requested but unavailable)")
        if not config.network and caps.network != "appcontainer":
            missing.append("network isolation (appcontainer requested but unavailable)")

    if config.filesystem and caps.filesystem is None:
        missing.append("filesystem (no enforcement mechanism available)")
    if not config.network and caps.network is None:
        missing.append("network isolation (no enforcement mechanism available)")
    if (config.max_memory_mb or config.max_cpu_seconds) and (
        caps.memory is None or caps.cpu is None
    ):
        missing.append("memory/cpu caps (no enforcement mechanism available)")
    if missing and reason:
        missing.append(f"reason: {reason}")
    return missing


class SandboxedExecutor:
    """Run CodeAct cells in a guarded worker process with a hard timeout."""

    def __init__(
        self,
        agent: Any,
        config: SandboxConfig,
        *,
        cell_timeout: float | None,
        framework_builtins: dict[str, Any] | None = None,
        restrictions: Any = None,
        max_error: int | None = None,
        error_tail: int | None = None,
    ):
        if config.start_method not in mp.get_all_start_methods():
            hint = (
                " On Windows use start_method='spawn': the container process is created by "
                "the AppContainer launcher, which cannot inherit live objects."
                if os.name == "nt"
                else " Run the agent in a supported Linux environment to use the sandbox backend."
            )
            raise SandboxUnavailable(
                f"Sandbox execution requires the {config.start_method!r} process start method, "
                f"which is unavailable on this platform.{hint}"
            )
        self._agent = agent
        self._config = config
        self._cell_timeout = cell_timeout
        self._framework_builtins = framework_builtins or {}
        self._restrictions = restrictions
        self._max_error = effective_error_limit(max_error)
        self._error_tail = error_tail
        self._spec: ResolvedSpec = resolve_spec(config)
        self._codec = wire.Codec(wire.agent_types(agent))

        # A spawn worker cannot inherit anything, so its blueprint is built here:
        # an unimportable agent class or a non-picklable parameter is a
        # configuration error, and surfacing it at construction is far more
        # actionable than failing on the first cell.
        self._blueprint: Any = None
        if config.start_method == "spawn":
            from nooa.runtime.sandbox.blueprint import BlueprintError, WorkerBlueprint

            try:
                self._blueprint = WorkerBlueprint.from_agent(agent, self._framework_builtins)
            except BlueprintError as exc:
                raise SandboxUnavailable(str(exc)) from exc

        # The probe must validate the root this config names, which is the one
        # the launcher will run from; probing any other lets the two disagree.
        caps = _capabilities(config.runtime_root)
        missing = check_enforceable(config, caps)
        if missing and config.require:
            raise SandboxUnavailable(
                "Cannot enforce requested sandbox guardrails: "
                + "; ".join(missing)
                + ". Set sandbox.require=False to run without them (unsafe), or "
                "disable the affected guardrail."
            )
        self._degraded = missing  # non-empty only when require=False
        if missing:
            # require=False: drop the guards this host can't enforce so the worker
            # actually runs (unguarded for those) instead of the worker's
            # install_guards raising and failing every cell.
            self._spec = self._prune_unenforceable(self._spec, caps)
            logger.warning(
                "sandbox running with UNENFORCED guardrails (require=False): %s",
                "; ".join(missing),
            )

        self._ctx = mp.get_context(config.start_method)
        # Windows has its own launcher: SECURITY_CAPABILITIES is a creation-time
        # attribute, so multiprocessing's CreateProcess cannot enter a container.
        self._uses_container = os.name == "nt" and caps.filesystem == "appcontainer"
        self._runtime_root: Any = None
        # Last ``prepare_root`` summary. Carried into startup failures so the
        # real precondition problem is visible instead of only its symptom.
        self._runtime_summary: str = ""
        if self._uses_container:
            from nooa.runtime.sandbox.win_runtime import resolve_root

            self._runtime_root = resolve_root(config.runtime_root)
            if config.start_method != "spawn":
                raise SandboxUnavailable(
                    "the AppContainer backend requires start_method='spawn': the launcher "
                    "creates the process itself and cannot inherit live objects"
                )
        self._proc: BaseProcess | None = None
        self._conn: Connection | None = None
        self._lock = asyncio.Lock()
        self._req_id = 0
        self._closed = False
        self._disabled = False  # set when recovery="disabled" after a kill

        # A typo'd/missing workspace would otherwise become a silently unwritable
        # sandbox; create it up front (it's the parent's own filesystem).
        if config.filesystem and config.workspace:
            try:
                os.makedirs(config.workspace, exist_ok=True)
            except OSError as exc:
                raise SandboxUnavailable(
                    f"sandbox workspace {config.workspace!r} could not be created: {exc}"
                ) from exc

    @staticmethod
    def _prune_unenforceable(spec: ResolvedSpec, caps: Capabilities) -> ResolvedSpec:
        """Drop guards the host can't enforce (used only on the require=False path)."""
        from dataclasses import replace

        resource_ok = caps.memory is not None and caps.cpu is not None
        return replace(
            spec,
            filesystem=spec.filesystem and caps.filesystem is not None,
            path_grants=spec.path_grants if caps.filesystem is not None else (),
            block_network=spec.block_network and caps.network is not None,
            max_memory_mb=spec.max_memory_mb if resource_ok else 0,
            max_cpu_seconds=spec.max_cpu_seconds if resource_ok else 0,
        )

    @property
    def degraded_guards(self) -> list[str]:
        """Guardrails that could not be enforced (only when require=False)."""
        return list(self._degraded)

    # --- worker lifecycle --------------------------------------------------
    def _start_worker(self) -> None:
        if self._uses_container:
            self._start_container_worker()
            return
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        init: dict[str, Any] = {
            "restrictions": self._restrictions,
            "spec": self._spec,
            "max_error": self._max_error,
            "error_tail": self._error_tail,
        }
        if self._config.start_method == "spawn":
            # Nothing can be inherited: the child imports the agent class and
            # rebuilds the namespace, brokering every self.* access.
            #
            # ``framework_builtins`` is deliberately NOT sent — it holds the
            # return_result closure (and _call, a CurrentCall), which cannot be
            # pickled. The blueprint carries the picklable equivalent instead.
            init["mode"] = "blueprint"
            init["blueprint"] = self._blueprint
        else:
            init["mode"] = "fork"
            init["agent"] = self._agent
            init["framework_builtins"] = self._framework_builtins
        proc = self._ctx.Process(
            target=worker_main, args=(child_conn, init), daemon=True, name="nooa-sandbox-worker"
        )
        proc.start()
        child_conn.close()
        self._conn = parent_conn
        self._proc = proc

    def _start_container_worker(self) -> None:
        """Start the worker inside an AppContainer (Windows).

        The launcher grants the ACLs, creates the Job Object, spawns suspended,
        assigns the job, resumes, and completes a ready handshake. The broker
        connection it returns is a plain ``multiprocessing`` pipe — the container
        can use it because the child receives a duplicated handle, not a name
        lookup, which is why none of the broker code branches on platform.
        """
        from nooa.runtime.sandbox.win_launcher import start_container_worker
        from nooa.runtime.sandbox.win_runtime import _workspace_packages, prepare_root

        # Logged, not discarded: the summary says whether the package copy and
        # the tokenizer cache are usable. Both are silent preconditions of the
        # launch, and a failure here shows up later as an opaque worker-start
        # error inside the sandbox.
        #
        # Every distribution this process imported goes into the root: a spawned
        # worker imports the agent's module by name, and an agent that lives in
        # a host package (nooa_cli's coding agent, for instance) would otherwise
        # fail with ModuleNotFoundError before its first cell.
        summary = prepare_root(self._runtime_root, sync=True, packages=_workspace_packages())
        logger.debug("sandbox runtime prepared: %s", summary)
        self._runtime_summary = summary
        init: dict[str, Any] = {
            "restrictions": self._restrictions,
            "spec": self._spec,
            "max_error": self._max_error,
            "error_tail": self._error_tail,
            "mode": "blueprint",
            "blueprint": self._blueprint,
        }
        try:
            handles = start_container_worker(
                init,
                self._spec,
                self._runtime_root,
                # None lets the launcher mint one per start. The worker is
                # restarted after a kill, and a restarted worker is a new
                # process that should not inherit the dead one's identity.
                container_name=self._config.container_name,
                workspace=self._config.workspace,
                env_passthrough=self._config.env_passthrough,
            )
        except SandboxUnavailable as exc:
            # The worker's own error says what failed inside the sandbox; the
            # runtime summary says whether the preconditions were ever met
            # (package copy, tokenizer cache). Without both, diagnosing a clean
            # CI runner means guessing.
            raise SandboxUnavailable(f"{exc} [runtime: {self._runtime_summary}]") from exc
        self._conn = handles.conn
        self._proc = handles.proc

    def _detach_worker(self) -> Any:
        """Clear the proc/conn refs and close the pipe; return the proc to reap."""
        proc, conn = self._proc, self._conn
        self._proc = self._conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        return proc

    def _terminate_worker(self) -> None:
        """Synchronous teardown (sync cleanup paths only, e.g. ``close_sync``).

        The blocking ``proc.join`` runs on the caller's thread — do NOT call this
        from the event loop; use :meth:`_aterminate_worker` there instead.
        """
        proc = self._detach_worker()
        if proc is None:
            return
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1.0)
        _release_worker(proc)

    async def _aterminate_worker(self) -> None:
        """Async teardown: send signals inline (fast, non-blocking) but run the
        blocking ``proc.join`` off the event loop so a worker kill/restart does
        not stall the loop (and other concurrent sessions) for up to ~2s."""
        proc = self._detach_worker()
        if proc is None:
            return
        if proc.is_alive():
            proc.terminate()
            await asyncio.to_thread(proc.join, 1.0)
        if proc.is_alive():
            proc.kill()
            await asyncio.to_thread(proc.join, 1.0)
        # Revoking container ACLs is a few Win32 calls per path; keep it off the
        # loop for the same reason as the join.
        if _is_container_proc(proc):
            await asyncio.to_thread(_release_worker, proc)

    async def _restart_worker(self) -> None:
        await self._aterminate_worker()
        if self._config.recovery == "disabled":
            # Do not resurrect the worker; subsequent cells fail deterministically.
            self._disabled = True
            return
        self._start_worker()  # fork stays on the loop thread (fork-from-thread is unsafe)

    async def _ensure_worker(self) -> None:
        if self._disabled:
            raise WorkerDiedError(
                "sandbox worker was killed and recovery='disabled'; no further cells can run"
            )
        if self._proc is None or not self._proc.is_alive():
            await self._aterminate_worker()
            self._start_worker()  # fork on the loop thread

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    # --- running a cell ----------------------------------------------------
    async def run_cell(self, code: str, *, execution_count: int = 1) -> ExecutionResult:
        """Execute one cell in the worker and return an ``ExecutionResult``.

        Output is buffered in the worker and transferred with the final result.
        If the process is hard-killed (timeout, resource limit, or crash), output
        written by that cell before death cannot be recovered.
        """
        if self._closed:
            raise WorkerDiedError("sandbox executor is closed")
        async with self._lock:
            if self._disabled:
                # recovery="disabled": a prior kill retired the worker; report a
                # cell error rather than raising out of the strategy loop.
                return self._synth_error(
                    WorkerDiedError(
                        "a prior cell was killed and recovery='disabled'; the sandbox "
                        "worker will not restart, so no further cells can run"
                    )
                )
            await self._ensure_worker()
            assert self._conn is not None
            req_id = self._next_id()
            try:
                self._conn.send(
                    {"op": "run", "id": req_id, "code": code, "execution_count": execution_count}
                )
            except (BrokenPipeError, OSError) as exc:
                # The worker died between _ensure_worker and this send (e.g. an
                # OOM kill); treat it like any other worker death, not an escape.
                await self._restart_worker()
                return self._synth_error(self._classify_worker_death(WorkerDiedError(str(exc))))
            deadline = None
            if self._cell_timeout:
                deadline = self._cell_timeout + self._config.timeout_grace_s
            loop = asyncio.get_running_loop()
            try:
                response = await asyncio.to_thread(self._recv_until_result, req_id, deadline, loop)
            except CellTimeoutError as exc:
                await self._restart_worker()
                return self._synth_error(exc)
            except WorkerDiedError as exc:
                err = self._classify_worker_death(exc)
                await self._restart_worker()
                return self._synth_error(err)
            try:
                dto: ResultDTO = dto_from_wire(response.get("result"))
            except CellSerializationError as exc:
                # Only a tampered worker produces a malformed result: retire it.
                await self._restart_worker()
                return self._synth_error(WorkerDiedError(f"{exc}.{self._RESET_NOTE}"))
            return dto_to_result(dto, signal_factory=self._signal_factory, codec=self._codec)

    def _recv_until_result(
        self, req_id: int, deadline: float | None, loop: asyncio.AbstractEventLoop
    ) -> dict[str, Any]:
        """Block (in a thread) until the worker answers, servicing broker calls.

        ``deadline`` seconds after now, a still-silent worker is declared timed
        out. Broker ``tool_call`` messages are dispatched onto the parent loop so
        ``self.*`` runs against the live agent while we wait.
        """
        end = time.monotonic() + deadline if deadline else None
        poll = self._config.rss_poll_s if self._config.rss_poll_s > 0 else 0.1
        while True:
            conn = self._conn
            if conn is None:
                raise WorkerDiedError("sandbox worker connection lost")
            if end is not None:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise CellTimeoutError(
                        f"cell exceeded its {self._cell_timeout}s deadline and was killed"
                    )
                wait = min(remaining, poll)
            else:
                wait = poll
            if not conn.poll(max(0.01, wait)):
                if self._proc is None or not self._proc.is_alive():
                    raise WorkerDiedError("sandbox worker exited unexpectedly")
                continue
            try:
                raw = conn.recv_bytes()
            except (EOFError, OSError) as exc:
                raise WorkerDiedError("sandbox worker pipe closed") from exc
            # Never ``conn.recv()`` here: that would unpickle worker bytes.
            try:
                msg = self._codec.loads(raw)
                if not isinstance(msg, dict):
                    raise CellSerializationError("not a message dict")
            except CellSerializationError as exc:
                raise WorkerDiedError(
                    f"sandbox worker sent a malformed message ({exc}).{self._RESET_NOTE}"
                ) from exc
            mtype = msg.get("type")
            if mtype == "tool_call":
                # Pause the cell clock while the parent services the brokered
                # call: the worker is idle-waiting, not running cell code, so
                # this time must not count against the cell deadline.
                broker_started = time.monotonic()
                self._service_tool_call(msg, loop)
                if end is not None:
                    end += time.monotonic() - broker_started
                continue
            if mtype == "fatal":
                raise WorkerDiedError(f"sandbox worker fatal: {msg.get('error')}")
            if mtype == "response" and msg.get("id") == req_id:
                return msg

    def _service_tool_call(self, msg: dict[str, Any], loop: asyncio.AbstractEventLoop) -> None:
        try:
            call = self._decode_tool_call(msg)
        except CellSerializationError as exc:
            # Refused value or malformed request: answer this call with a normal
            # tool error so the cell sees it and the worker (and namespace) survives.
            response: dict[str, Any] = {
                "ok": False,
                "error_type": "CellSerializationError",
                "error": _bounded_text(exc, "CellSerializationError", limit=self._max_error),
            }
        else:
            future = asyncio.run_coroutine_threadsafe(self._dispatch_tool_call(call), loop)
            # Brokered ``self.*`` work runs parent-side while the worker idles — it
            # gets its OWN bound (broker_timeout_s), not the cell deadline: killing
            # the worker because the parent was slow (e.g. memory consolidation LLM
            # calls) wiped REPL state and swallowed queued submits in the ARC fleet.
            broker_timeout = self._config.broker_timeout_s or None
            try:
                response = future.result(timeout=broker_timeout)
            except futures.TimeoutError:
                future.cancel()
                raise CellTimeoutError(
                    f"brokered self.* call exceeded broker_timeout_s={broker_timeout}s"
                ) from None
        response["type"] = "tool_result"
        response["tool_call_id"] = msg.get("tool_call_id")
        conn = self._conn
        if conn is not None:
            try:
                conn.send(response)
            except (BrokenPipeError, OSError) as exc:
                # Worker died mid-brokered-call; surface as a worker death so
                # run_cell restarts it instead of aborting the whole generation.
                raise WorkerDiedError("sandbox worker pipe closed during tool call") from exc

    def _decode_tool_call(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Check the broker request shape and decode its cell-supplied payload.

        Raises ``CellSerializationError`` for a refused value (a type the agent
        did not declare) or a malformed request.
        """
        kind, path = msg.get("kind"), msg.get("path") or []
        if kind not in ("call", "attr", "setattr", "iter", "introspect"):
            raise CellSerializationError(f"malformed sandbox broker request: kind {kind!r}")
        if not isinstance(path, list) or not all(isinstance(p, str) for p in path):
            raise CellSerializationError("malformed sandbox broker request: bad path")
        call: dict[str, Any] = {"kind": kind, "path": path}
        if kind == "introspect":
            name = msg.get("name")
            if not isinstance(name, str):
                raise CellSerializationError(
                    "malformed sandbox broker request: bad introspect name"
                )
            call["name"] = name
        if kind not in ("call", "setattr", "introspect"):
            return call
        what = (
            f"{call.get('name', '?')}(self)"
            if kind == "introspect"
            else f"self.{'.'.join(path)}" + ("" if kind == "setattr" else "(...)")
        )
        payload = msg.get("payload")
        if not isinstance(payload, bytes):
            raise CellSerializationError(f"malformed sandbox broker request for {what}")
        try:
            value = self._codec.loads(payload)
        except CellSerializationError as exc:
            raise CellSerializationError(
                f"Value passed to {what} cannot cross the sandbox boundary: {exc}. {wire.HINT}"
            ) from exc
        if kind == "setattr":
            call["value"] = value
        elif (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], tuple)
            and isinstance(value[1], dict)
        ):
            call["args"], call["kwargs"] = value
        else:
            raise CellSerializationError(f"malformed sandbox broker request for {what}")
        return call

    def _walk_path(self, path: list[str]) -> Any:
        """Resolve a dotted attribute path (``["memory", "remember"]``) on the agent."""
        obj: Any = self._agent
        for part in path:
            obj = getattr(obj, part)
        return obj

    async def _dispatch_tool_call(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Run a brokered ``self.<path>`` access against the parent's live agent."""
        from nooa.events import ExecutionSignal

        path = msg.get("path") or []
        display = ".".join(path)
        kind = msg.get("kind")
        target: Any = None
        try:
            if kind == "setattr":
                # self.<path> = value on the parent's live agent.
                obj = self._walk_path(path[:-1])
                setattr(obj, path[-1], msg.get("value"))
                return {"ok": True, "result": None}
            if kind == "introspect":
                # Spawned workers have no local instance, so doc(self) /
                # methods(self) / variables(self) render here against the live
                # agent and cross back as a string.
                from nooa.agentdoc import doc
                from nooa.agentdoc.introspect import methods, variables

                fn = {"doc": doc, "methods": methods, "variables": variables}.get(
                    msg.get("name", "")
                )
                if fn is None:
                    return {
                        "ok": False,
                        "error_type": "AttributeError",
                        "error": f"unknown introspection function {msg.get('name')!r}",
                    }
                result = fn(self._agent, *msg.get("args", ()), **msg.get("kwargs", {}))
                return {"ok": True, "result": result}
            if kind == "iter":
                # Materialize list(obj) on the parent (the iterator isn't picklable).
                items = list(self._walk_path(path))
                if not is_picklable(items):
                    return {
                        "ok": False,
                        "error_type": "CellSerializationError",
                        "error": (
                            f"iterating self.{display} produced non-picklable items that "
                            "cannot cross the sandbox boundary."
                        ),
                    }
                return {"ok": True, "result": items}
            target = self._walk_path(path)
            if kind == "attr":
                # A spawned worker has no local agent, so it cannot tell a method
                # from a value; report that classification with the value itself.
                is_callable = callable(target) and not isinstance(target, type)
                flags = {
                    "callable": is_callable,
                    "is_async": is_callable and asyncio.iscoroutinefunction(target),
                }
                # Picklable state crosses; a live object becomes a nested proxy.
                if is_picklable(target):
                    return {"ok": True, "result": target, **flags}
                return {"ok": True, "result": None, "proxy": True, **flags}
            value = target(*msg.get("args", ()), **msg.get("kwargs", {}))
            # Record whether the call was async so the worker can re-wrap the
            # (already-resolved) result in an awaitable — otherwise a cell doing
            # ``await proxy(...)`` on an async callable reached via a nested proxy
            # would await a plain value and raise TypeError.
            was_async = asyncio.iscoroutine(value) or asyncio.isfuture(value)
            if was_async:
                value = await value
            if not is_picklable(value):
                return {
                    "ok": False,
                    "error_type": "CellSerializationError",
                    "error": (
                        f"self.{display} returned a {type(value).__name__!r} that is not "
                        "picklable and cannot cross the sandbox boundary. Return a "
                        "picklable summary instead."
                    ),
                }
            return {"ok": True, "result": value, "was_async": was_async}
        except ExecutionSignal as sig:
            # Some tools end the turn by *raising* a control-flow signal (e.g. an
            # ARC submit_actions -> return_result). Marshal it back to the cell so
            # it re-raises there and flows through the normal signal path.
            payload = getattr(sig, "result", None)
            payload_is_picklable = is_picklable(payload)
            return {
                "ok": False,
                "error_type": "ExecutionSignal",
                "error": _bounded_text(sig, "ExecutionSignal", limit=self._max_error),
                "signal_result": payload if payload_is_picklable else None,
                "signal_result_dropped": not payload_is_picklable,
            }
        except CellSerializationError as exc:
            return {
                "ok": False,
                "error_type": "CellSerializationError",
                "error": _bounded_text(exc, "CellSerializationError", limit=self._max_error),
            }
        except Exception as exc:  # noqa: BLE001 - surface tool errors to the cell
            # The parent still has the resolved target and can safely derive its
            # agentdoc. The child cannot: it only has a proxy and a traceback-local
            # surrogate exception. Transport a bounded string hint, never the
            # callable or traceback itself.
            from nooa.errors.formatting import _bad_call_agentdoc

            call_hint = _bad_call_agentdoc(exc, target=target)
            response = {
                "ok": False,
                "error_type": _bounded_text(type(exc).__name__, "Exception", limit=self._max_error),
                "error": _bounded_text(exc, type(exc).__name__, limit=self._max_error),
            }
            if call_hint:
                response["call_hint"] = _bounded_text(call_hint, "", limit=self._max_error)
            return response

    # --- error synthesis ---------------------------------------------------
    def _classify_worker_death(self, exc: WorkerDiedError) -> Exception:
        proc = self._proc
        code = getattr(proc, "exitcode", None)
        if os.name == "nt":
            # A Job Object terminates the process with STATUS_CONTROL_C_EXIT when
            # the per-job CPU limit is reached (measured by the M0 probe).
            from nooa.runtime.sandbox.win_launcher import EXIT_JOB_TIME_LIMIT

            if code == EXIT_JOB_TIME_LIMIT:
                return CellTimeoutError("cell exceeded its CPU-time limit and was killed")
        # Neither SIGXCPU nor SIGKILL exists on Windows; a missing constant must
        # not turn a worker death into an AttributeError here.
        sig_xcpu = getattr(signal, "SIGXCPU", None)
        sig_kill = getattr(signal, "SIGKILL", None)
        if sig_xcpu is not None and code == -sig_xcpu:
            return CellTimeoutError("cell exceeded its CPU-time limit and was killed")
        if sig_kill is not None and code == -sig_kill:
            return CellMemoryError(
                "worker was killed (out-of-memory or resource limit). "
                "Reduce the cell's memory use or raise max_memory_mb."
            )
        return exc

    _RESET_NOTE = (
        " The worker was restarted, so variables/functions defined in earlier "
        "cells are gone — recompute any state you need."
    )

    def _synth_error(self, error: Exception) -> ExecutionResult:
        # A kill/restart wipes the persistent namespace; tell the model so it
        # rebuilds state instead of referencing now-undefined earlier-cell names.
        if isinstance(error, CellTimeoutError | CellMemoryError) and not self._disabled:
            error = type(error)(str(error) + self._RESET_NOTE)
        return ExecutionResult(stdout="", stderr="", error=error, defined_methods={})

    @staticmethod
    def _signal_factory(payload: Any) -> Any:
        from nooa.strategies.codeact import _ReturnResultSignal

        return _ReturnResultSignal(result=payload)

    # --- teardown ----------------------------------------------------------
    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        async with self._lock:
            conn = self._conn
            if conn is not None and self._proc is not None and self._proc.is_alive():
                try:
                    conn.send({"op": "shutdown", "id": self._next_id()})
                    await asyncio.to_thread(self._proc.join, 0.5)
                except Exception:
                    pass
            await self._aterminate_worker()

    def close_sync(self) -> None:
        """Best-effort synchronous teardown (for non-async cleanup paths)."""
        if self._closed:
            return
        self._closed = True
        self._terminate_worker()
