# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The router role: one editor connection, one worker process per root session.

The client (an editor such as Pool) speaks ACP to the router on standard
input and output. The router answers ``initialize`` and ``session/list``
itself and forwards everything else, unchanged, to the worker that runs
the session (``nooa_coder.acp.worker``). Each worker is a plain ACP server
on one end of a Unix socket pair.

The router is a JSON-lines proxy. It parses only the envelope of each
frame (``id``, ``method``, ``params.sessionId``) and forwards the original
bytes. Routing, client to worker:

- ``initialize``: answered by the router (``initialize_response``); the
  raw params are kept and replayed as each new worker's first request.
- ``session/list``: answered from the store, read-only.
- ``session/new``: a new worker; the session id in its answer is mapped to
  that worker before the answer is forwarded.
- ``session/load`` (and ``_nooa/session/delete``) of an id that is not
  mapped: the store's ``parent_id`` chain gives the root; if a live worker
  runs that root, the request goes there, else to a new worker. A load is
  mapped when forwarded and unmapped if it fails.
- ``$/cancel_request``: to the worker running that request.
- anything else with ``params.sessionId``: to the session's worker; an
  unknown id is ``resource_not_found`` for a request and dropped for a
  notification. Other requests are ``method_not_found``.
- replies to worker requests: to worker ``k = id >> 32`` (each worker's
  request ids start at ``k << 32``); other ids are dropped.

Nothing on the client's input path waits on a worker: each worker has an
unbounded outbound queue and its own writer task, and spawning runs in a
task. Frames to the client go through one bounded queue and one writer
task, so each worker's frames keep their order.

The router alone stops workers: when a worker has no sessions left and no
requests in flight (its root was closed, a new or load failed, a delete
of an idle session finished), the router closes its end of the socket and
the worker exits. On end of input or SIGTERM every worker is closed the
same way, with 5 seconds in total to exit, and then every process group
is killed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NamedTuple

from acp import PROTOCOL_VERSION, RequestError
from acp.connection import StreamDirection, StreamEvent
from acp.schema import InitializeRequest, ListSessionsRequest
from pydantic import ValidationError

from nooa_coder.acp.framing import FRAME_LIMIT, Frame, encode, read_frame
from nooa_coder.acp.server import initialize_response, list_sessions
from nooa_coder.session.store import InvalidSessionIdError, SessionNotFoundError, SessionStore

logger = logging.getLogger(__name__)

INIT_REQUEST_ID = "nooa-router-init"
"""Id of the ``initialize`` the router replays to each new worker; its answer stays here."""

ID_SHIFT = 32
"""Worker ``k`` sends requests with ids from ``k << ID_SHIFT``; ``k`` starts at 1."""

MAX_WORKERS = 1 << 21
"""``k << 32`` must stay below 2**53 so every id is exact in JavaScript clients."""

SPAWN_TIMEOUT = 30.0
STOP_GRACE = 5.0
STDOUT_QUEUE_SIZE = 1000

_DELETE_METHOD = "_nooa/session/delete"
_ROUTED_BY_STORE = ("session/load", _DELETE_METHOD)
_MAX_TREE_DEPTH = 64


class WorkerProcess(NamedTuple):
    """What a spawn function returns for worker ``k``.

    ``reader``/``writer`` are the router's end of the worker's socket.
    ``wait()`` waits for the process to exit and returns its exit code;
    ``kill()`` kills its process group (it must tolerate a group that is
    already gone).
    """

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    wait: Callable[[], Awaitable[int | None]]
    kill: Callable[[], None]
    pid: int | None = None


Spawn = Callable[[int], Awaitable[WorkerProcess]]
Observer = Callable[[StreamEvent], Any]


def process_spawn(command: Sequence[str]) -> Spawn:
    """The real spawn: ``command --worker-fd N --id-base B`` on one end of a socket pair.

    The worker gets its own session and process group (so the router can
    kill everything it started), no standard input, and standard output
    sent to standard error: only the router writes to the client's
    standard output, and the client must see end of stream when the router
    dies. Standard error is inherited. The router's copy of the worker's
    end is closed right after the spawn, so the router sees end of stream
    when the worker dies.
    """

    async def spawn(k: int) -> WorkerProcess:
        ours, theirs = socket.socketpair()
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                "--worker-fd",
                str(theirs.fileno()),
                "--id-base",
                str(k << ID_SHIFT),
                stdin=subprocess.DEVNULL,
                stdout=2,
                start_new_session=True,
                pass_fds=(theirs.fileno(),),
            )
        except BaseException:
            ours.close()
            raise
        finally:
            theirs.close()
        logger.info("worker %d pid %d started", k, process.pid)
        reader, writer = await asyncio.open_unix_connection(sock=ours, limit=FRAME_LIMIT)

        def kill() -> None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass  # the group is gone (PermissionError: macOS, a zombie leader)

        return WorkerProcess(reader, writer, process.wait, kill, pid=process.pid)

    return spawn


class _Worker:
    def __init__(self, k: int) -> None:
        self.k = k
        self.outbound: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.process: WorkerProcess | None = None
        self.root_id: str | None = None
        self.sessions: set[str] = set()
        self.live = True
        self.stopping = False
        self.exited = asyncio.Event()
        self.tasks: set[asyncio.Task[Any]] = set()
        self.spawn_ms = 0.0
        self.handshake_ms = 0.0
        self.ready_at: float | None = None


class _Pending:
    """A client request forwarded to a worker and not yet answered."""

    def __init__(self, frame: Frame, worker: _Worker, *, mapped: bool) -> None:
        self.id = frame.id
        self.method = frame.method or ""
        self.session_id = frame.session_id
        self.worker = worker
        self.mapped = mapped
        self.sent_at = time.perf_counter()


def _key(request_id: Any) -> str:
    # 1, 1.0, "1" and true are different JSON-RPC ids.
    return json.dumps(request_id)


class Router:
    """Routes one ACP client connection to one worker process per root session."""

    def __init__(
        self,
        *,
        spawn: Spawn,
        store: SessionStore,
        observers: list[Observer] | None = None,
        spawn_timeout: float = SPAWN_TIMEOUT,
        stop_grace: float = STOP_GRACE,
    ) -> None:
        self._spawn = spawn
        self._store = store
        self._observers = list(observers or [])
        self._spawn_timeout = spawn_timeout
        self._stop_grace = stop_grace
        self._workers: dict[int, _Worker] = {}
        self._sessions: dict[str, _Worker] = {}
        self._pending: dict[str, _Pending] = {}
        self._init_params: Any = None
        self._next_k = 1
        self._stdout: asyncio.Queue[tuple[bytes, dict[str, Any]] | None] = asyncio.Queue(
            STDOUT_QUEUE_SIZE
        )
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closing = False
        self._stop = asyncio.Event()

    # ---- lifetime ------------------------------------------------------------

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Route frames from ``reader`` until it ends or ``stop()``; then stop every worker."""
        stdout_task = asyncio.create_task(self._write_client(writer), name="nooa-router-stdout")
        read_task = asyncio.create_task(self._read_client(reader), name="nooa-router-stdin")
        stop_task = asyncio.create_task(self._stop.wait())
        try:
            await asyncio.wait({read_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            read_task.cancel()
            stop_task.cancel()
            await asyncio.gather(read_task, stop_task, return_exceptions=True)
            try:
                await self._shutdown()
            finally:
                await self._close_client(stdout_task, writer)

    async def serve_stdio(
        self, *, input_fd: int | None = None, output_fd: int | None = None
    ) -> None:
        """Serve the client on this process's standard input and output.

        ``input_fd``/``output_fd`` are the real standard input and output
        when the entry point reserved them for ACP
        (``cli.reserve_stdio_for_acp``), so a stray print in the router
        cannot reach the client. The first SIGTERM
        starts the same shutdown as end of input; a second one gets the
        default action, so a hung shutdown can still be killed.
        """
        from nooa_coder.acp.server import open_stdio

        loop = asyncio.get_running_loop()
        previous = signal.getsignal(signal.SIGTERM)

        def terminate() -> None:
            self.stop()
            loop.remove_signal_handler(signal.SIGTERM)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)

        installed = False
        try:
            loop.add_signal_handler(signal.SIGTERM, terminate)
            installed = True
        except (NotImplementedError, RuntimeError):
            pass  # not the main thread
        try:
            reader, writer = await open_stdio(input_fd, output_fd)
            await self.serve(reader, writer)
        finally:
            if installed:
                loop.remove_signal_handler(signal.SIGTERM)
                signal.signal(signal.SIGTERM, previous)

    def stop(self) -> None:
        """Begin the same shutdown as end of input (used for SIGTERM)."""
        self._stop.set()

    async def _shutdown(self) -> None:
        self._closing = True
        workers = list(self._workers.values())
        for worker in workers:
            if worker.ready_at is None:
                # Still starting: stop the spawn; its cancellation kills the process.
                for task in list(worker.tasks):
                    task.cancel()
            self._close_worker(worker)
        if workers:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*(worker.exited.wait() for worker in workers)),
                    self._stop_grace,
                )
        for worker in workers:
            self._kill(worker)
            for task in list(worker.tasks):
                task.cancel()
        tasks = [task for worker in workers for task in worker.tasks] + list(self._tasks)
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _close_client(
        self, stdout_task: asyncio.Task[None], writer: asyncio.StreamWriter
    ) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self._stdout.put_nowait(None)
        try:
            await asyncio.wait_for(asyncio.shield(stdout_task), 1.0)
        except Exception:  # a timeout, or the writer failed
            stdout_task.cancel()
            await asyncio.gather(stdout_task, return_exceptions=True)
        with contextlib.suppress(Exception):
            writer.close()

    def _task(self, coroutine: Awaitable[Any], *, name: str, worker: _Worker | None = None) -> None:
        task = asyncio.ensure_future(coroutine)
        task.set_name(name)
        owner = worker.tasks if worker is not None else self._tasks
        owner.add(task)

        def done(finished: asyncio.Task[Any]) -> None:
            owner.discard(finished)
            if not finished.cancelled() and finished.exception() is not None:
                logger.error("%s failed", name, exc_info=finished.exception())

        task.add_done_callback(done)

    # ---- client side -------------------------------------------------------------

    async def _read_client(self, reader: asyncio.StreamReader) -> None:
        try:
            while (frame := await read_frame(reader)) is not None:
                self._observe(StreamDirection.INCOMING, frame.message)
                self._on_client_frame(frame)
        except (ConnectionError, OSError) as exc:
            logger.info("The client connection failed: %s", exc)

    async def _write_client(self, writer: asyncio.StreamWriter) -> None:
        while (item := await self._stdout.get()) is not None:
            raw, message = item
            self._observe(StreamDirection.OUTGOING, message)
            try:
                writer.write(raw)
                await writer.drain()
            except (ConnectionError, OSError) as exc:
                logger.info("The client stopped reading: %s", exc)
                self._stop.set()
                # Keep draining so no worker reader blocks on a full queue.
                while await self._stdout.get() is not None:
                    pass
                return

    def _observe(self, direction: StreamDirection, message: dict[str, Any]) -> None:
        for observer in self._observers:
            try:
                observer(StreamEvent(direction, message))
            except Exception:
                logger.exception("Stream observer failed")

    async def _emit(self, raw: bytes, message: dict[str, Any]) -> None:
        await self._stdout.put((raw, message))

    async def _emit_message(self, message: dict[str, Any]) -> None:
        await self._emit(_encode_lenient(message), message)

    def _emit_soon(self, message: dict[str, Any]) -> None:
        """Queue a frame the router writes itself, without waiting on the client."""
        raw = _encode_lenient(message)
        try:
            self._stdout.put_nowait((raw, message))
        except asyncio.QueueFull:
            self._task(self._emit(raw, message), name="nooa-router-emit")

    def _answer(self, frame: Frame, result: Any) -> None:
        self._emit_soon({"jsonrpc": "2.0", "id": frame.id, "result": result})

    def _fail(self, request_id: Any, error: RequestError) -> None:
        self._emit_soon({"jsonrpc": "2.0", "id": request_id, "error": error.to_error_obj()})

    def _on_client_frame(self, frame: Frame) -> None:
        if frame.is_response:
            self._route_reply(frame)
            return
        method = frame.method
        if method is None:
            logger.warning("Dropped a client frame with neither method nor id")
            return
        if frame.is_notification:
            self._on_client_notification(frame, method)
            return
        session_id = frame.session_id
        if method == "initialize":
            self._initialize(frame)
        elif method == "session/list":
            self._task(self._list_sessions(frame), name="nooa-router-list")
        elif method == "session/new":
            self._forward(self._new_worker(), frame)
        elif method in _ROUTED_BY_STORE and session_id is not None:
            worker = self._sessions.get(session_id)
            if worker is not None:
                self._forward(worker, frame)
            else:
                self._task(self._route_by_store(frame, session_id), name="nooa-router-resolve")
        elif session_id is not None:
            worker = self._sessions.get(session_id)
            if worker is None:
                self._fail(frame.id, RequestError.resource_not_found(session_id))
            else:
                self._forward(worker, frame)
        else:
            self._fail(frame.id, RequestError.method_not_found(method))

    def _on_client_notification(self, frame: Frame, method: str) -> None:
        if method == "$/cancel_request":
            pending = self._pending.get(_key(frame.params.get("requestId")))
            if pending is not None and pending.worker.live:
                pending.worker.outbound.put_nowait(frame.raw)
            return
        session_id = frame.session_id
        worker = self._sessions.get(session_id) if session_id is not None else None
        if worker is not None:
            worker.outbound.put_nowait(frame.raw)
        else:
            logger.debug("Dropped notification %s for session %s", method, session_id)

    def _route_reply(self, frame: Frame) -> None:
        request_id = frame.id
        if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
            logger.info("Dropped a client reply with id %r: not a worker request id", request_id)
            return
        worker = self._workers.get(request_id >> ID_SHIFT)
        if worker is None or not worker.live:
            logger.info("Dropped a client reply with id %d: no live worker owns it", request_id)
            return
        worker.outbound.put_nowait(frame.raw)

    def _initialize(self, frame: Frame) -> None:
        try:
            request = InitializeRequest.model_validate(frame.message.get("params"))
        except ValidationError as exc:
            self._fail(frame.id, RequestError.invalid_params({"errors": exc.errors()}))
            return
        self._init_params = frame.message.get("params")
        response = initialize_response(request.protocol_version)
        self._answer(
            frame,
            response.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True),
        )

    async def _list_sessions(self, frame: Frame) -> None:
        try:
            request = ListSessionsRequest.model_validate(frame.message.get("params") or {})
            response = await list_sessions(
                self._store, cwd=request.cwd, cursor=request.cursor, live=self._live_status
            )
        except RequestError as exc:
            self._fail(frame.id, exc)
            return
        except ValidationError as exc:
            self._fail(frame.id, RequestError.invalid_params({"errors": exc.errors()}))
            return
        except Exception as exc:
            logger.exception("session/list failed")
            self._fail(frame.id, RequestError.internal_error({"details": str(exc)}))
            return
        self._answer(
            frame,
            response.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True),
        )

    def _live_status(self, session_id: str) -> tuple[str, str | None] | None:
        if session_id not in self._sessions:
            return None
        running = any(
            p.method == "session/prompt" and p.session_id == session_id
            for p in self._pending.values()
        )
        return ("running" if running else "idle"), None

    async def _route_by_store(self, frame: Frame, session_id: str) -> None:
        """Send a load or delete of an unmapped id to its root's worker, or a new one."""
        try:
            root_id = await asyncio.to_thread(self._root_of, session_id)
        except (SessionNotFoundError, InvalidSessionIdError):
            self._fail(frame.id, RequestError.resource_not_found(session_id))
            return
        # No await from here on: the lookups and the mapping are one step, so a
        # second load of the same tree finds the worker this one chose.
        worker = self._sessions.get(session_id) or self._root_worker(root_id)
        if worker is None:
            worker = self._new_worker()
            worker.root_id = root_id
        mapped = False
        if frame.method == "session/load" and session_id not in self._sessions:
            self._map(session_id, worker)
            mapped = True
        self._forward(worker, frame, mapped=mapped)

    def _root_of(self, session_id: str) -> str:
        info = self._store.get(session_id)
        for _ in range(_MAX_TREE_DEPTH):
            if info.parent_id is None:
                return info.id
            try:
                info = self._store.get(info.parent_id)
            except (SessionNotFoundError, InvalidSessionIdError):
                return info.id
        return info.id

    def _root_worker(self, root_id: str) -> _Worker | None:
        for worker in self._workers.values():
            if worker.live and not worker.stopping and worker.root_id == root_id:
                return worker
        return None

    def _forward(self, worker: _Worker, frame: Frame, *, mapped: bool = False) -> None:
        if not worker.live:
            self._fail(frame.id, _exited_error(frame.session_id, None))
            return
        self._pending[_key(frame.id)] = _Pending(frame, worker, mapped=mapped)
        worker.outbound.put_nowait(frame.raw)

    def _map(self, session_id: str, worker: _Worker) -> None:
        self._sessions[session_id] = worker
        worker.sessions.add(session_id)

    def _unmap(self, session_id: str, worker: _Worker) -> None:
        if self._sessions.get(session_id) is worker:
            del self._sessions[session_id]
        worker.sessions.discard(session_id)

    # ---- workers -------------------------------------------------------------------

    def _new_worker(self) -> _Worker:
        k = self._next_k
        if k >= MAX_WORKERS:
            raise RuntimeError("worker id space exhausted")
        self._next_k += 1
        worker = _Worker(k)
        self._workers[k] = worker
        self._task(self._start_worker(worker), name=f"nooa-router-spawn-{k}", worker=worker)
        return worker

    async def _start_worker(self, worker: _Worker) -> None:
        started = time.perf_counter()
        spawned = started
        try:
            async with asyncio.timeout(self._spawn_timeout):
                process = await self._spawn(worker.k)
                worker.process = process
                spawned = time.perf_counter()
                params = self._init_params
                if params is None:
                    params = {"protocolVersion": PROTOCOL_VERSION}
                process.writer.write(
                    encode(
                        {
                            "jsonrpc": "2.0",
                            "id": INIT_REQUEST_ID,
                            "method": "initialize",
                            "params": params,
                        }
                    )
                )
                await process.writer.drain()
                answer = await self._read_init_answer(worker, process)
        except asyncio.CancelledError:
            self._kill(worker)
            worker.live = False
            worker.exited.set()
            raise
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            if isinstance(exc, TimeoutError):
                reason = f"no answer within {self._spawn_timeout:.0f} s"
            logger.error("worker %d failed to start: %s", worker.k, reason)
            await self._worker_failed(worker, f"the worker process failed to start: {reason}")
            return
        if "error" in answer.message:
            logger.error("worker %d refused initialize: %s", worker.k, answer.message["error"])
            await self._worker_failed(
                worker, f"the worker refused initialize: {answer.message['error']}"
            )
            return
        worker.ready_at = time.perf_counter()
        worker.spawn_ms = (spawned - started) * 1000
        worker.handshake_ms = (worker.ready_at - spawned) * 1000
        logger.info(
            "worker %d pid %s ready: spawn_ms=%.0f handshake_ms=%.0f",
            worker.k,
            process.pid,
            worker.spawn_ms,
            worker.handshake_ms,
        )
        self._task(
            self._write_worker(worker, process), name=f"nooa-router-to-{worker.k}", worker=worker
        )
        self._task(
            self._read_worker(worker, process), name=f"nooa-router-from-{worker.k}", worker=worker
        )

    async def _read_init_answer(self, worker: _Worker, process: WorkerProcess) -> Frame:
        while True:
            frame = await read_frame(process.reader)
            if frame is None:
                code = None
                with contextlib.suppress(Exception):
                    code = await asyncio.wait_for(process.wait(), 1.0)
                raise ConnectionError(f"the worker exited during initialize (exit code {code})")
            if frame.is_response and frame.id == INIT_REQUEST_ID:
                return frame
            logger.warning(
                "worker %d: dropped a frame sent before initialize was answered", worker.k
            )

    async def _write_worker(self, worker: _Worker, process: WorkerProcess) -> None:
        writer = process.writer
        try:
            while (data := await worker.outbound.get()) is not None:
                writer.write(data)
                await writer.drain()
            if writer.can_write_eof():
                writer.write_eof()
            else:
                writer.close()
        except (ConnectionError, OSError) as exc:
            logger.info("worker %d: the socket closed while writing: %s", worker.k, exc)

    async def _read_worker(self, worker: _Worker, process: WorkerProcess) -> None:
        try:
            while (frame := await read_frame(process.reader)) is not None:
                await self._on_worker_frame(worker, frame)
        except (ConnectionError, OSError) as exc:
            logger.info("worker %d: the socket failed: %s", worker.k, exc)
        await self._worker_exited(worker, process)

    async def _on_worker_frame(self, worker: _Worker, frame: Frame) -> None:
        if not frame.is_response:
            if frame.is_request and (
                isinstance(frame.id, bool)
                or not isinstance(frame.id, int)
                or frame.id >> ID_SHIFT != worker.k
            ):
                logger.warning(
                    "worker %d sent request id %r outside its range; replies will be dropped",
                    worker.k,
                    frame.id,
                )
            await self._emit(frame.raw, frame.message)
            return
        pending = self._pending.get(_key(frame.id))
        if pending is None or pending.worker is not worker:
            logger.warning(
                "worker %d: dropped a response to unknown request %r", worker.k, frame.id
            )
            return
        del self._pending[_key(frame.id)]
        self._settle(worker, pending, frame)
        await self._emit(frame.raw, frame.message)
        if not worker.sessions and not self._has_pending(worker):
            self._close_worker(worker)

    def _settle(self, worker: _Worker, pending: _Pending, frame: Frame) -> None:
        """Update the session map for a worker's answer, before the client sees it."""
        ok = "error" not in frame.message
        session_id = pending.session_id
        method = pending.method
        if method == "session/new" and ok:
            result = frame.message.get("result")
            new_id = result.get("sessionId") if isinstance(result, dict) else None
            if isinstance(new_id, str):
                self._map(new_id, worker)
                worker.root_id = worker.root_id or new_id
                session_id = new_id
        elif method == "session/load" and not ok and pending.mapped and session_id is not None:
            self._unmap(session_id, worker)
        elif method == "session/load" and ok and session_id is not None and not worker.stopping:
            # A close answered while this load was in flight unmapped the id;
            # the load succeeded, so the session is open on this worker again.
            self._map(session_id, worker)
        elif method in ("session/close", _DELETE_METHOD) and ok and session_id is not None:
            ids = list(worker.sessions) if session_id == worker.root_id else [session_id]
            for each in ids:
                self._unmap(each, worker)
        if method in ("session/new", "session/load"):
            forward_ms = (time.perf_counter() - max(pending.sent_at, worker.ready_at or 0)) * 1000
            logger.info(
                "worker %d %s %s: spawn_ms=%.0f handshake_ms=%.0f forward_ms=%.0f",
                worker.k,
                method,
                session_id,
                worker.spawn_ms,
                worker.handshake_ms,
                forward_ms,
            )

    def _has_pending(self, worker: _Worker) -> bool:
        return any(pending.worker is worker for pending in self._pending.values())

    def _close_worker(self, worker: _Worker) -> None:
        """Close the router's end of the worker's socket; the worker exits on end of stream."""
        if worker.stopping:
            return
        worker.stopping = True
        for session_id in list(worker.sessions):
            self._unmap(session_id, worker)
        worker.outbound.put_nowait(None)
        if not self._closing:
            self._task(self._reap_later(worker), name=f"nooa-router-reap-{worker.k}")

    async def _reap_later(self, worker: _Worker) -> None:
        try:
            await asyncio.wait_for(worker.exited.wait(), self._stop_grace)
        except TimeoutError:
            logger.warning("worker %d did not exit after its socket closed; killing it", worker.k)
            self._kill(worker)

    def _kill(self, worker: _Worker) -> None:
        if worker.process is None:
            return
        try:
            worker.process.kill()
        except Exception:
            logger.debug("worker %d: kill failed", worker.k, exc_info=True)

    async def _worker_exited(self, worker: _Worker, process: WorkerProcess) -> None:
        """End of a worker's socket: tell the client, then reap the process."""
        worker.live = False
        self._workers.pop(worker.k, None)
        sessions = sorted(worker.sessions)
        for session_id in sessions:
            self._unmap(session_id, worker)
        code: int | None = None
        with contextlib.suppress(Exception):
            code = await asyncio.wait_for(process.wait(), 1.0)
        if not self._closing:
            if not worker.stopping:
                logger.warning("worker %d pid %s exited (code %s)", worker.k, process.pid, code)
            for session_id in sessions:
                await self._emit_message(_worker_exited_update(session_id))
            await self._answer_orphans(worker, code)
        # The group may hold processes the worker started; kill it even after a clean exit.
        self._kill(worker)
        worker.exited.set()

    async def _worker_failed(self, worker: _Worker, reason: str) -> None:
        worker.live = False
        self._workers.pop(worker.k, None)
        for session_id in list(worker.sessions):
            self._unmap(session_id, worker)
        self._kill(worker)
        worker.exited.set()
        for key, pending in list(self._pending.items()):
            if pending.worker is worker:
                del self._pending[key]
                error = RequestError.internal_error({"details": reason})
                await self._emit_message(
                    {"jsonrpc": "2.0", "id": pending.id, "error": error.to_error_obj()}
                )

    async def _answer_orphans(self, worker: _Worker, code: int | None) -> None:
        for key, pending in list(self._pending.items()):
            if pending.worker is not worker:
                continue
            del self._pending[key]
            if pending.method == "session/close":
                # The session is gone with its worker, which is what close asked for.
                await self._emit_message({"jsonrpc": "2.0", "id": pending.id, "result": {}})
            else:
                error = _exited_error(pending.session_id, code)
                await self._emit_message(
                    {"jsonrpc": "2.0", "id": pending.id, "error": error.to_error_obj()}
                )


def _worker_exited_update(session_id: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": {
                "sessionUpdate": "session_info_update",
                "_meta": {"status": "worker_exited"},
            },
        },
    }


def _encode_lenient(message: dict[str, Any]) -> bytes:
    try:
        return encode(message)
    except (TypeError, ValueError):
        # Validation error details can hold objects JSON cannot encode.
        return (json.dumps(message, separators=(",", ":"), default=str) + "\n").encode()


def _exited_error(session_id: str | None, code: int | None) -> RequestError:
    what = f"the worker for session {session_id}" if session_id else "the worker"
    status = f" (exit code {code})" if code is not None else ""
    return RequestError.internal_error({"details": f"{what} exited{status}"})
