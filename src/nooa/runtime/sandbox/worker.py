# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox worker: the locked-down child process that runs cells.

Lifecycle inside the child (all before the op loop, so no cell ever runs
unguarded):

1. Build the persistent cell namespace from the forked agent (module globals,
   framework builtins, ``self`` -> :class:`ParentAgentProxy`).
2. ``install_guards`` — apply the irrevocable OS guardrails to this process.
3. Serve ``run`` / ``ping`` / ``shutdown`` requests over the pipe, executing each
   cell with :func:`run_cell_source` and streaming ``self.*`` calls back to the
   parent via the broker.
"""

from __future__ import annotations

import asyncio
import os
import threading
import weakref
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any, cast

from nooa.errors.formatting import IPythonErrorFormatter
from nooa.runtime.sandbox import wire
from nooa.runtime.sandbox.cell_core import run_cell_source
from nooa.runtime.sandbox.serialization import ResultDTO, dto_to_wire, result_to_dto

# Private state (the broker, and thus the parent pipe) for every proxy handed to
# the cell is kept OUT of the object's ``__dict__``, in this module-private
# weak-keyed registry. So ``self._broker`` is not a proxy attribute: it resolves
# through ``__getattr__`` against the agent (which has no such attribute) and
# fails, rather than handing the cell the live pipe. Legitimately-exposed private
# agent attributes (``self._foo``) still broker normally, matching in-process.
# Hiding the pipe is hygiene, not the boundary: the parent treats everything this
# process sends as hostile and decodes it as msgpack (``wire.py``), never pickle.
_PROXY_STATE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

_NO_PAYLOAD = object()
_CODEC = wire.Codec()

# Capture a dedicated formatter before untrusted cell code runs. In particular,
# mutating the module-level default formatter cannot forge worker diagnostics.
_WORKER_ERROR_FORMATTER = IPythonErrorFormatter().format


class ParentToolError(RuntimeError):
    """Raised in the worker when a parent-brokered ``self.*`` call fails."""


class ChildBroker:
    """Child-side RPC client: forward ``self.*`` access to the parent's agent.

    Access is addressed by a dotted *path* (e.g. ``["memory", "remember"]`` for
    ``self.memory.remember``), so nested attributes of the live agent are reachable
    without their intermediate objects ever crossing the process boundary.
    """

    def __init__(self, conn: Connection, send_lock: threading.RLock):
        self._conn = conn
        self._send_lock = send_lock
        self._n = 0

    def _rpc(self, envelope: dict[str, Any], payload: Any = _NO_PAYLOAD) -> dict[str, Any]:
        # The cell's value (call arguments or an assigned attribute) is encoded
        # separately into ``payload`` so the parent can refuse it per call.
        # Hold the lock across the whole request/response so concurrent self.*
        # calls from threads inside a cell serialize on the single pipe (the id
        # counter and the send/recv pair must not interleave).
        with self._send_lock:
            self._n += 1
            call_id = self._n
            envelope["tool_call_id"] = call_id
            if payload is not _NO_PAYLOAD:
                try:
                    envelope["payload"] = _CODEC.dumps(payload)
                except Exception as exc:  # noqa: BLE001 - unsupported type, depth, user __str__...
                    from nooa.runtime.sandbox.errors import CellSerializationError

                    target = ".".join(envelope.get("path") or [])
                    what = (
                        f"The value assigned to self.{target}"
                        if envelope.get("kind") == "setattr"
                        else f"Arguments to self.{target}(...)"
                    )
                    raise CellSerializationError(
                        f"{what} cannot cross the sandbox boundary: {exc}. {wire.HINT}"
                    ) from exc
            wire.send(self._conn, envelope)
            response = self._conn.recv()
        if (
            not isinstance(response, dict)
            or response.get("type") != "tool_result"
            or response.get("tool_call_id") != call_id
        ):
            raise ParentToolError("sandbox broker got an unexpected response")
        if not response.get("ok"):
            _raise_broker_error(response)
        return response

    def call(self, path: list[str], args: tuple, kwargs: dict) -> Any:
        return self.call_result(path, args, kwargs)[0]

    def call_result(self, path: list[str], args: tuple, kwargs: dict) -> tuple[Any, bool]:
        """Like :meth:`call` but also reports whether the parent awaited a coroutine.

        Returns ``(result, was_async)``. Callers that must preserve ``await``
        semantics on a proxy (see :meth:`_NestedProxy.__call__`) re-wrap the
        already-resolved ``result`` in an awaitable when ``was_async`` is True.
        """
        resp = self._rpc({"type": "tool_call", "kind": "call", "path": path}, (args, kwargs))
        return resp.get("result"), bool(resp.get("was_async"))

    def get_attr(self, path: list[str]) -> tuple[Any, bool, bool, bool]:
        """Return ``(value, is_proxy, is_callable, is_async)``.

        The callable/async flags let a spawned worker — which has no local agent
        to classify against — decide whether to hand back a broker callable or a
        plain value. They come from the parent's live object, so they are also
        correct for attributes the forked copy never saw.
        """
        resp = self._rpc({"type": "tool_call", "kind": "attr", "path": path})
        return (
            resp.get("result"),
            bool(resp.get("proxy")),
            bool(resp.get("callable")),
            bool(resp.get("is_async")),
        )

    def introspect(self, name: str, args: tuple, kwargs: dict) -> str:
        """Run ``doc`` / ``methods`` / ``variables`` against the parent's agent."""
        resp = self._rpc(
            {"type": "tool_call", "kind": "introspect", "path": [], "name": name},
            (args, kwargs),
        )
        return str(resp.get("result") or "")

    def set_attr(self, path: list[str], value: Any) -> None:
        """Assign ``value`` at ``path`` on the parent's live agent (``self.x = value``)."""
        self._rpc({"type": "tool_call", "kind": "setattr", "path": path}, value)

    def iterate(self, path: list[str]) -> list[Any]:
        """Return ``list(obj)`` for the live object at ``path`` (materialized on the parent)."""
        return self._rpc({"type": "tool_call", "kind": "iter", "path": path}).get("result") or []


def _raise_broker_error(response: dict[str, Any]) -> None:
    import builtins as _bi

    from nooa.runtime.sandbox.errors import CellSerializationError

    err_type = str(response.get("error_type") or "RuntimeError")
    message = str(response.get("error") or "")
    if err_type == "ExecutionSignal":
        # A brokered tool raised a control-flow signal (e.g. return_result via an
        # ARC submit_actions). Re-raise it in the cell so it flows through the
        # normal signal path (cell_core catches it -> ExecutionResult.signal).
        from nooa.strategies.codeact import _ReturnResultSignal

        signal_result = response.get("signal_result")
        if response.get("signal_result_dropped", False):
            raise CellSerializationError(
                "A control-flow signal carried a value that could not cross the "
                "sandbox boundary. Return a JSON/pickle-safe value instead."
            )
        raise _ReturnResultSignal(result=cast(Any, signal_result))
    if err_type == "CellSerializationError":
        raise CellSerializationError(message)
    exc_cls = getattr(_bi, err_type, ParentToolError)
    if not isinstance(exc_cls, type) or not issubclass(exc_cls, BaseException):
        exc_cls = ParentToolError
    try:
        error = exc_cls(message)
    except Exception:
        # Some builtin exceptions (for example UnicodeDecodeError) require a
        # structured constructor. Preserve the remote type name rather than
        # replacing the actual failure with that constructor's TypeError.
        error = ParentToolError(message)
        error.original_type = err_type  # type: ignore[attr-defined]
    call_hint = response.get("call_hint")
    if isinstance(call_hint, str) and call_hint:
        error._nooa_call_hint = call_hint  # type: ignore[attr-defined]
    raise error


def _is_async_callable(obj: Any) -> bool:
    """True if calling ``obj`` yields a coroutine (``async def`` method/function).

    Callable *instances* whose ``__call__`` is async are not detected here; agent
    tools and generation methods are plain ``async def`` methods, which are.
    """
    import inspect

    return inspect.iscoroutinefunction(obj)


class _BrokerCallable:
    """A ``self.<path>`` handle that RPCs the call to the parent's live agent.

    For an ``async def`` method the cell writes ``await self.method(...)``; the
    parent resolves the coroutine and returns the value, so here we hand back an
    awaitable that yields that already-resolved value. For a plain method we
    return the value directly. The broker (and its pipe) is held in
    :data:`_PROXY_STATE`, never as an instance attribute.
    """

    def __init__(self, broker: ChildBroker, path: list[str], is_async: bool):
        _PROXY_STATE[self] = broker
        self._path = path
        self._is_async = is_async
        self.__name__ = path[-1] if path else "?"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        broker = _PROXY_STATE[self]
        path = self._path
        if self._is_async:

            async def _awaitable() -> Any:
                return broker.call(path, args, kwargs)

            return _awaitable()
        return broker.call(path, args, kwargs)

    def __repr__(self) -> str:
        return f"<sandbox parent-proxy self.{'.'.join(self._path)}>"


def _resolve_child_attr(broker: ChildBroker, local_obj: Any, path: list[str], name: str) -> Any:
    """Resolve ``local_obj.name`` for the cell: broker callable, value, or nested proxy.

    ``path`` is the dotted address of ``local_obj`` under ``self`` (empty for the
    agent itself). Callables become broker callables; picklable values are fetched
    fresh from the parent; non-picklable live objects become a nested proxy so
    ``self.a.b(...)`` keeps working without ``a`` ever crossing the boundary.

    ``local_obj`` is ``None`` in a spawned worker, which has no inherited agent.
    The callable/async classification then comes from the parent's response
    instead of a local ``getattr``.
    """
    new_path = [*path, name]
    if local_obj is None:
        value, is_proxy, is_callable, is_async = broker.get_attr(new_path)
        if is_proxy:
            return _NestedProxy(broker, new_path, None)
        if is_callable:
            return _BrokerCallable(broker, new_path, is_async)
        return value
    try:
        local_attr = getattr(local_obj, name)
    except AttributeError:
        value, is_proxy, _callable, _async = broker.get_attr(new_path)
        return _NestedProxy(broker, new_path, None) if is_proxy else value
    if callable(local_attr) and not isinstance(local_attr, type):
        return _BrokerCallable(broker, new_path, _is_async_callable(local_attr))
    value, is_proxy, _callable, _async = broker.get_attr(new_path)
    return _NestedProxy(broker, new_path, local_attr) if is_proxy else value


class _NestedProxy:
    """Proxy for a non-picklable nested attribute (e.g. ``self.memory``).

    Method calls broker to the parent's live object at this path; deeper attribute
    access recurses. Broker/path/local state lives in :data:`_PROXY_STATE`, not on
    the instance, so the pipe stays unreachable from cell code.
    """

    def __init__(self, broker: ChildBroker, path: list[str], local_obj: Any):
        _PROXY_STATE[self] = (broker, path, local_obj)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        broker, path, local = _PROXY_STATE[self]
        if local is None:
            # Unknown local shape (spawned worker, or a non-picklable parent
            # object): let the parent classify callable vs value.
            value, is_proxy, is_callable, is_async = broker.get_attr([*path, name])
            if is_callable:
                return _BrokerCallable(broker, [*path, name], is_async)
            return _NestedProxy(broker, [*path, name], None) if is_proxy else value
        return _resolve_child_attr(broker, local, path, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("__") and name.endswith("__"):
            object.__setattr__(self, name, value)
            return
        broker, path, _local = _PROXY_STATE[self]
        broker.set_attr([*path, name], value)

    # Container/callable protocol: forward the operator to the parent's live object
    # so documented patterns like ``self.context["k"] = v`` keep working. Each
    # brokers the corresponding dunder as a method call on the parent object.
    def __getitem__(self, key: Any) -> Any:
        broker, path, _ = _PROXY_STATE[self]
        return broker.call([*path, "__getitem__"], (key,), {})

    def __setitem__(self, key: Any, value: Any) -> None:
        broker, path, _ = _PROXY_STATE[self]
        broker.call([*path, "__setitem__"], (key, value), {})

    def __delitem__(self, key: Any) -> None:
        broker, path, _ = _PROXY_STATE[self]
        broker.call([*path, "__delitem__"], (key,), {})

    def __contains__(self, item: Any) -> bool:
        broker, path, _ = _PROXY_STATE[self]
        return bool(broker.call([*path, "__contains__"], (item,), {}))

    def __len__(self) -> int:
        broker, path, _ = _PROXY_STATE[self]
        return int(broker.call([*path, "__len__"], (), {}))

    def __iter__(self) -> Any:
        broker, path, _ = _PROXY_STATE[self]
        # The parent's iterator isn't picklable, so materialize a list there.
        return iter(broker.iterate(path))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        broker, path, _ = _PROXY_STATE[self]
        # local is None here (unknown shape), so async-ness isn't known up front
        # like it is for _BrokerCallable. Ask the parent: if it awaited a
        # coroutine, re-wrap the resolved value in an awaitable so the cell's
        # ``await proxy(...)`` works (and a bare call still yields a coroutine,
        # matching in-process semantics).
        result, was_async = broker.call_result(path, args, kwargs)
        if was_async:

            async def _awaitable() -> Any:
                return result

            return _awaitable()
        return result

    def __bool__(self) -> bool:
        # A live object exists -> truthy. Prevents Python's implicit __len__
        # fallback from brokering a __len__ call for a plain ``if self.x:`` check.
        return True

    def __repr__(self) -> str:
        return f"<sandbox nested-proxy self.{'.'.join(_PROXY_STATE[self][1])}>"


class ParentAgentProxy:
    """Stands in for ``self`` in the cell namespace.

    Method access returns a broker callable (the call runs on the parent's live
    agent); non-callable attribute access fetches a fresh picklable snapshot from
    the parent, so agent state the parent mutates between turns stays authoritative;
    a non-picklable live attribute becomes a nested proxy.

    ``local_agent`` is the forked copy when the start method can inherit one, and
    ``None`` in a spawned worker. Without it, callable/value classification is
    delegated to the parent rather than guessed locally.

    The broker and forked agent live in :data:`_PROXY_STATE`, not in instance
    attributes, so ``self._broker`` resolves against the agent (no such attribute)
    and fails instead of exposing the parent pipe.
    """

    def __init__(self, broker: ChildBroker, local_agent: Any | None):
        _PROXY_STATE[self] = (broker, local_agent)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        broker, local = _PROXY_STATE[self]
        return _resolve_child_attr(broker, local, [], name)

    def __setattr__(self, name: str, value: Any) -> None:
        # ``self.x = value`` in a cell must reach the parent's live agent, not bind
        # onto the worker-side proxy (which would silently diverge and be lost on
        # worker restart). Broker the assignment to the parent.
        if name.startswith("__") and name.endswith("__"):
            object.__setattr__(self, name, value)
            return
        broker, _local = _PROXY_STATE[self]
        broker.set_attr([name], value)

    def __repr__(self) -> str:
        local = _PROXY_STATE[self][1]
        kind = type(local).__name__ if local is not None else "spawned"
        return f"<sandbox self-proxy for {kind}>"


def _introspect_wrapper(fn: Any, proxy: ParentAgentProxy, agent: Any) -> Any:
    """Wrap ``doc``/``methods``/``variables`` so ``doc(self)`` works on the proxy.

    With a forked agent the call runs locally against that copy. In a spawned
    worker there is no local instance, so the call is brokered to the parent,
    which renders against the live agent and returns the string.
    """

    def wrapper(obj: Any = proxy, *args: Any, **kwargs: Any) -> Any:
        if obj is not proxy:
            return fn(obj, *args, **kwargs)
        if agent is None:
            broker = _PROXY_STATE[proxy][0]
            # The proxy itself must not cross the boundary; the parent renders
            # against its own live agent.
            rest = tuple(a for a in args if a is not proxy)
            return broker.introspect(fn.__name__, rest, kwargs)
        return fn(agent, *args, **kwargs)

    return wrapper


def build_namespace(
    agent: Any,
    framework_builtins: dict[str, Any],
    proxy: ParentAgentProxy,
    restrictions: Any,
    *,
    agent_class: type | None = None,
) -> dict[str, Any]:
    """Build the worker's persistent cell namespace, mirroring actor exec_globals.

    ``agent`` is the inherited instance under ``fork``. A spawned worker passes
    ``agent=None`` plus the imported ``agent_class``: module-level globals only
    depend on the class MRO, so the namespace is identical either way.
    """
    import typing as _typing

    from nooa.agentdoc import doc
    from nooa.agentdoc.introspect import methods, variables
    from nooa.agentdoc.visibility import filter_mro_module_globals
    from nooa.decorators import strategy
    from nooa.media import Audio, File, Image, Media, Video
    from nooa.runtime.media_capture import show
    from nooa.runtime.pprint import pprint
    from nooa.runtime.restrictions import RestrictionsConfig

    cls = agent_class if agent_class is not None else type(agent)

    # Only ``self.*`` brokers to the live parent; module-level state is a private
    # copy in this worker, so a cell mutating it would silently diverge from the
    # in-process backend. Wrap mutable module globals read-only so such a mutation
    # fails loud (SandboxStateError) instead. Applied to the module globals only,
    # before ``self`` / the framework builtins are added below.
    from nooa.runtime.sandbox.readonly import freeze_module_state
    from nooa.strategies import (
        CodeActStrategy,
        CompositeStrategy,
        PredictStrategy,
        ReflexionStrategy,
        TemplateStrategy,
    )
    from nooa.strategies.pure_python import PurePythonStrategy

    ns: dict[str, Any] = freeze_module_state(filter_mro_module_globals(cls))
    ns.update(
        {
            "self": proxy,
            "asyncio": asyncio,
            "typing": _typing,
            "Annotated": _typing.Annotated,
            "Any": _typing.Any,
            "Literal": _typing.Literal,
            "Optional": _typing.Optional,
            "Union": _typing.Union,
            "doc": _introspect_wrapper(doc, proxy, agent),
            "methods": _introspect_wrapper(methods, proxy, agent),
            "variables": _introspect_wrapper(variables, proxy, agent),
            "help": _introspect_wrapper(doc, proxy, agent),
            "pprint": pprint,
            "show": show,
            "Media": Media,
            "Image": Image,
            "Audio": Audio,
            "Video": Video,
            "File": File,
            "strategy": strategy,
            "PurePythonStrategy": PurePythonStrategy,
            "CodeActStrategy": CodeActStrategy,
            "ReflexionStrategy": ReflexionStrategy,
            "PredictStrategy": PredictStrategy,
            "TemplateStrategy": TemplateStrategy,
            "CompositeStrategy": CompositeStrategy,
        }
    )
    if framework_builtins:
        ns.update(framework_builtins)

    from nooa.runtime.actor import _strip_blocked_modules

    effective = restrictions or RestrictionsConfig()
    ns = _strip_blocked_modules(ns, effective.blocked_modules)
    return ns


def worker_main(  # pragma: no cover - child
    conn: Connection,
    init: dict[str, Any],
    *,
    on_ready: Callable[[], None] | None = None,
) -> None:
    """Child entrypoint: build namespace, install guards, serve requests.

    ``init`` has one of two shapes:

    * ``{"mode": "fork", ...}`` — inherited live agent and framework builtins;
    * ``{"mode": "blueprint", "blueprint": WorkerBlueprint, ...}`` — the spawned
      path, where everything is rebuilt from picklable data.

    ``on_ready`` runs once the namespace exists and guards are installed, just
    before the op loop. Callers that create the process themselves (the Windows
    AppContainer launcher) use it to tell the parent the worker is usable — the
    handshake must not be sent earlier, or the parent's first cell would be
    charged for the child's import cost.
    """
    from nooa.runtime.sandbox.guards import install_guards

    send_lock = threading.RLock()
    try:
        broker = ChildBroker(conn, send_lock)
        if init.get("mode") == "blueprint":
            blueprint = init["blueprint"]
            agent = None
            agent_class = blueprint.import_agent_class()
            framework_builtins = blueprint.rebuild_builtins()
            proxy = ParentAgentProxy(broker, None)
            namespace = build_namespace(
                agent,
                framework_builtins,
                proxy,
                init.get("restrictions"),
                agent_class=agent_class,
            )
        else:
            proxy = ParentAgentProxy(broker, init["agent"])
            namespace = build_namespace(
                init["agent"],
                init.get("framework_builtins") or {},
                proxy,
                init.get("restrictions"),
            )
        # Guards go on AFTER the (trusted) namespace build and BEFORE the op loop,
        # so every cell — and nothing else — runs under the full lock-down.
        install_guards(init["spec"])
        if on_ready is not None:
            on_ready()
    except BaseException as exc:  # noqa: BLE001 - report and exit; parent fails closed
        try:
            wire.send(conn, {"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
        os._exit(3)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while True:
            try:
                request = conn.recv()  # parent -> worker: the parent is trusted
            except (EOFError, OSError):
                return
            op = request.get("op")
            req_id = request.get("id")
            if op == "shutdown":
                return
            if op == "ping":
                wire.send(conn, {"type": "response", "id": req_id, "ok": True})
                continue
            if op == "run":
                dto = _run_one(
                    loop,
                    namespace,
                    request,
                    max_error=init.get("max_error"),
                    tail_chars=init.get("error_tail"),
                )
                wire.send(
                    conn,
                    {"type": "response", "id": req_id, "ok": True, "result": dto_to_wire(dto)},
                )
                continue
            wire.send(
                conn,
                {"type": "response", "id": req_id, "ok": False, "error": f"unknown op {op!r}"},
            )
    finally:
        loop.close()


def _run_one(
    loop: asyncio.AbstractEventLoop,
    namespace: dict[str, Any],
    request: dict[str, Any],
    _serialize: Any = result_to_dto,
    _error_formatter: Any = _WORKER_ERROR_FORMATTER,
    *,
    max_error: int | None = None,
    tail_chars: int | None = None,
) -> ResultDTO:
    result = loop.run_until_complete(
        run_cell_source(
            request["code"],
            namespace,
            execution_count=int(request.get("execution_count", 1)),
        )
    )
    return _serialize(
        result,
        error_formatter=_error_formatter,
        max_error=max_error,
        tail_chars=tail_chars,
    )
