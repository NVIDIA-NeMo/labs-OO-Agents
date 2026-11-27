# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ACP agent over the Session layer: protocol translation, no session policy.

``AtomACPAgent`` serves ACP v2 on a ``SessionRegistry``: each request
method takes the request's v2 params and returns its v2 result, and the
client is a ``Peer`` (``nooa_atom.acp.v2``). Sessions, turns, steering,
cancellation and checkpoints are the Session's; this module maps requests
onto it and keeps one ``ACPEventBridge`` per session id that sends the
session's activity to the client.

``serve_connection`` serves it to a v2 client directly (``v2.serve_v2``)
and to a v1 client through the v1 shim (``nooa_atom.acp.v1_shim``), which
also uses the few methods documented as its own here (``start_prompt``,
``finish_item``...). See ``docs/acp-v2.md``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import signal
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext, suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from acp import RequestError

from nooa.errors import GenerationError
from nooa.mcp import MCPManager, MCPTool
from nooa.slash_dispatch import CoercionError
from nooa.storage.sqlite import SessionAlreadyActiveError
from nooa_atom.acp import v2
from nooa_atom.acp.event_bridge import (
    VERBOSITY_LEVELS,
    ACPEventBridge,
    agent_message,
    cancel_text,
    end_line,
    limit_stop_reason,
    pool_input_event,
)
from nooa_atom.acp.framing import FRAME_LIMIT, read_frame
from nooa_atom.acp.listing import list_sessions, validate_workspace
from nooa_atom.acp.protocol import (
    INJECT_CAPABILITY,
    POOL_DIALECT,
    POOLSIDE_DIALECT,
    PoolDialect,
    open_stdio,
    pool_dialects,
)
from nooa_atom.acp.recover import COMMAND as RECOVER
from nooa_atom.acp.recover import recover
from nooa_atom.acp.v2 import Peer
from nooa_atom.agent.slash_commands import RESERVED_COMMAND_NAMES
from nooa_atom.session.items import (
    CommandInfo,
    CommandsChangedUpdate,
    Receipt,
    TurnCancelledOutcome,
)
from nooa_atom.session.loader import ATOM_AGENT, canonical_agent_spec
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import ChildActiveElsewhereError, SessionRegistry
from nooa_atom.session.session import (
    ItemWithdrawnError,
    Session,
    SessionClosedError,
    TurnFailedError,
)
from nooa_atom.session.store import (
    InvalidSessionIdError,
    SessionNotFoundError,
    SessionStore,
    sessions_root,
)
from nooa_atom.session.tracing import trace_in_relay
from nooa_atom.workspace.settings import load_acp_verbosity, write_settings_updates

_VERBOSITY_HELP = (
    "silent: no activity; concise: one-line notices; verbose: full activity (default)."
)


logger = logging.getLogger(__name__)

_DELETE_METHOD = "nooa/session/delete"
"""``_nooa/session/delete`` as ``ext_method`` receives it (without the underscore)."""
# The reasoning option's first choice: no level chosen, the model's own default applies.
DEFAULT_REASONING = "default"

_INJECT_METHOD = "nooa/session/inject"
_REVOKE_METHOD = "nooa/session/revoke_inject"
_POOL_STEER_METHODS = {
    dialect.steer_method: dialect for dialect in (POOLSIDE_DIALECT, POOL_DIALECT)
}
"""``_poolside/session_steer`` and ``_pool/session_steer`` as ``ext_method``
receives them (without the underscore), whatever the client's dialect."""

_POOL_SHELL_METHOD = "pool/execute_shell"
"""Pool 2's ``!`` shortcut as ``ext_method`` receives it. Atom does not run it."""

_POOL_SHELL_UNSUPPORTED = (
    "NOOA Atom does not support ! shell commands. Ask the agent to run the command instead."
)

SOURCE = "acp"
"""Already-visible client prompts/Pool inputs and permission answers; do not echo."""

INJECT_SOURCE = "acp:inject"
"""Injected inputs are not ordinary prompts: echo their admission once."""

_CANCELLED = object()  # a client request stopped by session/cancel
_DISMISSED = object()  # the person declined or dismissed a local form

_CONNECT_TEXT = (
    "Connecting a model provider needs the terminal for now: run `nooa connect` in a "
    "shell, then pick the new model alias here."
)


MODES = ("auto",)
"""The permission modes a session can have: only ``auto`` until tools can ask first."""


_detached_closes: set[asyncio.Future[Any]] = set()
"""Closes started by a cancelled ``_close_in_order``; kept so they are not collected."""


def _detached_close_done(task: asyncio.Future[Any]) -> None:
    _detached_closes.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.warning("A close failed after cancellation", exc_info=task.exception())


async def _close_in_order(*closers: Callable[[], Any] | None) -> None:
    """Run each closer in order, tolerating failures, then re-raise.

    Equivalent to nesting one ``try/finally`` per closer: every closer runs
    even if an earlier one fails, and the last failure propagates with the
    earlier ones chained as its ``__context__``.

    A cancellation propagates at once: the closers not yet run are started
    in the background, not waited for, so cleanup does not hold it up.
    """
    remaining = [close for close in closers if close is not None]
    pending: BaseException | None = None
    try:
        while remaining:
            close = remaining.pop(0)
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                if pending is not None:
                    exc.__context__ = pending
                pending = exc
    finally:
        for close in remaining:  # left over only when cancelled
            try:
                result = close()
                if inspect.isawaitable(result):
                    started = asyncio.ensure_future(result)
                    _detached_closes.add(started)
                    started.add_done_callback(_detached_close_done)
            except Exception:
                logger.warning("A close failed after cancellation", exc_info=True)
    if pending is not None:
        raise pending


def _default_model(workspace: Path) -> str | None:
    """The first configured alias; the client can change it with ``session/set_config_option``."""
    from nooa_atom.workspace.models import default_model

    try:
        return default_model(workspace)
    except Exception:
        logger.warning("Could not load the model configuration of %s", workspace, exc_info=True)
        return None


def model_aliases(workspace: Path) -> list[str]:
    """The model aliases configured for ``workspace``, sorted.

    Its own ``.nooa/llm_config.yaml`` included (``workspace_models``). The
    files are read again whenever one was added, removed or changed, so a
    model connected with ``nooa connect`` in a terminal shows up in the
    picker without restarting the server.
    """
    from nooa_atom.workspace.models import workspace_models

    try:
        return sorted(workspace_models(workspace))
    except Exception:
        logger.warning("Could not load the model configuration of %s", workspace, exc_info=True)
        return []


class AtomACPAgent:
    """One ACP connection's view of the sessions this process runs.

    Sessions are stored per workspace (``sessions_root``): a request's
    ``cwd`` picks the store, and each store has its own
    ``SessionRegistry``, built by ``new_registry(store)`` the first time
    the client names that workspace. ``sessions_dir`` is one directory
    for every workspace instead (``None``: ``NOOA_SESSIONS_DIR``, else each
    workspace's ``.nooa/sessions``).

    ``agent_spec`` names the agent class for new sessions (``None``: the
    workspace's ``atom.agent_spec`` setting, else the Atom agent).
    ``model`` is the model alias new sessions start with; the registry's
    ``llm_factory`` builds the client.
    """

    def __init__(
        self,
        new_registry: Callable[[SessionStore], SessionRegistry],
        *,
        sessions_dir: Path | None = None,
        agent_spec: str | None = None,
        model: str | None = None,
    ) -> None:
        self._new_registry = new_registry
        self._sessions_dir = sessions_dir
        # By store directory: two workspaces sharing one directory share a registry.
        self._registries: dict[Path, SessionRegistry] = {}
        self._agent_spec = agent_spec
        self._model = model
        self._conn: Peer | None = None
        # The client's v2 ``capabilities`` and ``info`` from ``initialize``.
        self.client_capabilities: dict[str, Any] = {}
        self.client_info: dict[str, Any] | None = None
        # The client is Pool (``clientInfo.name``): it gets Pool's input events.
        self._pool_client = False
        # The Pool dialects of this connection (``pool_dialects``): which
        # meta keys input events carry and which form request Pool gets.
        self._pool_dialects: tuple[PoolDialect, ...] = ()
        # The client is Pool and has not failed a Pool form request.
        self._pool_forms = False
        self._bridges: dict[str, ACPEventBridge] = {}
        self._background: set[asyncio.Task[None]] = set()
        self._title_checked: set[str] = set()
        # Receipts of prompts still waiting for their turn, by session: Stop
        # answers them "cancelled" and withdraws the ones not yet consumed.
        self._prompts: dict[str, list[Receipt]] = {}
        # Message ids of answered prompts, by session and idempotency key.
        self._idempotent: dict[tuple[str, str], str] = {}
        # Receipts of injected messages, by session and item id, for revoke.
        self._injects: dict[str, dict[str, Receipt]] = {}
        # Client requests (forms, permissions) a prompt is waiting on, by
        # session: session/cancel stops them.
        self._asks: dict[str, set[asyncio.Future[Any]]] = {}
        self._ask_epochs: dict[str, int] = {}
        # The question each session's prompt is asking, so two prompts that
        # returned with the same turn ask it once.
        # Workspaces whose model configuration files were logged.
        self._logged_config: set[Path] = set()

    def on_connect(self, conn: Peer) -> None:
        self._conn = conn

    def _log_llm_config(self, workspace: Path) -> None:
        """Log the model configuration files of a workspace, the first time it is used."""
        if workspace not in self._logged_config:
            from nooa_atom.acp.cli import llm_config_summary

            self._logged_config.add(workspace)
            logger.info(llm_config_summary(workspace))

    # ---- registries ----------------------------------------------------

    def registry_for(self, workspace: Path) -> SessionRegistry:
        """The registry of the store that holds ``workspace``'s sessions."""
        root = sessions_root(workspace, self._sessions_dir).resolve()
        registry = self._registries.get(root)
        if registry is None:
            registry = self._registries[root] = self._new_registry(SessionStore(root))
        return registry

    def session(self, session_id: str) -> Session | None:
        """The live session with this id, in any of this connection's registries."""
        for registry in self._registries.values():
            session = registry.get(session_id)
            if session is not None:
                return session
        return None

    def _registry_holding(self, session_id: str) -> SessionRegistry | None:
        """The registry whose store has this session: live, else on disk."""
        registries = list(self._registries.values())
        for registry in registries:
            if registry.get(session_id) is not None:
                return registry
        for registry in registries:
            with suppress(InvalidSessionIdError):
                if registry.store.path_for(session_id).exists():
                    return registry
        return None

    # ---- initialize ----------------------------------------------------

    async def initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """``initialize`` (v2 params): note the client's capabilities and who it is."""
        capabilities = params.get("capabilities")
        self.client_capabilities = capabilities if isinstance(capabilities, dict) else {}
        info = params.get("info")
        self.client_info = info if isinstance(info, dict) else None
        self._pool_dialects = pool_dialects(self.client_info)
        self._pool_client = bool(self._pool_dialects)
        self._pool_forms = self._pool_client
        return v2.initialize_result(self.client_info)

    def _use_pool_dialect(self, dialect: PoolDialect) -> None:
        """Speak only ``dialect`` from now on: the client just used its steer method.

        Only narrows a Pool connection that was speaking several dialects
        (its version did not say which); a known dialect stays as it is.
        """
        if len(self._pool_dialects) > 1 and dialect in self._pool_dialects:
            self._pool_dialects = (dialect,)
            for bridge in self._bridges.values():
                bridge.pool_dialects = self._pool_dialects

    # ---- new -----------------------------------------------------------

    async def new_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """``session/new``: ``{cwd, additionalDirectories?, mcpServers?}`` -> ``{sessionId, ...}``."""
        root = validate_workspace(params.get("cwd"), params.get("additionalDirectories"))
        self._log_llm_config(root)
        options = SessionOptions(
            workspace=root,
            agent_spec=self._agent_spec_for(root),
            model=self._model,
            host="acp",
        )
        warnings: list[str] = []
        attached: list[ACPEventBridge] = []
        mcp_servers = params.get("mcpServers")

        async def prepare(session: Session) -> None:
            _trace_as(session)
            trace_in_relay(session)
            attached.append(self._attach(session))
            warnings.extend(await self._prepare_tools(session, mcp_servers))

        try:
            session = await self.registry_for(root).create(options, prepare=prepare)
        except BaseException:
            for bridge in attached:
                self._bridges.pop(bridge.session_id, None)
                await bridge.close(finish_open=False)
            raise
        self._defer_bootstrap_updates(session, warnings)
        return {"sessionId": session.id, **self._setup_result(session)}

    def _setup_result(self, session: Session) -> dict[str, Any]:
        """What ``session/new`` and ``session/resume`` answer besides the id."""
        return {
            "configOptions": self._config_options(session),
            "availableCommands": _available_commands(session.commands()),
        }

    # ---- resume --------------------------------------------------------

    async def resume_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """``session/resume``: open a stored or live session, replaying it from ``replayFrom``.

        With ``replayFrom`` the transcript is replayed, then
        ``replay_complete``; the session's foreground state follows either
        way, before the answer.
        """
        session_id = params.get("sessionId")
        if not isinstance(session_id, str):
            raise v2.invalid("sessionId is required")
        replay = params.get("replayFrom") is not None
        root = validate_workspace(params.get("cwd"), params.get("additionalDirectories"))
        self._log_llm_config(root)
        registry = self.registry_for(root)
        live = self.session(session_id)
        if live is not None:
            # Attach: the same Session; reuse this adapter's bridge if it has
            # one (a second bridge would send every update twice) and replay.
            bridge = self._bridges.get(session_id) or self._attach(live)
            if replay:
                self._replay(bridge, live, pool=self._pool_client)
            bridge.snapshot_state()
            await bridge.flush()
            self._defer_bootstrap_updates(live, [])
            return self._setup_result(live)

        warnings: list[str] = []
        attached: list[ACPEventBridge] = []
        mcp_servers = params.get("mcpServers")

        async def prepare(session: Session) -> None:
            _trace_as(session)
            trace_in_relay(session)
            bridge = self._attach(session)
            attached.append(bridge)
            if replay:
                self._replay(bridge, session, pool=self._pool_client)
            warnings.extend(await self._prepare_tools(session, mcp_servers))

        try:
            session = await registry.load(session_id, prepare=prepare, host="acp")
        except BaseException as exc:
            for bridge in attached:
                self._bridges.pop(bridge.session_id, None)
                await bridge.close(finish_open=False)
            raise _load_error(session_id, exc) from exc
        bridge = self._bridges[session.id]
        bridge.snapshot_state()
        await bridge.flush()
        self._defer_bootstrap_updates(session, warnings)
        return self._setup_result(session)

    # ---- list ----------------------------------------------------------

    async def list_sessions(self, params: dict[str, Any]) -> dict[str, Any]:
        """Root sessions with at least one message, most recent first (see ``list_sessions``).

        Without ``cwd``, the stores of the workspaces this connection has
        named; there is no index of every workspace.
        """
        cwd, cursor = params.get("cwd"), params.get("cursor")
        if cwd is not None:
            self.registry_for(validate_workspace(cwd, None))

        def live(session_id: str) -> tuple[str, str | None] | None:
            for registry in self._registries.values():
                session = registry.get(session_id)
                if session is not None:
                    return registry.live_info(session).status, session.info.title
            return None

        response = await list_sessions(
            self._store_for,
            cwd=cwd,
            cursor=cursor,
            live=live,
            known=[registry.store for registry in self._registries.values()],
        )
        return response.model_dump(mode="json", by_alias=True, exclude_none=True)

    def _store_for(self, workspace: Path) -> SessionStore:
        return self.registry_for(workspace).store

    # ---- close and delete ----------------------------------------------

    async def close_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """Close a session; for a child whose parent is live, only stop following it."""
        session_id = params.get("sessionId")
        session = self.session(session_id) if isinstance(session_id, str) else None
        bridge = self._bridges.pop(session_id, None) if isinstance(session_id, str) else None
        if isinstance(session_id, str):
            self._cancel_client_calls(session_id)
        if session is None and bridge is None:
            raise RequestError.resource_not_found(session_id)
        if session is not None and self._parent_is_live(session):
            if bridge is not None:
                await bridge.close(finish_open=False)
            return {}
        if session is not None:
            await session.close()
        if bridge is not None:
            await bridge.close()
        return {}

    def is_subagent(self, session_id: str) -> bool:
        """Whether ``session_id`` is a live child this connection shows, not one it follows."""
        session = self.session(session_id)
        return (
            session is not None
            and session_id not in self._bridges
            and self._descends_from_followed(session)
        )

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """``_nooa/session/inject``, ``revoke_inject`` (see ``_inject``), Pool's
        ``_poolside/session_steer`` or ``_pool/session_steer`` (see
        ``_pool_steer``) and ``_nooa/session/delete`` (``delete_session``). Pool's
        ``_pool/execute_shell`` (its ``!`` shortcut) is refused with a message saying so.
        """
        if method == _INJECT_METHOD:
            return await self._inject(params)
        if method == _REVOKE_METHOD:
            return self._revoke_inject(params)
        dialect = _POOL_STEER_METHODS.get(method)
        if dialect is not None:
            return await self._pool_steer(params, dialect)
        if method == _POOL_SHELL_METHOD:
            # Still "method not found", with a message Pool shows to the user.
            raise RequestError(-32601, _POOL_SHELL_UNSUPPORTED, {"method": f"_{method}"})
        if method != _DELETE_METHOD:
            raise RequestError.method_not_found(f"_{method}")
        return await self.delete_session(params)

    async def delete_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """``session/delete``: ``sessionId``, optional ``keepFiles`` and ``cwd``.

        ``cwd`` names the workspace whose store holds the session; without
        it the stores of the workspaces this connection has named are searched.
        """
        session_id = params.get("sessionId")
        if not isinstance(session_id, str):
            raise RequestError.invalid_params({"reason": "sessionId must be a string"})
        cwd = params.get("cwd")
        if cwd is not None:
            # Only the store is needed: the workspace itself may be gone.
            if not isinstance(cwd, str) or not Path(cwd).is_absolute():
                raise RequestError.invalid_params({"cwd": cwd, "reason": "cwd must be absolute"})
            self.registry_for(Path(cwd))
        registry = self._registry_holding(session_id)
        if registry is None:
            raise RequestError.resource_not_found(session_id)
        bridge = self._bridges.pop(session_id, None)
        self._cancel_client_calls(session_id)
        try:
            await registry.delete(session_id, keep_files=bool(params.get("keepFiles")))
        except (SessionNotFoundError, InvalidSessionIdError):
            raise RequestError.resource_not_found(session_id) from None
        if bridge is not None:
            await bridge.close()
        return {}

    async def _inject(self, params: dict[str, Any]) -> dict[str, Any]:
        """``_nooa/session/inject``: ``{sessionId, mode: "queue"|"steer", prompt|text}``.

        ``queue`` submits the message; ``steer`` hands it to the running
        turn's next model call (queued when there is none). Answers
        ``{messageId, delivered: "queued"|"steered"}`` at once. An injected
        message is not a prompt request: Stop leaves it queued.
        """
        session_id = params.get("sessionId")
        if not isinstance(session_id, str):
            raise RequestError.invalid_params({"reason": "sessionId must be a string"})
        mode = params.get("mode")
        if mode not in ("queue", "steer"):
            raise RequestError.invalid_params({"mode": mode, "reason": "mode is queue or steer"})
        text = params.get("text")
        if not isinstance(text, str):
            blocks = params.get("prompt")
            if not isinstance(blocks, list) or not all(isinstance(b, dict) for b in blocks):
                raise RequestError.invalid_params({"reason": "prompt or text is required"})
            text = self._prompt_text([SimpleNamespace(**block) for block in blocks])
        elif not text.strip():
            raise RequestError.invalid_params({"reason": "Prompt text must not be empty"})
        session, _bridge = self._followed(session_id)
        slash = _slash_invocation(text)
        if slash is not None and slash[0] == "verbosity":
            raise RequestError.invalid_params(
                {"reason": "ACP /verbosity is a local command; send it as session/prompt"}
            )
        try:
            if mode == "steer":
                receipt = await session.steer(text, source=INJECT_SOURCE)
            else:
                receipt = await session.submit(text, source=INJECT_SOURCE)
        except SessionClosedError:
            raise RequestError.resource_not_found(session_id) from None
        self._injects.setdefault(session_id, {})[receipt.item_id] = receipt
        return {"messageId": receipt.item_id, "delivered": receipt.delivered}

    async def _pool_steer(
        self,
        params: dict[str, Any],
        dialect: PoolDialect | None = None,
        *,
        on_submitted: Callable[[str, Receipt, str], None] | None = None,
    ) -> dict[str, Any]:
        """``_poolside/session_steer``: ``{sessionId, inputId, prompt}`` -> ``{inputId}``.

        Pool 2's ``_pool/session_steer`` has the same params and answer.

        Pool sends what the person types during a running prompt. The text
        is queued for the next turn on ``user_messages``, as ``session/prompt``
        text sent during a turn is: the running turn sees it only as a pending
        message, and the next turn gets it in its notification. A slash
        command is refused, and Pool then sends it as a normal prompt after
        the turn.

        Pool (a v1 client) keeps its turn open until the messages it handed
        over are handled: ``on_submitted(session_id, receipt, text)`` lets the
        v1 shim hold its open prompt for this one.

        Pool shows the message in the conversation when it sees the steer's
        ``inputId`` in an input event (``pool_input_event``), which the bridge
        sends when a turn takes the message. The answer follows that event,
        as Pool's own agent answers; it also goes out when the message is
        withdrawn or its turn ends without it.
        """
        session_id, input_id = params.get("sessionId"), params.get("inputId")
        if not isinstance(session_id, str) or not isinstance(input_id, str):
            raise RequestError.invalid_params({"reason": "sessionId and inputId are strings"})
        blocks = params.get("prompt")
        if not isinstance(blocks, list) or not all(isinstance(b, dict) for b in blocks):
            raise RequestError.invalid_params({"reason": "prompt is required"})
        session, bridge = self._followed(session_id)
        text = self._prompt_text([SimpleNamespace(**block) for block in blocks])
        if _slash_invocation(text) is not None:
            raise RequestError.invalid_params(
                {"reason": "A slash command cannot steer a turn; send it after the turn ends"}
            )
        # Only a valid steer says which dialect the client speaks.
        if dialect is not None:
            self._use_pool_dialect(dialect)
        # Queued, not steered, on purpose until the Pool team says whether
        # this request means queue or steer. Handling it at all keeps the
        # message out of Pool's own queue, where Esc drops it.
        try:
            receipt = await session.submit(text, source=SOURCE)
        except SessionClosedError:
            raise RequestError.resource_not_found(session_id) from None
        shown = bridge.client_input_taken(receipt.item_id, input_id)
        if on_submitted is not None:
            on_submitted(session_id, receipt, text)
        request_cancelled = False
        try:
            await _taken_or_settled(
                session,
                receipt.item_id,
                shown,
                was_taken=lambda: bridge.client_input_was_taken(receipt.item_id),
            )
        except asyncio.CancelledError:
            request_cancelled = True
            raise
        finally:
            # Request cancellation does not withdraw its admitted input; it may
            # be consumed later and still needs its original client-ID event.
            bridge.release_client_input_waiter(receipt.item_id, keep_pending=request_cancelled)
        return {"inputId": input_id}

    def _revoke_inject(self, params: dict[str, Any]) -> dict[str, Any]:
        """``_nooa/session/revoke_inject``: ``{sessionId, messageId}`` -> ``{revoked}``."""
        session_id, message_id = params.get("sessionId"), params.get("messageId")
        if not isinstance(session_id, str) or not isinstance(message_id, str):
            raise RequestError.invalid_params({"reason": "sessionId and messageId are strings"})
        session, _bridge = self._followed(session_id)
        receipt = self._injects.get(session_id, {}).pop(message_id, None)
        return {"revoked": receipt is not None and session.withdraw(receipt)}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        del method, params

    async def close(self) -> None:
        """Close every session this process runs and every bridge."""
        bridges = list(self._bridges.values())
        self._bridges.clear()
        for session_id in tuple(self._asks):
            self._cancel_client_calls(session_id)
        for task in list(self._background):
            task.cancel()
        await asyncio.gather(*self._background, return_exceptions=True)
        empty = [
            (registry, session.id)
            for registry in self._registries.values()
            for session in registry.sessions.values()
            if session.parent_id is None and session.info.turn_count == 0
        ]
        await _close_in_order(
            *(registry.close_all for registry in self._registries.values()),
            *(bridge.close for bridge in bridges),
        )
        for registry, session_id in empty:
            await _discard_empty(registry, session_id)

    # ---- prompt and cancel ---------------------------------------------

    async def prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        """``session/prompt``: answered with the user message's ``messageId`` once it is inserted.

        The message is shown as ``user_message``; the turn's progress follows
        as updates and ``state_update`` (``running``, ``idle`` with its stop
        reason). A repeated ``idempotencyKey`` gets the first answer again.
        """
        session_id = params.get("sessionId")
        if not isinstance(session_id, str):
            raise v2.invalid("sessionId is required")
        key = params.get("idempotencyKey")
        if isinstance(key, str) and (session_id, key) in self._idempotent:
            return {"messageId": self._idempotent[(session_id, key)]}
        message_id, completion = await self.start_prompt(session_id, params.get("prompt") or [])
        completion.add_done_callback(_observe_completion)
        if isinstance(key, str):
            self._idempotent[(session_id, key)] = message_id
        return {"messageId": message_id}

    async def start_prompt(
        self, session_id: str, blocks: list[Any], *, echo: bool = True
    ) -> tuple[str, asyncio.Future[str]]:
        """Insert a prompt; return its message id and a future of its stop reason.

        The text is submitted on ``user_messages``; the future resolves when
        the turn that consumes it ends: the next turn, or the running one if
        the model takes the pending message from its queue (the queues
        context block lists it). A ``Waiting`` outcome keeps it pending.
        Stop resolves it ``cancelled`` and withdraws the message if no turn
        took it. A failed turn resolves it with its stop reason
        (``max_tokens``...) or a ``RequestError``. ``/name`` prompts naming a
        session command run the command first (a local command resolves at
        once). To steer a running turn, clients use
        ``_nooa/session/inject`` with ``mode: "steer"``.

        ``echo`` shows the prompt as a ``user_message`` (the v1 shim does not:
        a v1 client shows its own prompt).
        """
        session, bridge = self._followed(session_id)
        text = self._prompt_text(blocks)
        content = [block for block in blocks if isinstance(block, dict)] or [v2.text(text)]
        loop = asyncio.get_running_loop()
        slash = _slash_invocation(text)
        try:
            if slash is not None:
                bridge.begin_local()
                try:
                    handled = await self._run_command(session, bridge, *slash)
                except TurnFailedError as exc:
                    bridge.end_local("error")
                    reason = await self._turn_failed(bridge, exc)
                    failed: asyncio.Future[str] = loop.create_future()
                    failed.set_result(reason)
                    return v2.new_message_id(), failed
                except BaseException:
                    bridge.end_local("error")
                    raise
                if isinstance(handled, Receipt):
                    bridge.end_local()
                    if echo:
                        bridge.publish(v2.message_update("user_message", handled.item_id, content))
                    return handled.item_id, self._complete(session, bridge, handled)
                if handled is not None:
                    message_id = v2.new_message_id()
                    if echo:
                        bridge.publish(v2.message_update("user_message", message_id, content))
                    bridge.end_local(handled)
                    await bridge.flush()
                    done: asyncio.Future[str] = loop.create_future()
                    done.set_result(handled)
                    return message_id, done
                bridge.end_local()
            await self._request_title(session, text)
            receipt = await session.submit(text, source=SOURCE)
        except SessionClosedError:
            raise RequestError.resource_not_found(session_id) from None
        if echo:
            bridge.publish(v2.message_update("user_message", receipt.item_id, content))
        if self._pool_client:
            bridge.publish(pool_input_event(receipt.item_id, dialects=self._pool_dialects))
        return receipt.item_id, self._complete(session, bridge, receipt)

    def _complete(
        self, session: Session, bridge: ACPEventBridge, receipt: Receipt
    ) -> asyncio.Future[str]:
        """Track a prompt's receipt until its turn ends; the future of its stop reason."""
        self._prompts.setdefault(session.id, []).append(receipt)

        async def complete() -> str:
            try:
                return await self.finish_item(session.id, receipt.item_id)
            finally:
                with suppress(ValueError):
                    self._prompts.get(session.id, []).remove(receipt)

        return asyncio.ensure_future(complete())

    async def finish_item(self, session_id: str, item_id: str) -> str:
        """The stop reason of the turn that consumes ``item_id``, once it ends.

        ``cancelled`` when it was withdrawn or the turn was cancelled. A
        ``NeedInput`` is never a dialog: the bridge already sent its question
        as the turn's final message, so it ends the item with ``end_turn``
        and the person's next prompt answers it as text. A failed turn is
        its limit's stop reason, else a ``RequestError``.
        """
        session, bridge = self._followed(session_id)
        try:
            outcome = await session.outcome(item_id)
        except ItemWithdrawnError:
            await bridge.flush()
            return "cancelled"
        except SessionClosedError:
            raise RequestError.resource_not_found(session_id) from None
        except TurnFailedError as exc:
            return await self._turn_failed(bridge, exc)
        # The bridge closed the open cards when the Session reported a
        # cancel, which happens before this outcome resolves.
        await bridge.flush()
        if isinstance(outcome, TurnCancelledOutcome):
            return "cancelled"
        return "end_turn"

    async def _request_title(self, session: Session, text: str) -> None:
        """On a session's first prompt, ask the agent to title it (once per session).

        The request goes on ``system_messages`` (housekeeping, not part of
        the conversation) just before the prompt, so the same turn handles
        both. Agents without that channel, and sessions that already have a
        title or a message, are left alone.
        """
        if session.id in self._title_checked:
            return
        self._title_checked.add(session.id)
        if session.info.title or "system_messages" not in session.channels():
            return
        if any(entry.role == "user" for entry in session.transcript()):
            return
        from nooa_atom.agent.base import session_title_request

        await session.submit(session_title_request(text), channel="system_messages", source="host")

    async def _ask_verbosity(self, session: Session, question: str) -> Any:
        """Ask the client to pick a verbosity level in a form of its own.

        Pool's ``_poolside/elicitation`` (Pool 2: ``_pool/elicitation``) when
        the client is Pool, else a standard ``elicitation/create`` form when
        the client advertised ``elicitation.form``. Returns the chosen level,
        ``_DISMISSED`` when the person declined or dismissed it, ``_CANCELLED``
        when Stop ended it, or ``None`` when no form could be shown or the
        answer is not a level. A failed Pool request turns Pool forms off for this connection.
        """
        conn = self._require_conn()
        via_pool = self._pool_forms
        if via_pool:
            # Pool declines non-string inputs: one required string picker.
            picker = {
                "type": "string",
                "title": "Verbosity",
                "description": _VERBOSITY_HELP,
                "oneOf": [{"const": level, "title": level} for level in VERBOSITY_LEVELS],
            }
            params = {
                "sessionId": session.id,
                "mode": "form",
                "message": question,
                "requestedSchema": {
                    "type": "object",
                    "properties": {"verbosity": picker},
                    "required": ["verbosity"],
                },
            }
            # A Pool whose version did not say gets Pool 1's request.
            dialect = self._pool_dialects[0]
            if dialect.form_has_id:
                # Pool 2 requires it: it ties the answer to the request.
                params["elicitationId"] = uuid.uuid4().hex
            request = conn.request(f"_{dialect.form_method}", params)
        elif self._client_shows_forms():
            field = {
                "type": "string",
                "title": "Verbosity",
                "description": _VERBOSITY_HELP,
                "enum": list(VERBOSITY_LEVELS),
            }
            request = conn.request(
                "elicitation/create",
                {
                    "sessionId": session.id,
                    "mode": "form",
                    "message": question,
                    "requestedSchema": {
                        "type": "object",
                        "properties": {"verbosity": field},
                        "required": ["verbosity"],
                    },
                },
            )
        else:
            return None
        response = await self._client_call(session.id, request)
        if response is _CANCELLED:
            return response
        if response is None:
            if via_pool:
                self._pool_forms = False
            return None
        if isinstance(response, dict):
            action, content = response.get("action"), response.get("content")
        else:
            action, content = getattr(response, "action", None), getattr(response, "content", None)
        if action != "accept":
            return _DISMISSED
        level = content.get("verbosity") if isinstance(content, dict) else None
        return level if level in VERBOSITY_LEVELS else None

    def _client_shows_forms(self) -> bool:
        """Whether the client advertised standard ACP form elicitation."""
        elicitation = self.client_capabilities.get("elicitation")
        return isinstance(elicitation, dict) and isinstance(elicitation.get("form"), dict)

    def _client_answer_is_current(
        self, session: Session, bridge: ACPEventBridge, epoch: int
    ) -> bool:
        return (
            self._bridges.get(session.id) is bridge
            and self.session(session.id) is session
            and self._ask_epochs.get(session.id, 0) == epoch
        )

    def _cancel_client_calls(self, session_id: str) -> None:
        # Invalidate answers even if a client coroutine swallows cancellation.
        self._ask_epochs[session_id] = self._ask_epochs.get(session_id, 0) + 1
        for task in tuple(self._asks.get(session_id, ())):
            task.cancel()

    async def _client_call(self, session_id: str, request: Any) -> Any:
        """Await a request to the client that session/cancel can stop.

        Returns the response, ``_CANCELLED`` when cancelled, or ``None`` when
        the client failed it (the caller falls back to text).
        """
        epoch = self._ask_epochs.get(session_id, 0)
        task = asyncio.ensure_future(request)
        tasks = self._asks.setdefault(session_id, set())
        tasks.add(task)
        bridge = self._bridges.get(session_id)
        if bridge is not None:
            bridge.ask_started()
        try:
            answer = await task
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise asyncio.CancelledError
            if self._ask_epochs.get(session_id, 0) != epoch:
                return _CANCELLED
            return answer
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if task.cancelled() and not (current is not None and current.cancelling()):
                return _CANCELLED
            raise
        except Exception:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise asyncio.CancelledError from None
            if self._ask_epochs.get(session_id, 0) != epoch:
                return _CANCELLED
            logger.warning("The client failed a request in session %s", session_id, exc_info=True)
            return None
        finally:
            if bridge is not None:
                bridge.ask_ended()
            tasks.discard(task)
            if not tasks and self._asks.get(session_id) is tasks:
                del self._asks[session_id]

    async def _turn_failed(self, bridge: ACPEventBridge, exc: TurnFailedError) -> str:
        """Map a failed turn: generation limits are stop reasons, anything else an error."""
        await bridge.flush()  # includes the "Unfinished" cards for the failed turn
        error = getattr(exc, "error", None) or exc.__cause__ or exc
        message = str(error)
        if isinstance(error, GenerationError):
            stop_reason = limit_stop_reason(message)
            if stop_reason is not None:
                return stop_reason
        raise RequestError(-32603, message, {"details": message}) from exc

    async def _run_command(
        self, session: Session, bridge: ACPEventBridge, name: str, raw_args: str
    ) -> str | Receipt | None:
        """Run ``/name args``; ``None`` when it is not a command, so the text is a prompt.

        A local command returns its stop reason; a command whose output goes
        to the agent returns the receipt of that message.
        """
        if name == "verbosity":
            level = raw_args.strip().lower()
            if not level:
                return await self._pick_verbosity(session, bridge)
            elif level not in VERBOSITY_LEVELS:
                message = "Usage: /verbosity silent|concise|verbose"
            else:
                try:
                    await self._set_verbosity(session, bridge, level)
                except RequestError as exc:
                    return await self._say(
                        bridge, f"/verbosity: could not save setting: {exc.data}"
                    )
                bridge.publish(
                    {
                        "sessionUpdate": "config_option_update",
                        "configOptions": self._config_options(session),
                    }
                )
                message = f"ACP verbosity: {bridge.verbosity}."
            return await self._say(bridge, message)
        if name == RECOVER.name:
            return await self._say(bridge, await self._recover(session, raw_args))
        if name not in {command.name for command in session.commands()}:
            if name == "connect":
                message = _CONNECT_TEXT
            elif name in RESERVED_COMMAND_NAMES:
                available = ", ".join(f"/{command.name}" for command in session.commands())
                message = (
                    f"NOOA /{name} is not available through ACP yet. "
                    f"Commands here: {available or 'none'}."
                )
            else:
                return None
            return await self._say(bridge, message)
        try:
            with _trace_scope(session):
                result = await session.invoke_command(name, raw_args)
        except CoercionError as exc:
            message = f"/{name}: {exc.message}"
            if exc.hint:
                message += f"\n\nUsage: `/{name} {exc.hint}`"
            return await self._say(bridge, message)
        except (GenerationError, SessionClosedError, TurnFailedError):
            raise
        except Exception as exc:
            # Command bodies are third-party code from workspace and installed
            # skills; a failure is the command's, not a protocol error.
            logger.warning(
                "Slash command /%s failed in session %s", name, session.id, exc_info=True
            )
            return await self._say(bridge, f"/{name} failed: {exc}")
        if result.output_to_agent:
            if not result.text.strip():
                # An empty message would spend a turn on nothing (and some
                # providers reject empty content); ending silently hides that
                # the command ran at all.
                return await self._say(
                    bridge, f"/{name} produced no output, so nothing was sent to the agent."
                )
            channels = session.channels()
            channel = "slash_commands" if "slash_commands" in channels else "user_messages"
            return await session.submit(result.text, channel=channel, source=SOURCE)
        return await self._say(bridge, result.text)

    async def _pick_verbosity(self, session: Session, bridge: ACPEventBridge) -> str:
        """Ask the client locally; neither the command nor its answer enters the agent."""
        question = f"ACP verbosity: {bridge.verbosity}. Choose activity detail."
        epoch = self._ask_epochs.get(session.id, 0)
        level = await self._ask_verbosity(session, question)
        if not self._client_answer_is_current(session, bridge, epoch):
            return "cancelled"
        if level is _CANCELLED:
            await bridge.flush()
            return "cancelled"
        if level is None:
            return await self._say(
                bridge,
                f"ACP verbosity: {bridge.verbosity}. Usage: /verbosity silent|concise|verbose",
            )
        if level is _DISMISSED:
            return await self._say(bridge, f"ACP verbosity unchanged: {bridge.verbosity}.")
        # Reuse local dispatch for validation, confirmation and config-picker sync.
        response = await self._run_command(session, bridge, "verbosity", level)
        assert isinstance(response, str)
        return response

    async def _recover(self, session: Session, raw_args: str) -> str:
        """``/recover``: over the store that holds ``session``, for its workspace."""
        registry = self._registry_holding(session.id)
        if registry is None:
            return "/recover failed: this session's store is not known here."
        open_here = {
            session_id for known in self._registries.values() for session_id in known.sessions
        }
        try:
            return await asyncio.to_thread(
                recover,
                registry.store,
                session.info.workspace or None,
                raw_args,
                open_here=open_here,
            )
        except Exception as exc:
            logger.warning("/recover failed in session %s", session.id, exc_info=True)
            return f"/recover failed: {exc}"

    @staticmethod
    async def _say(bridge: ACPEventBridge, message: str) -> str:
        if message:
            bridge.say(message)
        await bridge.flush()
        return "end_turn"

    async def cancel(self, params: dict[str, Any]) -> None:
        """``session/cancel``: stop the running turn; the bridge closes its cards when told.

        A child this connection shows as a subagent (announced with
        ``capabilities.cancel``) stops on its own; its parent goes on.
        """
        session_id = params.get("sessionId")
        session = self.session(session_id) if isinstance(session_id, str) else None
        if session is None:
            return  # a notification: nothing to answer
        if session_id not in self._bridges:
            if self._descends_from_followed(session):
                await session.cancel(by="user")
            return
        self._cancel_client_calls(session_id)
        # A prompt whose message no turn took is withdrawn and ends
        # "cancelled"; the one the running turn took ends "cancelled" with it.
        # Injected messages stay queued.
        for receipt in self._prompts.pop(session_id, []):
            session.withdraw(receipt)
        await session.cancel(by="user")

    async def mcp_notification(self, params: dict[str, Any]) -> None:
        """Receive a provider notification without ever answering or misrouting it.

        ``MCPTool`` has no progress/subscription consumer. Unknown and late
        notifications are therefore ignored; they must not be delivered to a
        different operation. Tool results and errors use the request path.
        """
        logger.debug(
            "Ignoring MCP notification %s for server %s request %s",
            params.get("method"),
            params.get("serverId"),
            params.get("requestId"),
        )

    # ---- client-initiated cancellation ---------------------------------

    def _activity_session(self, session_id: str) -> tuple[Session, ACPEventBridge]:
        """A session and the bridge showing its activity (an unloaded child's ancestor's)."""
        session = self.session(session_id)
        if session is None:
            raise RequestError.resource_not_found(session_id)
        bridge = self._bridges.get(session_id)
        parent_id = session.parent_id
        for _ in range(64):
            if bridge is not None:
                return session, bridge
            if parent_id is None:
                break
            bridge = self._bridges.get(parent_id)
            parent = self.session(parent_id)
            parent_id = parent.parent_id if parent is not None else None
        raise RequestError.resource_not_found(session_id)

    async def cancel_tool_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """``tool_call/cancel``: stop one advertised background job, not the foreground turn."""
        session_id = params.get("sessionId")
        tool_call_id = params.get("toolCallId")
        if not isinstance(session_id, str) or not isinstance(tool_call_id, str):
            raise v2.invalid("sessionId and toolCallId are required")
        session, bridge = self._activity_session(session_id)
        if not tool_call_id.startswith("job:"):
            raise v2.invalid(
                "Tool call is unknown, final, or not cancellable", toolCallId=tool_call_id
            )
        name = tool_call_id.removeprefix("job:")
        job = session._agent.jobs.get(name)
        if job is None or not await bridge.cancel_background(tool_call_id, job):
            raise v2.invalid(
                "Tool call is unknown, final, or not cancellable", toolCallId=tool_call_id
            )
        return {}

    async def cancel_terminal(self, params: dict[str, Any]) -> dict[str, Any]:
        """``terminal/cancel``: interrupt one command; its tool card and turn state stay put."""
        session_id = params.get("sessionId")
        terminal_id = params.get("terminalId")
        if not isinstance(session_id, str) or not isinstance(terminal_id, str):
            raise v2.invalid("sessionId and terminalId are required")
        _session, bridge = self._activity_session(session_id)
        if not await bridge.cancel_terminal(terminal_id):
            raise v2.invalid(
                "Terminal is unknown, exited, or not cancellable", terminalId=terminal_id
            )
        return {}

    # ---- modes and models ----------------------------------------------

    async def set_mode(self, session_id: str, mode_id: str) -> None:
        """Set the permission mode (a v1 concept, used by the v1 shim). Only ``auto`` exists."""
        session, _bridge = self._followed(session_id)
        if mode_id not in MODES:
            raise RequestError.invalid_params(
                {"modeId": mode_id, "reason": "Unknown mode; this agent offers 'auto' only"}
            )
        await session.set_mode(mode_id)

    async def set_config_option(self, params: dict[str, Any]) -> dict[str, Any]:
        """``session/set_config_option``: a model, reasoning level, or ACP-local verbosity."""
        config_id, session_id, value = (
            params.get("configId"),
            params.get("sessionId"),
            params.get("value"),
        )
        if not isinstance(session_id, str):
            raise v2.invalid("sessionId is required")
        session, _bridge = self._followed(session_id)
        if config_id == "verbosity":
            if not isinstance(value, str) or value not in VERBOSITY_LEVELS:
                raise RequestError.invalid_params(
                    {
                        "configId": config_id,
                        "value": value,
                        "reason": "Use silent, concise, or verbose",
                    }
                )
            await self._set_verbosity(session, _bridge, value)
            return {"configOptions": self._config_options(session)}
        if config_id == "reasoning" and isinstance(value, str):
            await self._set_reasoning(session, value)
            return {"configOptions": self._config_options(session)}
        if config_id != "model" or not isinstance(value, str):
            raise RequestError.invalid_params(
                {"configId": config_id, "reason": "Unknown configuration option"}
            )
        try:
            await session.set_model(value)
        except SessionClosedError:
            raise RequestError.resource_not_found(session_id) from None
        except Exception as exc:
            raise RequestError.invalid_params(
                {"configId": config_id, "value": value, "reason": str(exc)}
            ) from exc
        return {"configOptions": self._config_options(session)}

    async def _set_verbosity(self, session: Session, bridge: ACPEventBridge, level: str) -> None:
        followed, current_bridge = self._followed(session.id)
        if followed is not session or current_bridge is not bridge:
            raise RequestError.resource_not_found(session.id)
        # Save only this key at project scope, retaining unrelated settings.
        # A failed write must not report a successful persistent change.
        try:
            # Match existing workspace controls: this small local write and
            # bridge update have no await between them, so concurrent ACP
            # requests in this process cannot reorder persistence/application.
            write_settings_updates(
                {("atom", "verbosity"): level}, workspace=session.options.workspace
            )
        except Exception as exc:
            raise RequestError.internal_error(
                {"reason": f"Could not save ACP verbosity: {exc}"}
            ) from exc
        bridge.set_verbosity(level)

    @staticmethod
    async def _set_reasoning(session: Session, level: str) -> None:
        """Select a reasoning level the session's model declares (``Session.set_reasoning``).

        The level applies from the next model call on; switching the model
        later starts from that alias's default again.
        """
        try:
            await session.set_reasoning(None if level == DEFAULT_REASONING else level)
        except SessionClosedError:
            raise RequestError.resource_not_found(session.id) from None
        except ValueError as exc:
            raise RequestError.invalid_params(
                {"configId": "reasoning", "value": level, "reason": str(exc)}
            ) from exc

    def _config_options(self, session: Session) -> list[dict[str, Any]]:
        """The model select option (the registry's aliases plus the current model),
        a reasoning option when the current client declares levels, and ACP verbosity."""
        options: list[dict[str, Any]] = []
        current = session.info.model or self._model or _default_model(session.options.workspace)
        aliases = model_aliases(session.options.workspace)
        if current and current not in aliases:
            aliases = [current, *aliases]
        if current and aliases:
            options.append(
                v2.select_option(
                    "model",
                    "Model",
                    current,
                    [(alias, alias) for alias in aliases],
                    category="model",
                    description="The model this session uses from its next turn on.",
                )
            )
        reasoning = self._reasoning_option(session)
        if reasoning is not None:
            options.append(reasoning)
        bridge = self._bridges.get(session.id)
        options.append(
            v2.select_option(
                "verbosity",
                "Verbosity",
                bridge.verbosity if bridge is not None else "verbose",
                [(level, level.title()) for level in VERBOSITY_LEVELS],
                description="ACP activity detail only; does not change the agent or its reasoning.",
            )
        )
        return options

    @staticmethod
    def _reasoning_option(session: Session) -> dict[str, Any] | None:
        """The reasoning select option: the client's declared levels, or nothing.

        Offered only when the client declares levels and either a level was
        chosen or the route has a default; otherwise there is no honest
        current value to show.
        """
        model = session.model_info()
        levels = model.reasoning_levels
        if not levels:
            return None
        default_name = (
            f"Model default ({model.reasoning_default})"
            if model.reasoning_default
            else "Model default"
        )
        return v2.select_option(
            "reasoning",
            "Reasoning",
            model.reasoning_level or DEFAULT_REASONING,
            [(DEFAULT_REASONING, default_name), *((level, level) for level in levels)],
            category="thought_level",
            description="How much the model reasons before answering, from its next call on.",
        )

    def _followed(self, session_id: str) -> tuple[Session, ACPEventBridge]:
        session = self.session(session_id)
        bridge = self._bridges.get(session_id)
        if session is None or bridge is None:
            raise RequestError.resource_not_found(session_id)
        return session, bridge

    # ---- helpers -------------------------------------------------------

    def bridge(self, session_id: str) -> ACPEventBridge:
        """The bridge of a session this adapter follows; resource-not-found otherwise."""
        bridge = self._bridges.get(session_id)
        if bridge is None:
            raise RequestError.resource_not_found(session_id)
        return bridge

    def _attach(self, session: Session) -> ACPEventBridge:
        conn = self._require_conn()
        bridge = ACPEventBridge(session, conn, resolve_child=self.session)
        if self._pool_dialects:
            bridge.pool_dialects = self._pool_dialects
        bridge.set_verbosity(load_acp_verbosity(session.options.workspace))
        self._bridges[session.id] = bridge

        def on_update(update: Any) -> None:
            if isinstance(update, CommandsChangedUpdate):
                bridge.publish(_available_commands_update(update.commands))
            elif getattr(update, "kind", None) == "closed":
                if self._bridges.get(session.id) is bridge:
                    del self._bridges[session.id]
                    self._cancel_client_calls(session.id)
                unsubscribe()

        unsubscribe = session.subscribe(on_update)
        return bridge

    def _descends_from_followed(self, session: Session) -> bool:
        """Whether a session is a child (at any depth) of one this connection follows."""
        parent_id = session.parent_id
        for _ in range(64):
            if parent_id is None:
                return False
            if parent_id in self._bridges:
                return True
            parent = self.session(parent_id)
            parent_id = parent.parent_id if parent is not None else None
        return False

    def _parent_is_live(self, session: Session) -> bool:
        return session.parent_id is not None and self.session(session.parent_id) is not None

    def _require_conn(self) -> Peer:
        if self._conn is None:
            raise RequestError.internal_error({"reason": "ACP client is not connected"})
        return self._conn

    def _agent_spec_for(self, root: Path) -> str:
        if self._agent_spec:
            return canonical_agent_spec(self._agent_spec)
        from nooa_atom.workspace.options import AtomOptions

        configured = AtomOptions.load(root).agent_spec
        return canonical_agent_spec(configured) if configured else ATOM_AGENT

    async def _prepare_tools(self, session: Session, mcp_servers: list[Any] | None) -> list[str]:
        """Give a new or loaded session its tools; return warnings.

        Runs in the registry's ``prepare`` step, before any turn. The
        agent's own set-up runs first (an Atom agent connects the MCP
        servers its workspace remembers), then the MCP servers the client
        sent are registered as ``mcp.<name>``. A name that collides with
        one the agent provides (``shell``, ``repo``) is skipped with a
        warning, so the session stays usable instead of failing.
        """
        warnings = await session.prepare_tools()
        tools, mcp_warnings = await self._create_mcp_tools(mcp_servers)
        warnings.extend(mcp_warnings)
        failed = await session.register_tools({f"mcp.{name}": tool for name, tool in tools.items()})
        warnings.extend(
            f"MCP server {name!r} was not registered: {failed[f'mcp.{name}']}"
            for name in tools
            if f"mcp.{name}" in failed
        )
        return warnings

    async def _create_mcp_tools(
        self,
        mcp_servers: list[Any] | None,
    ) -> tuple[dict[str, MCPTool], tuple[str, ...]]:
        """Connect the client's MCP servers (v2 configs; ``sse`` too, for v1 clients).

        A server of another type, or with a name already taken, is skipped
        with a warning.
        """
        servers = list(mcp_servers or [])
        tools: dict[str, MCPTool] = {}
        warnings: list[str] = []
        seen_names: set[str] = set()
        seen_server_ids: set[str] = set()
        for server in servers:
            server = server if isinstance(server, dict) else {}
            name = server.get("name", "<unnamed>")
            kind = server.get("type")
            if kind not in ("stdio", "http", "sse", "acp"):
                warnings.append(
                    f"MCP server {name!r} was not loaded: unsupported ACP server type "
                    f"{server.get('_typeName') or kind}."
                )
                continue
            if name in seen_names:
                warnings.append(
                    f"MCP server {name!r} was not loaded: another server has the same name."
                )
                continue
            server_id = server.get("serverId")
            if kind == "acp" and (
                not isinstance(server_id, str) or not server_id or server_id in seen_server_ids
            ):
                warnings.append(
                    f"MCP server {name!r} was not loaded: serverId must be non-empty and unique."
                )
                continue
            seen_names.add(name)
            if isinstance(server_id, str):
                seen_server_ids.add(server_id)
            try:
                if kind == "acp":
                    from nooa_atom.acp.mcp_acp import create_acp_server

                    tools[name] = await create_acp_server(self._require_conn(), name, server_id)
                elif kind == "stdio":
                    env = {item["name"]: item["value"] for item in server.get("env") or []}
                    tools[name] = await MCPManager.create_stdio_server(
                        name,
                        command=server["command"],
                        args=list(server.get("args") or []),
                        env=env,
                    )
                else:
                    headers = {item["name"]: item["value"] for item in server.get("headers") or []}
                    tools[name] = await MCPManager.create_url_server(
                        name,
                        server["url"],
                        headers=headers,
                        transport="streamable-http" if kind == "http" else "sse",
                    )
            except Exception as exc:
                warnings.append(f"MCP server {name!r} was not loaded: {exc}")
        return tools, tuple(warnings)

    def _defer_bootstrap_updates(self, session: Session, warnings: list[str]) -> None:
        """Send commands and startup warnings after the response can reach clients such as Zed."""
        bridge = self._bridges.get(session.id)
        if bridge is None:
            return

        async def publish() -> None:
            await asyncio.sleep(0)
            bridge.publish_best_effort(_available_commands_update(session.commands()))
            if warnings:
                details = "\n".join(f"- {warning}" for warning in warnings)
                text = (
                    "NOOA started with configuration warnings. The session is still "
                    f"usable.\n\n{details}"
                )
                for update in agent_message(text):
                    bridge.publish_best_effort(update)
            await bridge.flush()

        self._spawn(publish(), name="nooa-acp-bootstrap")

    def _spawn(self, coroutine: Any, *, name: str) -> None:
        task = asyncio.ensure_future(coroutine)
        task.set_name(name)
        self._background.add(task)

        def finished(done: asyncio.Task[None]) -> None:
            self._background.discard(done)
            if not done.cancelled() and done.exception() is not None:
                logger.debug("%s failed", name, exc_info=done.exception())

        task.add_done_callback(finished)

    @staticmethod
    def _replay(bridge: ACPEventBridge, session: Session, *, pool: bool = False) -> None:
        """Queue the transcript as whole user and agent messages, then ``replay_complete``.

        A user message keeps its item id as its message id, the id its
        prompt was answered with. For Pool, each user message is followed
        by its input event, as Pool's own agent replays them.
        """
        for entry in session.transcript():
            if entry.role == "cancelled":
                by = entry.content.removeprefix("Stopped by ").strip() or "user"
                content = cancel_text(by)
            else:
                content = entry.content
            # Clients that join chunks would join two entries from one
            # speaker in a row: end each entry with a newline.
            content = content if content.endswith("\n") else content + "\n"
            if entry.role != "user":
                content = end_line(content)  # as the live message ended
            kind = "user_message" if entry.role == "user" else "agent_message"
            message_id = (entry.item_id if entry.role == "user" else None) or v2.new_message_id()
            bridge.publish(v2.message_update(kind, message_id, [v2.text(content)]))
            if pool and entry.role == "user" and entry.item_id:
                bridge.publish(pool_input_event(entry.item_id, dialects=bridge.pool_dialects))
        bridge.publish({"sessionUpdate": "replay_complete"})

    @staticmethod
    def _prompt_text(prompt: list[Any]) -> str:
        parts: list[str] = []
        for block in prompt:
            block_type = _field(block, "type")
            if block_type == "text":
                parts.append(str(_field(block, "text")))
            elif block_type == "resource_link":
                parts.append(f"Resource {_field(block, 'name')}: {_field(block, 'uri')}")
            else:
                raise RequestError.invalid_params(
                    {"reason": f"Unsupported prompt content type: {block_type!r}"}
                )
        text = "\n\n".join(parts)
        if not text.strip():
            raise RequestError.invalid_params({"reason": "Prompt text must not be empty"})
        return text


def _field(block: Any, name: str) -> Any:
    """A content block's field, whether the block is a dict or an object."""
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


def _observe_completion(future: asyncio.Future[str]) -> None:
    """A v2 prompt's completion nobody awaits: its failure is the turn's, already reported."""
    if not future.cancelled() and future.exception() is not None:
        logger.debug("A prompt ended with an error", exc_info=future.exception())


def _trace_as(session: Session) -> None:
    """Make the ACP session id the trace session of the session's turns.

    Turns run in the session's own loop context, so the id is set there by
    a loop hook, and also in the calling request's context. The loop's
    context is a fresh one, so the tracing hooks (kept in a context variable
    where tracing was enabled) are registered there too; without that the
    turns produce no spans at all. A child session is not traced here: the
    registry continues its creator's trace (``session.tracing``).
    """
    try:
        import nooa.tracing as tracing
    except ImportError:
        return

    def enter() -> None:
        tracing.set_session(session.id)
        tracing.register_hooks_in_current_context()

    session.add_loop_context_hook(enter)
    tracing.set_session(session.id)


def _trace_scope(session: Session) -> AbstractContextManager[object]:
    """The session's trace session for a command run in a request's context."""
    try:
        from nooa.tracing import session_scope
    except ImportError:
        return nullcontext()
    return session_scope(session.id)


def _slash_invocation(text: str) -> tuple[str, str] | None:
    """``(name, raw_args)`` for ``/name args`` text, else ``None``. Parsing only."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped[1:].split(maxsplit=1)
    name = parts[0].lower() if parts else ""
    if not name:
        return None
    return name, parts[1] if len(parts) == 2 else ""


async def _discard_empty(registry: SessionRegistry, session_id: str) -> None:
    """At shutdown, a root session the client never wrote to leaves no file.

    Clients open sessions they then abandon (a picker, a restart); without
    this the store fills with files that list nothing. Only at shutdown: a
    client may close a fresh session and load it again by id meanwhile.
    """
    try:
        await registry.delete(session_id, keep_files=False)
    except Exception:
        logger.warning("Session %s: could not remove its empty file", session_id, exc_info=True)


def _load_error(session_id: str, exc: BaseException) -> BaseException:
    """The protocol error for a failed ``registry.load``; other errors pass through."""
    if isinstance(exc, (SessionNotFoundError, InvalidSessionIdError)):
        return RequestError.resource_not_found(session_id)
    if isinstance(exc, ChildActiveElsewhereError):
        # Clients may display only error.message, without error.data.
        return RequestError(
            -32600,
            f"Session {session_id[:8]!r} cannot be opened: {exc} "
            "Close that subagent's session in the other client, then try again.",
            {"sessionId": session_id, "reason": str(exc)},
        )
    if isinstance(exc, SessionAlreadyActiveError):
        owner = f" by process {exc.owner_pid}" if exc.owner_pid else ""
        return RequestError(
            -32600,
            f"Session {session_id[:8]!r} is already open{owner}. "
            "Close it in the other client or tab, then try resuming again.",
            {"sessionId": session_id, "ownerPid": exc.owner_pid, "reason": str(exc)},
        )
    return exc


def _available_commands(commands: list[CommandInfo]) -> list[dict[str, Any]]:
    """The session's commands, and the ones this adapter runs itself (``/recover`` and ``/verbosity``)."""
    names = {command.name for command in commands}
    verbosity = CommandInfo(
        name="verbosity",
        description="Set how much agent activity is shown: silent, concise, or verbose.",
        input_hint="[silent|concise|verbose]",
    )
    # The ACP-local command wins over a skill command with the same name.
    return [
        v2.command(command.name, command.description, command.input_hint or None)
        for command in [
            *(c for c in commands if c.name != verbosity.name),
            *(c for c in (RECOVER,) if c.name not in names),
            verbosity,
        ]
    ]


def _available_commands_update(commands: list[CommandInfo]) -> dict[str, Any]:
    return {
        "sessionUpdate": "available_commands_update",
        "availableCommands": _available_commands(commands),
    }


async def _taken_or_settled(
    session: Session,
    item_id: str,
    shown: asyncio.Future[None],
    *,
    was_taken: Callable[[], bool],
) -> None:
    """Wait until Pool was told ``item_id`` was taken, or the item's outcome settles first.

    Only a never-taken item's withdrawal/discard/close ends the wait without
    a send. A consumed item's outcome cannot overtake its acknowledgement.
    """

    async def settled() -> None:
        with suppress(Exception):
            await session.outcome(item_id)

    outcome = asyncio.ensure_future(settled())
    try:
        await asyncio.wait({shown, outcome}, return_when=asyncio.FIRST_COMPLETED)
        # A consumed item may settle before the queued send completes. Its
        # outcome (even cancellation/error) is not evidence of client delivery.
        if shown.done():
            shown.result()
        elif was_taken():
            await asyncio.shield(shown)
    finally:
        outcome.cancel()
        await asyncio.gather(outcome, return_exceptions=True)


async def serve_connection(
    agent: AtomACPAgent,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    id_base: int | None = None,
    observers: list[Callable[[Any], None]] | None = None,
) -> None:
    """Serve ACP for ``agent`` on one stream pair until the peer closes it.

    Used on standard input and output (``serve``) and on a router's
    socket (the worker). ``id_base`` is the first id of the requests this
    side sends to the client (form and extension requests); a router gives
    each worker its own range so replies can be routed by id alone.

    The client's ``initialize`` picks the version (``v2.negotiate``): v2 is
    served directly, v1 through the v1 shim. This is the one place that
    chooses; without the shim, a v1 client would get v2's answer to a v1
    request.
    """
    first = await read_frame(reader)
    if first is None:
        return
    inner = _Replayed(reader, first.raw)
    if first.method == "initialize" and v2.negotiate(first.params) == v2.V2:
        logger.info("ACP: speaking protocol v2")
        await v2.serve_v2(agent, inner.reader, writer, id_base=id_base, observers=observers)
        return
    from nooa_atom.acp.v1_shim import serve_v1

    await serve_v1(agent, inner.reader, writer, id_base=id_base, observers=observers)


class _Replayed:
    """A reader that yields ``first`` (already read from ``reader``) and then the rest of ``reader``."""

    def __init__(self, reader: asyncio.StreamReader, first: bytes) -> None:
        self.reader = asyncio.StreamReader(limit=FRAME_LIMIT)
        self.reader.feed_data(first)
        self._source = reader
        self._task = asyncio.ensure_future(self._pump())

    async def _pump(self) -> None:
        try:
            while chunk := await self._source.read(1 << 16):
                self.reader.feed_data(chunk)
        except (ConnectionError, OSError) as exc:
            logger.info("ACP: the client connection failed: %s", exc)
        finally:
            self.reader.feed_eof()


async def serve(
    new_registry: Callable[[SessionStore], SessionRegistry],
    *,
    sessions_dir: Path | None = None,
    agent_spec: str | None = None,
    model: str | None = None,
    observers: list[Callable[[Any], None]] | None = None,
    input_fd: int | None = None,
    output_fd: int | None = None,
) -> None:
    """Serve ACP on this process's standard input and output until the client leaves.

    ``input_fd`` and ``output_fd`` are the descriptors frames are read from
    and written to when stdio was reserved for ACP
    (``cli.reserve_stdio_for_acp``); ``None`` uses the process's stdio.
    ``new_registry`` and ``sessions_dir`` are as for ``AtomACPAgent``.
    """
    adapter = AtomACPAgent(
        new_registry, sessions_dir=sessions_dir, agent_spec=agent_spec, model=model
    )
    # ACP clients may terminate their subprocess instead of closing stdin.
    # Let normal teardown checkpoint sessions and release their file claims.
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    terminating = False

    def terminate() -> None:
        nonlocal terminating
        if not terminating and task is not None:
            terminating = True
            task.cancel()
            # A second SIGTERM must be able to kill the process outright even
            # if cleanup hangs; the inherited handler could be SIG_IGN, so
            # install the default action right away.
            loop.remove_signal_handler(signal.SIGTERM)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)

    signal_installed = False
    try:
        loop.add_signal_handler(signal.SIGTERM, terminate)
        signal_installed = True
    except (NotImplementedError, RuntimeError):
        pass  # Non-Unix event loops or an embedded server outside the main thread.
    try:
        reader, writer = await open_stdio(input_fd, output_fd)
        await serve_connection(adapter, reader, writer, observers=observers)
    except asyncio.CancelledError:
        if not terminating:
            raise
    finally:
        try:
            with suppress(Exception):
                await adapter.close()
        finally:
            if signal_installed:
                loop.remove_signal_handler(signal.SIGTERM)
                signal.signal(signal.SIGTERM, previous_sigterm)


__all__ = [
    "INJECT_CAPABILITY",
    "AtomACPAgent",
    "list_sessions",
    "open_stdio",
    "serve",
    "serve_connection",
]
