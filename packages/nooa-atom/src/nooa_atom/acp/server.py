# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ACP agent over the Session layer: protocol translation, no session policy.

``AtomACPAgent`` implements the ``acp`` library's ``Agent`` interface on a
``SessionRegistry``. Sessions, turns, steering, cancellation and
checkpoints are the Session's; this module maps requests onto it and keeps
one ``ACPEventBridge`` per session id that sends the session's activity
to the client.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import signal
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext, suppress
from pathlib import Path
from string import Formatter
from types import SimpleNamespace
from typing import Any, cast

from acp import (
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    RequestError,
    text_block,
    update_agent_message,
    update_user_message,
)
from acp.agent.connection import AgentSideConnection
from acp.helpers import update_available_commands
from acp.interfaces import Agent, Client
from acp.schema import (
    AvailableCommand,
    AvailableCommandInput,
    ClientCapabilities,
    CloseSessionResponse,
    ConfigOptionUpdate,
    ElicitationFormSessionMode,
    ElicitationSchema,
    ElicitationStringPropertySchema,
    HttpMcpServer,
    Implementation,
    ListSessionsResponse,
    McpServerStdio,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionMode,
    SessionModeState,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
    SseMcpServer,
    UnstructuredCommandInput,
)

from nooa.errors import GenerationError
from nooa.mcp import MCPManager, MCPTool
from nooa.slash_dispatch import CoercionError
from nooa.storage.sqlite import SessionAlreadyActiveError
from nooa.strategies.codeact import MAX_ITERATIONS_MESSAGE, OUTPUT_TOKENS_EXHAUSTED_MESSAGE
from nooa_atom.acp.event_bridge import (
    VERBOSITY_LEVELS,
    ACPEventBridge,
    cancel_text,
    pool_input_event,
)
from nooa_atom.acp.listing import list_sessions, validate_workspace
from nooa_atom.acp.protocol import (
    INJECT_CAPABILITY,
    POOL_DIALECT,
    POOLSIDE_DIALECT,
    PoolDialect,
    initialize_response,
    open_stdio,
    pool_dialects,
)
from nooa_atom.acp.recover import COMMAND as RECOVER
from nooa_atom.acp.recover import recover
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


def _template_pattern(template: str) -> re.Pattern[str]:
    """A pattern matching ``template`` (a ``str.format`` string) with any field values."""
    return re.compile(
        "".join(
            re.escape(literal) + (".*?" if field is not None else "")
            for literal, field, _spec, _conversion in Formatter().parse(template)
        )
    )


_MAX_ITERATIONS = _template_pattern(MAX_ITERATIONS_MESSAGE)

_MODES = [
    SessionMode(
        id="auto",
        name="Auto",
        description="The agent runs its tools without asking first.",
    )
]


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
        self._conn: Client | None = None
        self.client_capabilities: ClientCapabilities | None = None
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
        # How many session/prompt requests of each session are waiting in
        # ``_finish``, and the messages Pool handed over with
        # ``_poolside/session_steer`` meanwhile (receipt, text), in order:
        # the prompt stays open until they are handled (see ``_finish``).
        self._open: dict[str, int] = {}
        self._followers: dict[str, list[tuple[Receipt, str]]] = {}
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

    def on_connect(self, conn: Client) -> None:
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

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        del kwargs
        self.client_capabilities = client_capabilities
        self._pool_dialects = pool_dialects(client_info)
        self._pool_client = bool(self._pool_dialects)
        self._pool_forms = self._pool_client
        return initialize_response(protocol_version, client_info)

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

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        del kwargs
        root = validate_workspace(cwd, additional_directories)
        self._log_llm_config(root)
        options = SessionOptions(
            workspace=root,
            agent_spec=self._agent_spec_for(root),
            model=self._model,
            host="acp",
        )
        warnings: list[str] = []
        attached: list[ACPEventBridge] = []

        async def prepare(session: Session) -> None:
            _trace_as(session)
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
        return NewSessionResponse(
            session_id=session.id,
            modes=self._modes(session),
            config_options=self._config_options(session),
        )

    # ---- load ----------------------------------------------------------

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        del kwargs
        root = validate_workspace(cwd, additional_directories)
        self._log_llm_config(root)
        registry = self.registry_for(root)
        live = self.session(session_id)
        if live is not None:
            # Attach: the same Session; reuse this adapter's bridge if it has
            # one (a second bridge would send every update twice) and replay.
            bridge = self._bridges.get(session_id) or self._attach(live)
            self._replay(bridge, live, pool=self._pool_client)
            await bridge.flush()
            self._defer_bootstrap_updates(live, [])
            return LoadSessionResponse(
                modes=self._modes(live), config_options=self._config_options(live)
            )

        warnings: list[str] = []
        attached: list[ACPEventBridge] = []

        async def prepare(session: Session) -> None:
            _trace_as(session)
            bridge = self._attach(session)
            attached.append(bridge)
            self._replay(bridge, session, pool=self._pool_client)
            warnings.extend(await self._prepare_tools(session, mcp_servers))

        try:
            session = await registry.load(session_id, prepare=prepare, host="acp")
        except BaseException as exc:
            for bridge in attached:
                self._bridges.pop(bridge.session_id, None)
                await bridge.close(finish_open=False)
            raise _load_error(session_id, exc) from exc
        await self._bridges[session.id].flush()
        self._defer_bootstrap_updates(session, warnings)
        return LoadSessionResponse(
            modes=self._modes(session), config_options=self._config_options(session)
        )

    # ---- list ----------------------------------------------------------

    async def list_sessions(
        self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        """Root sessions with at least one message, most recent first (see ``list_sessions``).

        Without ``cwd``, the stores of the workspaces this connection has
        named; there is no index of every workspace.
        """
        del kwargs
        if cwd is not None:
            self.registry_for(validate_workspace(cwd, None))

        def live(session_id: str) -> tuple[str, str | None] | None:
            for registry in self._registries.values():
                session = registry.get(session_id)
                if session is not None:
                    return registry.live_info(session).status, session.info.title
            return None

        return await list_sessions(
            self._store_for,
            cwd=cwd,
            cursor=cursor,
            live=live,
            known=[registry.store for registry in self._registries.values()],
        )

    def _store_for(self, workspace: Path) -> SessionStore:
        return self.registry_for(workspace).store

    # ---- close and delete ----------------------------------------------

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse:
        """Close a session; for a child whose parent is live, only stop following it."""
        del kwargs
        session = self.session(session_id)
        bridge = self._bridges.pop(session_id, None)
        self._cancel_client_calls(session_id)
        if session is None and bridge is None:
            raise RequestError.resource_not_found(session_id)
        if session is not None and self._parent_is_live(session):
            if bridge is not None:
                await bridge.close(finish_open=False)
            return CloseSessionResponse()
        if session is not None:
            await session.close()
        if bridge is not None:
            await bridge.close()
        return CloseSessionResponse()

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """``_nooa/session/inject``, ``revoke_inject`` (see ``_inject``), Pool's
        ``_poolside/session_steer`` or ``_pool/session_steer`` (see
        ``_pool_steer``) and ``delete``.

        ``_nooa/session/delete``: ``sessionId``, optional ``keepFiles`` and ``cwd``.

        ``cwd`` names the workspace whose store holds the session; without
        it the stores of the workspaces this connection has named are searched.
        """
        if method == _INJECT_METHOD:
            return await self._inject(params)
        if method == _REVOKE_METHOD:
            return self._revoke_inject(params)
        dialect = _POOL_STEER_METHODS.get(method)
        if dialect is not None:
            return await self._pool_steer(params, dialect)
        if method != _DELETE_METHOD:
            raise RequestError.method_not_found(f"_{method}")
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
        self, params: dict[str, Any], dialect: PoolDialect | None = None
    ) -> dict[str, Any]:
        """``_poolside/session_steer``: ``{sessionId, inputId, prompt}`` -> ``{inputId}``.

        Pool 2's ``_pool/session_steer`` has the same params and answer.

        Pool sends what the person types during a running prompt. The text
        is queued for the next turn on ``user_messages``, as ``session/prompt``
        text sent during a turn is: the running turn sees it only as a pending
        message, and the next turn gets it in its notification. A slash
        command is refused, and Pool then sends it as a normal prompt after
        the turn.

        Pool keeps its turn open until the messages it handed over are
        handled, so the open prompt follows this one: it returns only after
        the turn that handles it (``_finish``), and Stop withdraws it if no
        turn took it yet (``cancel``).

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
        if self._open.get(session_id):
            self._followers.setdefault(session_id, []).append((receipt, text))
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

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        """Run a prompt; a prompt sent while a turn runs is queued behind it.

        The text is submitted on ``user_messages`` and the request waits for
        the outcome of the turn that consumes it: the next turn, or the
        running one if the model takes the pending message from its queue
        (the queues context block lists it). A ``Waiting`` outcome keeps the
        request open, and so do messages Pool hands over during the prompt
        (``_pool_steer``) until they are handled. Stop answers a waiting
        prompt ``cancelled`` and withdraws its message if no turn took it.
        ``/name`` prompts naming a
        session command run the command. To steer a running turn, clients
        use ``_nooa/session/inject`` with ``mode: "steer"``.
        """
        del kwargs
        session, bridge = self._followed(session_id)
        text = self._prompt_text(prompt)
        try:
            slash = _slash_invocation(text)
            if slash is not None:
                handled = await self._run_command(session, bridge, *slash)
                if handled is not None:
                    return handled
            await self._request_title(session, text)
            receipt = await session.submit(text, source=SOURCE)
            if self._pool_client:
                bridge.publish(pool_input_event(receipt.item_id, dialects=self._pool_dialects))
            self._prompts.setdefault(session_id, []).append(receipt)
            try:
                return await self._finish(session, bridge, receipt.item_id)
            finally:
                with suppress(ValueError):
                    self._prompts.get(session_id, []).remove(receipt)
        except ItemWithdrawnError:
            await bridge.flush()
            return PromptResponse(stop_reason="cancelled")
        except SessionClosedError:
            raise RequestError.resource_not_found(session_id) from None
        except TurnFailedError as exc:
            return await self._turn_failed(bridge, exc)

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
        from nooa_atom.agent.agent import session_title_request

        await session.submit(session_title_request(text), channel="system_messages", source="host")

    async def _finish(
        self, session: Session, bridge: ACPEventBridge, item_id: str
    ) -> PromptResponse:
        """Wait for the turn that consumes ``item_id``, then for the prompt's followers.

        Followers are the messages Pool handed over with
        ``_poolside/session_steer`` while this prompt was open, in the order
        they were admitted, including ones admitted while this loop runs.
        Each is waited for as ``item_id`` is (``_finish_item``); the prompt
        returns ``end_turn`` when all are done, and stops at the first
        other answer.
        """
        self._open[session.id] = self._open.get(session.id, 0) + 1
        # Taken now: Stop pops the list but leaves in it the followers a
        # turn took, whose outcome is then the cancelled turn's.
        followers = self._followers.setdefault(session.id, [])
        try:
            response = await self._finish_item(session, bridge, item_id)
            while response.stop_reason == "end_turn" and followers:
                receipt, _text = followers[0]
                try:
                    response = await self._finish_item(session, bridge, receipt.item_id)
                finally:
                    if followers and followers[0][0] is receipt:
                        followers.pop(0)
            return response
        finally:
            self._open[session.id] -= 1
            if not self._open[session.id]:
                # Left over only when the prompt failed: they stay queued.
                del self._open[session.id]
                self._followers.pop(session.id, None)

    async def _finish_item(
        self, session: Session, bridge: ACPEventBridge, item_id: str
    ) -> PromptResponse:
        """Wait for the turn that consumes ``item_id``.

        A ``NeedInput`` is never a dialog: the bridge already sent its
        question as the turn's final message, so it ends the item with
        ``end_turn`` and the person's next prompt answers it as text.
        """
        outcome = await session.outcome(item_id)
        # The bridge closed the open cards when the Session reported a
        # cancel, which happens before this outcome resolves.
        await bridge.flush()
        if isinstance(outcome, TurnCancelledOutcome):
            return PromptResponse(stop_reason="cancelled")
        return PromptResponse(stop_reason="end_turn")

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
            request = conn.ext_method(dialect.form_method, params)
        elif self._client_shows_forms():
            field = ElicitationStringPropertySchema(
                type="string",
                title="Verbosity",
                description=_VERBOSITY_HELP,
                enum=list(VERBOSITY_LEVELS),
            )
            schema = ElicitationSchema(properties={"verbosity": field}, required=["verbosity"])
            request = conn.create_elicitation(
                message=question,
                mode=ElicitationFormSessionMode(session_id=session.id, requested_schema=schema),
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
        capabilities = self.client_capabilities
        return (
            capabilities is not None
            and capabilities.elicitation is not None
            and capabilities.elicitation.form is not None
        )

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
            tasks.discard(task)
            if not tasks and self._asks.get(session_id) is tasks:
                del self._asks[session_id]

    async def _turn_failed(self, bridge: ACPEventBridge, exc: TurnFailedError) -> PromptResponse:
        """Map a failed turn: generation limits are stop reasons, anything else an error."""
        await bridge.flush()  # includes the "Unfinished" cards for the failed turn
        error = getattr(exc, "error", None) or exc.__cause__ or exc
        message = str(error)
        if isinstance(error, GenerationError):
            # The messages CodeActStrategy raises, from its own constants.
            # max_retries ("failed after N errors") is repeated invalid output,
            # not a limit on turn requests, so it stays an error.
            if OUTPUT_TOKENS_EXHAUSTED_MESSAGE in message:
                return PromptResponse(stop_reason="max_tokens")
            if _MAX_ITERATIONS.search(message):
                return PromptResponse(stop_reason="max_turn_requests")
        raise RequestError(-32603, message, {"details": message}) from exc

    async def _run_command(
        self, session: Session, bridge: ACPEventBridge, name: str, raw_args: str
    ) -> PromptResponse | None:
        """Run ``/name args``; ``None`` when it is not a command, so the text is a prompt."""
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
                    ConfigOptionUpdate(
                        session_update="config_option_update",
                        config_options=self._config_options(session) or [],
                    )
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
            receipt = await session.submit(result.text, channel=channel, source=SOURCE)
            return await self._finish(session, bridge, receipt.item_id)
        return await self._say(bridge, result.text)

    async def _pick_verbosity(self, session: Session, bridge: ACPEventBridge) -> PromptResponse:
        """Ask the client locally; neither the command nor its answer enters the agent."""
        question = f"ACP verbosity: {bridge.verbosity}. Choose activity detail."
        epoch = self._ask_epochs.get(session.id, 0)
        level = await self._ask_verbosity(session, question)
        if not self._client_answer_is_current(session, bridge, epoch):
            return PromptResponse(stop_reason="cancelled")
        if level is _CANCELLED:
            await bridge.flush()
            return PromptResponse(stop_reason="cancelled")
        if level is None:
            return await self._say(
                bridge,
                f"ACP verbosity: {bridge.verbosity}. Usage: /verbosity silent|concise|verbose",
            )
        if level is _DISMISSED:
            return await self._say(bridge, f"ACP verbosity unchanged: {bridge.verbosity}.")
        # Reuse local dispatch for validation, confirmation and config-picker sync.
        response = await self._run_command(session, bridge, "verbosity", level)
        assert response is not None
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
    async def _say(bridge: ACPEventBridge, message: str) -> PromptResponse:
        if message:
            bridge.publish(update_agent_message(text_block(message)))
        await bridge.flush()
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Stop the running turn. The bridge closes its cards when the Session reports it."""
        del kwargs
        session = self.session(session_id)
        if session is None or session_id not in self._bridges:
            return  # a notification: nothing to answer
        self._cancel_client_calls(session_id)
        # A prompt request must be answered: a prompt whose message no turn
        # took is withdrawn and returns "cancelled"; the one the running turn
        # took returns "cancelled" with it. Injected messages stay queued.
        for receipt in self._prompts.pop(session_id, []):
            session.withdraw(receipt)
        # Messages Pool handed over during the prompt go with it: the ones
        # no turn took are withdrawn, so they do not run with no prompt
        # open, and listed, so none is lost silently.
        stopped = [
            text
            for receipt, text in self._followers.pop(session_id, [])
            if session.withdraw(receipt)
        ]
        if stopped:
            listed = "\n".join(f"- {text}" for text in stopped)
            self._bridges[session_id].publish(
                update_agent_message(
                    text_block(f"Stopped before these messages were handled:\n\n{listed}")
                )
            )
        await session.cancel(by="user")

    # ---- modes and models ----------------------------------------------

    async def set_session_mode(
        self, session_id: str, mode_id: str, **kwargs: Any
    ) -> SetSessionModeResponse:
        """Set the permission mode. Only ``auto`` exists until tools can ask first."""
        del kwargs
        session, _bridge = self._followed(session_id)
        if mode_id not in {mode.id for mode in _MODES}:
            raise RequestError.invalid_params(
                {"modeId": mode_id, "reason": "Unknown mode; this agent offers 'auto' only"}
            )
        await session.set_mode(mode_id)
        return SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: Any
    ) -> SetSessionConfigOptionResponse:
        """Set a model, reasoning level, or ACP-local activity verbosity."""
        del kwargs
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
            return SetSessionConfigOptionResponse(
                config_options=self._config_options(session) or []
            )
        if config_id == "reasoning" and isinstance(value, str):
            await self._set_reasoning(session, value)
            return SetSessionConfigOptionResponse(
                config_options=self._config_options(session) or []
            )
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
        return SetSessionConfigOptionResponse(config_options=self._config_options(session) or [])

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

    def _config_options(self, session: Session) -> list[Any] | None:
        """The model select option (the registry's aliases plus the current model),
        a reasoning option when the current client declares levels, and ACP verbosity."""
        options: list[Any] = []
        current = session.info.model or self._model or _default_model(session.options.workspace)
        aliases = model_aliases(session.options.workspace)
        if current and current not in aliases:
            aliases = [current, *aliases]
        if current and aliases:
            options.append(
                SessionConfigOptionSelect(
                    id="model",
                    name="Model",
                    category="model",
                    description="The model this session uses from its next turn on.",
                    type="select",
                    current_value=current,
                    options=[
                        SessionConfigSelectOption(value=alias, name=alias) for alias in aliases
                    ],
                )
            )
        reasoning = self._reasoning_option(session)
        if reasoning is not None:
            options.append(reasoning)
        bridge = self._bridges.get(session.id)
        options.append(
            SessionConfigOptionSelect(
                id="verbosity",
                name="Verbosity",
                description="ACP activity detail only; does not change the agent or its reasoning.",
                type="select",
                current_value=bridge.verbosity if bridge is not None else "verbose",
                options=[
                    SessionConfigSelectOption(value=level, name=level.title())
                    for level in VERBOSITY_LEVELS
                ],
            )
        )
        return options

    @staticmethod
    def _reasoning_option(session: Session) -> Any | None:
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
        return SessionConfigOptionSelect(
            id="reasoning",
            name="Reasoning",
            category="thought_level",
            description="How much the model reasons before answering, from its next call on.",
            type="select",
            current_value=model.reasoning_level or DEFAULT_REASONING,
            options=[
                SessionConfigSelectOption(value=DEFAULT_REASONING, name=default_name),
                *(SessionConfigSelectOption(value=level, name=level) for level in levels),
            ],
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

    def _parent_is_live(self, session: Session) -> bool:
        return session.parent_id is not None and self.session(session.parent_id) is not None

    def _require_conn(self) -> Client:
        if self._conn is None:
            raise RequestError.internal_error({"reason": "ACP client is not connected"})
        return self._conn

    def _agent_spec_for(self, root: Path) -> str:
        if self._agent_spec:
            return canonical_agent_spec(self._agent_spec)
        from nooa_atom.workspace.options import AtomOptions

        configured = AtomOptions.load(root).agent_spec
        return canonical_agent_spec(configured) if configured else ATOM_AGENT

    def _modes(self, session: Session) -> SessionModeState:
        return SessionModeState(current_mode_id=session.info.mode or "auto", available_modes=_MODES)

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
        servers = list(mcp_servers or [])
        supported_types = (McpServerStdio, HttpMcpServer, SseMcpServer)
        tools: dict[str, MCPTool] = {}
        warnings: list[str] = []
        seen_names: set[str] = set()
        for server in servers:
            name = getattr(server, "name", "<unnamed>")
            if not isinstance(server, supported_types):
                warnings.append(
                    f"MCP server {name!r} was not loaded: unsupported ACP server type "
                    f"{type(server).__name__}."
                )
                continue
            if name in seen_names:
                warnings.append(
                    f"MCP server {name!r} was not loaded: another server has the same name."
                )
                continue
            seen_names.add(name)
            try:
                if isinstance(server, McpServerStdio):
                    env = {item.name: item.value for item in server.env}
                    tools[name] = await MCPManager.create_stdio_server(
                        name,
                        command=server.command,
                        args=server.args,
                        env=env,
                    )
                elif isinstance(server, (HttpMcpServer, SseMcpServer)):
                    headers = {item.name: item.value for item in server.headers}
                    tools[name] = await MCPManager.create_url_server(
                        name,
                        server.url,
                        headers=headers,
                        transport=(
                            "streamable-http" if isinstance(server, HttpMcpServer) else "sse"
                        ),
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
                bridge.publish_best_effort(
                    update_agent_message(
                        text_block(
                            "NOOA started with configuration warnings. The session is still "
                            f"usable.\n\n{details}"
                        )
                    )
                )
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
        """Queue the transcript as user and agent message chunks on the bridge.

        For Pool, each user message is followed by its input event, as Pool's
        own agent replays them.
        """
        for entry in session.transcript():
            if entry.role == "cancelled":
                by = entry.content.removeprefix("Stopped by ").strip() or "user"
                content = cancel_text(by)
            else:
                content = entry.content
            # Each update is a chunk and ACP has no end-of-message marker, so
            # two entries from one speaker in a row would join into one
            # bubble. End each entry with a newline to keep them apart.
            content = content if content.endswith("\n") else content + "\n"
            block = text_block(content)
            bridge.publish(
                update_user_message(block) if entry.role == "user" else update_agent_message(block)
            )
            if pool and entry.role == "user" and entry.item_id:
                bridge.publish(pool_input_event(entry.item_id, dialects=bridge.pool_dialects))

    @staticmethod
    def _prompt_text(prompt: list[Any]) -> str:
        parts: list[str] = []
        for block in prompt:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                parts.append(block.text)
            elif block_type == "resource_link":
                parts.append(f"Resource {block.name}: {block.uri}")
            else:
                raise RequestError.invalid_params(
                    {"reason": f"Unsupported prompt content type: {block_type!r}"}
                )
        text = "\n\n".join(parts)
        if not text.strip():
            raise RequestError.invalid_params({"reason": "Prompt text must not be empty"})
        return text


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


def _available_commands_update(commands: list[CommandInfo]) -> Any:
    """The session's commands, and the ones this adapter runs itself (``/recover`` and ``/verbosity``)."""
    available: list[AvailableCommand] = []
    names = {command.name for command in commands}
    verbosity = CommandInfo(
        name="verbosity",
        description="Set how much agent activity is shown: silent, concise, or verbose.",
        input_hint="[silent|concise|verbose]",
    )
    # The ACP-local command wins over a skill command with the same name.
    for command in [
        *(c for c in commands if c.name != verbosity.name),
        *(c for c in (RECOVER,) if c.name not in names),
        verbosity,
    ]:
        input_spec = (
            AvailableCommandInput(UnstructuredCommandInput(hint=command.input_hint))
            if command.input_hint
            else None
        )
        available.append(
            AvailableCommand(
                name=command.name,
                description=command.description,
                input=input_spec,
            )
        )
    return update_available_commands(available)


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
    agent: Any,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    id_base: int | None = None,
    observers: list[Callable[[Any], None]] | None = None,
) -> None:
    """Serve ACP for ``agent`` on one stream pair until the peer closes it.

    Used on standard input and output (``serve``) and on a router's
    socket (the worker). ``id_base`` is the first id of the requests this
    side sends to the client (permission, elicitation, file and terminal
    requests); a router gives each worker its own range so replies can be
    routed by id alone.
    """
    # session/close is registered by the library as unstable. initialize()
    # advertises the close capability, so without this flag the agent
    # promises a method that answers "method not found".
    conn = AgentSideConnection(
        cast(Agent, agent),
        writer,
        reader,
        listening=False,
        use_unstable_protocol=True,
        observers=list(observers or []),
    )
    if id_base is not None:
        # acp 0.12 has no option for the first request id: every
        # agent-to-client request takes Connection._next_request_id and
        # increments it. Seed it before listen() so no request can go out
        # with the library's default of 0.
        # TODO(upstream): ask agent-client-protocol for a request-id option.
        conn._conn._next_request_id = id_base
    try:
        await conn.listen()
    finally:
        await asyncio.shield(conn.close())


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
    "initialize_response",
    "list_sessions",
    "open_stdio",
    "serve",
    "serve_connection",
]
