# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ACP agent over the Session layer: protocol translation, no session policy.

``CoderACPAgent`` implements the ``acp`` library's ``Agent`` interface on a
``SessionRegistry``. Sessions, turns, steering, cancellation and
checkpoints are the Session's; this module maps requests onto it and keeps
one ``ACPEventBridge`` per session id that sends the session's activity
to the client.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import signal
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext, suppress
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from string import Formatter
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    RequestError,
    start_tool_call,
    text_block,
    update_agent_message,
    update_tool_call,
    update_user_message,
)
from acp.agent.connection import AgentSideConnection
from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES
from acp.helpers import update_available_commands
from acp.interfaces import Agent, Client
from acp.schema import (
    AgentCapabilities,
    AvailableCommand,
    AvailableCommandInput,
    ClientCapabilities,
    CloseSessionResponse,
    ElicitationFormSessionMode,
    HttpMcpServer,
    Implementation,
    ListSessionsResponse,
    McpCapabilities,
    McpServerStdio,
    PermissionOption,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionListCapabilities,
    SessionMode,
    SessionModeState,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
    SseMcpServer,
    ToolCallUpdate,
    UnstructuredCommandInput,
)
from acp.schema import SessionInfo as ACPSessionInfo

from nooa.errors import GenerationError
from nooa.interactive import NeedInput
from nooa.mcp import MCPManager, MCPTool
from nooa.slash_dispatch import CoercionError
from nooa.storage.sqlite import SessionAlreadyActiveError
from nooa.strategies.codeact import MAX_ITERATIONS_MESSAGE, OUTPUT_TOKENS_EXHAUSTED_MESSAGE
from nooa_coder.acp.event_bridge import ACPEventBridge, cancel_text
from nooa_coder.acp.need_input import answer_from_content, need_input_schema
from nooa_coder.coding.identity import CODING_AGENT, canonical_agent_spec
from nooa_coder.coding.slash_commands import RESERVED_COMMAND_NAMES
from nooa_coder.session.items import CommandInfo, Receipt, TurnCancelledOutcome
from nooa_coder.session.options import SessionOptions
from nooa_coder.session.registry import ChildActiveElsewhereError, SessionRegistry
from nooa_coder.session.session import (
    ItemWithdrawnError,
    Session,
    SessionClosedError,
    TurnFailedError,
)
from nooa_coder.session.store import (
    InvalidSessionIdError,
    SessionNotFoundError,
    SessionStore,
    sessions_root,
)

logger = logging.getLogger(__name__)

_SESSION_PAGE_SIZE = 50
_DELETE_METHOD = "nooa/session/delete"
"""``_nooa/session/delete`` as ``ext_method`` receives it (without the underscore)."""
_INJECT_METHOD = "nooa/session/inject"
_REVOKE_METHOD = "nooa/session/revoke_inject"
INJECT_CAPABILITY = {"dev.nooa/inject": {"queue": {}, "steer": {}, "revoke": {}}}
"""``agentCapabilities._meta`` for ``_nooa/session/inject`` and ``revoke_inject``.

They follow the ACP RFD for message injection (agent-client-protocol PR #1261).
"""

SOURCE = "acp"
"""The source of items this adapter admits (the bridge does not echo them back)."""

DECLINED = "(declined to answer)"
"""What the agent receives when the person declines or dismisses a question."""
DECLINED_SOURCE = "user:declined"

_CANCELLED = object()  # a client request stopped by session/cancel

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


def model_aliases() -> list[str]:
    """The model aliases configured in the NOOA model registry, sorted."""
    from nooa.unifiedllm.registry import MODELS, ensure_loaded

    try:
        ensure_loaded()
    except Exception:
        logger.warning("Could not load the model registry", exc_info=True)
    return sorted(MODELS)


def initialize_response(protocol_version: int) -> InitializeResponse:
    """The static answer to ``initialize``: what this agent supports.

    ``session/delete`` and ``logout`` are not advertised: the 0.12 library
    does not route them, so a client calling them would get "method not
    found". Deleting is the ``_nooa/session/delete`` extension method.
    """
    try:
        package_version = version("nooa-coder")
    except PackageNotFoundError:
        package_version = "0.0.0"
    return InitializeResponse(
        protocol_version=min(protocol_version, PROTOCOL_VERSION),
        agent_capabilities=AgentCapabilities(
            load_session=True,
            # McpCapabilities defaults to all-false, and a client that honours
            # the handshake then filters its HTTP/SSE servers out of
            # session/new. _create_mcp_tools connects both transports, so say
            # so. `acp` stays off: it is unstable in the spec and not
            # implemented here.
            mcp_capabilities=McpCapabilities(http=True, sse=True),
            session_capabilities=SessionCapabilities(
                list=SessionListCapabilities(),
                close=SessionCloseCapabilities(),
            ),
            field_meta=INJECT_CAPABILITY,
        ),
        auth_methods=[],
        agent_info=Implementation(
            name="nooa-coder",
            title="NVIDIA Labs Object Oriented Agents (NOOA)",
            version=package_version,
        ),
    )


class CoderACPAgent:
    """One ACP connection's view of the sessions this process runs.

    Sessions are stored per workspace (``sessions_root``): a request's
    ``cwd`` picks the store, and each store has its own
    ``SessionRegistry``, built by ``new_registry(store)`` the first time
    the client names that workspace. ``sessions_dir`` is one directory
    for every workspace instead (``None``: ``NOOA_SESSIONS_DIR``, else each
    workspace's ``.nooa/sessions``).

    ``agent_spec`` names the agent class for new sessions (``None``: the
    workspace's ``coding.agent_spec`` setting, else the coding agent).
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
        self._bridges: dict[str, ACPEventBridge] = {}
        self._background: set[asyncio.Task[None]] = set()
        self._title_checked: set[str] = set()
        self._chosen_models: dict[str, str] = {}
        # Receipts of prompts still waiting for their turn, by session: Stop
        # answers them "cancelled" and withdraws the ones not yet consumed.
        self._prompts: dict[str, list[Receipt]] = {}
        # Receipts of injected messages, by session and item id, for revoke.
        self._injects: dict[str, dict[str, Receipt]] = {}
        # Client requests (forms, permissions) a prompt is waiting on, by
        # session: session/cancel stops them.
        self._asks: dict[str, asyncio.Task[Any]] = {}
        # The question each session's prompt is asking, so two prompts that
        # returned with the same turn ask it once.
        self._asking: dict[str, NeedInput] = {}

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

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
        del client_info, kwargs
        self.client_capabilities = client_capabilities
        return initialize_response(protocol_version)

    # ---- new -----------------------------------------------------------

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        del kwargs
        root = self._validate_workspace(cwd, additional_directories)
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
            warnings.extend(await self._prepare_agent(session, mcp_servers))

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
        root = self._validate_workspace(cwd, additional_directories)
        registry = self.registry_for(root)
        live = self.session(session_id)
        if live is not None:
            # Attach: the same Session; reuse this adapter's bridge if it has
            # one (a second bridge would send every update twice) and replay.
            bridge = self._bridges.get(session_id) or self._attach(live)
            self._replay(bridge, live)
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
            self._replay(bridge, session)
            warnings.extend(await self._prepare_agent(session, mcp_servers))

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
            self.registry_for(self._validate_workspace(cwd, None))

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
        """``_nooa/session/inject``, ``revoke_inject`` (see ``_inject``) and ``delete``.

        ``_nooa/session/delete``: ``sessionId``, optional ``keepFiles`` and ``cwd``.

        ``cwd`` names the workspace whose store holds the session; without
        it the stores of the workspaces this connection has named are searched.
        """
        if method == _INJECT_METHOD:
            return await self._inject(params)
        if method == _REVOKE_METHOD:
            return self._revoke_inject(params)
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
        try:
            if mode == "steer":
                receipt = await session.steer(text, source=SOURCE)
            else:
                receipt = await session.submit(text, source=SOURCE)
        except SessionClosedError:
            raise RequestError.resource_not_found(session_id) from None
        self._injects.setdefault(session_id, {})[receipt.item_id] = receipt
        return {"messageId": receipt.item_id, "delivered": receipt.delivered}

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
        for task in list(self._background):
            task.cancel()
        await asyncio.gather(*self._background, return_exceptions=True)
        bridges = list(self._bridges.values())
        self._bridges.clear()
        await _close_in_order(
            *(registry.close_all for registry in self._registries.values()),
            *(bridge.close for bridge in bridges),
        )

    # ---- prompt and cancel ---------------------------------------------

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        """Run a prompt; a prompt sent while a turn runs is queued behind it.

        The text is submitted on ``user_messages`` and the request waits for
        the outcome of the turn that consumes it: the next turn, or the
        running one if the model takes the pending message from its queue
        (the queues context block lists it). A ``Waiting`` outcome keeps the
        request open. Stop answers a waiting prompt ``cancelled`` and
        withdraws its message if no turn took it. ``/name`` prompts naming a
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
        if session.info.title or "system_messages" not in session.agent.queue_manager.channels():
            return
        if any(entry.role == "user" for entry in session.transcript()):
            return
        from nooa_coder.coding.agent import session_title_request

        await session.submit(session_title_request(text), channel="system_messages", source="host")

    async def _finish(
        self, session: Session, bridge: ACPEventBridge, item_id: str
    ) -> PromptResponse:
        """Wait for the turn that consumes ``item_id``; answer questions until it is done.

        A ``NeedInput`` the client can answer (a form, or a yes/no
        permission) is submitted and the same prompt waits for the next
        turn; otherwise the question, already sent by the bridge as the
        turn's final message, ends the prompt with ``end_turn``.
        """
        while True:
            outcome = await session.outcome(item_id)
            if isinstance(outcome, TurnCancelledOutcome):
                # The bridge closed the open cards when the Session reported
                # the cancel, which happens before this outcome resolves.
                await bridge.flush()
                return PromptResponse(stop_reason="cancelled")
            if not isinstance(outcome, NeedInput) or self._asking.get(session.id) is outcome:
                # Done; or a question another prompt of this turn is asking.
                await bridge.flush()
                return PromptResponse(stop_reason="end_turn")
            self._asking[session.id] = outcome
            try:
                await bridge.flush()
                answer = await self._ask(session, bridge, outcome)
            finally:
                if self._asking.get(session.id) is outcome:
                    del self._asking[session.id]
            if answer is _CANCELLED:
                await bridge.flush()
                return PromptResponse(stop_reason="cancelled")
            if answer is None:
                return PromptResponse(stop_reason="end_turn")
            value, source = answer
            receipt = await session.submit(value, source=source)
            item_id = receipt.item_id

    async def _ask(self, session: Session, bridge: ACPEventBridge, need: NeedInput) -> Any:
        """Ask the client to answer ``need``: ``(item, source)``, ``None`` or ``_CANCELLED``.

        A form when the client advertised ``elicitation.form`` and the
        question flattens; else a permission request for a yes/no question;
        else ``None`` (the question stays as text). A client error falls
        back to ``None``.
        """
        conn = self._require_conn()
        capabilities = self.client_capabilities
        forms = (
            capabilities is not None
            and capabilities.elicitation is not None
            and capabilities.elicitation.form is not None
        )
        schema = need_input_schema(need)
        if forms and schema is not None:
            mode = ElicitationFormSessionMode(session_id=session.id, requested_schema=schema)
            response = await self._client_call(
                session.id, conn.create_elicitation(message=need.question, mode=mode)
            )
            if response is None or response is _CANCELLED:
                return response
            if response.action == "accept":
                return answer_from_content(need, response.content), SOURCE
            return DECLINED, DECLINED_SOURCE
        options = need.options or []
        if sorted(option.lower() for option in options) == ["no", "yes"]:
            return await self._ask_yes_no(session, bridge, need.question, options)
        return None

    async def _ask_yes_no(
        self, session: Session, bridge: ACPEventBridge, question: str, options: list[str]
    ) -> Any:
        """A yes/no question as a pending card and a permission request."""
        conn = self._require_conn()
        yes = next(option for option in options if option.lower() == "yes")
        no = next(option for option in options if option.lower() == "no")
        tool_call_id = f"question-{uuid4()}"
        bridge.publish(start_tool_call(tool_call_id, question, kind="other", status="pending"))
        await bridge.flush()
        response = await self._client_call(
            session.id,
            conn.request_permission(
                session_id=session.id,
                tool_call=ToolCallUpdate(
                    tool_call_id=tool_call_id, title=question, status="pending"
                ),
                options=[
                    PermissionOption(option_id=yes, name=yes, kind="allow_once"),
                    PermissionOption(option_id=no, name=no, kind="reject_once"),
                ],
            ),
        )
        chosen = getattr(getattr(response, "outcome", None), "option_id", None)
        if response is _CANCELLED or response is None or chosen not in (yes, no):
            bridge.publish(
                update_tool_call(
                    tool_call_id,
                    status="failed",
                    title="Cancelled" if response is _CANCELLED else question,
                )
            )
            await bridge.flush()
            if response is None or response is _CANCELLED:
                return response
            return DECLINED, DECLINED_SOURCE
        bridge.publish(
            update_tool_call(tool_call_id, status="completed", content=None, raw_output=chosen)
        )
        await bridge.flush()
        return chosen, SOURCE

    async def _client_call(self, session_id: str, request: Any) -> Any:
        """Await a request to the client that session/cancel can stop.

        Returns the response, ``_CANCELLED`` when cancelled, or ``None`` when
        the client failed it (the caller falls back to text).
        """
        task = asyncio.ensure_future(request)
        self._asks[session_id] = task
        try:
            return await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if task.cancelled() and not (current is not None and current.cancelling()):
                return _CANCELLED
            raise
        except Exception:
            logger.warning("The client failed a request in session %s", session_id, exc_info=True)
            return None
        finally:
            if self._asks.get(session_id) is task:
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
            channels = session.agent.queue_manager.channels()
            channel = "slash_commands" if "slash_commands" in channels else "user_messages"
            receipt = await session.submit(result.text, channel=channel, source=SOURCE)
            return await self._finish(session, bridge, receipt.item_id)
        return await self._say(bridge, result.text)

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
        ask = self._asks.get(session_id)
        if ask is not None:
            ask.cancel()
        # A prompt request must be answered: a prompt whose message no turn
        # took is withdrawn and returns "cancelled"; the one the running turn
        # took returns "cancelled" with it. Injected messages stay queued.
        for receipt in self._prompts.pop(session_id, []):
            session.withdraw(receipt)
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
        """``model``: switch the session's model from its next turn on."""
        del kwargs
        session, _bridge = self._followed(session_id)
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
        self._chosen_models[session.id] = value
        return SetSessionConfigOptionResponse(config_options=self._config_options(session) or [])

    def _config_options(self, session: Session) -> list[Any] | None:
        """The model select option: the registry's aliases plus the current model."""
        current = (
            self._chosen_models.get(session.id)
            or session.info.model
            or session.options.model
            or self._model
        )
        aliases = model_aliases()
        if current and current not in aliases:
            aliases = [current, *aliases]
        if not current or not aliases:
            return None
        return [
            SessionConfigOptionSelect(
                id="model",
                name="Model",
                category="model",
                description="The model this session uses from its next turn on.",
                type="select",
                current_value=current,
                options=[SessionConfigSelectOption(value=alias, name=alias) for alias in aliases],
            )
        ]

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
        self._bridges[session.id] = bridge

        def forget_on_close(update: Any) -> None:
            if getattr(update, "kind", None) == "closed":
                if self._bridges.get(session.id) is bridge:
                    del self._bridges[session.id]
                unsubscribe()

        unsubscribe = session.subscribe(forget_on_close)
        commands = getattr(session.agent, "slash_commands", None)
        set_on_change = getattr(commands, "set_on_change", None)
        if callable(set_on_change):
            set_on_change(lambda _commands: bridge.publish(_available_commands_update(session)))
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
        from nooa_coder.workspace.options import CoderOptions

        configured = CoderOptions.load(root).agent_spec
        return canonical_agent_spec(configured) if configured else CODING_AGENT

    def _modes(self, session: Session) -> SessionModeState:
        return SessionModeState(current_mode_id=session.info.mode or "auto", available_modes=_MODES)

    async def _prepare_agent(self, session: Session, mcp_servers: list[Any] | None) -> list[str]:
        """Give a new or loaded agent its host controls and MCP tools; return warnings.

        Runs in the registry's ``prepare`` step, before any turn. A coding
        agent gets the ``/skills`` and ``/mcp`` controls and connects the
        MCP servers its workspace remembers; every agent with skills gets
        the MCP servers the client sent.
        """
        from nooa_coder.coding.agent import CodingAgent
        from nooa_coder.workspace.controls import behavior_commands
        from nooa_coder.workspace.options import CoderOptions, connect_session_mcp

        agent = session.agent
        warnings: list[str] = []
        if isinstance(agent, CodingAgent):
            coder_options = CoderOptions.load(session.options.workspace)
            agent.slash_commands.set_controls(
                behavior_commands(
                    agent,
                    coder_options,
                    workspace=Path(coder_options.working_dir),
                    command_registry=agent.slash_commands,
                )
            )
            warnings.extend(await connect_session_mcp(agent, coder_options))
        tools, mcp_warnings = await self._create_mcp_tools(mcp_servers)
        warnings.extend(mcp_warnings)
        skills = getattr(agent, "skills", None)
        for name, tool in tools.items():
            registry_name = f"mcp.{name}"
            if skills is None:
                warnings.append(f"MCP server {name!r} was not registered: the agent has no skills")
                continue
            try:
                skills.register(registry_name, tool)
                skills.activate([registry_name])
            except ValueError as exc:
                # A server name can collide with a core agent attribute
                # (`shell`, `repo`) or a reserved one. Skipping it keeps the
                # session usable instead of failing session/new.
                warnings.append(f"MCP server {name!r} was not registered: {exc}")
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
            bridge.publish_best_effort(_available_commands_update(session))
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
    def _replay(bridge: ACPEventBridge, session: Session) -> None:
        """Queue the transcript as user and agent message chunks on the bridge."""
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

    @staticmethod
    def _validate_workspace(cwd: str, additional_directories: list[str] | None) -> Path:
        if additional_directories:
            raise RequestError.invalid_params(
                {"reason": "Additional directories are not supported"}
            )
        root = Path(cwd).expanduser()
        if not root.is_absolute() or not root.is_dir():
            raise RequestError.invalid_params(
                {"cwd": cwd, "reason": "cwd must be an existing absolute directory"}
            )
        return root.resolve()


def _trace_as(session: Session) -> None:
    """Make the ACP session id the trace session of the session's turns.

    Turns run in the session's own loop context, so the id is set there by
    a loop hook, and also in the calling request's context. A subagent's
    session has no hook and so no trace session of its own yet.
    """
    try:
        from nooa.tracing import set_session
    except ImportError:
        return
    session.add_loop_context_hook(lambda: set_session(session.id))
    set_session(session.id)


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


async def list_sessions(
    store_for: Callable[[Path], SessionStore],
    *,
    cwd: str | None = None,
    cursor: str | None = None,
    live: Callable[[str], tuple[str, str | None] | None] = lambda _session_id: None,
    known: Iterable[SessionStore] = (),
) -> ListSessionsResponse:
    """Root sessions with at least one message, most recent first.

    ``cwd`` lists that workspace's sessions from ``store_for(cwd)``.
    Without it, every session in the ``known`` stores is listed: sessions
    are stored per workspace and there is no index of every workspace, so
    the caller passes the stores of the workspaces it has seen. Before any
    workspace is named, the server's own working directory is the workspace.
    ``live(session_id)`` returns ``(status,
    title)`` for a session that runs in this process (or, for the router,
    in one of its workers), else ``None``. Sessions held by another
    process are left out: opening them would fail. ``_meta["dev.nooa/status"]``
    is the live status (``running``, ``idle`` or ``retained``) or ``on_disk``.

    Read-only: the store is scanned without claiming any session, so the
    router can answer ``session/list`` without a worker.
    """
    root = CoderACPAgent._validate_workspace(cwd, None) if cwd is not None else None
    try:
        offset = int(cursor) if cursor is not None else 0
    except ValueError:
        raise RequestError.invalid_params({"cursor": cursor, "reason": "Invalid cursor"}) from None
    if offset < 0:
        raise RequestError.invalid_params({"cursor": cursor, "reason": "Invalid cursor"})

    if root is not None:
        stores = [store_for(root)]
    else:
        # Workspaces sharing one directory give the same store more than once.
        stores = list({store.root.resolve(): store for store in known}.values())
        if not stores:
            # Pool 1.0.16 lists without ``cwd`` before it opens any session;
            # the client starts the server in the directory it works in.
            stores = [store_for(Path.cwd())]

    def scan() -> list[tuple[Any, bool]]:
        # Pure filesystem work, one lock probe per session: off the loop.
        infos = [
            (info, store)
            for store in stores
            for info in store.list(workspace=root, roots_only=True)
            if info.turn_count > 0
        ]
        infos.sort(key=lambda pair: pair[0].last_active, reverse=True)
        return [(info, store.is_active(info.id)) for info, store in infos]

    found: list[tuple[Any, str]] = []
    for info, active in await asyncio.to_thread(scan):
        here = live(info.id)
        if here is not None:
            status, title = here
            info = info.model_copy(update={"title": title or info.title})
        elif active:
            continue
        else:
            status = "on_disk"
        workspace = info.workspace if Path(info.workspace).is_absolute() else None
        workspace = workspace or (str(root) if root is not None else None)
        if workspace is None:
            continue  # ACP requires an absolute cwd for every entry
        found.append((info.model_copy(update={"workspace": workspace}), status))
    page = found[offset : offset + _SESSION_PAGE_SIZE]
    sessions = [
        ACPSessionInfo(
            session_id=info.id,
            cwd=info.workspace,
            title=info.title or f"Untitled session [{info.id[:8]}]",
            updated_at=datetime.fromtimestamp(info.last_active, UTC).isoformat(),
            field_meta={"dev.nooa/status": status},
        )
        for info, status in page
    ]
    next_cursor = str(offset + len(page)) if len(found) > offset + len(page) else None
    return ListSessionsResponse(sessions=sessions, next_cursor=next_cursor)


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


def _available_commands_update(session: Session) -> Any:
    commands: list[CommandInfo] = session.commands()
    available: list[AvailableCommand] = []
    for command in commands:
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


class _WritePipeProtocol(asyncio.BaseProtocol):
    """Flow control for the output pipe (as ``acp.stdio`` does for stdout)."""

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._paused = False
        self._drain_waiter: asyncio.Future[None] | None = None

    def pause_writing(self) -> None:
        self._paused = True
        if self._drain_waiter is None:
            self._drain_waiter = self._loop.create_future()

    def resume_writing(self) -> None:
        self._paused = False
        if self._drain_waiter is not None and not self._drain_waiter.done():
            self._drain_waiter.set_result(None)
        self._drain_waiter = None

    async def _drain_helper(self) -> None:
        if self._paused and self._drain_waiter is not None:
            await self._drain_waiter


async def _stdio_streams(
    input_fd: int, output_fd: int
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """A reader on ``input_fd`` and a writer on ``output_fd`` (the reserved real stdio)."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=DEFAULT_STDIO_BUFFER_LIMIT_BYTES)
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader),
        os.fdopen(input_fd, "rb", buffering=0, closefd=False),
    )
    protocol = _WritePipeProtocol()
    transport, _ = await loop.connect_write_pipe(
        lambda: protocol, os.fdopen(output_fd, "wb", buffering=0, closefd=False)
    )
    return reader, asyncio.StreamWriter(transport, protocol, None, loop)


async def open_stdio(
    input_fd: int | None = None, output_fd: int | None = None
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Streams for ACP frames: on the reserved descriptors when given, else stdin and stdout."""
    if input_fd is not None and output_fd is not None:
        return await _stdio_streams(input_fd, output_fd)
    from acp.stdio import stdio_streams

    return await stdio_streams(limit=DEFAULT_STDIO_BUFFER_LIMIT_BYTES)


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
        # acp 0.11 has no option for the first request id: every
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
    ``new_registry`` and ``sessions_dir`` are as for ``CoderACPAgent``.
    """
    adapter = CoderACPAgent(
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
    "CoderACPAgent",
    "initialize_response",
    "list_sessions",
    "open_stdio",
    "serve",
    "serve_connection",
]
