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
import signal
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from acp import (
    PROTOCOL_VERSION,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    RequestError,
    run_agent,
    text_block,
    update_agent_message,
    update_user_message,
)
from acp.helpers import update_available_commands
from acp.interfaces import Agent, Client
from acp.schema import (
    AgentCapabilities,
    AvailableCommand,
    AvailableCommandInput,
    ClientCapabilities,
    CloseSessionResponse,
    HttpMcpServer,
    Implementation,
    ListSessionsResponse,
    McpCapabilities,
    McpServerStdio,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionListCapabilities,
    SessionMode,
    SessionModeState,
    SseMcpServer,
    UnstructuredCommandInput,
)
from acp.schema import SessionInfo as ACPSessionInfo

from nooa.errors import GenerationError
from nooa.mcp import MCPManager, MCPTool
from nooa.slash_dispatch import CoercionError
from nooa.storage.sqlite import SessionAlreadyActiveError
from nooa_coder.acp.event_bridge import ACPEventBridge, cancel_text
from nooa_coder.coding.identity import CODING_AGENT, canonical_agent_spec
from nooa_coder.coding.slash_commands import RESERVED_COMMAND_NAMES
from nooa_coder.session.items import CommandInfo, TurnCancelledOutcome
from nooa_coder.session.options import SessionOptions
from nooa_coder.session.registry import ChildActiveElsewhereError, SessionRegistry
from nooa_coder.session.session import Session, SessionClosedError, TurnFailedError
from nooa_coder.session.store import InvalidSessionIdError, SessionNotFoundError

logger = logging.getLogger(__name__)

_SESSION_PAGE_SIZE = 50
_DELETE_METHOD = "nooa/session/delete"
"""``_nooa/session/delete`` as ``ext_method`` receives it (without the underscore)."""

SOURCE = "acp"
"""The source of items this adapter admits (the bridge does not echo them back)."""

_CONNECT_TEXT = (
    "Connecting a model provider needs the terminal for now: run `nooa connect` in a "
    "shell, then pick the new model alias here."
)

_MODES = [
    SessionMode(
        id="auto",
        name="Auto",
        description="The agent runs its tools without asking first.",
    )
]


async def _close_in_order(*closers: Callable[[], Any] | None) -> None:
    """Run each closer in order, tolerating failures, then re-raise.

    Equivalent to nesting one ``try/finally`` per closer: every closer runs
    even if an earlier one fails, and the last failure propagates with the
    earlier ones chained as its ``__context__``.
    """
    pending: BaseException | None = None
    for close in closers:
        if close is None:
            continue
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except BaseException as exc:
            if pending is not None:
                exc.__context__ = pending
            pending = exc
    if pending is not None:
        raise pending


def initialize_response(protocol_version: int) -> InitializeResponse:
    """The static answer to ``initialize``: what this agent supports.

    ``session/delete`` and ``logout`` are not advertised: the 0.11 library
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
        ),
        auth_methods=[],
        agent_info=Implementation(
            name="nooa-coder",
            title="NVIDIA Labs Object Oriented Agents (NOOA)",
            version=package_version,
        ),
    )


class CoderACPAgent:
    """One ACP connection's view of a ``SessionRegistry``.

    ``agent_spec`` names the agent class for new sessions (``None``: the
    workspace's ``coding.agent_spec`` setting, else the coding agent).
    ``model`` is the model alias new sessions start with; the registry's
    ``llm_factory`` builds the client.
    """

    def __init__(
        self,
        registry: SessionRegistry,
        *,
        agent_spec: str | None = None,
        model: str | None = None,
    ) -> None:
        self.registry = registry
        self._agent_spec = agent_spec
        self._model = model
        self._conn: Client | None = None
        self.client_capabilities: ClientCapabilities | None = None
        self._bridges: dict[str, ACPEventBridge] = {}
        self._background: set[asyncio.Task[None]] = set()

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

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
            attached.append(self._attach(session))
            warnings.extend(await self._prepare_agent(session, mcp_servers))

        try:
            session = await self.registry.create(options, prepare=prepare)
        except BaseException:
            for bridge in attached:
                self._bridges.pop(bridge.session_id, None)
                await bridge.close(finish_open=False)
            raise
        self._defer_bootstrap_updates(session, warnings)
        return NewSessionResponse(session_id=session.id, modes=self._modes(session))

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
        self._validate_workspace(cwd, additional_directories)
        live = self.registry.get(session_id)
        if live is not None:
            # Attach: the same Session; reuse this adapter's bridge if it has
            # one (a second bridge would send every update twice) and replay.
            bridge = self._bridges.get(session_id) or self._attach(live)
            self._replay(bridge, live)
            await bridge.flush()
            self._defer_bootstrap_updates(live, [])
            return LoadSessionResponse(modes=self._modes(live))

        warnings: list[str] = []
        attached: list[ACPEventBridge] = []

        async def prepare(session: Session) -> None:
            bridge = self._attach(session)
            attached.append(bridge)
            self._replay(bridge, session)
            warnings.extend(await self._prepare_agent(session, mcp_servers))

        try:
            session = await self.registry.load(session_id, prepare=prepare, host="acp")
        except BaseException as exc:
            for bridge in attached:
                self._bridges.pop(bridge.session_id, None)
                await bridge.close(finish_open=False)
            raise _load_error(session_id, exc) from exc
        await self._bridges[session.id].flush()
        self._defer_bootstrap_updates(session, warnings)
        return LoadSessionResponse(modes=self._modes(session))

    # ---- list ----------------------------------------------------------

    async def list_sessions(
        self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        """Root sessions with at least one message, most recent first.

        ``cwd`` keeps one workspace; without it every workspace is listed
        (the store is per user). Sessions held by another process are left
        out: opening them would fail. ``_meta["dev.nooa/status"]`` is
        ``running``, ``idle`` or ``retained`` for sessions live here, else
        ``on_disk``.
        """
        del kwargs
        root = self._validate_workspace(cwd, None) if cwd is not None else None
        try:
            offset = int(cursor) if cursor is not None else 0
        except ValueError:
            raise RequestError.invalid_params(
                {"cursor": cursor, "reason": "Invalid cursor"}
            ) from None
        if offset < 0:
            raise RequestError.invalid_params({"cursor": cursor, "reason": "Invalid cursor"})
        store = self.registry.store

        def scan() -> list[tuple[Any, bool]]:
            # Pure filesystem work, one lock probe per session: off the loop.
            return [
                (info, store.is_active(info.id))
                for info in store.list(workspace=root, roots_only=True)
                if info.turn_count > 0
            ]

        found: list[tuple[Any, str]] = []
        for info, active in await asyncio.to_thread(scan):
            live = self.registry.get(info.id)
            if live is not None:
                status = self.registry.live_info(live).status
                info = info.model_copy(update={"title": live.info.title or info.title})
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

    # ---- close and delete ----------------------------------------------

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse:
        """Close a session; for a child whose parent is live, only stop following it."""
        del kwargs
        session = self.registry.get(session_id)
        bridge = self._bridges.pop(session_id, None)
        if session is None and bridge is None:
            raise RequestError.resource_not_found(session_id)
        if session is not None and self._parent_is_live(session):
            if bridge is not None:
                await bridge.close(finish_open=False)
            return CloseSessionResponse()
        if session is not None:
            await self.registry.close(session_id)
        if bridge is not None:
            await bridge.close()
        return CloseSessionResponse()

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method != _DELETE_METHOD:
            raise RequestError.method_not_found(f"_{method}")
        session_id = params.get("sessionId")
        if not isinstance(session_id, str):
            raise RequestError.invalid_params({"reason": "sessionId must be a string"})
        bridge = self._bridges.pop(session_id, None)
        try:
            await self.registry.delete(session_id, keep_files=bool(params.get("keepFiles")))
        except (SessionNotFoundError, InvalidSessionIdError):
            raise RequestError.resource_not_found(session_id) from None
        if bridge is not None:
            await bridge.close()
        return {}

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
            self.registry.close_all,
            *(bridge.close for bridge in bridges),
        )

    # ---- prompt and cancel ---------------------------------------------

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        """Run a prompt; a prompt sent while a turn runs steers that turn.

        Text goes in with ``steer`` (a plain submit while idle) and the
        request waits for the outcome of the turn that consumed it. When a
        turn is running, that turn's next model call sees the text and both
        prompts return together; if the running turn was already finishing,
        the text is handled by the next turn and this prompt returns with
        that one. A ``Waiting`` outcome keeps the request open. ``/name``
        prompts naming a session command run the command.
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
            receipt = await session.steer(text, source=SOURCE)
            return await self._finish(session, bridge, receipt.item_id)
        except SessionClosedError:
            raise RequestError.resource_not_found(session_id) from None
        except TurnFailedError as exc:
            return await self._turn_failed(bridge, exc)

    async def _finish(
        self, session: Session, bridge: ACPEventBridge, item_id: str
    ) -> PromptResponse:
        """Wait for the outcome of the turn that consumes ``item_id`` and map it to a stop reason."""
        outcome = await session.outcome(item_id)
        if isinstance(outcome, TurnCancelledOutcome):
            # The bridge closed the open cards when the Session reported the
            # cancel, which happens before this outcome resolves.
            await bridge.flush()
            return PromptResponse(stop_reason="cancelled")
        # NeedInput: the bridge sent the question as the turn's final message.
        await bridge.flush()
        return PromptResponse(stop_reason="end_turn")

    async def _turn_failed(self, bridge: ACPEventBridge, exc: TurnFailedError) -> PromptResponse:
        """Map a failed turn: generation limits are stop reasons, anything else an error."""
        await bridge.flush()  # includes the "Unfinished" cards for the failed turn
        error = getattr(exc, "error", None) or exc.__cause__ or exc
        message = str(error)
        if isinstance(error, GenerationError):
            if message.startswith("Empty response: the model used all available output tokens"):
                return PromptResponse(stop_reason="max_tokens")
            if message.startswith("Generation failed after ") and (
                "max_iterations=" in message or "max_retries=" in message
            ):
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
        if result.output_to_agent and result.text:
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
        session = self.registry.get(session_id)
        if session is None or session_id not in self._bridges:
            return  # a notification: nothing to answer
        await session.cancel(by="user")

    def _followed(self, session_id: str) -> tuple[Session, ACPEventBridge]:
        session = self.registry.get(session_id)
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
        bridge = ACPEventBridge(session, conn, resolve_child=self.registry.get)
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
        return session.parent_id is not None and self.registry.get(session.parent_id) is not None

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


async def serve(
    registry: SessionRegistry,
    *,
    agent_spec: str | None = None,
    model: str | None = None,
    observers: list[Callable[[Any], None]] | None = None,
) -> None:
    """Serve ACP on this process's standard input and output until the client leaves."""
    adapter = CoderACPAgent(registry, agent_spec=agent_spec, model=model)
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
        # session/close is registered by the router as unstable. initialize()
        # advertises the close capability, so without this flag the agent
        # promises a method that answers "method not found".
        await run_agent(
            cast(Agent, adapter),
            use_unstable_protocol=True,
            observers=list(observers or []),
        )
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


__all__ = ["CoderACPAgent", "initialize_response", "serve"]
