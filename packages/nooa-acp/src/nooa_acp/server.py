# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP adapter for the host-neutral NOOA coding agent."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
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
from acp.interfaces import Agent, Client
from acp.schema import (
    AgentCapabilities,
    CloseSessionResponse,
    Implementation,
    ListSessionsResponse,
    McpServerStdio,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionListCapabilities,
)
from acp.schema import (
    SessionInfo as ACPSessionInfo,
)

from nooa.errors import GenerationError
from nooa.mcp import MCPManager, MCPTool
from nooa.sessions import (
    InvalidSessionIdError,
    SessionBusyError,
    SessionHandle,
    SessionNotFoundError,
    SessionRuntime,
    SessionRuntimePool,
    SessionStore,
)
from nooa.unifiedllm import UnifiedLLM
from nooa_acp.coding_agent import CodingInteractiveAgent
from nooa_acp.dispatcher import InteractiveSessionDispatcher
from nooa_acp.event_bridge import ACPEventBridge

_SESSION_PAGE_SIZE = 50


@dataclass(slots=True)
class _ACPSession:
    """Live resources owned by one ACP session runtime."""

    handle: SessionHandle
    agent: CodingInteractiveAgent
    dispatcher: InteractiveSessionDispatcher
    bridge: ACPEventBridge
    cancel_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancel_complete: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.cancel_complete.set()

    async def close(self) -> None:
        try:
            await self.bridge.close()
        finally:
            try:
                await self.dispatcher.close()
            finally:
                self.handle.close()


class CodingACPAdapter:
    def __init__(self, llm_factory: Callable[[], UnifiedLLM]) -> None:
        self._llm_factory = llm_factory
        self._client: Client | None = None
        self._sessions: SessionRuntimePool[_ACPSession] = SessionRuntimePool()

    def on_connect(self, conn: Client) -> None:
        self._client = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        del protocol_version, client_capabilities, client_info, kwargs
        try:
            package_version = version("nooa-acp")
        except PackageNotFoundError:
            package_version = "0.0.0"
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                session_capabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                    close=SessionCloseCapabilities(),
                ),
            ),
            auth_methods=[],
            agent_info=Implementation(
                name="nooa-acp",
                title="NVIDIA Labs Object Oriented Agents (NOOA)",
                version=package_version,
            ),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        del kwargs
        root = self._validate_workspace(cwd, additional_directories)
        llm = self._llm_factory()
        try:
            handle = self._store(root).create(
                model=llm.model,
                agent="CodingAgent",
                working_directory=str(root),
                host="acp",
                check_same_thread=False,
            )
        except BaseException:
            await llm.aclose()
            raise
        try:
            await self._create_runtime(handle, root, mcp_servers, llm=llm)
        except BaseException:
            handle.close()
            self._store(root).delete(handle.id)
            raise
        return NewSessionResponse(session_id=handle.id)

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
        try:
            handle = self._store(root).open(session_id, check_same_thread=False)
        except (InvalidSessionIdError, SessionNotFoundError):
            raise RequestError.resource_not_found(session_id) from None
        runtime: SessionRuntime[_ACPSession] | None = None
        try:
            runtime = await self._create_runtime(handle, root, mcp_servers)
            await self._replay_session(handle)
        except BaseException:
            if runtime is not None:
                with suppress(KeyError):
                    await self._sessions.remove(session_id)
            else:
                handle.close()
            raise
        return LoadSessionResponse()

    async def list_sessions(
        self,
        cwd: str | None = None,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        del kwargs
        root = self._validate_workspace(cwd or str(Path.cwd()), None)
        try:
            offset = int(cursor) if cursor is not None else 0
        except ValueError:
            raise RequestError.invalid_params(
                {"cursor": cursor, "reason": "Invalid cursor"}
            ) from None
        if offset < 0:
            raise RequestError.invalid_params({"cursor": cursor, "reason": "Invalid cursor"})

        found = self._store(root).list(limit=offset + _SESSION_PAGE_SIZE + 1)
        page = found[offset : offset + _SESSION_PAGE_SIZE]
        sessions = [
            ACPSessionInfo(
                session_id=info.id,
                cwd=info.working_directory or str(root),
                title=info.title,
                updated_at=datetime.fromtimestamp(info.last_active, UTC).isoformat(),
            )
            for info in page
        ]
        next_cursor = str(offset + len(page)) if len(found) > offset + len(page) else None
        return ListSessionsResponse(sessions=sessions, next_cursor=next_cursor)

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse:
        del kwargs
        try:
            await self._sessions.remove(session_id)
        except KeyError:
            raise RequestError.resource_not_found(session_id) from None
        return CloseSessionResponse()

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        del kwargs
        runtime = await self._get_runtime(session_id)
        text = self._prompt_text(prompt)
        try:
            async with runtime.turn():
                session = runtime.value
                session.handle.record_user_message(text)
                session.cancel_complete.clear()
                try:
                    result = await session.dispatcher.submit(text)
                except GenerationError as exc:
                    await session.bridge.flush()
                    message = str(exc)
                    if message.startswith(
                        "Empty response: the model used all available output tokens"
                    ):
                        return PromptResponse(stop_reason="max_tokens")
                    if message.startswith("Generation failed after ") and (
                        "max_iterations=" in message or "max_retries=" in message
                    ):
                        return PromptResponse(stop_reason="max_turn_requests")
                    raise
                if result is None:
                    await session.cancel_complete.wait()
                    return PromptResponse(stop_reason="cancelled")
                await session.bridge.flush()
                return PromptResponse(stop_reason="end_turn")
        except SessionBusyError:
            raise RequestError.invalid_request(
                {"sessionId": session_id, "reason": "A prompt is already running"}
            ) from None

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        del kwargs
        runtime = await self._get_runtime(session_id)
        session = runtime.value
        async with session.cancel_lock:
            try:
                if await session.dispatcher.cancel():
                    await session.bridge.fail_open_tools("Cancelled by user.")
                    await session.bridge.flush()
            finally:
                session.cancel_complete.set()

    async def _create_runtime(
        self,
        handle: SessionHandle,
        root: Path,
        mcp_servers: list[Any] | None,
        *,
        llm: UnifiedLLM | None = None,
    ) -> SessionRuntime[_ACPSession]:
        llm = llm or self._llm_factory()
        if self._client is None:
            await llm.aclose()
            raise RequestError.internal_error({"reason": "ACP client is not connected"})
        agent: CodingInteractiveAgent | None = None
        value: _ACPSession | None = None
        try:
            mcp = await self._create_mcp_tools(mcp_servers)
            agent = CodingInteractiveAgent(llm=llm, cwd=root, storage=handle.storage)
            for name, tool in mcp.items():
                registry_name = f"mcp.{name}"
                agent.skills.register(registry_name, tool)
                agent.skills.activate([registry_name])
            dispatcher = InteractiveSessionDispatcher(agent)
            bridge = ACPEventBridge(agent, self._client, handle.id)
            value = _ACPSession(handle, agent, dispatcher, bridge)
            try:
                return await self._sessions.add(handle.id, value)
            except ValueError:
                raise RequestError.invalid_request(
                    {"sessionId": handle.id, "reason": "Session is already loaded"}
                ) from None
        except BaseException:
            if value is not None:
                await value.close()
            elif agent is not None:
                await agent.close()
            else:
                await llm.aclose()
            raise

    async def _create_mcp_tools(self, mcp_servers: list[Any] | None) -> dict[str, MCPTool]:
        unsupported = [
            type(server).__name__
            for server in mcp_servers or []
            if not isinstance(server, McpServerStdio)
        ]
        if unsupported:
            raise RequestError.invalid_params(
                {"reason": f"Unsupported MCP server type(s): {', '.join(unsupported)}"}
            )
        stdio_servers = [
            server for server in mcp_servers or [] if isinstance(server, McpServerStdio)
        ]
        names = [server.name for server in stdio_servers]
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicates:
            raise RequestError.invalid_params(
                {"reason": f"Duplicate MCP server name(s): {', '.join(duplicates)}"}
            )

        tools: dict[str, MCPTool] = {}
        for server in stdio_servers:
            env = {item.name: item.value for item in server.env}
            tools[server.name] = await MCPManager.create_stdio_server(
                server.name,
                command=server.command,
                args=server.args,
                env=env,
            )
        return tools

    async def _get_runtime(self, session_id: str) -> SessionRuntime[_ACPSession]:
        try:
            return await self._sessions.get(session_id)
        except KeyError:
            raise RequestError.resource_not_found(session_id) from None

    async def _replay_session(self, handle: SessionHandle) -> None:
        if self._client is None:
            return
        for turn in handle.turns():
            block = text_block(turn.content)
            update = (
                update_user_message(block) if turn.role == "user" else update_agent_message(block)
            )
            await self._client.session_update(handle.id, update)

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

    @staticmethod
    def _store(root: Path) -> SessionStore:
        return SessionStore(root / ".nooa" / "sessions")

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

    async def close(self) -> None:
        await self._sessions.close()


async def serve(llm_factory: Callable[[], UnifiedLLM]) -> None:
    adapter = CodingACPAdapter(llm_factory)
    try:
        await run_agent(cast(Agent, adapter))
    finally:
        with suppress(Exception):
            await adapter.close()
