# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP server for the NOOA coding agent."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    RequestError,
    run_agent,
)
from acp.interfaces import Agent, Client
from acp.schema import AgentCapabilities, Implementation, McpServerStdio

from nooa.errors import GenerationError
from nooa.mcp import MCPManager, MCPTool
from nooa.unifiedllm import UnifiedLLM
from nooa_acp.coding_agent import CodingInteractiveAgent
from nooa_acp.dispatcher import InteractiveSessionDispatcher
from nooa_acp.event_bridge import ACPEventBridge


class CodingACPAdapter:
    def __init__(self, llm_factory: Callable[[], UnifiedLLM]) -> None:
        self._llm_factory = llm_factory
        self._client: Client | None = None
        self._session_id: str | None = None
        self._dispatcher: InteractiveSessionDispatcher | None = None
        self._bridge: ACPEventBridge | None = None
        self._creating_session = False
        self._prompt_lock = asyncio.Lock()
        self._cancel_lock = asyncio.Lock()
        self._cancel_complete = asyncio.Event()
        self._cancel_complete.set()

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
            agent_capabilities=AgentCapabilities(),
            auth_methods=[],
            agent_info=Implementation(
                name="nooa-acp",
                title="NVIDIA OO Agents",
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
        if self._session_id is not None or self._creating_session:
            raise RequestError.invalid_request({"reason": "Only one session is supported"})
        if additional_directories:
            raise RequestError.invalid_params(
                {"reason": "Additional directories are not supported"}
            )
        unsupported_mcp = [
            type(server).__name__
            for server in mcp_servers or []
            if not isinstance(server, McpServerStdio)
        ]
        if unsupported_mcp:
            raise RequestError.invalid_params(
                {"reason": f"Unsupported MCP server type(s): {', '.join(unsupported_mcp)}"}
            )

        root = Path(cwd).expanduser()
        if not root.is_absolute() or not root.is_dir():
            raise RequestError.invalid_params(
                {"cwd": cwd, "reason": "cwd must be an existing absolute directory"}
            )
        if self._client is None:
            raise RequestError.internal_error({"reason": "ACP client is not connected"})

        session_id = str(uuid4())
        self._creating_session = True
        try:
            stdio_servers = [
                server for server in mcp_servers or [] if isinstance(server, McpServerStdio)
            ]
            server_names = [server.name for server in stdio_servers]
            duplicate_names = sorted(
                name for name in set(server_names) if server_names.count(name) > 1
            )
            if duplicate_names:
                raise RequestError.invalid_params(
                    {"reason": f"Duplicate MCP server name(s): {', '.join(duplicate_names)}"}
                )
            mcp: dict[str, MCPTool] = {}
            for server in stdio_servers:
                env = {item.name: item.value for item in server.env}
                mcp[server.name] = await MCPManager.create_stdio_server(
                    server.name,
                    command=server.command,
                    args=server.args,
                    env=env,
                )

            agent = CodingInteractiveAgent(llm=self._llm_factory(), cwd=root, mcp=mcp)
            self._session_id = session_id
            self._dispatcher = InteractiveSessionDispatcher(agent)
            self._bridge = ACPEventBridge(agent, self._client, session_id)
            return NewSessionResponse(session_id=session_id)
        finally:
            self._creating_session = False

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        del kwargs
        dispatcher, bridge = self._get_session(session_id)
        if self._prompt_lock.locked():
            raise RequestError.invalid_request({"reason": "A prompt is already running"})
        text_parts: list[str] = []
        for block in prompt:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "resource_link":
                text_parts.append(f"Resource {block.name}: {block.uri}")
            else:
                raise RequestError.invalid_params(
                    {"reason": f"Unsupported prompt content type: {block_type!r}"}
                )
        text = "\n\n".join(text_parts)
        if not text.strip():
            raise RequestError.invalid_params({"reason": "Prompt text must not be empty"})

        async with self._prompt_lock:
            self._cancel_complete.clear()
            try:
                result = await dispatcher.submit(text)
            except RuntimeError as exc:
                raise RequestError.invalid_request({"reason": str(exc)}) from None
            except GenerationError as exc:
                await bridge.flush()
                message = str(exc)
                if message.startswith("Empty response: the model used all available output tokens"):
                    return PromptResponse(stop_reason="max_tokens")
                if message.startswith("Generation failed after ") and (
                    "max_iterations=" in message or "max_retries=" in message
                ):
                    return PromptResponse(stop_reason="max_turn_requests")
                raise
            if result is None:
                await self._cancel_complete.wait()
                return PromptResponse(stop_reason="cancelled")
            await bridge.flush()
            return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        del kwargs
        dispatcher, bridge = self._get_session(session_id)
        async with self._cancel_lock:
            try:
                if await dispatcher.cancel():
                    await bridge.fail_open_tools("Cancelled by user.")
                    await bridge.flush()
            finally:
                self._cancel_complete.set()

    def _get_session(self, session_id: str) -> tuple[InteractiveSessionDispatcher, ACPEventBridge]:
        if session_id != self._session_id or self._dispatcher is None or self._bridge is None:
            raise RequestError.resource_not_found(session_id)
        return self._dispatcher, self._bridge

    async def close(self) -> None:
        if (
            self._session_id is not None
            and self._dispatcher is not None
            and self._bridge is not None
        ):
            with suppress(Exception):
                await self.cancel(self._session_id)
        self._cancel_complete.set()
        if self._bridge is not None:
            await self._bridge.close()
            self._bridge = None
        if self._dispatcher is not None:
            await self._dispatcher.close()
            self._dispatcher = None
        self._session_id = None


async def serve(llm_factory: Callable[[], UnifiedLLM]) -> None:
    adapter = CodingACPAdapter(llm_factory)
    try:
        await run_agent(cast(Agent, adapter))
    finally:
        with suppress(Exception):
            await adapter.close()
