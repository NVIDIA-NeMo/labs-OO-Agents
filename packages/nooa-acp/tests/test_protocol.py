# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end ACP JSON-RPC subprocess test."""

import asyncio
import sys
from pathlib import Path

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.schema import (
    AgentMessageChunk,
    ContentToolCallContent,
    EmbeddedResourceContentBlock,
    TextResourceContents,
    ToolCallProgress,
    ToolCallStart,
)


class _RecordingClient:
    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []
        self.tool_started = asyncio.Event()

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append((session_id, update))
        if isinstance(update, ToolCallStart):
            self.tool_started.set()


async def test_acp_subprocess_transcript(tmp_path):
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"

    async with spawn_agent_process(
        client,  # type: ignore[arg-type]
        sys.executable,
        str(fixture),
        cwd=tmp_path,
    ) as (connection, _process):
        initialized = await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        response = await connection.prompt(session.session_id, [text_block("run smoke test")])

    assert initialized.agent_info is not None
    assert initialized.agent_info.name == "nooa-acp"
    assert response.stop_reason == "end_turn"
    assert {update_session for update_session, _ in client.updates} == {session.session_id}
    started = next(update for _, update in client.updates if isinstance(update, ToolCallStart))
    assert started.content is not None
    source_content = started.content[0]
    assert isinstance(source_content, ContentToolCallContent)
    assert isinstance(source_content.content, EmbeddedResourceContentBlock)
    assert isinstance(source_content.content.resource, TextResourceContents)
    assert source_content.content.resource.mime_type == "text/x-python"
    assert "return_result" in source_content.content.resource.text

    completed = next(update for _, update in client.updates if isinstance(update, ToolCallProgress))
    assert completed.content is not None
    assert len(completed.content) == 2
    assert completed.title == "Ran Python"
    assert any(
        isinstance(update, AgentMessageChunk)
        and update.content.text == "NOOA ACP smoke test passed."
        for _, update in client.updates
    )


async def test_acp_subprocess_cancellation_finishes_open_tools(tmp_path):
    client = _RecordingClient()
    fixture = Path(__file__).parent / "fixtures" / "fake_agent.py"

    async with spawn_agent_process(
        client,  # type: ignore[arg-type]
        sys.executable,
        str(fixture),
        "--blocking",
        cwd=tmp_path,
    ) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        prompt_task = asyncio.create_task(
            connection.prompt(session.session_id, [text_block("wait forever")])
        )
        await asyncio.wait_for(client.tool_started.wait(), timeout=5)
        await connection.cancel(session.session_id)
        response = await asyncio.wait_for(prompt_task, timeout=5)

    assert response.stop_reason == "cancelled"
    started = next(update for _, update in client.updates if isinstance(update, ToolCallStart))
    failed = next(
        update
        for _, update in client.updates
        if isinstance(update, ToolCallProgress) and update.status == "failed"
    )
    assert failed.tool_call_id == started.tool_call_id
