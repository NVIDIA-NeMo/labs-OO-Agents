# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the scripted NOOA demo over real ACP stdio; no live LLM or Aion renderer."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.interfaces import Client
from acp.schema import (
    AgentMessageChunk,
    FileEditToolCallContent,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
)

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(__file__).with_name("launch.sh")
TIMEOUT = 45


class RecordingClient:
    def __init__(self) -> None:
        self.updates: list[Any] = []
        self.tool_started = asyncio.Event()
        self.updated = asyncio.Event()

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        del session_id, kwargs
        self.updates.append(update)
        self.updated.set()
        if isinstance(update, ToolCallStart):
            self.tool_started.set()

    async def wait_for_updates(self, predicate: Callable[[], bool]) -> None:
        """Wait for the SDK's notification worker to finish the expected updates."""
        async with asyncio.timeout(5):
            while not predicate():
                self.updated.clear()
                await self.updated.wait()


async def check_turn(client: RecordingClient, before: int) -> dict[str, int]:
    await client.wait_for_updates(
        lambda: (
            sum(isinstance(update, AgentMessageChunk) for update in client.updates[before:]) >= 2
            and sum(
                isinstance(update, ToolCallProgress) and update.status == "completed"
                for update in client.updates[before:]
            )
            >= 2
        )
    )
    updates = client.updates[before:]
    starts = [update for update in updates if isinstance(update, ToolCallStart)]
    kinds = Counter(update.kind for update in starts)
    assert kinds == {"other": 1, "edit": 1, "execute": 1}, kinds
    edit = next(update for update in starts if update.kind == "edit")
    assert edit.status == "completed"
    assert any(isinstance(content, FileEditToolCallContent) for content in edit.content or [])
    progress = [update for update in updates if isinstance(update, ToolCallProgress)]
    for start in starts:
        if start.kind != "edit":
            assert any(
                update.tool_call_id == start.tool_call_id and update.status == "completed"
                for update in progress
            )
    assert not any(update.status == "failed" for update in progress)
    messages = [update for update in updates if isinstance(update, AgentMessageChunk)]
    assert len(messages) == 2, messages
    assert "Verified:" in messages[-1].content.text
    return {
        "message_chunks": len(messages),
        "tool_cards": len(starts),
        "tool_updates": len(progress),
    }


async def main() -> None:
    started = time.monotonic()
    output_root = ROOT / "tmp" / "aion-ui-spike"
    run_dir = output_root / f"run-{uuid4().hex[:12]}"
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    env = {
        "NEMO_OO_USER_DIR": str(run_dir / "user-config"),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "TMPDIR": str(run_dir),
    }
    client = RecordingClient()
    turn_counts: list[dict[str, int]] = []
    async with asyncio.timeout(TIMEOUT):
        async with spawn_agent_process(
            cast(Client, client),
            str(LAUNCHER),
            "--demo",
            env=env,
            cwd=workspace,
            transport_kwargs={"stderr": None},
        ) as (connection, _process):
            initialized = await connection.initialize(PROTOCOL_VERSION)
            assert initialized.agent_info is not None and initialized.agent_info.name == "nooa-acp"
            assert initialized.agent_capabilities.load_session
            session = await connection.new_session(str(workspace))
            for text in ("Run the scripted demo.", "Run it a second time."):
                before = len(client.updates)
                response = await connection.prompt(session.session_id, [text_block(text)])
                assert response.stop_reason == "end_turn", response
                turn_counts.append(await check_turn(client, before))
            artifact = workspace / "nooa_aion_demo.py"
            assert artifact.read_text().startswith("# NOOA AionUi scripted spike artifact\n")
            assert "DEMO_TURN = 2" in artifact.read_text()
            listed = await connection.list_sessions(cwd=str(workspace))
            assert session.session_id not in {item.session_id for item in listed.sessions}
            await connection.close_session(session.session_id)
            listed = await connection.list_sessions(cwd=str(workspace))
            assert session.session_id in {item.session_id for item in listed.sessions}

    replay_client = RecordingClient()
    async with asyncio.timeout(TIMEOUT):
        async with spawn_agent_process(
            cast(Client, replay_client),
            str(LAUNCHER),
            "--demo",
            env=env,
            cwd=workspace,
            transport_kwargs={"stderr": None},
        ) as (connection, _process):
            await connection.initialize(PROTOCOL_VERSION)
            await connection.load_session(cwd=str(workspace), session_id=session.session_id)
            await replay_client.wait_for_updates(
                lambda: (
                    sum(isinstance(update, AgentMessageChunk) for update in replay_client.updates)
                    >= 4
                )
            )
            replay_users = sum(
                isinstance(update, UserMessageChunk) for update in replay_client.updates
            )
            replay_agents = sum(
                isinstance(update, AgentMessageChunk) for update in replay_client.updates
            )
            assert replay_users == 2 and replay_agents == 4, (replay_users, replay_agents)
            before = len(replay_client.updates)
            response = await connection.prompt(
                session.session_id, [text_block("Continue after restart.")]
            )
            assert response.stop_reason == "end_turn", response
            turn_counts.append(await check_turn(replay_client, before))
            assert "DEMO_TURN = 1" in artifact.read_text()

    cancel_workspace = run_dir / "cancel-workspace"
    cancel_workspace.mkdir()
    cancel_client = RecordingClient()
    async with asyncio.timeout(TIMEOUT):
        async with spawn_agent_process(
            cast(Client, cancel_client),
            str(LAUNCHER),
            "--demo",
            "--blocking",
            env=env,
            cwd=cancel_workspace,
            transport_kwargs={"stderr": None},
        ) as (connection, _process):
            await connection.initialize(PROTOCOL_VERSION)
            cancel_session = await connection.new_session(str(cancel_workspace))
            pending = asyncio.create_task(
                connection.prompt(cancel_session.session_id, [text_block("Wait.")])
            )
            try:
                await cancel_client.tool_started.wait()
                cancel_started = time.monotonic()
                await connection.cancel(cancel_session.session_id)
                response = await pending
                cancel_seconds = time.monotonic() - cancel_started
            finally:
                if not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
            assert response.stop_reason == "cancelled", response
            assert not (cancel_workspace / "nooa_aion_demo.py").exists()
            await cancel_client.wait_for_updates(
                lambda: any(
                    isinstance(update, ToolCallProgress) and update.status == "failed"
                    for update in cancel_client.updates
                )
            )
            before = len(cancel_client.updates)
            response = await connection.prompt(
                cancel_session.session_id, [text_block("Run after cancellation.")]
            )
            assert response.stop_reason == "end_turn", response
            turn_counts.append(await check_turn(cancel_client, before))
            assert "DEMO_TURN = 2" in (cancel_workspace / "nooa_aion_demo.py").read_text()

    summary = {
        "result": "passed",
        "scope": "Real NOOA ExperimentalCodingAgent and tools over ACP stdio; scripted provider; Aion renderer not exercised",
        "live_llm_calls": 0,
        "completed_turns": len(turn_counts),
        "turn_activity": turn_counts,
        "process_starts": 3,
        "replayed_user_messages": replay_users,
        "replayed_agent_messages": replay_agents,
        "cancelled_turns": 1,
        "cancel_seconds": round(cancel_seconds, 4),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "run_directory": str(run_dir),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
