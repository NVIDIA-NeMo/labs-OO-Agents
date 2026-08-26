# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the renderer-free ``nooa run`` runtime."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from nooa_cli.coding import CodingAgent
from nooa_cli.headless import resolve_session, run_headless

from nooa.events import LLMComplete
from nooa.interactive import RespondReason, RespondResult
from nooa.sessions import SessionStore
from nooa.unifiedllm import FakeLLMClient


def _config(
    tmp_path,
    *,
    agent_spec: str | None = None,
    mcp_servers: dict[str, dict[str, Any]] | None = None,
    mcp_auto_connect: list[str] | None = None,
):
    return SimpleNamespace(
        no_trace=True,
        agent=SimpleNamespace(working_dir=str(tmp_path)),
        tui=SimpleNamespace(
            default_model="fake/model",
            agent_spec=agent_spec,
            skills_dirs=[],
            active_skills=[],
            inactive_skills=[],
            mcp_file=tmp_path / ".mcp.json",
            mcp_servers=mcp_servers or {},
            mcp_auto_connect=mcp_auto_connect or [],
        ),
    )


def _completed_llm(message: str = "Finished **successfully**.") -> FakeLLMClient:
    return FakeLLMClient.with_tool_call(
        "execute_python",
        {
            "code": (
                f"self.message({message!r})\n"
                "return_result(RespondReason.DONE, explanation='completed and verified')"
            )
        },
    )


class _ResultAgent(CodingAgent):
    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult:
        assert notification["user_messages"]
        self.message("first")
        self.message("second")
        return RespondResult(kind=RespondReason.DONE, explanation="complete")


class _UsageAgent(CodingAgent):
    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult:
        self.event_manager.add(
            LLMComplete(
                prompt_tokens=11,
                completion_tokens=7,
                cached_tokens=3,
                reasoning_tokens=2,
                cost_usd=0.125,
            )
        )
        self.event_manager.add(LLMComplete(prompt_tokens=5, completion_tokens=4, cost_usd=0.25))
        return RespondResult(kind=RespondReason.DONE, explanation="complete")


class _CleanupFailingAgent(_ResultAgent):
    async def close(self) -> None:
        await super().close()
        raise RuntimeError("cleanup exploded")


class _CancelledAgent(CodingAgent):
    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult:
        raise asyncio.CancelledError


class _FailingAgent(CodingAgent):
    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult:
        raise RuntimeError("tool exploded")


class _BlockedAgent(CodingAgent):
    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult:
        self.message("Which branch should I use?")
        return RespondResult(kind=RespondReason.NEED_INPUT, explanation="branch required")


class _WaitingAgent(CodingAgent):
    calls = 0

    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult:
        self.calls += 1
        if self.calls == 1:
            self.queue_manager.get_channel("system_messages").put("ready")
            return RespondResult(kind=RespondReason.WAIT, explanation="waiting")
        assert notification == {"system_messages": ["ready"]}
        self.message("done after wait")
        return RespondResult(kind=RespondReason.DONE, explanation="complete")


@pytest.mark.parametrize(
    ("agent_cls", "status", "messages"),
    [
        (_ResultAgent, "done", ["first", "second"]),
        (_BlockedAgent, "blocked", ["Which branch should I use?"]),
        (_WaitingAgent, "done", ["done after wait"]),
    ],
)
async def test_run_headless_collects_messages_and_terminal_status(
    tmp_path, agent_cls, status, messages
):
    result = await run_headless(
        "do it",
        config=_config(tmp_path),
        ephemeral=True,
        llm=FakeLLMClient(),
        agent_cls=agent_cls,
    )

    assert result.session_id is None
    assert result.run_id
    assert result.status == status
    assert result.messages == messages


async def test_run_headless_persists_turns_snapshot_and_resume(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    first = await run_headless(
        "first prompt",
        config=_config(tmp_path),
        store=store,
        llm=FakeLLMClient(),
        agent_cls=_ResultAgent,
    )

    assert first.session_id is not None
    info = store.get(first.session_id)
    assert info.host == "run"
    assert [turn.content for turn in store.load_turns(first.session_id)] == [
        "first prompt",
        "first",
        "second",
    ]

    resumed = await run_headless(
        "follow up",
        config=_config(tmp_path),
        store=store,
        resume_session_id=first.session_id[:8],
        llm=FakeLLMClient(),
        agent_cls=_ResultAgent,
    )

    assert resumed.session_id == first.session_id
    assert [turn.content for turn in store.load_turns(first.session_id)][-3:] == [
        "follow up",
        "first",
        "second",
    ]


async def test_run_headless_ephemeral_creates_no_session(tmp_path):
    store = SessionStore(tmp_path / "sessions")

    await run_headless(
        "temporary",
        config=_config(tmp_path),
        ephemeral=True,
        store=store,
        llm=FakeLLMClient(),
        agent_cls=_ResultAgent,
    )

    assert store.list() == []


def test_resolve_session_continue_is_scoped_to_workspace(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    other = store.create(working_directory=str(tmp_path / "other"), host="run")
    other.record_user_message("other")
    other.close()
    wanted = store.create(working_directory=str(tmp_path / "wanted"), host="run")
    wanted.record_user_message("wanted")
    wanted_id = wanted.id
    wanted.close()

    handle, resumed = resolve_session(
        store,
        working_directory=tmp_path / "wanted",
        model="fake/model",
        agent_name="CodingAgent",
        continue_session=True,
    )
    try:
        assert resumed is True
        assert handle.id == wanted_id
    finally:
        handle.close()


def test_resolve_session_rejects_missing_resume_prefix(tmp_path):
    prefix = "missing"
    store = SessionStore(tmp_path / "sessions")

    with pytest.raises(ValueError, match="was not found"):
        resolve_session(
            store,
            working_directory=tmp_path,
            model="fake/model",
            agent_name="CodingAgent",
            resume_session_id=prefix,
        )


async def test_run_headless_executes_real_coding_agent_with_fake_llm(tmp_path):
    result = await run_headless(
        "inspect the repository",
        config=_config(tmp_path),
        ephemeral=True,
        llm=_completed_llm(),
    )

    assert result.status == "done"
    assert result.messages == ["Finished **successfully**."]
    assert result.explanation == "completed and verified"


async def test_run_headless_streams_ordered_terminal_events(tmp_path):
    events = []

    result = await run_headless(
        "do it",
        config=_config(tmp_path),
        ephemeral=True,
        llm=FakeLLMClient(),
        agent_cls=_ResultAgent,
        on_event=events.append,
    )

    assert result.status == "done"
    assert [event["type"] for event in events] == [
        "session.started",
        "turn.started",
        "agent.message",
        "agent.message",
        "turn.completed",
    ]
    assert {event["session_id"] for event in events} == {None}
    assert len({event["run_id"] for event in events}) == 1
    assert all(event["schema_version"] == 1 for event in events)
    assert all("timestamp" in event for event in events)


async def test_run_headless_aggregates_usage(tmp_path):
    events = []
    result = await run_headless(
        "measure it",
        config=_config(tmp_path),
        ephemeral=True,
        llm=FakeLLMClient(),
        agent_cls=_UsageAgent,
        on_event=events.append,
    )

    assert result.usage.prompt_tokens == 16
    assert result.usage.completion_tokens == 11
    assert result.usage.cached_tokens == 3
    assert result.usage.reasoning_tokens == 2
    assert result.usage.cost_usd == pytest.approx(0.375)
    assert [event["type"] for event in events].count("usage.updated") == 2


async def test_run_headless_blocks_unapproved_startup_mcp_without_connecting(tmp_path):
    result = await run_headless(
        "use the server",
        config=_config(
            tmp_path,
            mcp_servers={"local": {"command": "printf", "args": ["hello"]}},
            mcp_auto_connect=["local"],
        ),
        ephemeral=True,
        llm=FakeLLMClient(),
        agent_cls=_ResultAgent,
    )

    assert result.status == "blocked"
    assert result.explanation == "MCP approval required"
    assert "Approval required for MCP server 'local'" in result.messages[0]


async def test_run_headless_returns_correlated_failure_event(tmp_path):
    events = []
    result = await run_headless(
        "break it",
        config=_config(tmp_path),
        ephemeral=True,
        llm=FakeLLMClient(),
        agent_cls=_FailingAgent,
        on_event=events.append,
    )

    assert result.status == "failed"
    assert result.error == {"type": "RuntimeError", "message": "tool exploded"}
    assert events[-1]["type"] == "turn.failed"
    assert events[-1]["session_id"] is None
    assert events[-1]["run_id"] == result.run_id


async def test_run_headless_persists_blocked_turn(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    result = await run_headless(
        "choose a branch",
        config=_config(tmp_path),
        store=store,
        llm=FakeLLMClient(),
        agent_cls=_BlockedAgent,
    )

    assert result.status == "blocked"
    assert result.session_id is not None
    assert [turn.content for turn in store.load_turns(result.session_id)] == [
        "choose a branch",
        "Which branch should I use?",
    ]


def test_resolve_session_rejects_ambiguous_resume_prefix(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    for session_id in (
        "aaaaaaaa-0000-0000-0000-000000000001",
        "aaaaaaaa-0000-0000-0000-000000000002",
    ):
        handle = store.create(session_id=session_id)
        handle.close()

    with pytest.raises(ValueError, match="is ambiguous"):
        resolve_session(
            store,
            working_directory=tmp_path,
            model="fake/model",
            agent_name="CodingAgent",
            resume_session_id="aaaaaaaa",
        )


async def test_run_headless_emits_one_failed_terminal_after_cleanup_error(tmp_path):
    events = []
    result = await run_headless(
        "do it",
        config=_config(tmp_path),
        ephemeral=True,
        llm=FakeLLMClient(),
        agent_cls=_CleanupFailingAgent,
        on_event=events.append,
    )

    terminal = [
        event
        for event in events
        if event["type"].startswith("turn.") and event["type"] != "turn.started"
    ]
    assert result.status == "failed"
    assert result.explanation == "Resource cleanup failed: cleanup exploded"
    assert [event["type"] for event in terminal] == ["turn.failed"]


async def test_run_headless_emits_cancelled_terminal_and_propagates_cancellation(tmp_path):
    events = []

    with pytest.raises(asyncio.CancelledError):
        await run_headless(
            "cancel it",
            config=_config(tmp_path),
            ephemeral=True,
            llm=FakeLLMClient(),
            agent_cls=_CancelledAgent,
            on_event=events.append,
        )

    terminal = [
        event
        for event in events
        if event["type"].startswith("turn.") and event["type"] != "turn.started"
    ]
    assert [event["type"] for event in terminal] == ["turn.cancelled"]


async def test_run_headless_uses_run_id_for_ephemeral_trace_correlation(tmp_path):
    trace_sessions = []
    with patch(
        "nooa_cli.tui.bootstrap._enable_tracing",
        return_value=(True, trace_sessions.append),
    ):
        result = await run_headless(
            "trace it",
            config=_config(tmp_path),
            ephemeral=True,
            llm=FakeLLMClient(),
            agent_cls=_ResultAgent,
        )

    assert result.session_id is None
    assert len(trace_sessions) == 1
    assert trace_sessions[0].endswith(result.run_id[:8])
