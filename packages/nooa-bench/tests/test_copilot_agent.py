# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Copilot SDK-backed benchmark agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from copilot import ToolInvocation
from copilot.session_events import (
    AssistantMessageData,
    AssistantUsageData,
    SessionErrorData,
    SessionIdleData,
)
from nooa_bench import copilot_agent as copilot_agent_module
from nooa_bench.copilot_agent import CopilotBenchAgent, TaskResult
from pydantic import ValidationError


class _CallReturnTool:
    def __init__(self, arguments: dict[str, Any]) -> None:
        self.arguments = arguments


def _model(
    model_id: str = "gpt-5.6-sol",
    *,
    supports_reasoning: bool = True,
    supported_reasoning_efforts: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=model_id,
        capabilities=SimpleNamespace(
            supports=SimpleNamespace(reasoning_effort=supports_reasoning)
        ),
        supported_reasoning_efforts=supported_reasoning_efforts
        or ["low", "medium", "high", "xhigh"],
    )


class _FakeSession:
    def __init__(
        self,
        events: list[Any],
        *,
        hang_send: bool = False,
        send_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.hang_send = hang_send
        self.send_error = send_error
        self.handlers: list[Any] = []
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.disconnected = False
        self.unsubscribed = False
        self.tools: list[Any] = []

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.disconnect()

    async def disconnect(self) -> None:
        self.disconnected = True

    def on(self, handler: Any) -> Any:
        self.handlers.append(handler)

        def unsubscribe() -> None:
            self.unsubscribed = True

        return unsubscribe

    async def send(self, prompt: str, **kwargs: Any) -> str:
        if self.hang_send:
            await asyncio.Event().wait()
        if self.send_error is not None:
            raise self.send_error
        self.sent.append((prompt, kwargs))
        for data in self.events:
            if isinstance(data, _CallReturnTool):
                tool = next(tool for tool in self.tools if tool.name == "return_task_result")
                await tool.handler(ToolInvocation(arguments=data.arguments))
                continue
            for handler in list(self.handlers):
                handler(SimpleNamespace(data=data))
        return "message-id"


class _FakeClient:
    def __init__(
        self,
        session: _FakeSession,
        models: list[Any],
        list_models_error: Exception | None,
        create_session_error: Exception | None,
        hang_start: bool,
        hang_stop: bool,
        **kwargs: Any,
    ) -> None:
        self.session = session
        self.models = models
        self.list_models_error = list_models_error
        self.create_session_error = create_session_error
        self.hang_start = hang_start
        self.hang_stop = hang_stop
        self.kwargs = kwargs
        self.create_session_kwargs: dict[str, Any] | None = None
        self.started = False
        self.stopped = False
        self.force_stopped = False
        self.list_models_calls = 0

    async def __aenter__(self) -> _FakeClient:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.stop()

    async def start(self) -> None:
        if self.hang_start:
            await asyncio.Event().wait()
        self.started = True

    async def stop(self) -> None:
        if self.hang_stop:
            await asyncio.Event().wait()
        self.stopped = True

    async def force_stop(self) -> None:
        self.force_stopped = True

    async def list_models(self) -> list[Any]:
        self.list_models_calls += 1
        if self.list_models_error is not None:
            raise self.list_models_error
        return self.models

    async def create_session(self, **kwargs: Any) -> _FakeSession:
        if self.create_session_error is not None:
            raise self.create_session_error
        self.create_session_kwargs = kwargs
        self.session.tools = kwargs["tools"]
        return self.session


class _ClientFactory:
    def __init__(
        self,
        events: list[Any],
        models: list[Any] | None = None,
        *,
        hang_send: bool = False,
        send_error: Exception | None = None,
        list_models_error: Exception | None = None,
        create_session_error: Exception | None = None,
        hang_start: bool = False,
        hang_stop: bool = False,
    ) -> None:
        self.session = _FakeSession(events, hang_send=hang_send, send_error=send_error)
        self.models = models or [_model()]
        self.list_models_error = list_models_error
        self.create_session_error = create_session_error
        self.hang_start = hang_start
        self.hang_stop = hang_stop
        self.clients: list[_FakeClient] = []

    def __call__(self, **kwargs: Any) -> _FakeClient:
        client = _FakeClient(
            self.session,
            self.models,
            self.list_models_error,
            self.create_session_error,
            self.hang_start,
            self.hang_stop,
            **kwargs,
        )
        self.clients.append(client)
        return client


def test_task_result_rejects_blank_fields_and_extras():
    with pytest.raises(ValidationError, match="non-blank"):
        TaskResult(
            solution_description="fixed it",
            evidence="   ",
            command_to_verify="pytest -q",
        )

    with pytest.raises(ValidationError, match="Extra inputs"):
        TaskResult(
            solution_description="fixed it",
            evidence="pytest passed",
            command_to_verify="pytest -q",
            extra="not allowed",
        )

    with pytest.raises(ValidationError, match="string_type"):
        TaskResult(
            solution_description=123,
            evidence="pytest passed",
            command_to_verify="pytest -q",
        )


@pytest.mark.asyncio
async def test_copilot_agent_returns_structured_result_and_usage(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    factory = _ClientFactory(
        [
            AssistantUsageData(
                model="gpt-5.6-sol",
                api_call_id="call-1",
                input_tokens=11,
                output_tokens=3,
            ),
            AssistantUsageData(
                model="gpt-5.6-sol",
                api_call_id="call-2",
                input_tokens=7,
                output_tokens=5,
            ),
            AssistantUsageData(
                model="gpt-5.6-sol",
                api_call_id="call-1",
                input_tokens=1000,
                output_tokens=1000,
            ),
            SimpleNamespace(input_tokens=1000, output_tokens=1000),
            AssistantMessageData(
                content="I verified the task and will return the structured result.",
                message_id="m1",
                output_tokens=1000,
            ),
            _CallReturnTool(
                {
                    "solution_description": "fixed it",
                    "evidence": "pytest -q passed",
                    "command_to_verify": "pytest -q",
                }
            ),
            SessionIdleData(),
        ]
    )
    agent = CopilotBenchAgent(
        model="gpt-5.6-sol",
        reasoning_effort="xhigh",
        context_tier="long_context",
        timeout_seconds=1,
        client_factory=factory,
    )

    result = await agent._run_evaluation(
        {"user_message": "fix the bug", "working_dir": str(Path.cwd())}
    )

    assert result["success"] is True
    assert result["response"] == "pytest -q"
    assert result["result"] == {
        "solution_description": "fixed it",
        "evidence": "pytest -q passed",
        "command_to_verify": "pytest -q",
    }
    assert result["n_input_tokens"] == 18
    assert result["n_output_tokens"] == 8

    client = factory.clients[-1]
    assert client.list_models_calls == 1
    assert client.kwargs == {"working_directory": str(Path.cwd()), "use_logged_in_user": True}
    assert client.create_session_kwargs is not None
    assert client.create_session_kwargs["model"] == "gpt-5.6-sol"
    assert client.create_session_kwargs["reasoning_effort"] == "xhigh"
    assert client.create_session_kwargs["context_tier"] == "long_context"
    assert client.create_session_kwargs["system_message"]["mode"] == "append"
    assert client.create_session_kwargs["on_permission_request"] is not None
    assert [tool.name for tool in client.create_session_kwargs["tools"]] == ["return_task_result"]
    assert factory.session.sent[-1][1] == {"agent_mode": "autopilot"}
    assert factory.session.disconnected is True
    assert client.stopped is True


@pytest.mark.asyncio
async def test_copilot_agent_reports_idle_without_return_tool(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    factory = _ClientFactory(
        [AssistantMessageData(content='{"solution_description":"text only"}', message_id="m1"), SessionIdleData()]
    )
    agent = CopilotBenchAgent(timeout_seconds=1, client_factory=factory)

    result = await agent._run_evaluation(
        {"problem_statement": "fix the bug", "working_dir": str(Path.cwd())}
    )

    assert result["success"] is False
    assert result["response"] == '{"solution_description":"text only"}'
    assert "return_task_result" in result["error"]
    assert result["n_input_tokens"] == 0
    assert result["n_output_tokens"] == 0


@pytest.mark.asyncio
async def test_copilot_agent_reports_session_error(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    factory = _ClientFactory(
        [
            AssistantUsageData(
                model="gpt-5.6-sol",
                api_call_id="call-1",
                input_tokens=4,
                output_tokens=1,
            ),
            SessionErrorData(error_type="provider", message="provider failed"),
        ]
    )
    agent = CopilotBenchAgent(timeout_seconds=1, client_factory=factory)

    result = await agent._run_evaluation(
        {"task_description": "fix the bug", "working_dir": str(Path.cwd())}
    )

    assert result == {
        "response": "",
        "success": False,
        "error": "provider failed",
        "n_input_tokens": 4,
        "n_output_tokens": 1,
    }
    assert factory.session.disconnected is True
    assert factory.clients[-1].stopped is True


@pytest.mark.asyncio
async def test_copilot_agent_reports_timeout_and_cleans_up(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    factory = _ClientFactory([])
    agent = CopilotBenchAgent(timeout_seconds=0.01, client_factory=factory)

    result = await agent._run_evaluation(
        {"user_message": "fix the bug", "working_dir": str(Path.cwd())}
    )

    assert result["success"] is False
    assert "timed out" in result["error"]
    assert factory.session.disconnected is True
    assert factory.clients[-1].stopped is True


@pytest.mark.asyncio
async def test_copilot_agent_times_out_hanging_send_and_cleans_up(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    factory = _ClientFactory([], hang_send=True)
    agent = CopilotBenchAgent(timeout_seconds=0.01, client_factory=factory)

    result = await agent._run_evaluation(
        {"user_message": "fix the bug", "working_dir": str(Path.cwd())}
    )

    assert result == {
        "response": "",
        "success": False,
        "error": "Copilot session timed out after 0.01 seconds",
        "n_input_tokens": 0,
        "n_output_tokens": 0,
    }
    assert factory.session.disconnected is True
    assert factory.clients[-1].stopped is True


@pytest.mark.asyncio
async def test_copilot_agent_timeout_preserves_partial_usage(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    factory = _ClientFactory(
        [
            AssistantUsageData(
                model="gpt-5.6-sol",
                api_call_id="call-before-stall",
                input_tokens=13,
                output_tokens=8,
            )
        ]
    )
    agent = CopilotBenchAgent(timeout_seconds=0.01, client_factory=factory)

    result = await agent._run_evaluation(
        {"user_message": "fix the bug", "working_dir": str(Path.cwd())}
    )

    assert result == {
        "response": "",
        "success": False,
        "error": "Copilot session timed out after 0.01 seconds",
        "n_input_tokens": 13,
        "n_output_tokens": 8,
    }
    assert factory.session.disconnected is True
    assert factory.clients[-1].stopped is True


@pytest.mark.asyncio
async def test_copilot_agent_timeout_during_start_stops_client(monkeypatch):
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    factory = _ClientFactory([], hang_start=True)
    agent = CopilotBenchAgent(timeout_seconds=0.01, client_factory=factory)

    result = await agent._run_evaluation(
        {"user_message": "fix the bug", "working_dir": str(Path.cwd())}
    )

    assert result == {
        "response": "",
        "success": False,
        "error": "Copilot session timed out after 0.01 seconds",
        "n_input_tokens": 0,
        "n_output_tokens": 0,
    }
    assert factory.clients[-1].started is False
    assert factory.clients[-1].stopped is True
    assert factory.session.disconnected is False


@pytest.mark.asyncio
async def test_copilot_agent_force_stops_client_when_stop_hangs(monkeypatch):
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(copilot_agent_module, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    factory = _ClientFactory(
        [
            _CallReturnTool(
                {
                    "solution_description": "fixed it",
                    "evidence": "pytest passed",
                    "command_to_verify": "pytest -q",
                }
            ),
            SessionIdleData(),
        ],
        hang_stop=True,
    )
    agent = CopilotBenchAgent(timeout_seconds=1, client_factory=factory)

    result = await agent._run_evaluation(
        {"user_message": "fix the bug", "working_dir": str(Path.cwd())}
    )

    assert result["success"] is True
    assert factory.session.disconnected is True
    assert factory.clients[-1].stopped is False
    assert factory.clients[-1].force_stopped is True


@pytest.mark.asyncio
async def test_copilot_agent_returns_failure_for_send_exception(monkeypatch):
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    factory = _ClientFactory([], send_error=RuntimeError("send failed"))
    agent = CopilotBenchAgent(timeout_seconds=1, client_factory=factory)

    result = await agent._run_evaluation(
        {"user_message": "fix the bug", "working_dir": str(Path.cwd())}
    )

    assert result == {
        "response": "",
        "success": False,
        "error": "send failed",
        "n_input_tokens": 0,
        "n_output_tokens": 0,
    }
    assert factory.session.disconnected is True
    assert factory.clients[-1].stopped is True


@pytest.mark.asyncio
async def test_copilot_agent_returns_failure_for_create_session_exception(monkeypatch):
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    factory = _ClientFactory([], create_session_error=RuntimeError("session failed"))
    agent = CopilotBenchAgent(timeout_seconds=1, client_factory=factory)

    result = await agent._run_evaluation(
        {"user_message": "fix the bug", "working_dir": str(Path.cwd())}
    )

    assert result == {
        "response": "",
        "success": False,
        "error": "session failed",
        "n_input_tokens": 0,
        "n_output_tokens": 0,
    }
    assert factory.clients[-1].stopped is True


@pytest.mark.asyncio
async def test_copilot_agent_uses_explicit_container_token(monkeypatch):
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "token-value")
    factory = _ClientFactory(
        [
            _CallReturnTool(
                {
                    "solution_description": "fixed it",
                    "evidence": "pytest passed",
                    "command_to_verify": "pytest -q",
                }
            ),
            SessionIdleData(),
        ]
    )
    agent = CopilotBenchAgent(timeout_seconds=1, client_factory=factory)

    result = await agent._run_evaluation(
        {"user_message": "fix the bug", "working_dir": str(Path.cwd())}
    )

    assert result["success"] is True
    assert factory.clients[-1].kwargs == {
        "working_directory": str(Path.cwd()),
        "github_token": "token-value",
        "use_logged_in_user": False,
    }


@pytest.mark.asyncio
async def test_copilot_agent_fails_when_model_unavailable(monkeypatch):
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    factory = _ClientFactory([], models=[_model("gpt-5")])
    agent = CopilotBenchAgent(model="gpt-5.6-sol", timeout_seconds=1, client_factory=factory)

    result = await agent._run_evaluation(
        {"user_message": "fix the bug", "working_dir": str(Path.cwd())}
    )

    assert result["success"] is False
    assert "not available" in result["error"]
    assert factory.clients[-1].create_session_kwargs is None


@pytest.mark.asyncio
async def test_copilot_agent_returns_failure_for_list_models_exception(monkeypatch):
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    factory = _ClientFactory([], list_models_error=RuntimeError("not authenticated"))
    agent = CopilotBenchAgent(timeout_seconds=1, client_factory=factory)

    result = await agent._run_evaluation(
        {"user_message": "fix the bug", "working_dir": str(Path.cwd())}
    )

    assert result == {
        "response": "",
        "success": False,
        "error": "not authenticated",
        "n_input_tokens": 0,
        "n_output_tokens": 0,
    }
    assert factory.clients[-1].stopped is True


@pytest.mark.asyncio
async def test_copilot_agent_fails_for_unsupported_reasoning_effort(monkeypatch):
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    factory = _ClientFactory(
        [],
        models=[
            _model(
                "gpt-5.6-sol",
                supports_reasoning=True,
                supported_reasoning_efforts=["low", "medium", "high"],
            )
        ],
    )
    agent = CopilotBenchAgent(reasoning_effort="xhigh", timeout_seconds=1, client_factory=factory)

    result = await agent._run_evaluation(
        {"user_message": "fix the bug", "working_dir": str(Path.cwd())}
    )

    assert result["success"] is False
    assert "does not support reasoning effort" in result["error"]
    assert factory.clients[-1].create_session_kwargs is None
