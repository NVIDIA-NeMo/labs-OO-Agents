# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Harbor runner dispatch paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from nooa_bench import runner

import nooa.runtime.token_usage as token_usage
import nooa.unifiedllm as unifiedllm


@pytest.mark.asyncio
async def test_runner_copilot_dispatch_does_not_construct_litellm(monkeypatch):
    constructed: dict[str, Any] = {}

    class FakeCopilotAgent:
        def __init__(self, **kwargs: Any) -> None:
            constructed.update(kwargs)

        async def _run_evaluation(self, task_input: dict[str, Any]) -> dict[str, Any]:
            assert task_input == {"user_message": "fix it", "working_dir": str(Path.cwd())}
            return {
                "response": "pytest -q",
                "success": True,
                "n_input_tokens": 2,
                "n_output_tokens": 3,
            }

    written: dict[str, Any] = {}

    monkeypatch.setattr(runner, "_import_agent_class", lambda agent_type: FakeCopilotAgent)
    monkeypatch.setattr(
        runner, "_write_result", lambda result, model, agent_type: written.update(result)
    )
    monkeypatch.setattr(runner, "_write_answer", lambda result: None)

    exit_code = await runner._run(
        "fix it",
        "gpt-5.6-sol",
        "copilot",
        api_base=None,
        working_dir=str(Path.cwd()),
        reasoning_effort="xhigh",
        context_tier="long_context",
        timeout_seconds=10,
    )

    assert exit_code == 0
    assert constructed == {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "xhigh",
        "context_tier": "long_context",
        "timeout_seconds": 10,
    }
    assert written["n_input_tokens"] == 2
    assert written["n_output_tokens"] == 3


@pytest.mark.asyncio
async def test_runner_legacy_dispatch_preserves_token_accounting(monkeypatch):
    started = False
    llm_client = object()
    written: dict[str, Any] = {}

    class FakeBenchAgent:
        def __init__(self, *, llm: Any) -> None:
            assert llm is llm_client

        async def _run_evaluation(self, task_input: dict[str, Any]) -> dict[str, Any]:
            assert task_input == {"user_message": "fix it", "working_dir": str(Path.cwd())}
            return {"response": "pytest -q", "success": True}

    def start_task_tokens() -> None:
        nonlocal started
        started = True

    monkeypatch.setattr(runner, "_import_agent_class", lambda agent_type: FakeBenchAgent)
    monkeypatch.setattr(unifiedllm, "get_llm_client", lambda model, **kwargs: llm_client)
    monkeypatch.setattr(token_usage, "start_task_tokens", start_task_tokens)
    monkeypatch.setattr(
        token_usage,
        "get_task_tokens",
        lambda: {"n_input_tokens": 11, "n_output_tokens": 7},
    )
    monkeypatch.setattr(
        runner, "_write_result", lambda result, model, agent_type: written.update(result)
    )
    monkeypatch.setattr(runner, "_write_answer", lambda result: None)

    exit_code = await runner._run(
        "fix it",
        "openai/test-model",
        "bench",
        api_base=None,
        working_dir=str(Path.cwd()),
    )

    assert exit_code == 0
    assert started is True
    assert written["n_input_tokens"] == 11
    assert written["n_output_tokens"] == 7


@pytest.mark.asyncio
async def test_runner_writes_failure_result_when_agent_raises(monkeypatch):
    class BrokenAgent:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def _run_evaluation(self, task_input: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("sdk exploded")

    written: dict[str, Any] = {}

    monkeypatch.setattr(runner, "_import_agent_class", lambda agent_type: BrokenAgent)
    monkeypatch.setattr(
        runner, "_write_result", lambda result, model, agent_type: written.update(result)
    )
    monkeypatch.setattr(runner, "_write_answer", lambda result: None)

    exit_code = await runner._run(
        "fix it",
        "gpt-5.6-sol",
        "copilot",
        api_base=None,
        working_dir=str(Path.cwd()),
    )

    assert exit_code == 1
    assert written == {"response": "", "success": False, "error": "sdk exploded"}


@pytest.mark.asyncio
async def test_runner_writes_failure_result_when_agent_returns_none(monkeypatch):
    class EmptyAgent:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def _run_evaluation(self, task_input: dict[str, Any]) -> None:
            return None

    written: dict[str, Any] = {}

    monkeypatch.setattr(runner, "_import_agent_class", lambda agent_type: EmptyAgent)
    monkeypatch.setattr(
        runner, "_write_result", lambda result, model, agent_type: written.update(result)
    )
    monkeypatch.setattr(runner, "_write_answer", lambda result: None)

    exit_code = await runner._run(
        "fix it",
        "gpt-5.6-sol",
        "copilot",
        api_base=None,
        working_dir=str(Path.cwd()),
    )

    assert exit_code == 1
    assert written == {
        "response": "",
        "success": False,
        "error": "Agent returned no result",
    }


@pytest.mark.asyncio
async def test_runner_copilot_rejects_api_base_before_agent_construction(monkeypatch):
    def fail_import(agent_type: str) -> Any:
        raise AssertionError("agent should not be imported when --api-base is rejected")

    written: dict[str, Any] = {}

    monkeypatch.setattr(runner, "_import_agent_class", fail_import)
    monkeypatch.setattr(
        runner, "_write_result", lambda result, model, agent_type: written.update(result)
    )
    monkeypatch.setattr(runner, "_write_answer", lambda result: None)

    exit_code = await runner._run(
        "fix it",
        "gpt-5.6-sol",
        "copilot",
        api_base="https://example.test/v1",
        working_dir=str(Path.cwd()),
        reasoning_effort="xhigh",
        context_tier="long_context",
        timeout_seconds=10,
    )

    assert exit_code == 1
    assert written == {
        "response": "",
        "success": False,
        "error": (
            "--api-base is not supported for --agent-type copilot; "
            "BYOK provider wiring is not implemented"
        ),
    }


@pytest.mark.asyncio
async def test_runner_writes_failure_result_when_copilot_setup_fails(monkeypatch):
    def fail_import(agent_type: str) -> Any:
        raise RuntimeError("copilot import failed")

    written: dict[str, Any] = {}

    monkeypatch.setattr(runner, "_import_agent_class", fail_import)
    monkeypatch.setattr(
        runner, "_write_result", lambda result, model, agent_type: written.update(result)
    )
    monkeypatch.setattr(runner, "_write_answer", lambda result: None)

    exit_code = await runner._run(
        "fix it",
        "gpt-5.6-sol",
        "copilot",
        api_base=None,
        working_dir=str(Path.cwd()),
    )

    assert exit_code == 1
    assert written == {
        "response": "",
        "success": False,
        "error": "copilot import failed",
    }
