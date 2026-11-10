# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the CyberGym agent entry point."""

from __future__ import annotations

import pytest

pytest.importorskip("nooa")

from examples.cybergym.nooa_cybergym import main as nooa_cybergym_main


def test_cli_default_comes_from_agent_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--prompt", "test"])

    args = nooa_cybergym_main._parse_args()

    assert args.model == "glm-5.2"
    assert args.model == nooa_cybergym_main.DEFAULT_MODEL_NAME


def test_llm_client_kwargs_uses_gateway_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("NVIDIA_INTERNAL_API_KEY", "test-key")

    kwargs = nooa_cybergym_main._llm_client_kwargs(max_output_tokens=32768)

    assert kwargs["api_key"] == "test-key"
    assert kwargs["api_base"] == nooa_cybergym_main.DEFAULT_API_BASE
    assert kwargs["max_tokens"] == 32768
    assert "reasoning" not in kwargs
    assert "reasoning_effort" not in kwargs


def test_reasoning_effort_uses_responses_shape_from_registry_config():
    class FakeResponsesLLM:
        config = {}
        _registry_config = {"client_type": "responses"}

    llm = FakeResponsesLLM()

    nooa_cybergym_main._apply_reasoning_effort(llm, "xhigh")

    assert llm.config["reasoning"] == {"effort": "xhigh"}
    assert "reasoning_effort" not in llm.config


def test_shutdown_tracing_with_timeout_returns_when_shutdown_stalls(monkeypatch):
    import threading
    import time

    started = threading.Event()

    def slow_shutdown():
        started.set()
        time.sleep(1)

    monkeypatch.setattr(nooa_cybergym_main, "shutdown_tracing", slow_shutdown)

    before = time.monotonic()
    ok = nooa_cybergym_main._shutdown_tracing_with_timeout(timeout_sec=0.01)
    elapsed = time.monotonic() - before

    assert started.wait(0.1)
    assert ok is False
    assert elapsed < 0.5


def test_soft_timeout_writes_output_before_tracing_shutdown_and_exit(monkeypatch):
    import asyncio

    events = []

    class ExitCalled(Exception):
        def __init__(self, code):
            self.code = code

    class FakeLLM:
        context_window = 100_000

    class FakeAgent:
        def __init__(self, llm):
            self.llm = llm

        async def solve(self, prompt):
            await asyncio.sleep(10)

        def timeout_summary(self):
            return "timed out summary"

    def fake_write_output(result):
        events.append(("write", result))

    def fake_shutdown():
        events.append(("shutdown", None))
        return True

    def fake_exit(code):
        events.append(("exit", code))
        raise ExitCalled(code)

    monkeypatch.setattr(nooa_cybergym_main, "SOFT_TIMEOUT_SEC", 0.01)
    monkeypatch.setattr(nooa_cybergym_main, "make_llm", lambda *args, **kwargs: FakeLLM())
    monkeypatch.setattr(nooa_cybergym_main, "CyberGymAgent", FakeAgent)
    monkeypatch.setattr(nooa_cybergym_main, "configure_tracing", lambda *args, **kwargs: None)
    monkeypatch.setattr(nooa_cybergym_main, "install_summarizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(nooa_cybergym_main, "_write_output", fake_write_output)
    monkeypatch.setattr(nooa_cybergym_main, "_shutdown_tracing_with_timeout", fake_shutdown)
    monkeypatch.setattr(nooa_cybergym_main.os, "_exit", fake_exit)

    with pytest.raises(ExitCalled) as exc_info:
        asyncio.run(nooa_cybergym_main.amain("prompt", "model", None))

    assert exc_info.value.code == 0
    assert events == [
        ("write", "timed out summary"),
        ("shutdown", None),
        ("exit", 0),
    ]
