# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the CyberGym agent entry point."""

from __future__ import annotations

import pytest

pytest.importorskip("nooa")

from examples.cybergym.nooa_cybergym import main as nooa_cybergym_main
from examples.cybergym.nooa_cybergym import util as nooa_cybergym_util


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
    assert kwargs["output_token_margin"] == 64000
    assert kwargs["reasoning_output_floor"] == 128000
    assert kwargs["usage_log_path"] == "/logs/artifacts/llm_usage.jsonl"
    assert "reasoning" not in kwargs
    assert "reasoning_effort" not in kwargs


def test_summarizer_has_independent_llm_with_thinking_disabled(monkeypatch):
    class FakeLLM:
        model = "deepseek/deepseek-v4-flash"
        context_window = 1_000_000
        config = {"reasoning_effort": "max"}

    summary_llm = FakeLLM()
    installed = {}

    monkeypatch.setattr(
        nooa_cybergym_util,
        "make_llm",
        lambda *args, **kwargs: installed.update(make_kwargs=kwargs) or summary_llm,
    )
    monkeypatch.setattr(
        nooa_cybergym_util,
        "TokenBudgetSummarizer",
        type(
            "FakeSummarizer",
            (),
            {
                "install": staticmethod(
                    lambda agent, **kwargs: installed.update(install_kwargs=kwargs)
                )
            },
        ),
    )

    nooa_cybergym_util.install_summarizer(object(), FakeLLM())

    assert installed["make_kwargs"]["reasoning_effort"] == "none"
    assert summary_llm.config == {"extra_body": {"thinking": {"type": "disabled"}}}
    assert installed["install_kwargs"]["llm"] is summary_llm
    config = installed["install_kwargs"]["config"]
    assert config.context_window == 1_000_000
    assert config.output_margin == 64_000
    assert config.reasoning_output_floor == 128_000


def test_reasoning_effort_uses_responses_shape_from_registry_config():
    class FakeResponsesLLM:
        config = {}
        _registry_config = {"client_type": "responses"}

    llm = FakeResponsesLLM()

    nooa_cybergym_main._apply_reasoning_effort(llm, "xhigh")

    assert llm.config["reasoning"] == {"effort": "xhigh"}
    assert "reasoning_effort" not in llm.config


def test_completion_reasoning_effort_is_allowlisted_for_openai_compatible_endpoint():
    class FakeCompletionLLM:
        config = {"allowed_openai_params": ["seed"]}

    llm = FakeCompletionLLM()

    nooa_cybergym_main._apply_reasoning_effort(llm, "max")

    assert llm.config["reasoning_effort"] == "max"
    assert llm.config["allowed_openai_params"] == ["seed", "reasoning_effort"]


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


def test_soft_timeout_requests_clean_finalization_without_forced_exit(monkeypatch):
    import asyncio

    events = []

    class FakeLLM:
        context_window = 100_000

    class FakeAgent:
        def __init__(self, llm):
            self.llm = llm
            self.stop = asyncio.Event()

        async def solve(self, prompt):
            await self.stop.wait()
            events.append(("solve", "finalized"))
            return "finalized result"

        def request_stop(self):
            events.append(("stop", None))
            self.stop.set()

        async def shutdown(self):
            events.append(("agent_shutdown", None))

        def timeout_summary(self):
            return "timed out summary"

    monkeypatch.setattr(nooa_cybergym_main, "SOFT_TIMEOUT_SEC", 0.01)
    monkeypatch.setattr(nooa_cybergym_main, "FINALIZATION_GRACE_SEC", 1)
    monkeypatch.setattr(nooa_cybergym_main, "make_llm", lambda *args, **kwargs: FakeLLM())
    monkeypatch.setattr(nooa_cybergym_main, "CyberGymAgent", FakeAgent)
    monkeypatch.setattr(nooa_cybergym_main, "configure_tracing", lambda *args, **kwargs: None)
    monkeypatch.setattr(nooa_cybergym_main, "install_summarizer", lambda *args, **kwargs: None)
    result = asyncio.run(nooa_cybergym_main.amain("prompt", "model", None))

    assert result == "finalized result"
    assert events == [("stop", None), ("solve", "finalized")]
