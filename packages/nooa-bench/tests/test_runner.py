# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle tests for the benchmark runner."""

from types import SimpleNamespace

import pytest
from nooa_bench import runner


@pytest.mark.asyncio
async def test_run_closes_agent_and_llm_when_result_writing_fails(monkeypatch):
    calls: list[str] = []

    class FakeLLM:
        async def aclose(self):
            calls.append("llm")

    class FakeAgent:
        def __init__(self, llm):
            self.llm = llm
            self.event_manager = SimpleNamespace(items=lambda: [])

        async def _run_evaluation(self, task_input):
            return {"success": True, "response": "done"}

        async def close(self):
            calls.append("agent")

    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", lambda *args, **kwargs: FakeLLM())
    monkeypatch.setattr(runner, "_import_agent_class", lambda name: FakeAgent)
    monkeypatch.setattr(
        runner, "_write_result", lambda *args: (_ for _ in ()).throw(OSError("disk"))
    )

    with pytest.raises(OSError, match="disk"):
        await runner._run("task", "model", "bench", None)

    assert calls == ["agent", "llm"]


@pytest.mark.asyncio
async def test_run_closes_llm_when_agent_construction_fails(monkeypatch):
    calls: list[str] = []

    class FakeLLM:
        async def aclose(self):
            calls.append("llm")

    class BrokenAgent:
        def __init__(self, llm):
            raise RuntimeError("constructor failed")

    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", lambda *args, **kwargs: FakeLLM())
    monkeypatch.setattr(runner, "_import_agent_class", lambda name: BrokenAgent)

    with pytest.raises(RuntimeError, match="constructor failed"):
        await runner._run("task", "model", "bench", None)

    assert calls == ["llm"]
