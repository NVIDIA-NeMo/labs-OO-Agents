# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``nooa.config.effective_truncation``: method > agent instance > class > defaults."""

import asyncio
import contextvars
import threading

from nooa import Agent, CodeActStrategy, strategy
from nooa.config import CaptureConfig, TruncationConfig, effective_truncation
from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
from nooa.unifiedllm import FakeLLMClient


def _seen() -> int:
    return effective_truncation().capture.max_stdout


def _agent(*, cls_stdout=None, inst_stdout=None, method_stdout=None):
    """An agent whose ``implemented()`` and generated ``generated(probe)`` report ``max_stdout``."""
    llm = FakeLLMClient.with_tool_call(
        "execute_python", {"code": "return_result(probe())"}, strict_exhaustion=True
    )
    cls_kwargs = {}
    if cls_stdout is not None:
        cls_kwargs["truncation"] = TruncationConfig(capture=CaptureConfig(max_stdout=cls_stdout))
    method_kwargs = {}
    if method_stdout is not None:
        method_kwargs["truncation"] = TruncationConfig(
            capture=CaptureConfig(max_stdout=method_stdout)
        )

    class Probe(Agent, llm=llm, **cls_kwargs):
        async def implemented(self) -> int:
            return _seen()

        @strategy(CodeActStrategy(), **method_kwargs)
        async def generated(self, probe: object) -> int:
            """Call probe() and return its result."""
            ...

    inst_kwargs = {}
    if inst_stdout is not None:
        inst_kwargs["truncation"] = TruncationConfig(capture=CaptureConfig(max_stdout=inst_stdout))
    return Probe(**inst_kwargs)


def test_outside_any_agent_it_is_the_defaults():
    assert effective_truncation() is DEFAULT_TRUNCATION_CONFIG


async def test_no_override_is_the_defaults_in_agent_code():
    agent = _agent()
    try:
        assert await agent.implemented() == 50_000
        assert await agent.generated(_seen) == 50_000
    finally:
        await agent.aclose()


async def test_class_level_beats_the_defaults():
    agent = _agent(cls_stdout=111)
    try:
        assert await agent.implemented() == 111
        assert await agent.generated(_seen) == 111
    finally:
        await agent.aclose()


async def test_instance_level_beats_the_class():
    agent = _agent(cls_stdout=111, inst_stdout=222)
    try:
        assert await agent.implemented() == 222
        assert await agent.generated(_seen) == 222
    finally:
        await agent.aclose()


async def test_method_level_beats_the_instance_only_inside_that_method():
    agent = _agent(cls_stdout=111, inst_stdout=222, method_stdout=333)
    try:
        assert await agent.generated(_seen) == 333
        assert await agent.implemented() == 222
        assert effective_truncation() is DEFAULT_TRUNCATION_CONFIG
    finally:
        await agent.aclose()


async def test_an_explicit_agent_is_its_config_and_the_runtime_agrees():
    agent = _agent(inst_stdout=222)
    try:
        assert effective_truncation(agent).capture.max_stdout == 222
        assert agent.runtime.truncation_config is effective_truncation(agent)
    finally:
        await agent.aclose()


class _Threads(Agent, llm=FakeLLMClient([])):
    async def threads(self) -> dict[str, int]:
        seen: dict[str, int] = {}

        def record(key: str) -> None:
            seen[key] = _seen()

        plain = threading.Thread(target=record, args=("plain",))
        copied = threading.Thread(target=contextvars.copy_context().run, args=(record, "copied"))
        for thread in (plain, copied):
            thread.start()
            await asyncio.to_thread(thread.join)
        await asyncio.to_thread(record, "to_thread")
        return seen


async def test_plain_threads_need_copy_context():
    agent = _Threads(truncation=TruncationConfig(capture=CaptureConfig(max_stdout=222)))
    try:
        seen = await agent.threads()
    finally:
        await agent.aclose()
    assert seen == {"plain": 50_000, "copied": 222, "to_thread": 222}
