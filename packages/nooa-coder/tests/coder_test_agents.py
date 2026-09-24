# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agents and fake-model scripts for the nooa-coder tests.

The module has a unique name and sits on the pytest ``pythonpath`` so that
``"coder_test_agents:EchoAgent"`` loads as an agent spec. Generated cells
run with this module's globals, so the names imported here (``Done``,
``TaskResult``, the ``STARTED``/``BLOCK`` events...) are what a scripted
cell can use.
"""

import asyncio
import json
from typing import Any

from nooa_coder.session.items import ChildQuestion, ChildResult, TaskResult  # noqa: F401
from pydantic import BaseModel

from nooa.interactive import Done, InteractiveAgent, NeedInput, Waiting  # noqa: F401
from nooa.unifiedllm import FakeLLMClient, LLMResponse, LLMUsage, ToolCall

# Cells can reach these; tests replace them with fresh events per test.
STARTED: asyncio.Event | None = None
BLOCK: asyncio.Event | None = None


class EchoAgent(InteractiveAgent, llm=FakeLLMClient()):
    """An interactive agent whose turns are scripted through a fake model."""


class BatchAgent(InteractiveAgent, llm=FakeLLMClient()):
    """An agent run unattended (``handle_batch``) that reports a TaskResult."""


class FailingAgent(InteractiveAgent, llm=FakeLLMClient()):
    """An agent whose constructor fails, for the registry's two-phase create."""

    def __init__(self, **kwargs: Any) -> None:
        raise RuntimeError("agent construction failed")


class NotAnAgent:
    """Not an InteractiveAgent; loading it as an agent spec must fail."""


class Answer(BaseModel):
    """A typed answer a scripted agent can ask for."""

    branch: str


_counter = 0


def cell(code: str, *, usage: LLMUsage | None = None) -> LLMResponse:
    """One model response that runs ``code`` in an ``execute_python`` cell."""
    global _counter
    _counter += 1
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=f"call_{_counter}", name="execute_python", arguments=json.dumps({"code": code})
            )
        ],
        finish_reason="tool_calls",
        usage=usage,
    )


def reply(text: str, explanation: str = "answered", **kwargs: Any) -> LLMResponse:
    """A cell that sends ``text`` to the user and ends the turn with ``Done``."""
    return cell(
        f"self.message({text!r})\nreturn_result(Done(explanation={explanation!r}))", **kwargs
    )


def done(explanation: str = "done", **kwargs: Any) -> LLMResponse:
    """A cell that ends the turn with ``Done`` and nothing else."""
    return cell(f"return_result(Done(explanation={explanation!r}))", **kwargs)


def ask(question: str, **kwargs: Any) -> LLMResponse:
    """A cell that ends the turn with a ``NeedInput`` question."""
    return cell(f"return_result(NeedInput(question={question!r}))", **kwargs)


def wait_on(*names: str) -> LLMResponse:
    """A cell that ends the turn with ``Waiting`` on the given names."""
    return cell(f"return_result(Waiting(explanation='waiting', on={list(names)!r}))")


def fresh_events() -> tuple[asyncio.Event, asyncio.Event]:
    """New ``STARTED``/``BLOCK`` events for cells that pause mid-turn."""
    global STARTED, BLOCK
    STARTED, BLOCK = asyncio.Event(), asyncio.Event()
    return STARTED, BLOCK


BLOCKING_CELL = """\
print("cell started")
STARTED.set()
await BLOCK.wait()
print("cell released")
"""
