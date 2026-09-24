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
import contextvars
import json
from typing import Any, ClassVar

from nooa_coder.session.items import (  # noqa: F401
    ChildFailed,
    ChildFailedError,
    ChildQuestion,
    ChildResult,
    TaskResult,
)
from nooa_coder.session.loader import default_agent_factory
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


# A context variable a test sets outside the session; cells read it.
MARKER: contextvars.ContextVar[str] = contextvars.ContextVar("MARKER", default="unset")


class BlockingLLM(FakeLLMClient):
    """A fake model whose call number ``block_on`` never returns until cancelled."""

    def __init__(self, responses: list[LLMResponse] | None = None, *, block_on: int = 1) -> None:
        super().__init__(responses or [], strict_exhaustion=True)
        self.block_on = block_on
        self.entered = asyncio.Event()
        self._seen = 0

    async def acall(self, *args: Any, **kwargs: Any) -> LLMResponse:
        self._seen += 1
        if self._seen == self.block_on:
            self.entered.set()
            await asyncio.Event().wait()
        return await super().acall(*args, **kwargs)


class ScriptedModels:
    """An agent factory that gives each session its own strict fake model.

    Scripts are keyed by session name (``None`` for an unnamed root); a
    session whose name has no script gets an empty strict model.
    """

    def __init__(self, scripts: dict[str | None, list[LLMResponse]] | None = None) -> None:
        self.scripts = dict(scripts or {})
        self.llms: dict[str | None, FakeLLMClient] = {}
        self.built: list[Any] = []

    def __call__(self, options: Any, storage: Any) -> InteractiveAgent:
        llm = FakeLLMClient(list(self.scripts.get(options.name, [])), strict_exhaustion=True)
        self.llms[options.name] = llm
        self.built.append(options)
        return default_agent_factory(options.model_copy(update={"llm": llm}), storage)


class _Command:
    def __init__(self, name: str, description: str, argument_hint: str | None) -> None:
        self.name = name
        self.description = description
        self.argument_hint = argument_hint


class _CommandOutput:
    def __init__(self, text: str, value: Any, output_to_agent: bool) -> None:
        self.text = text
        self.value = value
        self.output_to_agent = output_to_agent


class FakeSlashCommands:
    """A slash command registry shaped like the coding agent's (duck-typed)."""

    def __init__(self) -> None:
        self.invoked: list[tuple[str, str]] = []

    def commands(self) -> tuple[_Command, ...]:
        return (
            _Command("model", "Show or switch the model", "[alias]"),
            _Command("clear", "Clear", None),
        )

    async def invoke(self, name: str, raw_args: str) -> _CommandOutput:
        if name not in ("model", "clear"):
            raise KeyError(name)
        self.invoked.append((name, raw_args))
        return _CommandOutput(f"model is {raw_args or 'default'}", {"alias": raw_args}, False)


class CommandAgent(InteractiveAgent, llm=FakeLLMClient()):
    """An agent with a slash command registry."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.slash_commands = FakeSlashCommands()


class TrackedLLM(FakeLLMClient):
    """A strict fake model that records whether it was closed."""

    def __init__(self, alias: str, responses: list[LLMResponse]) -> None:
        super().__init__(responses, strict_exhaustion=True)
        self.alias = alias
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class ModelFactory:
    """An ``llm_factory``: builds a TrackedLLM per call, scripted by alias."""

    def __init__(self, scripts: dict[str, list[list[LLMResponse]]] | None = None) -> None:
        self.scripts = {alias: list(queue) for alias, queue in (scripts or {}).items()}
        self.made: list[TrackedLLM] = []
        self.calls: list[tuple[str | None, Any]] = []

    def __call__(self, alias: str | None, workspace: Any) -> TrackedLLM:
        self.calls.append((alias, workspace))
        if alias == "bad-alias":
            raise ValueError("unknown model alias 'bad-alias'")
        resolved = alias or "default-model"  # None means the factory's default
        queue = self.scripts.get(resolved, [])
        llm = TrackedLLM(resolved, queue.pop(0) if queue else [])
        self.made.append(llm)
        return llm


class SelfCancelAgent(InteractiveAgent, llm=FakeLLMClient()):
    """Its first turn is cancelled from inside (not by Session.cancel()); later turns finish."""

    async def handle(self, notification: dict[str, list[Any]]) -> Done:
        if "cancelled_once" not in self.vars:
            self.vars["cancelled_once"] = True
            inner = asyncio.ensure_future(asyncio.sleep(10))
            await asyncio.sleep(0)
            inner.cancel()
            await inner
        return Done(explanation="finished")


class HiddenPortAgent(InteractiveAgent, llm=FakeLLMClient()):
    """An agent that exposes delegation its own way, so the port is hidden from the model."""

    session_port_visible: ClassVar[bool] = False


async def until(predicate: Any, timeout: float = 5) -> None:
    """Wait until ``predicate()`` is true; fail with TimeoutError after ``timeout`` seconds."""

    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


CODER_SPEC = "nooa_coder.coding.agent:CodingAgent"


class CoderModels(ScriptedModels):
    """An agent factory for coding-agent sessions, one strict fake model each.

    Like ``ScriptedModels``, scripts are keyed by session name. The agent is
    built for the session's workspace, with its libraries directory inside
    that workspace so tests never touch the process's project directory.
    """

    def __call__(self, options: Any, storage: Any) -> InteractiveAgent:
        from nooa_coder.session.loader import load_agent_class

        llm = options.llm or FakeLLMClient(
            list(self.scripts.get(options.name, [])), strict_exhaustion=True
        )
        self.llms[options.name] = llm
        self.built.append(options)
        agent_class = load_agent_class(options.agent_spec, base=options.workspace)
        return agent_class(
            llm=llm,
            storage=storage,
            cwd=options.workspace,
            libs_dir=options.workspace / ".nooa" / "libs",
        )
