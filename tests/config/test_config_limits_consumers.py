# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each configurable limit of the core package honours an agent-level and a method-level override.

With no override the output is the one of the old constant. The consumers:
``capture.max_shell_output`` (BashSession), ``state_format`` (the ``state``
context block, agent level only), ``prefill_format`` (InspectInputsPrefill without a config)
and ``placeholder_chars`` (the method's own docstring, the system prompt).
"""

from collections.abc import Callable
from typing import Any

import pytest

from nooa import Agent, CodeActStrategy, strategy
from nooa.config import CaptureConfig, FormatConfig, TruncationConfig
from nooa.strategies.current_call import CurrentCall
from nooa.strategies.prefill import InspectInputsPrefill
from nooa.tools._bash_session import BashSession
from nooa.unifiedllm import FakeLLMClient

SHELL = TruncationConfig(capture=CaptureConfig(max_shell_output=100))
STATE = TruncationConfig(state_format=FormatConfig(max_string=5, max_length=2, max_depth=2))
PREFILL = TruncationConfig(prefill_format=FormatConfig(max_string=7, max_length=3, max_depth=2))
PLACEHOLDER = TruncationConfig(placeholder_chars=100)


def _probe_agent(method_truncation: TruncationConfig | None, **agent_kwargs: Any) -> Any:
    """An agent with ``implemented(probe)`` and a generated ``generated(probe)``.

    Both call ``probe(agent)``; ``generated`` runs it from its generated code
    under ``@strategy(truncation=method_truncation)``.
    """
    llm = FakeLLMClient.with_tool_call(
        "execute_python", {"code": "return_result(await probe(self))"}, strict_exhaustion=False
    )
    kwargs: dict[str, Any] = {} if method_truncation is None else {"truncation": method_truncation}

    class Probe(Agent, llm=llm):
        """Probe {self.value}."""

        value: str = ""

        async def implemented(self, probe: Callable[..., Any]) -> Any:
            return await probe(self)

        @strategy(CodeActStrategy(), **kwargs)
        async def generated(self, probe: Callable[..., Any]) -> Any:
            """Await probe(self) and return its result."""
            ...

    return Probe(**agent_kwargs)


async def _both(probe: Callable[..., Any], config: TruncationConfig) -> tuple[Any, Any, Any]:
    """``probe`` with no override, with ``config`` on the agent, and on the method only."""
    results = []
    for method, agent_kwargs in ((None, {}), (None, {"truncation": config}), (config, {})):
        agent = _probe_agent(method, **agent_kwargs)
        try:
            if method is None:
                results.append(await agent.implemented(probe))
            else:
                results.append(await agent.generated(probe))
        finally:
            await agent.aclose()
    return results[0], results[1], results[2]


# capture.max_shell_output


async def _shell_output(_agent: Agent) -> int:
    session = BashSession()
    try:
        out, _err, _code = await session.run("python3 -c \"print('x' * 50000)\"")
    finally:
        await session.close()
    return len(out)


async def test_shell_output_follows_max_shell_output():
    default, agent_level, method_level = await _both(_shell_output, SHELL)
    # 30,000 kept + the cut line (old MAX_OUTPUT_CHARS).
    assert 30_000 < default < 30_100
    assert 100 < agent_level < 200
    assert 100 < method_level < 200


async def test_shell_output_default_is_the_old_constant_outside_an_agent():
    from nooa._bounded_text import cut_text
    from nooa.tools._bash_session import _bounded

    text = "".join(chr(ord("a") + i % 26) for i in range(100_000))
    assert _bounded(text) == cut_text(text, 30_000)


def test_max_shell_output_must_be_positive():
    with pytest.raises(ValueError, match="max_shell_output"):
        CaptureConfig(max_shell_output=0)


def test_tail_does_not_constrain_the_shell_and_job_limits():
    CaptureConfig(tail=40_000, max_stdout=100_000, max_stderr=50_000, max_error=50_000)


# state_format: the state context block (agent level; the prompt shows its expression)


async def _state_expr(agent: Agent) -> str:
    return agent.context_manager._blocks["state"].expr


async def test_state_block_follows_the_agent_state_format():
    default, agent_level, method_level = await _both(_state_expr, STATE)
    old = "pformat(self, max_length=50, max_string=500, max_depth=4)"
    assert default == old  # byte-identical to the old literal expression
    assert agent_level == "pformat(self, max_length=2, max_string=5, max_depth=2)"
    # Built with the agent: a method-level override does not change it.
    assert method_level == old


async def test_state_block_class_level_state_format():
    class Small(Agent, llm=FakeLLMClient([]), truncation=STATE):
        pass

    agent = Small()
    try:
        expr = agent.context_manager._blocks["state"].expr
        assert expr == "pformat(self, max_length=2, max_string=5, max_depth=2)"
    finally:
        await agent.aclose()


# prefill_format: InspectInputsPrefill.get_code(call) without a config


def _call() -> CurrentCall:
    return CurrentCall(
        id="c1",
        method_name="m",
        decorator="plan",
        signature="(self, data: str) -> str",
        docstring="",
        args=(),
        kwargs={"data": "x"},
    )


async def _prefill(_agent: Agent) -> str:
    return InspectInputsPrefill().get_code(_call()) or ""


async def test_prefill_without_a_config_follows_prefill_format():
    default, agent_level, method_level = await _both(_prefill, PREFILL)
    assert "max_length=25, max_string=2000, max_depth=4" in default
    assert "max_length=3, max_string=7, max_depth=2" in agent_level
    assert "max_length=3, max_string=7, max_depth=2" in method_level


def test_prefill_without_a_config_outside_an_agent_uses_the_prefill_defaults():
    # The one intended change: was max_length=50, max_string=500, max_depth=4
    # (bare FormatConfig defaults), now DEFAULT_TRUNCATION_CONFIG.prefill_format.
    code = InspectInputsPrefill().get_code(_call()) or ""
    assert "max_length=25, max_string=2000, max_depth=4" in code
    assert "max_length=50" not in code


# placeholder_chars: the method's own docstring and the system prompt


class _Doc(Agent):
    """System {self.value}."""

    value: str = ""

    @strategy(CodeActStrategy(), truncation=PLACEHOLDER)
    async def limited(self, text: str) -> str:
        """Echo {text}."""
        ...

    @strategy(CodeActStrategy())
    async def unlimited(self, text: str) -> str:
        """Echo {text}."""
        ...


async def _task_prompt(method_name: str, agent_kwargs: dict[str, Any]) -> tuple[str, str]:
    llm = FakeLLMClient.with_tool_call(
        "execute_python",
        {"code": "return_result(self._resolve_system_prompt())"},
        strict_exhaustion=False,
    )
    agent = _Doc(llm=llm, **agent_kwargs)
    agent.value = "S" * 300
    try:
        system = await getattr(agent, method_name)("T" * 300)
        return str(llm.calls[0].messages), system
    finally:
        await agent.aclose()


async def test_the_method_docstring_and_system_prompt_follow_a_method_override():
    request, system = await _task_prompt("limited", {})
    assert "Echo " + "T" * 300 not in request and "Echo " + "T" * 40 in request
    assert len(system) < 200 and system.startswith("System SSS")


async def test_the_method_docstring_and_system_prompt_follow_the_agent_config():
    request, system = await _task_prompt("unlimited", {"truncation": PLACEHOLDER})
    assert "Echo " + "T" * 300 not in request and "Echo " + "T" * 40 in request
    assert len(system) < 200


async def test_the_method_docstring_and_system_prompt_are_whole_by_default():
    request, system = await _task_prompt("unlimited", {})
    assert "Echo " + "T" * 300 in request
    assert system == "System " + "S" * 300 + "."
