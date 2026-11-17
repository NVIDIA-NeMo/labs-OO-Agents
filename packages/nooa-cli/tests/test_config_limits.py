# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa-cli's configurable limits honour agent-level and method-level overrides.

``capture.max_shell_output`` (terminal cards: half of it per stream, as
before) and ``instructions_file_chars`` / ``instructions_total_chars``
(repository instructions), read through ``nooa.config.effective_truncation``.
With no override the output is the one of the old constants.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from nooa_cli.coding import CodingAgent
from nooa_cli.coding.activity import ActivityShellTools, TerminalCommandOutput
from nooa_cli.coding.instructions import render_agent_instructions

from nooa import Agent, CodeActStrategy, strategy
from nooa.config import CaptureConfig, TruncationConfig
from nooa.runtime.event_manager import EventManager
from nooa.tools import ShellTools
from nooa.unifiedllm import FakeLLMClient

CARD = TruncationConfig(capture=CaptureConfig(max_shell_output=200))
INSTRUCTIONS = TruncationConfig(instructions_file_chars=60, instructions_total_chars=500)


def _probe_agent(method_truncation: TruncationConfig | None, **agent_kwargs: Any) -> Any:
    """``implemented(probe)`` and a generated ``generated(probe)`` that await ``probe()``."""
    llm = FakeLLMClient.with_tool_call(
        "execute_python", {"code": "return_result(await probe())"}, strict_exhaustion=False
    )
    kwargs: dict[str, Any] = {} if method_truncation is None else {"truncation": method_truncation}

    class Probe(Agent, llm=llm):
        async def implemented(self, probe: Callable[..., Any]) -> Any:
            return await probe()

        @strategy(CodeActStrategy(), **kwargs)
        async def generated(self, probe: Callable[..., Any]) -> Any:
            """Await probe() and return its result."""
            ...

    return Probe(**agent_kwargs)


async def _three(probe: Callable[..., Any], config: TruncationConfig) -> list[Any]:
    """``probe()`` with no override, with ``config`` on the agent, and on the method only."""
    results = []
    for method, agent_kwargs in ((None, {}), (None, {"truncation": config}), (config, {})):
        agent = _probe_agent(method, **agent_kwargs)
        try:
            call = agent.implemented if method is None else agent.generated
            results.append(await call(probe))
        finally:
            await agent.aclose()
    return results


def _card_probe(tmp_path: Path) -> Callable[..., Any]:
    async def probe() -> int:
        events: list[Any] = []
        manager = EventManager()
        manager.on("*", events.append)
        shell: Any = ActivityShellTools(shell=ShellTools(cwd=str(tmp_path)), event_manager=manager)
        try:
            await shell.run("python3 -c \"print('z' * 50000)\"")
        finally:
            await shell.close()
        [card] = [event for event in events if isinstance(event, TerminalCommandOutput)]
        return len(card.stdout)

    return probe


async def test_the_card_follows_max_shell_output(tmp_path):
    default, agent_level, method_level = await _three(_card_probe(tmp_path), CARD)
    assert 15_000 < default < 15_100  # the old _MAX_COMMAND_OUTPUT_CHARS // 2
    assert 100 < agent_level < 200
    assert 100 < method_level < 200


def _instructions_probe(tmp_path: Path) -> Callable[..., Any]:
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "AGENTS.md").write_text("r" * 5_000)

    async def probe() -> str:
        return render_agent_instructions(tmp_path)

    return probe


async def test_instructions_follow_the_instruction_budgets(tmp_path):
    default, *overridden = await _three(_instructions_probe(tmp_path), INSTRUCTIONS)
    assert "r" * 5_000 in default and "[... truncated ...]" not in default
    for text in overridden:
        assert "r" * 60 in text and "r" * 61 not in text
        assert text.endswith("[... truncated ...]")


def test_instruction_defaults_are_the_old_constants(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("q" * 150_000)
    rendered = render_agent_instructions(tmp_path)
    assert "q" * 100_000 in rendered and "q" * 100_001 not in rendered


async def test_the_coding_agent_builds_its_block_under_its_own_config(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("r" * 5_000)
    for truncation, whole in ((None, True), (INSTRUCTIONS, False)):
        agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path, truncation=truncation)
        try:
            block = agent.context_manager._blocks["repository_instructions"]
            text = block.value if hasattr(block, "value") else str(block)
            assert ("r" * 5_000 in text) is whole
            assert ("r" * 61 in text) is whole
        finally:
            await agent.close()
