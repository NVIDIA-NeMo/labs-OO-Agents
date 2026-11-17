# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa-atom's configurable limits honour agent-level and method-level overrides.

``capture.max_shell_output`` (terminal cards), ``capture.max_job_tail``
(a background run's done update) and ``instructions_file_chars`` /
``instructions_total_chars`` (repository instructions), all read through
``nooa.config.effective_truncation``. With no override the output is the one
of the old constants.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nooa_atom.agent.activity import ActivityShellTools, TerminalCommandOutput
from nooa_atom.agent.instructions import RepositoryInstructions, render_agent_instructions
from nooa_atom.agent.jobs import JobsApi

from nooa import Agent, CodeActStrategy, strategy
from nooa.config import CaptureConfig, TruncationConfig
from nooa.runtime.channels import QueueManager
from nooa.runtime.event_manager import EventManager
from nooa.tools import ShellTools
from nooa.unifiedllm import FakeLLMClient

CARD = TruncationConfig(capture=CaptureConfig(max_shell_output=100))
JOB = TruncationConfig(capture=CaptureConfig(max_job_tail=50))
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


# capture.max_shell_output: the terminal card


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
    assert 30_000 < default < 30_100  # the old _MAX_COMMAND_OUTPUT_CHARS
    assert 100 < agent_level < 200
    assert 100 < method_level < 200


# capture.max_job_tail: a background run's done update


def _job_probe(tmp_path: Path) -> Callable[..., Any]:
    class _Host:
        jobs: Any

        def __init__(self) -> None:
            self.queue_manager = QueueManager()
            self.queue_manager.queue("jobs")
            self.shell = ShellTools(cwd=str(tmp_path))
            self.cwd = tmp_path
            self._jobs_running: dict = {}

    async def probe() -> str:
        host = _Host()
        host.jobs = JobsApi(host)
        try:
            await host.jobs.run("for i in $(seq 5000); do echo line$i; done", name="big")
            update = await asyncio.wait_for(host.queue_manager.get_channel("jobs").get(), 15)
            return update.stdout
        finally:
            await host.queue_manager.shutdown()
            await host.shell.close()

    return probe


async def test_a_job_tail_follows_max_job_tail(tmp_path):
    default, agent_level, method_level = await _three(_job_probe(tmp_path), JOB)
    whole = "".join(f"line{i}\n" for i in range(1, 5001))
    omitted = "chars omitted; full output not kept]\n"
    assert default == f"[... {len(whole) - 20_000:,} {omitted}" + whole[-20_000:]
    for text in (agent_level, method_level):
        assert text == f"[... {len(whole) - 50:,} {omitted}" + whole[-50:]


# instructions_file_chars / instructions_total_chars


def _instructions_probe(tmp_path: Path) -> Callable[..., Any]:
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "AGENTS.md").write_text("r" * 5_000)
    cached = RepositoryInstructions(tmp_path)

    async def probe() -> tuple[str, str]:
        return render_agent_instructions(tmp_path), cached.render()

    return probe


async def test_instructions_follow_the_instruction_budgets(tmp_path):
    results = await _three(_instructions_probe(tmp_path), INSTRUCTIONS)
    (default, cached_default), *overridden = results
    assert default == cached_default
    assert "r" * 5_000 in default and "[... truncated ...]" not in default
    for direct, cached in overridden:
        # The cache keys on the limits too, so a changed limit re-renders.
        assert direct == cached
        assert "r" * 60 in direct and "r" * 61 not in direct
        assert direct.endswith("[... truncated ...]")
        assert len(direct) <= 500


def test_instruction_defaults_are_the_old_constants(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("q" * 150_000)
    rendered = render_agent_instructions(tmp_path)
    assert "q" * 100_000 in rendered and "q" * 100_001 not in rendered
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "AGENTS.md").write_text("s" * 150_000)
    both = render_agent_instructions(tmp_path / "sub")
    assert len(both) == 200_000
