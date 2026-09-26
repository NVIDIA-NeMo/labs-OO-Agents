# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token usage over a whole session: counted, rolled up, kept on disk, and /usage."""

import asyncio

import pytest
from coder_test_agents import ScriptedModels, cell, done
from nooa_coder.session.items import Usage
from nooa_coder.session.registry import SessionRegistry
from nooa_coder.session.store import SessionStore
from nooa_coder.workspace.controls import UsageControl

from nooa.unifiedllm import LLMUsage

TIMEOUT = 20

CALL = LLMUsage(
    input_tokens=100,
    output_tokens=20,
    cached_input_tokens=60,
    cache_write_input_tokens=10,
    reasoning_tokens=5,
    total_tokens=120,
    cost_usd=0.5,
)
TWO_CALLS = Usage(
    input_tokens=200,
    output_tokens=40,
    cached_input_tokens=120,
    cache_write_input_tokens=20,
    reasoning_tokens=10,
    total_tokens=240,
    cost_usd=1.0,
)


async def _reloaded(sessions_dir, session_id):
    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=ScriptedModels())
    session = await fresh.load(session_id)
    return fresh, session


async def test_every_token_count_is_kept_over_the_whole_trajectory(
    registry, root_options, models, sessions_dir
):
    models.scripts[None] = [done("one", usage=CALL), done("two", usage=CALL)]
    root = await registry.create(root_options)
    await asyncio.wait_for(root.prompt("one"), TIMEOUT)
    await asyncio.wait_for(root.prompt("two"), TIMEOUT)
    # The latest call's input is the context in use; it is live state, not a total.
    assert root.info.usage == TWO_CALLS.model_copy(update={"last_input_tokens": 100})
    await registry.close_all()

    assert SessionStore(sessions_dir).get(root.id).usage == TWO_CALLS
    fresh, loaded = await _reloaded(sessions_dir, root.id)
    try:
        assert loaded.info.usage == TWO_CALLS
    finally:
        await fresh.close_all()


async def test_every_model_call_is_announced_with_the_new_totals(registry, root_options, models):
    models.scripts[None] = [done("one", usage=CALL)]
    root = await registry.create(root_options)
    seen = []
    root.subscribe(lambda e: seen.append(e.usage) if e.kind == "usage_changed" else None)
    await asyncio.wait_for(root.prompt("one"), TIMEOUT)
    [usage] = seen
    assert (usage.input_tokens, usage.cost_usd, usage.last_input_tokens) == (100, 0.5, 100)


async def test_a_childs_cached_tokens_roll_up_and_survive_a_reload(
    registry, root_options, models, sessions_dir
):
    models.scripts[None] = [
        cell(
            "a = await self.session.delegate('A', 'a', retain=True)\n"
            "await a.wait()\n"
            "return_result(Done(explanation='ok'))"
        )
    ]
    models.scripts["A"] = [done("a done", usage=CALL)]
    root = await registry.create(root_options)
    await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    totals = root.agent.session.usage()
    assert (totals.cached_input_tokens, totals.attributed_cached_input_tokens) == (0, 60)
    assert totals.attributed_reasoning_tokens == 5
    await registry.close_all()

    fresh, loaded = await _reloaded(sessions_dir, root.id)
    try:
        assert loaded.info.usage == totals
    finally:
        await fresh.close_all()


async def test_usage_reports_this_session_then_with_its_subagents(registry, root_options, models):
    models.scripts[None] = [
        cell(
            "a = await self.session.delegate('A', 'a', retain=True)\n"
            "await a.wait()\n"
            "return_result(Done(explanation='ok'))",
            usage=CALL,
        )
    ]
    models.scripts["A"] = [done("a done", usage=CALL)]
    root = await registry.create(root_options)
    await asyncio.wait_for(root.prompt("go"), TIMEOUT)

    text = str(await UsageControl(root.agent, None).invoke(""))
    own, _, rest = text.partition("Including subagents")
    assert "This session" in own
    assert _row(own, "Input tokens") == "100"
    assert _row(own, "Cached input tokens (cache reads)") == "60"
    assert _row(own, "Cache-write input tokens") == "10"
    assert _row(own, "Reasoning tokens") == "5"
    assert _row(own, "Total tokens") == "120"
    assert _row(own, "Cost (USD)") == "0.5000"
    assert _row(rest, "Input tokens") == "200"
    assert _row(rest, "Cached input tokens (cache reads)") == "120"
    assert "Turns: 1" in rest


async def test_usage_without_subagents_has_one_block(registry, root_options, models):
    models.scripts[None] = [done("one", usage=LLMUsage(input_tokens=3, output_tokens=1))]
    root = await registry.create(root_options)
    await asyncio.wait_for(root.prompt("one"), TIMEOUT)
    text = str(await UsageControl(root.agent, None).invoke(""))
    assert "Including subagents" not in text
    assert "Cost (USD)" not in text  # nothing was spent
    assert _row(text, "Output tokens") == "1"


@pytest.mark.parametrize("agent", [object(), None])
async def test_usage_without_a_session_port_says_so(agent):
    result = await UsageControl(agent, None).invoke("")
    assert not result.success
    assert "not available" in str(result)


def _row(text: str, label: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith(label + " "):
            return line.strip()[len(label) :].strip().replace(",", "")
    raise AssertionError(f"no {label!r} row in:\n{text}")
