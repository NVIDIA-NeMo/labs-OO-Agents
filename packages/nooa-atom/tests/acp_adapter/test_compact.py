# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""/compact over ACP: listed, folds the history into one summary between turns."""

import asyncio
import json

from acp import text_block
from acp.schema import AgentMessageChunk, AvailableCommandsUpdate, UsageUpdate
from atom_test_agents import BLOCKING_CELL, ScriptedModels, cell, fresh_events, reply

from nooa.unifiedllm import LLMResponse, LLMUsage

TIMEOUT = 30


def _summary(text: str) -> LLMResponse:
    """The summarizer's answer, as its predict call reads it."""
    return LLMResponse(raw_response=None, content=json.dumps({"value": text}), finish_reason="stop")


async def _new(adapter, workspace):
    return (await adapter.new_session(str(workspace))).session_id


async def _say(adapter, client, session_id, text):
    response = await asyncio.wait_for(adapter.prompt(session_id, [text_block(text)]), TIMEOUT)
    assert response.stop_reason == "end_turn"
    return client.texts(AgentMessageChunk, session_id)[-1]


async def test_compact_is_advertised(atom_adapter, workspace, client):
    adapter = await atom_adapter([])
    session_id = await _new(adapter, workspace)
    await client.wait_for(lambda: client.updates(session_id, AvailableCommandsUpdate))
    [commands] = client.updates(session_id, AvailableCommandsUpdate)
    [compact] = [c for c in commands.available_commands if c.name == "compact"]
    assert "Summarize" in compact.description
    assert compact.input is None  # the summarizer takes no custom instructions


async def test_compact_folds_the_history_into_one_summary(atom_adapter, workspace, client):
    usage = LLMUsage(input_tokens=40_000, output_tokens=10, total_tokens=40_010)
    adapter = await atom_adapter(
        [reply("First answer.", usage=usage), _summary("SUMMARY: we said hello."), reply("Ok.")]
    )
    session_id = await _new(adapter, workspace)
    await _say(adapter, client, session_id, "hello")
    session = adapter.session(session_id)
    agent = session._agent
    before = agent.event_manager.keys()
    assert len(before) > 1

    text = await _say(adapter, client, session_id, "/compact")

    [summary_tag] = agent.event_manager.keys()
    assert summary_tag == f"{before[0]}..{before[-1]}"
    assert agent.event_manager[summary_tag].summary_text == "SUMMARY: we said hello."
    assert text.startswith(
        f"Compacted {len(before)} events \u2192 1 (summary block {summary_tag}); context ~40,000 \u2192 ~"
    )
    after = int(text.split("\u2192 ~")[-1].split()[0].replace(",", ""))
    assert 0 < after < 40_000
    # The status line and the model's context block show the estimate until the next call.
    assert session.info.usage.last_input_tokens == after
    assert client.updates(session_id, UsageUpdate)[-1].used == after
    assert agent.context_stats.prompt_tokens == after
    assert agent.context_stats.events_count == 1
    # The summary went to the session's model; the next turn sees it, not the old events.
    [llm] = adapter.test_models.made
    assert "First answer." in str(llm.calls[1].messages)
    await _say(adapter, client, session_id, "next")
    assert "SUMMARY: we said hello." in str(llm.calls[2].messages)


async def test_nothing_to_compact(atom_adapter, workspace, client):
    adapter = await atom_adapter([reply("Hi."), _summary("SUMMARY."), reply("Ok.")])
    session_id = await _new(adapter, workspace)
    assert await _say(adapter, client, session_id, "/compact") == (
        "Nothing to compact \u2014 history is already empty.\n\n"
    )
    await _say(adapter, client, session_id, "hello")
    assert (await _say(adapter, client, session_id, "/compact")).startswith("Compacted ")
    assert await _say(adapter, client, session_id, "/compact") == (
        "Nothing to compact \u2014 history is already one summary.\n\n"
    )
    [llm] = adapter.test_models.made
    assert len(llm.calls) == 2  # no model call when there is nothing to compact


async def test_compact_refuses_an_argument(atom_adapter, workspace, client):
    adapter = await atom_adapter([reply("Hi.")])
    session_id = await _new(adapter, workspace)
    await _say(adapter, client, session_id, "hello")
    keys = adapter.session(session_id)._agent.event_manager.keys()
    text = await _say(adapter, client, session_id, "/compact focus on the parser")
    assert text == "/compact takes no arguments: the summarizer has no custom instructions.\n\n"
    assert adapter.session(session_id)._agent.event_manager.keys() == keys


async def test_compact_during_a_turn_is_refused(make_adapter, workspace, client):
    started, block = fresh_events()
    adapter = await make_adapter(ScriptedModels({None: [cell(BLOCKING_CELL), reply("Done.")]}))
    session_id = await _new(adapter, workspace)
    turn = asyncio.create_task(adapter.prompt(session_id, [text_block("work")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    keys = adapter.session(session_id)._agent.event_manager.keys()

    text = await _say(adapter, client, session_id, "/compact")

    assert text == (
        "/compact runs between turns and a turn is running. Send /compact again when it ends.\n\n"
    )
    assert adapter.session(session_id)._agent.event_manager.keys() == keys
    block.set()
    assert (await asyncio.wait_for(turn, TIMEOUT)).stop_reason == "end_turn"


async def test_a_compacted_session_loads_with_its_summary(atom_adapter, workspace, client):
    adapter = await atom_adapter(
        [reply("First answer."), _summary("SUMMARY: we said hello.")], [reply("Resumed.")]
    )
    session_id = await _new(adapter, workspace)
    await _say(adapter, client, session_id, "hello")
    assert (await _say(adapter, client, session_id, "/compact")).startswith("Compacted ")
    await adapter.close_session(session_id)

    await adapter.load_session(str(workspace), session_id)
    assert await _say(adapter, client, session_id, "again") == "Resumed.\n\n"
    resumed = adapter.test_models.made[-1]
    sent = str(resumed.calls[0].messages)
    assert "SUMMARY: we said hello." in sent
    assert "First answer." not in sent
