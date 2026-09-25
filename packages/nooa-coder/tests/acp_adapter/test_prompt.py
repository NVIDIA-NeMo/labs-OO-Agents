# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""session/prompt and session/cancel over the Session: stop reasons, steer, commands."""

import asyncio

import pytest
from acp import RequestError, resource_link_block, text_block
from acp.schema import AgentMessageChunk, ImageContentBlock, ToolCallProgress, ToolCallStart
from coder_test_agents import (
    BLOCKING_CELL,
    CommandAgent,
    ScriptedModels,
    cell,
    fresh_events,
    reply,
    wait_on,
)

TIMEOUT = 30


async def _new(adapter, workspace):
    return (await adapter.new_session(str(workspace))).session_id


async def _prompt(adapter, session_id, text):
    return await asyncio.wait_for(adapter.prompt(session_id, [text_block(text)]), TIMEOUT)


async def test_a_done_turn_ends_the_prompt_after_its_messages(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels({None: [reply("Hello there.")]}))
    session_id = await _new(adapter, workspace)
    response = await _prompt(adapter, session_id, "hi")
    client.log.append(("response", "prompt", response))
    assert response.stop_reason == "end_turn"
    kinds = [
        entry[0]
        for entry in client.log
        if entry[0] != "update" or isinstance(entry[2], AgentMessageChunk)
    ]
    assert kinds[-2:] == ["update", "response"]
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Hello there.\n\n"


async def test_prompt_text_includes_resource_links(make_adapter, workspace):
    models = ScriptedModels({None: [reply("ok")]})
    adapter = await make_adapter(models)
    session_id = await _new(adapter, workspace)
    await asyncio.wait_for(
        adapter.prompt(
            session_id,
            [text_block("read"), resource_link_block("notes", "file:///tmp/notes.md")],
        ),
        TIMEOUT,
    )
    assert "Resource notes: file:///tmp/notes.md" in str(models.llms[None].calls[0].messages)


async def test_prompt_rejects_empty_text_and_unsupported_blocks(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels())
    session_id = await _new(adapter, workspace)
    for blocks in (
        [text_block("   ")],
        [ImageContentBlock(type="image", data="AAAA", mime_type="image/png")],
    ):
        with pytest.raises(RequestError) as caught:
            await adapter.prompt(session_id, blocks)
        assert caught.value.code == -32602


async def test_prompt_on_an_unknown_session_is_resource_not_found(make_adapter):
    adapter = await make_adapter(ScriptedModels())
    with pytest.raises(RequestError) as caught:
        await adapter.prompt("no-such-session", [text_block("hi")])
    assert caught.value.code == -32002


async def test_a_waiting_turn_keeps_the_prompt_open_until_a_later_turn_ends(
    make_adapter, workspace
):
    models = ScriptedModels({None: [wait_on("jobs"), reply("All done.")]})
    adapter = await make_adapter(models)
    session_id = await _new(adapter, workspace)
    first = asyncio.create_task(adapter.prompt(session_id, [text_block("start the job")]))
    session = adapter.session(session_id)
    await asyncio.wait_for(
        _until(lambda: len(models.llms[None].calls) == 1 and session.info.status == "idle"),
        TIMEOUT,
    )
    await asyncio.sleep(0.05)
    assert not first.done()
    second = await _prompt(adapter, session_id, "is it finished?")
    assert second.stop_reason == "end_turn"
    assert (await asyncio.wait_for(first, TIMEOUT)).stop_reason == "end_turn"


async def _until(predicate) -> None:
    while not predicate():
        await asyncio.sleep(0.01)


async def test_a_second_prompt_during_a_turn_steers_it_and_both_return_together(
    make_adapter, workspace
):
    started, block = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), reply("Did both.")]})
    adapter = await make_adapter(models)
    session_id = await _new(adapter, workspace)
    first = asyncio.create_task(adapter.prompt(session_id, [text_block("do the first thing")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    second = asyncio.create_task(adapter.prompt(session_id, [text_block("also do the second")]))
    await asyncio.sleep(0.05)
    block.set()
    responses = await asyncio.wait_for(asyncio.gather(first, second), TIMEOUT)
    assert [response.stop_reason for response in responses] == ["end_turn", "end_turn"]
    # The running turn's next model call saw the steer.
    assert "also do the second" in str(models.llms[None].calls[1].messages)
    assert len(models.llms[None].calls) == 2


async def test_cancel_closes_the_cards_before_the_prompt_answers_cancelled(
    make_adapter, workspace, client
):
    started, _block = fresh_events()
    adapter = await make_adapter(ScriptedModels({None: [cell(BLOCKING_CELL)]}))
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("wait forever")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await client.wait_for(lambda: client.updates(session_id, ToolCallStart))
    await adapter.cancel(session_id)
    response = await asyncio.wait_for(prompt, TIMEOUT)
    client.log.append(("response", "prompt", response))

    assert response.stop_reason == "cancelled"
    [started_card] = client.updates(session_id, ToolCallStart)
    failed = [u for u in client.updates(session_id, ToolCallProgress) if u.status == "failed"]
    assert [(u.tool_call_id, u.title) for u in failed] == [(started_card.tool_call_id, "Cancelled")]
    assert "cell started" in str(failed[0].content)  # partial output kept
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Stopped at your request.\n\n"
    order = [
        entry[0] if entry[0] == "response" else type(entry[2]).__name__ for entry in client.log
    ]
    assert order.index("ToolCallProgress") < order.index("response")
    assert order.index("AgentMessageChunk") < order.index("response")


async def test_stop_also_stops_a_prompt_that_steered_the_turn(make_adapter, workspace, client):
    """A steer the model has not seen yet must not run as a new turn after Stop."""
    started, _block = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), reply("Should not run.")]})
    adapter = await make_adapter(models)
    session_id = await _new(adapter, workspace)
    first = asyncio.create_task(adapter.prompt(session_id, [text_block("do the first thing")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    second = asyncio.create_task(adapter.prompt(session_id, [text_block("also do this")]))
    await asyncio.sleep(0.05)
    await adapter.cancel(session_id)
    responses = await asyncio.wait_for(asyncio.gather(first, second), TIMEOUT)
    assert [response.stop_reason for response in responses] == ["cancelled", "cancelled"]
    await asyncio.sleep(0.2)  # a re-admitted steer would start a turn now
    session = adapter.session(session_id)
    assert session.info.status == "idle"
    assert len(models.llms[None].calls) == 1
    entries = session.transcript()
    assert [entry.role for entry in entries][-1] == "cancelled"
    assert "Should not run." not in client.texts(AgentMessageChunk, session_id)


async def test_cancel_without_a_turn_is_harmless_and_the_session_goes_on(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels({None: [reply("Still here.")]}))
    session_id = await _new(adapter, workspace)
    await adapter.cancel(session_id)
    await adapter.cancel("no-such-session")
    assert (await _prompt(adapter, session_id, "hello")).stop_reason == "end_turn"


async def test_the_session_is_usable_after_a_cancel(make_adapter, workspace, client):
    started, _block = fresh_events()
    adapter = await make_adapter(
        ScriptedModels({None: [cell(BLOCKING_CELL), reply("Back again.")]})
    )
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("wait")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await adapter.cancel(session_id)
    assert (await asyncio.wait_for(prompt, TIMEOUT)).stop_reason == "cancelled"
    assert (await _prompt(adapter, session_id, "again")).stop_reason == "end_turn"
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Back again.\n\n"


# ---- errors from the turn ----------------------------------------------------


async def test_a_failed_turn_is_an_internal_error_with_its_message(
    make_adapter, workspace, file_spec
):
    adapter = await make_adapter(ScriptedModels(), agent_spec=file_spec("BrokenAgent"))
    session_id = await _new(adapter, workspace)
    with pytest.raises(RequestError) as caught:
        await _prompt(adapter, session_id, "go")
    assert caught.value.code == -32603
    assert "the turn broke" in str(caught.value)


@pytest.mark.parametrize(
    ("text", "stop_reason"), [("tokens", "max_tokens"), ("iterations", "max_turn_requests")]
)
async def test_generation_limits_map_to_stop_reasons(
    make_adapter, workspace, file_spec, text, stop_reason
):
    adapter = await make_adapter(ScriptedModels(), agent_spec=file_spec("LimitAgent"))
    session_id = await _new(adapter, workspace)
    assert (await _prompt(adapter, session_id, text)).stop_reason == stop_reason


@pytest.mark.parametrize(
    ("text", "expected"), [("other", "provider rejected"), ("retries", "max_retries")]
)
async def test_another_generation_error_is_an_internal_error(
    make_adapter, workspace, file_spec, text, expected
):
    """Repeated errors (max_retries) are a failure to report, not a turn-request limit."""
    adapter = await make_adapter(ScriptedModels(), agent_spec=file_spec("LimitAgent"))
    session_id = await _new(adapter, workspace)
    with pytest.raises(RequestError) as caught:
        await _prompt(adapter, session_id, text)
    assert caught.value.code == -32603
    assert expected in str(caught.value)


# ---- slash commands ----------------------------------------------------------


async def test_a_command_with_text_output_answers_without_a_turn(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels(), agent_spec="coder_test_agents:CommandAgent")
    session_id = await _new(adapter, workspace)
    response = await _prompt(adapter, session_id, "/model fast")
    assert response.stop_reason == "end_turn"
    assert client.texts(AgentMessageChunk, session_id)[-1] == "model is fast\n"
    agent = adapter.session(session_id).agent
    assert isinstance(agent, CommandAgent)
    assert agent.slash_commands.invoked == [("model", "fast")]


async def test_a_command_meant_for_the_agent_runs_a_turn(coder_adapter, workspace, client):
    skills = workspace / "skills" / "review"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        "---\nname: review-this\ndescription: Review\nargument-hint: [target]\n---\n"
        "Review $ARGUMENTS"
    )
    (workspace / ".nooa").mkdir()
    (workspace / ".nooa" / "settings.yaml").write_text(
        f"coding:\n  additional_skills_dirs:\n    - {workspace / 'skills'}\n"
    )
    adapter = await coder_adapter([reply("Reviewed.")])
    session_id = await _new(adapter, workspace)
    response = await _prompt(adapter, session_id, '/review-this "two words"')
    assert response.stop_reason == "end_turn"
    [llm] = adapter.test_models.made
    assert "Review two words" in str(llm.calls[0].messages)
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Reviewed.\n\n"


async def test_host_controls_answer_as_text(coder_adapter, workspace, client):
    adapter = await coder_adapter([])
    session_id = await _new(adapter, workspace)
    assert (await _prompt(adapter, session_id, "/skills list")).stop_reason == "end_turn"
    assert "Skills" in client.texts(AgentMessageChunk, session_id)[-1]


async def test_trace_url_answers_with_the_viewer_url(coder_adapter, workspace, client, monkeypatch):
    import nooa.tracing

    monkeypatch.setattr(nooa.tracing, "get_session", lambda: "trace-1")
    monkeypatch.setenv("OTLP_ENDPOINT", "http://viewer:5001/v1/traces")
    adapter = await coder_adapter([])
    session_id = await _new(adapter, workspace)
    assert (await _prompt(adapter, session_id, "/trace-url")).stop_reason == "end_turn"
    assert client.texts(AgentMessageChunk, session_id)[-1] == (
        "http://viewer:5001/traces/view?session_id=trace-1"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/compact", "/compact is not available through ACP"),
        ("/connect", "nooa connect"),
    ],
)
async def test_reserved_commands_without_an_acp_form_explain_themselves(
    coder_adapter, workspace, client, text, expected
):
    adapter = await coder_adapter([])
    session_id = await _new(adapter, workspace)
    assert (await _prompt(adapter, session_id, text)).stop_reason == "end_turn"
    assert expected in client.texts(AgentMessageChunk, session_id)[-1]


async def test_an_unknown_slash_command_is_an_ordinary_prompt(make_adapter, workspace):
    models = ScriptedModels({None: [reply("That is a path.")]})
    adapter = await make_adapter(models)
    session_id = await _new(adapter, workspace)
    assert (await _prompt(adapter, session_id, "/usr/bin is a directory")).stop_reason == (
        "end_turn"
    )
    assert "/usr/bin is a directory" in str(models.llms[None].calls[0].messages)


async def test_a_failing_command_reports_the_failure(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels(), agent_spec="coder_test_agents:CommandAgent")
    session_id = await _new(adapter, workspace)
    agent = adapter.session(session_id).agent

    async def broken(name, raw_args):
        raise RuntimeError("command exploded")

    agent.slash_commands.invoke = broken
    assert (await _prompt(adapter, session_id, "/model x")).stop_reason == "end_turn"
    assert client.texts(AgentMessageChunk, session_id)[-1] == "/model failed: command exploded\n\n"


async def test_a_command_for_the_agent_with_no_output_says_so(make_adapter, workspace, client):
    """Nothing to send the agent: the client is told, not left with a silent end_turn."""
    from coder_test_agents import _CommandOutput

    adapter = await make_adapter(ScriptedModels(), agent_spec="coder_test_agents:CommandAgent")
    session_id = await _new(adapter, workspace)
    agent = adapter.registry.get(session_id).agent

    async def empty(name, raw_args):
        return _CommandOutput("", None, True)

    agent.slash_commands.invoke = empty
    assert (await _prompt(adapter, session_id, "/model")).stop_reason == "end_turn"
    assert client.texts(AgentMessageChunk, session_id)[-1] == (
        "/model produced no output, so nothing was sent to the agent.\n\n"
    )


# ---- titles ------------------------------------------------------------------


async def test_the_first_prompt_asks_the_agent_for_a_title(coder_adapter, workspace, client):
    from acp.schema import SessionInfoUpdate

    adapter = await coder_adapter(
        [
            cell(
                "await self.rename_session('Parser fix')\n"
                "self.message('On it.')\n"
                "return_result(Done(explanation='titled'))"
            ),
            reply("Second."),
        ]
    )
    session_id = await _new(adapter, workspace)
    await _prompt(adapter, session_id, "fix the parser")
    await _prompt(adapter, session_id, "and the lexer")
    [llm] = adapter.test_models.made
    assert "[session-title]" in str(llm.calls[0].messages)
    assert "[session-title]" not in str(llm.calls[1].messages[-3:])
    titles = [u.title for u in client.updates(session_id, SessionInfoUpdate) if u.title]
    assert titles == ["Parser fix"]
    # The request is housekeeping: not part of the conversation.
    entries = adapter.session(session_id).transcript()
    assert [e.content for e in entries if e.role == "user"] == ["fix the parser", "and the lexer"]


async def test_a_session_that_has_a_title_is_not_asked_again(make_adapter, workspace):
    models = ScriptedModels({None: [reply("Hi.")]})
    adapter = await make_adapter(models, agent_spec="coder_test_agents:EchoAgent")
    session_id = await _new(adapter, workspace)
    await adapter.session(session_id).set_title("Chosen", user_set=True)
    await _prompt(adapter, session_id, "hello")
    assert "[session-title]" not in str(models.llms[None].calls[0].messages)
