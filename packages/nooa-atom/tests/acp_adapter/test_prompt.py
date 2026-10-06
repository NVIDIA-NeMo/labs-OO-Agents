# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""session/prompt and session/cancel over the Session: stop reasons, queueing, commands."""

import asyncio
from typing import Any

import pytest
from acp import RequestError, resource_link_block, text_block
from acp.schema import (
    AgentMessageChunk,
    ImageContentBlock,
    Implementation,
    ToolCallProgress,
    ToolCallStart,
)
from atom_test_agents import (
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
    adapter = await make_adapter(ScriptedModels({None: [reply("Hello there.\n")]}))
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


async def test_a_second_prompt_during_a_turn_is_queued_for_its_own_turn(make_adapter, workspace):
    """A queue is a queue: the running turn sees the message pending, not injected."""
    started, block = fresh_events()
    models = ScriptedModels(
        {None: [cell(BLOCKING_CELL), reply("Did the first."), reply("Did the second.")]}
    )
    adapter = await make_adapter(models)
    session_id = await _new(adapter, workspace)
    finished: list[str] = []

    async def run(text: str, label: str) -> Any:
        response = await adapter.prompt(session_id, [text_block(text)])
        finished.append(label)
        return response

    first = asyncio.create_task(run("do the first thing", "first"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    second = asyncio.create_task(run("also do the second", "second"))
    await asyncio.sleep(0.05)
    block.set()
    responses = await asyncio.wait_for(asyncio.gather(first, second), TIMEOUT)
    assert [response.stop_reason for response in responses] == ["end_turn", "end_turn"]
    assert finished == ["first", "second"]
    calls = models.llms[None].calls
    assert len(calls) == 3
    # The running turn's next model call lists it as pending, once, in the queues block.
    during = str(calls[1].messages)
    assert "user_messages: 1 pending" in during
    assert during.count("also do the second") == 1
    assert "also do the second" in str(calls[2].messages)


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
    assert [u.tool_call_id for u in failed] == [started_card.tool_call_id]
    assert failed[0].title == started_card.title + " (cancelled)"
    assert "cell started" in str(failed[0].content)  # partial output kept
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Stopped at your request.\n\n"
    order = [
        entry[0] if entry[0] == "response" else type(entry[2]).__name__ for entry in client.log
    ]
    assert order.index("ToolCallProgress") < order.index("response")
    assert order.index("AgentMessageChunk") < order.index("response")


async def test_stop_answers_a_queued_prompt_cancelled_and_withdraws_it(
    make_adapter, workspace, client
):
    """A prompt request must be answered: Stop cancels it and its item never runs."""
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
    await asyncio.sleep(0.2)  # a queued item left behind would start a turn now
    session = adapter.session(session_id)
    assert session.info.status == "idle"
    assert len(models.llms[None].calls) == 1
    entries = session.transcript()
    assert [entry.role for entry in entries][-1] == "cancelled"
    assert "Should not run.\n" not in client.texts(AgentMessageChunk, session_id)


async def test_cancel_without_a_turn_is_harmless_and_the_session_goes_on(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels({None: [reply("Still here.")]}))
    session_id = await _new(adapter, workspace)
    await adapter.cancel(session_id)
    await adapter.cancel("no-such-session")
    assert (await _prompt(adapter, session_id, "hello")).stop_reason == "end_turn"


async def test_the_session_is_usable_after_a_cancel(make_adapter, workspace, client):
    started, _block = fresh_events()
    adapter = await make_adapter(
        ScriptedModels({None: [cell(BLOCKING_CELL), reply("Back again.\n")]})
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
    adapter = await make_adapter(ScriptedModels(), agent_spec="atom_test_agents:CommandAgent")
    session_id = await _new(adapter, workspace)
    response = await _prompt(adapter, session_id, "/model fast")
    assert response.stop_reason == "end_turn"
    assert client.texts(AgentMessageChunk, session_id)[-1] == "model is fast\n\n"
    agent = adapter.session(session_id)._agent
    assert isinstance(agent, CommandAgent)
    assert agent.slash_commands.invoked == [("model", "fast")]


async def test_a_command_meant_for_the_agent_runs_a_turn(atom_adapter, workspace, client):
    skills = workspace / "skills" / "review"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        "---\nname: review-this\ndescription: Review\nargument-hint: [target]\n---\n"
        "Review $ARGUMENTS"
    )
    (workspace / ".nooa").mkdir()
    (workspace / ".nooa" / "settings.yaml").write_text(
        f"atom:\n  additional_skills_dirs:\n    - {workspace / 'skills'}\n"
    )
    adapter = await atom_adapter([reply("Reviewed.\n")])
    session_id = await _new(adapter, workspace)
    response = await _prompt(adapter, session_id, '/review-this "two words"')
    assert response.stop_reason == "end_turn"
    [llm] = adapter.test_models.made
    assert "Review two words" in str(llm.calls[0].messages)
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Reviewed.\n\n"


async def test_host_controls_answer_as_text(atom_adapter, workspace, client):
    adapter = await atom_adapter([])
    session_id = await _new(adapter, workspace)
    assert (await _prompt(adapter, session_id, "/skills list")).stop_reason == "end_turn"
    assert "Skills" in client.texts(AgentMessageChunk, session_id)[-1]


_TRACE_CELL = (
    "from nooa.tracing import get_session\n"
    "self.message(str(get_session()))\n"
    "return_result(Done(explanation='traced'))"
)


async def test_each_session_is_its_own_trace_session(atom_adapter, workspace, client, monkeypatch):
    """The adapter names the trace session after the ACP session: turns and /trace-url."""
    import nooa.tracing

    monkeypatch.setenv("OTLP_ENDPOINT", "http://viewer:5001/v1/traces")
    adapter = await atom_adapter([cell(_TRACE_CELL)], [cell(_TRACE_CELL)])
    first = await _new(adapter, workspace)
    assert nooa.tracing.get_session() == first
    second = await _new(adapter, workspace)
    for session_id in (first, second):
        await _prompt(adapter, session_id, "which trace?")
        assert client.texts(AgentMessageChunk, session_id)[-1] == session_id + "\n\n"
        await _prompt(adapter, session_id, "/trace-url")
        assert client.texts(AgentMessageChunk, session_id)[-1] == (
            f"Trace viewer:\n```text\nhttp://viewer:5001/traces/view?session_id={session_id}\n```\n\n"
        )


_HOOKS_CELL = (
    "from nooa.runtime.hooks import get_hooks\n"
    "self.message(type(get_hooks()).__name__)\n"
    "return_result(Done(explanation='hooked'))"
)


async def test_turns_run_with_the_tracing_hooks_registered(
    atom_adapter, workspace, client, monkeypatch
):
    """The loop runs in a fresh context; the hooks tracing registered must reach it."""
    import nooa.tracing
    from nooa.runtime.hooks import set_hooks

    class Hooks:
        pass

    monkeypatch.setattr(nooa.tracing, "_hooks", Hooks())
    set_hooks(None)
    adapter = await atom_adapter([cell(_HOOKS_CELL)])
    session_id = await _new(adapter, workspace)
    await _prompt(adapter, session_id, "hooked?")
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Hooks\n\n"


async def test_usage_answers_with_the_token_totals(atom_adapter, workspace, client):
    from nooa.unifiedllm import LLMUsage

    adapter = await atom_adapter(
        [reply("Hi.", usage=LLMUsage(input_tokens=11, output_tokens=2, cached_input_tokens=7))]
    )
    session_id = await _new(adapter, workspace)
    await _prompt(adapter, session_id, "hello")
    assert (await _prompt(adapter, session_id, "/usage")).stop_reason == "end_turn"
    text = client.texts(AgentMessageChunk, session_id)[-1]
    assert "This session" in text
    assert "Cached input tokens (cache reads)" in text and "Turns: 1" in text


async def test_trace_url_answers_with_the_viewer_url(atom_adapter, workspace, client, monkeypatch):
    import nooa.tracing

    monkeypatch.setattr(nooa.tracing, "get_session", lambda: "trace-1")
    monkeypatch.setenv("OTLP_ENDPOINT", "http://viewer:5001/v1/traces")
    adapter = await atom_adapter([])
    session_id = await _new(adapter, workspace)
    assert (await _prompt(adapter, session_id, "/trace-url")).stop_reason == "end_turn"
    assert client.texts(AgentMessageChunk, session_id)[-1] == (
        "Trace viewer:\n```text\nhttp://viewer:5001/traces/view?session_id=trace-1\n```\n\n"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/compact", "/compact is not available through ACP"),
        ("/connect", "nooa connect"),
    ],
)
async def test_reserved_commands_without_an_acp_form_explain_themselves(
    atom_adapter, workspace, client, text, expected
):
    adapter = await atom_adapter([])
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
    adapter = await make_adapter(ScriptedModels(), agent_spec="atom_test_agents:CommandAgent")
    session_id = await _new(adapter, workspace)
    agent = adapter.session(session_id)._agent

    async def broken(name, raw_args):
        raise RuntimeError("command exploded")

    agent.slash_commands.invoke = broken
    assert (await _prompt(adapter, session_id, "/model x")).stop_reason == "end_turn"
    assert client.texts(AgentMessageChunk, session_id)[-1] == "/model failed: command exploded\n\n"


async def test_a_command_for_the_agent_with_no_output_says_so(make_adapter, workspace, client):
    """Nothing to send the agent: the client is told, not left with a silent end_turn."""
    from atom_test_agents import _CommandOutput

    adapter = await make_adapter(ScriptedModels(), agent_spec="atom_test_agents:CommandAgent")
    session_id = await _new(adapter, workspace)
    agent = adapter.session(session_id)._agent

    async def empty(name, raw_args):
        return _CommandOutput("", None, True)

    agent.slash_commands.invoke = empty
    assert (await _prompt(adapter, session_id, "/model")).stop_reason == "end_turn"
    assert client.texts(AgentMessageChunk, session_id)[-1] == (
        "/model produced no output, so nothing was sent to the agent.\n\n"
    )


# ---- titles ------------------------------------------------------------------


async def test_the_first_prompt_asks_the_agent_for_a_title(atom_adapter, workspace, client):
    from acp.schema import SessionInfoUpdate

    adapter = await atom_adapter(
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
    adapter = await make_adapter(models, agent_spec="atom_test_agents:EchoAgent")
    session_id = await _new(adapter, workspace)
    await adapter.session(session_id).set_title("Chosen", user_set=True)
    await _prompt(adapter, session_id, "hello")
    assert "[session-title]" not in str(models.llms[None].calls[0].messages)


# ---- _nooa/session/inject ------------------------------------------------------


async def _inject(adapter, session_id, mode, text):
    return await adapter.ext_method(
        "nooa/session/inject",
        {"sessionId": session_id, "mode": mode, "prompt": [{"type": "text", "text": text}]},
    )


async def test_inject_is_advertised_for_the_router_too():
    from acp import PROTOCOL_VERSION
    from nooa_atom.acp.server import initialize_response

    meta = initialize_response(PROTOCOL_VERSION).agent_capabilities.field_meta
    assert meta == {
        "dev.nooa/inject": {"queue": {}, "steer": {}, "revoke": {}},
        "poolside/session_steer": True,
    }


async def test_inject_queue_starts_a_turn_when_idle(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels({None: [reply("Got it.")]}))
    session_id = await _new(adapter, workspace)
    answer = await _inject(adapter, session_id, "queue", "note this")
    assert answer["delivered"] == "queued" and answer["messageId"]
    await client.wait_for(lambda: "Got it.\n\n" in client.texts(AgentMessageChunk, session_id))


async def test_inject_steer_reaches_the_running_turns_next_model_call(make_adapter, workspace):
    started, block = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), reply("Adjusted.")]})
    adapter = await make_adapter(models)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("start")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    answer = await adapter.ext_method(
        "nooa/session/inject", {"sessionId": session_id, "mode": "steer", "text": "use tabs"}
    )
    assert answer["delivered"] == "steered"
    block.set()
    assert (await asyncio.wait_for(prompt, TIMEOUT)).stop_reason == "end_turn"
    assert "use tabs" in str(models.llms[None].calls[1].messages)
    assert len(models.llms[None].calls) == 2


async def test_revoke_takes_back_a_queued_inject(make_adapter, workspace):
    started, block = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), reply("Only the first.")]})
    adapter = await make_adapter(models)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("start")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    answer = await _inject(adapter, session_id, "queue", "never mind this")
    revoke = {"sessionId": session_id, "messageId": answer["messageId"]}
    assert await adapter.ext_method("nooa/session/revoke_inject", revoke) == {"revoked": True}
    assert await adapter.ext_method("nooa/session/revoke_inject", revoke) == {"revoked": False}
    block.set()
    assert (await asyncio.wait_for(prompt, TIMEOUT)).stop_reason == "end_turn"
    await asyncio.sleep(0.2)  # a queued item would start a turn now
    assert len(models.llms[None].calls) == 2


async def test_a_queued_inject_survives_cancel(make_adapter, workspace, client):
    started, _block = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), reply("Handled the note.")]})
    adapter = await make_adapter(models)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("start")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await _inject(adapter, session_id, "queue", "a note for later")
    await adapter.cancel(session_id)
    assert (await asyncio.wait_for(prompt, TIMEOUT)).stop_reason == "cancelled"
    await client.wait_for(
        lambda: "Handled the note.\n\n" in client.texts(AgentMessageChunk, session_id)
    )


async def test_inject_checks_its_params(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels())
    session_id = await _new(adapter, workspace)
    for params in (
        {"sessionId": session_id, "mode": "shout", "text": "x"},
        {"sessionId": session_id, "mode": "queue"},
    ):
        with pytest.raises(RequestError):
            await adapter.ext_method("nooa/session/inject", params)
    with pytest.raises(RequestError):
        await _inject(adapter, "no-such-session", "queue", "x")


# ---- _poolside/session_steer ---------------------------------------------------


def _steer_params(session_id, text, input_id="steer-1"):
    return {
        "sessionId": session_id,
        "inputId": input_id,
        "prompt": [{"type": "text", "text": text}],
    }


async def test_a_pool_steer_is_queued_for_the_next_turn(make_adapter, workspace, client):
    started, block = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), reply("Adjusted."), reply("Tabs.")]})
    adapter = await make_adapter(models)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("start")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    steer = await _pool_steer(adapter, session_id, "use tabs", "steer-2")
    await asyncio.sleep(0.05)
    assert not steer.done()  # answered when a turn takes the message
    block.set()
    assert await asyncio.wait_for(steer, TIMEOUT) == {"inputId": "steer-2"}
    assert (await asyncio.wait_for(prompt, TIMEOUT)).stop_reason == "end_turn"
    await client.wait_for(lambda: "Tabs.\n\n" in client.texts(AgentMessageChunk, session_id))

    calls = models.llms[None].calls
    assert len(calls) == 3
    running, following = (
        "".join(str(message.get("content")) for message in call.messages) for call in calls[1:]
    )
    # The running turn got no steer, only the pending message in its queues block.
    assert "Notification(" not in running
    assert "{'user_messages': ['use tabs']}" not in running
    assert "{'user_messages': ['use tabs']}" in following  # the next turn's input
    session = adapter.session(session_id)
    assert [entry.content for entry in session.transcript() if entry.role == "user"] == [
        "start",
        "use tabs",
    ]


async def test_a_pool_steer_without_a_turn_is_queued(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels({None: [reply("Got it.")]}))
    session_id = await _new(adapter, workspace)
    answer = await adapter.ext_method("poolside/session_steer", _steer_params(session_id, "hi"))
    assert answer == {"inputId": "steer-1"}
    await client.wait_for(lambda: "Got it.\n\n" in client.texts(AgentMessageChunk, session_id))


async def test_a_pool_steer_checks_its_params(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels())
    session_id = await _new(adapter, workspace)
    with pytest.raises(RequestError) as unknown:
        await adapter.ext_method("poolside/session_steer", _steer_params("no-such-session", "x"))
    assert unknown.value.code == RequestError.resource_not_found("x").code
    for params in (
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "x"}]},  # no inputId
        {"sessionId": session_id, "inputId": "steer-1"},  # no prompt
        _steer_params(session_id, "/usage"),  # a command is not steered
    ):
        with pytest.raises(RequestError) as invalid:
            await adapter.ext_method("poolside/session_steer", params)
        assert invalid.value.code == RequestError.invalid_params().code


# ---- a Pool steer during a prompt: the prompt stays open until it is handled ----

POOL = Implementation(name="pool", version="1.0.16")


async def _pool_steer(adapter, session_id, text, input_id="steer-1"):
    """Send a Pool steer; return its request as a task once the message is admitted.

    The request is answered when a turn takes the message (or it is
    withdrawn), so a test awaits the task where that should have happened.
    """
    admitted = asyncio.Event()

    def on_update(update):
        if getattr(update, "kind", "") == "item_admitted" and getattr(update, "text", "") == text:
            admitted.set()

    unsubscribe = adapter.session(session_id).subscribe(on_update)
    try:
        request = asyncio.create_task(
            adapter.ext_method("poolside/session_steer", _steer_params(session_id, text, input_id))
        )
        await asyncio.wait_for(admitted.wait(), TIMEOUT)
    finally:
        unsubscribe()
    return request


def _order(client, session_id):
    """The log as labels: agent message texts, ``ext`` requests and responses."""
    labels = []
    for entry in client.log:
        if entry[0] == "update" and entry[1] == session_id:
            if isinstance(entry[2], AgentMessageChunk):
                labels.append(entry[2].content.text)
        elif entry[0] in ("ext", "response"):
            labels.append(entry[0])
    return labels


async def test_a_pool_steer_keeps_the_prompt_open_until_its_turn_ends(
    make_adapter, workspace, client
):
    """Pool closes the turn at end_turn: the steered message's reply must come before it."""
    started, first_block = fresh_events()
    models = ScriptedModels(
        {None: [cell(BLOCKING_CELL), reply("First."), cell(BLOCKING_CELL), reply("Second.")]}
    )
    adapter = await make_adapter(models, client_info=POOL)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("first")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    steer = await _pool_steer(adapter, session_id, "queued second")
    started, second_block = fresh_events()
    first_block.set()
    await asyncio.wait_for(started.wait(), TIMEOUT)  # the steered message's turn runs
    # Answered once the turn took the message, before that turn ends.
    assert await asyncio.wait_for(steer, TIMEOUT) == {"inputId": "steer-1"}
    await asyncio.sleep(0.05)
    assert not prompt.done()
    second_block.set()
    response = await asyncio.wait_for(prompt, TIMEOUT)
    client.log.append(("response", "prompt", response))

    assert response.stop_reason == "end_turn"
    assert _order(client, session_id)[-3:] == ["First.\n\n", "Second.\n\n", "response"]
    assert len(models.llms[None].calls) == 4


async def test_pool_steers_are_handled_in_order_within_the_prompt(make_adapter, workspace, client):
    """A steer that arrives while an earlier steer's turn runs is picked up too."""
    started, first_block = fresh_events()
    models = ScriptedModels(
        {
            None: [
                cell(BLOCKING_CELL),
                reply("First."),
                cell(BLOCKING_CELL),
                reply("A."),
                reply("B."),
            ]
        }
    )
    adapter = await make_adapter(models, client_info=POOL)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("first")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await _pool_steer(adapter, session_id, "message a", "steer-a")
    started, second_block = fresh_events()  # the steer's turn blocks on these
    first_block.set()
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await _pool_steer(adapter, session_id, "message b", "steer-b")
    second_block.set()
    response = await asyncio.wait_for(prompt, TIMEOUT)
    client.log.append(("response", "prompt", response))

    assert response.stop_reason == "end_turn"
    assert _order(client, session_id)[-4:] == ["First.\n\n", "A.\n\n", "B.\n\n", "response"]
    calls = models.llms[None].calls
    assert len(calls) == 5
    assert "message a" in str(calls[2].messages) and "message b" not in str(calls[2].messages)
    assert "message b" in str(calls[4].messages)


async def test_pool_steers_taken_by_one_turn_end_the_prompt_with_it(
    make_adapter, workspace, client
):
    """Two steers queued during a turn are one turn's input; each resolves with it."""
    started, block = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), reply("First."), reply("Both.")]})
    adapter = await make_adapter(models, client_info=POOL)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("first")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await _pool_steer(adapter, session_id, "message a", "steer-a")
    await _pool_steer(adapter, session_id, "message b", "steer-b")
    block.set()
    response = await asyncio.wait_for(prompt, TIMEOUT)
    client.log.append(("response", "prompt", response))

    assert response.stop_reason == "end_turn"
    assert _order(client, session_id)[-3:] == ["First.\n\n", "Both.\n\n", "response"]
    assert len(models.llms[None].calls) == 3


async def test_a_pool_steer_the_running_turn_takes_is_done_with_it(make_adapter, workspace, client):
    """The model can take a pending message itself; its question is asked once."""
    started, block = fresh_events()
    models = ScriptedModels(
        {
            None: [
                cell(BLOCKING_CELL),
                cell(
                    "await self.queue_manager.get_channel('user_messages').get()\n"
                    "return_result(NeedInput(question='Which branch?'))"
                ),
                reply("Pushed."),
            ]
        }
    )
    client.ext_answers = [{"action": "accept", "content": {"answer": "main"}}]
    adapter = await make_adapter(models, client_info=POOL)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("first")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await _pool_steer(adapter, session_id, "push it too")
    block.set()
    response = await asyncio.wait_for(prompt, TIMEOUT)
    client.log.append(("response", "prompt", response))

    assert response.stop_reason == "end_turn"
    assert [entry[1] for entry in client.log if entry[0] == "ext"] == ["poolside/elicitation"]
    assert _order(client, session_id)[-2:] == ["Pushed.\n\n", "response"]
    await asyncio.sleep(0.1)  # nothing is left to start another turn
    assert len(models.llms[None].calls) == 3


async def test_a_pool_steers_question_is_a_pool_form_inside_the_prompt(
    make_adapter, workspace, client
):
    started, block = fresh_events()
    models = ScriptedModels(
        {
            None: [
                cell(BLOCKING_CELL),
                reply("First."),
                cell("return_result(NeedInput(question='Name the release?'))"),
                cell(
                    "[a] = notification['user_messages']\n"
                    "self.message(repr(a))\n"
                    "return_result(Done(explanation='answered'))"
                ),
            ]
        }
    )
    client.ext_answers = [{"action": "accept", "content": {"answer": "Aurora"}}]
    adapter = await make_adapter(models, client_info=POOL)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("first")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await _pool_steer(adapter, session_id, "tag a release")
    block.set()
    response = await asyncio.wait_for(prompt, TIMEOUT)
    client.log.append(("response", "prompt", response))

    assert response.stop_reason == "end_turn"
    [(_, method, params)] = [entry for entry in client.log if entry[0] == "ext"]
    assert method == "poolside/elicitation"
    assert params["message"] == "Name the release?"
    assert _order(client, session_id)[-2:] == ["'Aurora'\n\n", "response"]
    assert _order(client, session_id).index("ext") < _order(client, session_id).index("response")


async def test_cancel_withdraws_pool_steers_no_turn_took_and_says_so(
    make_adapter, workspace, client
):
    started, _block = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), reply("Should not run.")]})
    adapter = await make_adapter(models, client_info=POOL)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("first")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await _pool_steer(adapter, session_id, "message a", "steer-a")
    await _pool_steer(adapter, session_id, "message b", "steer-b")
    await adapter.cancel(session_id)
    response = await asyncio.wait_for(prompt, TIMEOUT)
    client.log.append(("response", "prompt", response))

    assert response.stop_reason == "cancelled"
    [stopped] = [
        text
        for text in _order(client, session_id)
        if text.startswith("Stopped before these messages were handled")
    ]
    assert "message a" in stopped and "message b" in stopped
    order = _order(client, session_id)
    assert order.index(stopped) < order.index("response")
    await asyncio.sleep(0.2)  # a queued message would start a turn now
    assert len(models.llms[None].calls) == 1
    session = adapter.session(session_id)
    assert session.info.status == "idle"


async def test_cancel_during_a_pool_steers_turn_ends_the_prompt_cancelled(
    make_adapter, workspace, client
):
    started, first_block = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), reply("First."), cell(BLOCKING_CELL)]})
    adapter = await make_adapter(models, client_info=POOL)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("first")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await _pool_steer(adapter, session_id, "message a")
    started, _second_block = fresh_events()
    first_block.set()
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await adapter.cancel(session_id)
    response = await asyncio.wait_for(prompt, TIMEOUT)

    assert response.stop_reason == "cancelled"
    assert not any(
        text.startswith("Stopped before these messages were handled")
        for text in client.texts(AgentMessageChunk, session_id)
    )


async def test_a_nooa_inject_does_not_hold_the_prompt_open(make_adapter, workspace, client):
    """``_nooa/session/inject`` keeps its meaning: a queued message is not the prompt's."""
    started, first_block = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), reply("First."), cell(BLOCKING_CELL)]})
    adapter = await make_adapter(models, client_info=POOL)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("first")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await _inject(adapter, session_id, "queue", "a note")
    started, second_block = fresh_events()
    first_block.set()
    # The prompt ends while the injected message's turn is still running.
    assert (await asyncio.wait_for(prompt, TIMEOUT)).stop_reason == "end_turn"
    await asyncio.wait_for(started.wait(), TIMEOUT)
    second_block.set()


async def test_a_pool_steer_from_another_client_is_still_followed(make_adapter, workspace, client):
    """Following is by method, not by client name: any client using the Pool method gets it."""
    started, block = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), reply("First."), reply("Second.")]})
    adapter = await make_adapter(models)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("first")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await _pool_steer(adapter, session_id, "queued second")
    block.set()
    response = await asyncio.wait_for(prompt, TIMEOUT)
    client.log.append(("response", "prompt", response))
    assert _order(client, session_id)[-2:] == ["Second.\n\n", "response"]


async def test_a_question_waits_while_a_pool_steer_is_queued(make_adapter, workspace, client):
    """The person's queued message comes first: its turn may answer the question."""
    started, block = fresh_events()
    models = ScriptedModels(
        {
            None: [
                cell(BLOCKING_CELL),
                cell("return_result(NeedInput(question='Which branch?'))"),
                reply("Using main."),
            ]
        }
    )
    adapter = await make_adapter(models, client_info=POOL)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("first")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await _pool_steer(adapter, session_id, "wait, use main")
    block.set()
    response = await asyncio.wait_for(prompt, TIMEOUT)
    client.log.append(("response", "prompt", response))

    assert response.stop_reason == "end_turn"
    assert [entry for entry in client.log if entry[0] in ("ext", "permission")] == []
    assert _order(client, session_id)[-3:] == ["Which branch?\n\n", "Using main.\n\n", "response"]
    calls = models.llms[None].calls
    assert len(calls) == 3
    assert "wait, use main" in str(calls[2].messages)


async def test_a_pool_steer_is_answered_while_a_form_is_open(make_adapter, workspace, client):
    client.ext_gate = asyncio.Event()
    client.ext_answers = [{"action": "accept", "content": {"answer": "Aurora"}}]
    models = ScriptedModels(
        {
            None: [
                cell("return_result(NeedInput(question='Name the release?'))"),
                reply("Noted."),
                cell(
                    "[a] = notification['user_messages']\n"
                    "self.message(repr(a))\n"
                    "return_result(Done(explanation='answered'))"
                ),
            ]
        }
    )
    adapter = await make_adapter(models, client_info=POOL)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("tag it")]))
    await client.wait_for(lambda: any(entry[0] == "ext" for entry in client.log))
    steer = await _pool_steer(adapter, session_id, "a note")
    assert await asyncio.wait_for(steer, TIMEOUT) == {"inputId": "steer-1"}
    assert not prompt.done()
    # The steered message's turn runs while the form is open (the session is idle).
    await client.wait_for(lambda: "Noted.\n\n" in client.texts(AgentMessageChunk, session_id))
    client.ext_gate.set()
    response = await asyncio.wait_for(prompt, TIMEOUT)
    client.log.append(("response", "prompt", response))

    assert response.stop_reason == "end_turn"
    order = _order(client, session_id)
    assert {"Noted.\n\n", "'Aurora'\n\n"} <= set(order[order.index("ext") :])
    assert order[-1] == "response"


# ---- Pool's input events: how Pool places a person's input in the conversation ----


def _input_events(client, session_id):
    """``(index in client.log, _meta)`` of each Pool input event sent for the session."""
    from acp.schema import SessionInfoUpdate

    return [
        (index, entry[2].field_meta)
        for index, entry in enumerate(client.log)
        if entry[0] == "update"
        and entry[1] == session_id
        and isinstance(entry[2], SessionInfoUpdate)
        and (entry[2].field_meta or {}).get("poolside/inputEventId")
    ]


def _user_item_ids(adapter, session_id):
    return {
        entry.content: entry.item_id
        for entry in adapter.session(session_id).transcript()
        if entry.role == "user"
    }


async def test_pool_is_told_when_a_turn_takes_its_steered_message(make_adapter, workspace, client):
    """Pool shows a steered message when it sees its inputId; the steer's answer comes after."""
    started, first_block = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), reply("First."), reply("Second.")]})
    adapter = await make_adapter(models, client_info=POOL)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("first")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    steer = await _pool_steer(adapter, session_id, "queued second", "steer-7")
    assert [meta for _, meta in _input_events(client, session_id)] == [
        {"poolside/inputEventId": _user_item_ids(adapter, session_id)["first"]}
    ]
    first_block.set()
    assert await asyncio.wait_for(steer, TIMEOUT) == {"inputId": "steer-7"}
    answered_at = len(client.log)
    assert (await asyncio.wait_for(prompt, TIMEOUT)).stop_reason == "end_turn"

    ids = _user_item_ids(adapter, session_id)
    events = _input_events(client, session_id)
    assert [meta for _, meta in events] == [
        {"poolside/inputEventId": ids["first"]},
        {"poolside/clientInputId": "steer-7", "poolside/inputEventId": ids["queued second"]},
    ]
    assert events[1][0] < answered_at  # the event went out before the answer
    assert "Second.\n\n" in client.texts(AgentMessageChunk, session_id)


async def test_stop_answers_a_steer_no_turn_took(make_adapter, workspace, client):
    started, _block = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL)]})
    adapter = await make_adapter(models, client_info=POOL)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("first")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    steer = await _pool_steer(adapter, session_id, "never mind")
    await adapter.cancel(session_id)
    assert await asyncio.wait_for(steer, TIMEOUT) == {"inputId": "steer-1"}
    assert (await asyncio.wait_for(prompt, TIMEOUT)).stop_reason == "cancelled"
    assert not any(
        "poolside/clientInputId" in meta for _, meta in _input_events(client, session_id)
    )


async def test_a_replay_to_pool_follows_each_user_message_with_its_input_event(
    make_adapter, workspace, client
):
    adapter = await make_adapter(ScriptedModels({None: [reply("Hi.")]}), client_info=POOL)
    session_id = await _new(adapter, workspace)
    await asyncio.wait_for(adapter.prompt(session_id, [text_block("hello")]), TIMEOUT)
    client.log.clear()
    await adapter.load_session(str(workspace), session_id)

    updates = [entry[2] for entry in client.log if entry[0] == "update" and entry[1] == session_id]
    kinds = [update.session_update for update in updates]
    user = kinds.index("user_message_chunk")
    assert kinds[user + 1] == "session_info_update"
    assert updates[user + 1].field_meta == {
        "poolside/inputEventId": _user_item_ids(adapter, session_id)["hello"]
    }


async def test_other_clients_get_no_pool_input_events(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels({None: [reply("Hi.")]}))
    session_id = await _new(adapter, workspace)
    await asyncio.wait_for(adapter.prompt(session_id, [text_block("hello")]), TIMEOUT)
    await adapter.load_session(str(workspace), session_id)
    assert _input_events(client, session_id) == []
