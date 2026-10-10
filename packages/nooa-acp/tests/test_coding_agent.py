# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the NOOA coding agent and dispatcher."""

import asyncio
from typing import Any

from nooa_acp.dispatcher import InteractiveSessionDispatcher
from nooa_cli.coding import CodingAgent

from nooa.context_blocks.events import ToolCallEvent
from nooa.events import PythonOutput
from nooa.interactive import AgentMessage, Done, NeedInput, NeedInputForm, TextQuestion, Waiting
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _completed_llm(message: str = "Finished **successfully**.") -> FakeLLMClient:
    return FakeLLMClient.with_tool_call(
        "execute_python",
        {
            "code": (
                f"self.message({message!r})\n"
                "return_result(Done(explanation='completed and verified'))"
            )
        },
    )


async def test_coding_agent_runs_through_nooa_codeact(tmp_path):
    agent = CodingAgent(llm=_completed_llm(), cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)

    result = await dispatcher.submit("inspect the repository")

    assert result is not None
    assert isinstance(result, Done)
    assert agent.cwd == tmp_path.resolve()
    assert agent.shell.session is agent.repo.session
    events = agent.event_manager.values()
    assert any(isinstance(event, AgentMessage) for event in events)
    assert any(isinstance(event, ToolCallEvent) for event in events)
    assert any(isinstance(event, PythonOutput) for event in events)
    await dispatcher.close()


class _BlockingLLM(FakeLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def acall(self, *args, **kwargs) -> LLMResponse:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _WaitingAgent(CodingAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.handle_calls = 0

    async def handle(self, notification: dict[str, list[Any]]) -> Done | Waiting:
        self.handle_calls += 1
        if self.handle_calls == 1:
            self.queue_manager.get_channel("system_messages").put("job finished")
            return Waiting(explanation="waiting for job", on=["system_messages"])
        assert notification == {"system_messages": ["job finished"]}
        return Done(explanation="job finished")


class _BackgroundAgent(CodingAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.job_started = asyncio.Event()
        self.job: Any = None

    async def handle(self, notification: dict[str, list[Any]]) -> Waiting:
        async def background_job() -> None:
            self.job_started.set()
            await asyncio.Event().wait()

        self.job = self.queue_manager.spawn(background_job(), channel="system_messages")
        return Waiting(explanation="waiting for job", on=["system_messages"])


class _RestartableAgent(CodingAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.started = asyncio.Event()
        self.handle_calls = 0

    async def handle(self, notification: dict[str, list[Any]]) -> Done:
        self.handle_calls += 1
        if self.handle_calls == 1:
            self.started.set()
            await asyncio.Event().wait()
        return Done(explanation="second prompt completed")


async def test_dispatcher_cancels_active_nooa_turn(tmp_path):
    llm = _BlockingLLM()
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)
    prompt_task = asyncio.create_task(dispatcher.submit("wait forever"))
    await asyncio.wait_for(llm.started.wait(), timeout=2)

    assert await dispatcher.cancel() is True
    assert await asyncio.wait_for(prompt_task, timeout=2) is None
    assert dispatcher.active is False
    await dispatcher.close()


async def test_dispatcher_accepts_another_prompt_after_cancellation(tmp_path):
    agent = _RestartableAgent(llm=FakeLLMClient(), cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)
    first = asyncio.create_task(dispatcher.submit("cancel this"))
    await asyncio.wait_for(agent.started.wait(), timeout=1)

    assert await dispatcher.cancel() is True
    assert await asyncio.wait_for(first, timeout=1) is None
    result = await asyncio.wait_for(dispatcher.submit("try again"), timeout=1)

    assert result is not None
    assert isinstance(result, Done)
    assert agent.handle_calls == 2
    await dispatcher.close()


async def test_dispatcher_resumes_after_wait_notification(tmp_path):
    agent = _WaitingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)

    result = await dispatcher.submit("wait for the job")

    assert result is not None
    assert isinstance(result, Done)
    assert agent.handle_calls == 2
    await dispatcher.close()


async def test_dispatcher_cancels_background_jobs(tmp_path):
    agent = _BackgroundAgent(llm=FakeLLMClient(), cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)
    prompt_task = asyncio.create_task(dispatcher.submit("start a background job"))
    await asyncio.wait_for(agent.job_started.wait(), timeout=1)

    assert await dispatcher.cancel() is True
    assert await asyncio.wait_for(prompt_task, timeout=1) is None
    assert agent.job is not None
    assert agent.job.state == "cancelled"
    await dispatcher.close()


class _TypedResultAgent(CodingAgent):
    """Returns results that carry text for the person: Waiting first, then Done."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.handle_calls = 0

    async def handle(self, notification: dict[str, list[Any]]) -> Done | Waiting:
        self.handle_calls += 1
        if self.handle_calls == 1:
            self.queue_manager.get_channel("system_messages").put("job finished")
            return Waiting(
                explanation="waiting for job",
                message="Waiting for the job.",
                on=["system_messages"],
            )
        return Done(message="All done.", explanation="job finished")


async def test_dispatcher_accepts_the_typed_turn_results(tmp_path):
    agent = _TypedResultAgent(llm=FakeLLMClient(), cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)

    result = await dispatcher.submit("wait for the job")

    assert result == Done(message="All done.", explanation="job finished")
    assert agent.handle_calls == 2
    await dispatcher.close()


def _agent_messages(agent: CodingAgent) -> list[str]:
    return [e.content for e in agent.event_manager.values() if isinstance(e, AgentMessage)]


async def test_dispatcher_shows_the_messages_of_typed_turn_results(tmp_path):
    """Done.message and Waiting.message reach the person as agent messages."""
    agent = _TypedResultAgent(llm=FakeLLMClient(), cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)

    await dispatcher.submit("wait for the job")

    assert _agent_messages(agent) == ["Waiting for the job.", "All done."]
    await dispatcher.close()


class _QuestionAgent(CodingAgent):
    async def handle(self, notification: dict[str, list[Any]]) -> NeedInput:
        return NeedInput(question="Which branch?", options=["main", "dev"])


async def test_dispatcher_shows_a_need_input_question_with_its_choices(tmp_path):
    agent = _QuestionAgent(llm=FakeLLMClient(), cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)

    result = await dispatcher.submit("push it")

    assert isinstance(result, NeedInput)
    assert _agent_messages(agent) == ["Which branch?\n\n- main\n- dev"]
    await dispatcher.close()


class _TypedQuestionAgent(CodingAgent):
    async def handle(self, notification: dict[str, list[Any]]) -> NeedInputForm:
        return NeedInputForm(
            heading="Which release?",
            reason="The version decides the changelog heading.",
            questions=[
                TextQuestion(id="version", label="Version?", help="The version number"),
                TextQuestion(id="notes", label="Notes?"),
            ],
        )


async def test_dispatcher_shows_a_need_input_reason_and_answer_fields(tmp_path):
    """The reason and the answer_type fields reach the person, who answers as text."""
    agent = _TypedQuestionAgent(llm=FakeLLMClient(), cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)

    await dispatcher.submit("release it")

    assert _agent_messages(agent) == [
        "Which release?\n\n"
        "The version decides the changelog heading.\n\n"
        "Explicit form requested; this host has no dialog. Text is unvalidated.\n"
        "Text replies require agent interpretation and targeted follow-up. Structured "
        "FormResponse submission requires a capable session host.\n\n"
        "- version: Version? — The version number\n"
        "- notes: Notes?"
    ]
    await dispatcher.close()


def test_coding_agent_generation_names_include_both_input_constructors():
    from nooa.agentdoc._visibility import filter_mro_module_globals

    assert {
        "NeedInput",
        "NeedInputForm",
        "FormResponse",
        "TextQuestion",
        "PickOneQuestion",
        "PickOneOrTextQuestion",
        "FormChoice",
        "FormQuestion",
        "InputRequest",
    } <= set(filter_mro_module_globals(CodingAgent))
