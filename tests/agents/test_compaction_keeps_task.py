# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compaction keeps the current task's input verbatim in history.

Regression: an automatic summary of events 1..21 archived the Task and the
inspect-inputs prefill that printed the user's message; the summary rewrote an
exact manifest spec as "names mapped to unique IDs" and the agent wrote a
wrong manifest.
"""

import pytest

from nooa import Agent
from nooa.agentdoc import truncating_pformat
from nooa.agents import TokenBudgetSummarizer
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.context_blocks import ResultStatus
from nooa.context_blocks.events import ToolCallEvent
from nooa.events import BeforeTurn, Message, PythonOutput, Summary, Task
from nooa.runtime import event_manager as event_manager_module
from nooa.runtime.context_vars import _pop_agent_call_id, _push_agent_call_id
from nooa.runtime.event_manager import EventManager
from nooa.runtime.middleware import LLMCallContext
from nooa.unifiedllm import FakeLLMClient, LLMResponse, LLMUsage

SPEC = (
    "- segment_manifest.json: "
    '{"segments":[{"segment_id":1,"name":"exact name"},...]}.\n'
    "  Extra factual source/hash/count fields are allowed."
)
PROMPT = "## Task: handle\n\nHandle what arrived in ``notification``."


def add_turn(em: EventManager, call_id: str, message: str, work: int) -> list[str]:
    """One handle() call: Task, inspect-inputs prefill showing ``message``, then work."""
    _push_agent_call_id(call_id)
    try:
        tags = [em.add(Task(prompt=PROMPT))]
        prefill = {"prefill": True, "prefill_type": "inspect_inputs"}
        tags.append(
            em.add(
                ToolCallEvent(
                    tool_call_id=f"prefill_{call_id}",
                    name="python_cell",
                    arguments={"code": "pprint(notification)"},
                    metadata=dict(prefill),
                )
            )
        )
        tags.append(
            em.add(
                PythonOutput(
                    tool_call_id=f"prefill_{call_id}",
                    execution_status=ResultStatus.COMPLETE,
                    execution_count=1,
                    stdout=f"{{'user_messages': ['''{message}''']}}",
                    metadata=dict(prefill),
                )
            )
        )
        for i in range(work):
            em.add(Message(content=f"{call_id} work {i}"))
        return tags
    finally:
        _pop_agent_call_id()


def rendered(em: EventManager) -> str:
    """Active history as the model reads it (one pformat per active event)."""
    return "\n".join(truncating_pformat(event) for event in em.values())


def test_collapse_keeps_current_task_input_verbatim():
    em = EventManager()
    old = add_turn(em, "old-call", "old request", work=2)
    task = add_turn(em, "current-call", f"Deliver:\n{SPEC}", work=5)
    last = em.keys()[-1]

    _push_agent_call_id("current-call")
    try:
        tag = em.collapse("1", last, summary_text="Manifest: names mapped to unique IDs.")
    finally:
        _pop_agent_call_id()

    assert tag == f"1..{last}"
    assert em.keys() == [tag, *task]
    assert set(old) <= set(em[tag].children_tags)
    assert not set(task) & set(em[tag].children_tags)
    output = em[task[2]]
    assert isinstance(output, PythonOutput) and SPEC in output.stdout
    assert SPEC in rendered(em)


def test_collapse_without_call_stack_protects_the_newest_task():
    em = EventManager()
    add_turn(em, "old-call", "old request", work=1)
    task = add_turn(em, "current-call", SPEC, work=3)
    tag = em.collapse("1", em.keys()[-1], summary_text="s")
    assert em.keys() == [tag, *task]


def test_collapse_of_only_the_task_input_is_refused():
    em = EventManager()
    task = add_turn(em, "current-call", SPEC, work=2)
    with pytest.raises(ValueError, match="keeps verbatim"):
        em.collapse(task[0], task[-1], summary_text="s")
    assert em.keys()[:3] == task


def test_oversized_task_input_is_compacted_with_a_note(monkeypatch):
    monkeypatch.setattr(event_manager_module, "TASK_VERBATIM_MAX_CHARS", 400)
    em = EventManager()
    task = add_turn(em, "current-call", SPEC + "x" * 1000, work=2)
    tag = em.collapse("1", em.keys()[-1], summary_text="s")
    summary = em[tag]
    assert isinstance(summary, Summary)
    assert task[0] in em.keys() and task[1] in em.keys()
    assert task[2] in summary.children_tags
    assert summary.summary_text.startswith(f"[Task input event(s) {task[2]} exceed the 400-")
    assert f'self.events["{task[2]}"]' in summary.summary_text
    assert summary.summary_text.endswith("\n\ns")


@pytest.mark.asyncio
async def test_token_budget_summary_keeps_task_spec_word_for_word():
    """The automatic fork summarizes the range; the task input stays active."""
    agent = Agent(llm=FakeLLMClient())
    em = agent.event_manager
    task = add_turn(em, "handle-call", f"Deliver:\n{SPEC}", work=6)
    summarizer = TokenBudgetSummarizer.install(
        agent, config=TokenBudgetConfig(max_tokens=100, preserve_recent=2)
    )

    async def summary_call(messages, **params):
        assert "Background memory compaction" in messages[-1]["content"]
        return LLMResponse(
            content="Manifest: exact semantic names mapped to unique IDs.",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )

    agent.llm.acall = summary_call
    ctx = LLMCallContext(
        agent=agent,
        runtime=agent.runtime,
        client=agent.llm,
        messages=[{"role": "system", "content": "stable"}, {"role": "user", "content": "x"}],
        params={"tools": [], "output_model": None},
    )

    async def core(request):
        request.response = LLMResponse(
            content="parent", usage=LLMUsage(input_tokens=1000, output_tokens=5)
        )
        return request

    _push_agent_call_id("handle-call")
    try:
        await em.run_middleware("llm_call", ctx, core)
        await summarizer._pending_task
        em.add(
            BeforeTurn(method_name="handle", strategy="CodeAct", generation_id="g", turn_number=2)
        )
    finally:
        _pop_agent_call_id()

    # The fork summarized 1..7; the task input (1-3) stays active ahead of the summary.
    keys = em.keys()
    assert keys == [*task, "4..7", "8", "9"]
    summary_tag = keys[3]
    assert em[summary_tag].summary_text.startswith("Manifest:")
    assert SPEC in em[task[2]].stdout
    assert SPEC in rendered(em)
    summarizer._uninstall()
