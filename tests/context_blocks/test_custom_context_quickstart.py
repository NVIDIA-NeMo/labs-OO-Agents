# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable contract for the custom-context quickstart."""

import importlib
import json
from copy import deepcopy
from typing import Any

import pytest

import nooa.agent as agent_module
import nooa.runtime.method_wrapper as method_wrapper
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

example = importlib.import_module("examples.quickstart.16_custom_context")


def _tool_response(name: str, arguments: dict[str, Any], call_id: str) -> LLMResponse:
    tool_call = ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[tool_call],
        finish_reason="tool_calls",
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        usage={"prompt_tokens": 20, "completion_tokens": 8},
    )


class RecordingFakeLLM(FakeLLMClient):
    def __init__(self, scripted_responses):
        super().__init__(scripted_responses)
        self.requests: list[list[dict[str, Any]]] = []

    async def acall(self, messages, *args, **kwargs):
        self.requests.append(deepcopy(messages))
        return await super().acall(messages, *args, **kwargs)


@pytest.mark.asyncio
async def test_custom_api_and_view_control_the_codeact_loop(monkeypatch):
    async def no_flush():
        pass

    monkeypatch.setattr(agent_module, "_auto_tracing_attempted", True)
    monkeypatch.setattr(method_wrapper, "_flush_litellm_journal", no_flush)
    code = "hits = self.research_context.search(question)\nprint(hits)"
    fake = RecordingFakeLLM(
        [
            _tool_response("execute_python", {"code": code}, "search"),
            _tool_response(
                "return_result",
                {
                    "result": {
                        "answer": "Q7-MANGO",
                        "sources": ["aurora-brief"],
                    }
                },
                "answer",
            ),
        ]
    )
    agent = example.ResearchAgent(llm=fake)
    agent.research_context.add("aurora-brief", "Project Aurora's verification code is Q7-MANGO.")
    agent.research_context.add("borealis-brief", "Project Borealis meets in Oslo.")
    agent.context["ignored_builtin_context"] = "THIS MUST NOT REACH THE MODEL"

    result = await agent.investigate("What is Project Aurora's verification code?")
    first_prompt, second_prompt = map(json.dumps, fake.requests)

    assert result == example.ResearchAnswer(answer="Q7-MANGO", sources=["aurora-brief"])
    assert fake.call_count == 2
    assert agent.research_context.selected() == {
        "aurora-brief": "Project Aurora's verification code is Q7-MANGO."
    }
    assert "What is Project Aurora's verification code?" in first_prompt
    assert "Q7-MANGO" not in first_prompt
    assert '"name": "execute_python"' in second_prompt
    assert "self.research_context.search" in second_prompt
    assert "Project Aurora's verification code is Q7-MANGO" in second_prompt
    assert "Project Borealis meets in Oslo" not in second_prompt
    assert "THIS MUST NOT REACH THE MODEL" not in second_prompt

    selected_messages = [
        message
        for message in fake.requests[1]
        if "<selected_research>" in str(message.get("content", ""))
    ]
    assert len(selected_messages) == 1
    without_selected = [message for message in fake.requests[1] if message not in selected_messages]
    assert "Q7-MANGO" not in json.dumps(without_selected)
