# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture deterministic, provider-ready requests for context parity."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypedDict
from unittest.mock import patch

from pydantic import BaseModel

import nooa.agent as agent_module
import nooa.runtime.method_wrapper as method_wrapper
from nooa import Agent, Context, EventQuery, strategy
from nooa.config.truncation_config import TruncationConfig
from nooa.context_blocks import ScopedContext
from nooa.context_blocks.events import AssistantEvent, UserEvent
from nooa.skill_registry import SkillRegistry
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


async def _no_flush() -> None:
    pass


# Keep the experiment hermetic: neither automatic dev-viewer probing nor trace
# callback draining contributes to context construction.
agent_module._auto_tracing_attempted = True
method_wrapper._flush_litellm_journal = _no_flush


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, type) and issubclass(value, BaseModel):
        return _jsonable(value.model_json_schema())
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


class CapturingFakeLLMClient(FakeLLMClient):
    """Fake client that retains every final request, not only the last one."""

    def __init__(self, responses: list[LLMResponse]):
        super().__init__(scripted_responses=responses)
        self.requests: list[dict[str, Any]] = []

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools=None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        tool_contracts = None
        if tools is not None:
            tool_contracts = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.get_parameter_schema(),
                }
                for tool in tools
            ]
        stable_kwargs = {key: value for key, value in kwargs.items() if key != "prompt_cache_key"}
        self.requests.append(
            {
                "messages": _jsonable(messages),
                "tools": _jsonable(tool_contracts),
                "output_schema": _jsonable(output_model),
                "options": _jsonable(stable_kwargs),
            }
        )
        return await super().acall(messages, tools=tools, output_model=output_model, **kwargs)


def _predict_response(payload: dict[str, Any]) -> LLMResponse:
    content = json.dumps(payload, separators=(",", ":"))
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
        reasoning=None,
        usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    )


def _tool_response(name: str, arguments: dict[str, Any], call_id: str) -> LLMResponse:
    encoded = json.dumps(arguments, separators=(",", ":"))
    tool_call = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": encoded},
    }
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=encoded)],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None, "tool_calls": [tool_call]},
        reasoning=None,
        usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    )


class Answer(BaseModel):
    value: str


class CodeResult(TypedDict):
    status: str
    phase: str


def _request_text(client: CapturingFakeLLMClient, call: int = -1) -> str:
    return json.dumps(client.requests[call]["messages"], sort_keys=True)


async def predict_context() -> dict[str, Any]:
    client = CapturingFakeLLMClient([_predict_response({"value": "predict-ok"})])

    class PredictContextAgent(Agent, llm=client):
        """Agent system prompt for the parity experiment."""

        def __init__(self):
            super().__init__()
            self.phase = "dynamic-v2"
            self.context["policy"] = Context("STATIC_POLICY_V1", prefix=True)
            self.context["dynamic_state"] = Context(expr="self.dynamic_state()")
            self.context["remove_me"] = "REMOVE_ME"

        def dynamic_state(self) -> str:
            return self.phase

        @strategy(PredictStrategy(), context={"decorator_block": "DECORATOR_V1"})
        async def run(self, task: str) -> Answer:
            """Return the requested marker."""
            ...

    agent = PredictContextAgent()
    agent.event_manager.add(UserEvent(id="prior-user", content="PRIOR_USER_V1"))
    agent.event_manager.add(AssistantEvent(id="prior-assistant", content="PRIOR_ASSISTANT_V1"))
    with ScopedContext(context={"scoped_block": "SCOPED_V1", "remove_me": None}):
        result = await agent.run("TASK_INPUT_V1")
    rendered = _request_text(client)
    markers = [
        "STATIC_POLICY_V1",
        "DECORATOR_V1",
        "SCOPED_V1",
        "PRIOR_USER_V1",
        "PRIOR_ASSISTANT_V1",
        "dynamic-v2",
        "TASK_INPUT_V1",
    ]
    assert all(marker in rendered for marker in markers)
    assert "REMOVE_ME" not in rendered
    assert client.requests[0]["output_schema"]
    return {"result": _jsonable(result), "requests": client.requests}


async def filtered_events() -> dict[str, Any]:
    client = CapturingFakeLLMClient([_predict_response({"value": "filtered-ok"})])

    class FilteredEventsAgent(Agent, llm=client):
        @strategy(PredictStrategy(), ScopedContext(events=EventQuery.current_call()))
        async def run(self, task: str) -> Answer:
            """Return the requested marker using only current-call events."""
            ...

    agent = FilteredEventsAgent()
    agent.event_manager.add(UserEvent(id="excluded-user", content="MUST_BE_FILTERED"))
    result = await agent.run("FILTERED_TASK_V1")
    rendered = _request_text(client)
    assert "FILTERED_TASK_V1" in rendered
    assert "MUST_BE_FILTERED" not in rendered
    return {"result": _jsonable(result), "requests": client.requests}


async def dynamic_failure() -> dict[str, Any]:
    client = CapturingFakeLLMClient([_predict_response({"value": "error-ok"})])

    class DynamicFailureAgent(Agent, llm=client):
        def __init__(self):
            super().__init__()
            self.context["broken_dynamic"] = Context(expr="1 / 0")

        @strategy(PredictStrategy())
        async def run(self) -> Answer:
            """Observe the dynamic-context failure and return the marker."""
            ...

    result = await DynamicFailureAgent().run()
    assert "ZeroDivisionError: division by zero" in _request_text(client)
    return {"result": _jsonable(result), "requests": client.requests}


async def skill_registry_context() -> dict[str, Any]:
    client = CapturingFakeLLMClient([_predict_response({"value": "skills-ok"})])

    class SkillAgent(Agent, llm=client):
        def __init__(self):
            super().__init__()
            with patch("nooa.skill_registry.entry_points", return_value=[]):
                self.skills = SkillRegistry(self)

        @strategy(PredictStrategy())
        async def run(self) -> Answer:
            """Observe the skill registry and return the marker."""
            ...

    agent = SkillAgent()
    agent.event_manager.add(UserEvent(id="skill-prior", content="SKILL_PRIOR_EVENT_V1"))
    result = await agent.run()
    rendered = _request_text(client)
    assert "SKILL_PRIOR_EVENT_V1" in rendered
    assert "<skills" in rendered
    return {"result": _jsonable(result), "requests": client.requests}


async def codeact_multiturn() -> dict[str, Any]:
    client = CapturingFakeLLMClient(
        [
            _tool_response(
                "execute_python",
                {"code": "phase = self.advance_phase()\nprint(phase)"},
                "call-execute-1",
            ),
            _tool_response(
                "return_result",
                {"result": {"status": "codeact-ok", "phase": "after"}},
                "call-return-2",
            ),
        ]
    )

    class CodeActAgent(Agent, llm=client):
        """Agent that exercises a deterministic two-turn CodeAct trajectory."""

        def __init__(self):
            super().__init__()
            self.phase = "before"
            self.context["live_phase"] = Context(expr="self.phase")

        def advance_phase(self) -> str:
            """Advance and return the current phase."""
            self.phase = "after"
            return self.phase

        async def run(self, values: list[int]) -> CodeResult:
            """Advance the phase and return the requested status."""
            ...

    result = await CodeActAgent().run([2, 3, 5])
    assert len(client.requests) == 2
    assert '<live_phase expr=\\"self.phase\\">\\nbefore\\n</live_phase>' in _request_text(client, 0)
    assert '<live_phase expr=\\"self.phase\\">\\nafter\\n</live_phase>' in _request_text(client, 1)
    assert client.requests[0]["tools"]
    return {"result": _jsonable(result), "requests": client.requests}


async def budget_eviction() -> dict[str, Any]:
    client = CapturingFakeLLMClient([_predict_response({"value": "budget-ok"})])
    truncation = TruncationConfig(max_context_tokens=700, response_reserve_tokens=0)

    class BudgetAgent(Agent, llm=client, truncation=truncation):
        def __init__(self):
            super().__init__()
            self.context["large_a"] = "A" * 2_000
            self.context["large_b"] = "B" * 2_000
            self.context["large_c"] = "C" * 2_000

        @strategy(PredictStrategy())
        async def run(self) -> Answer:
            """Return the marker after applying the context budget."""
            ...

    result = await BudgetAgent().run()
    rendered = _request_text(client)
    assert rendered.count("EVICTED: over context budget") == 3
    assert "A" * 2_000 not in rendered
    return {"result": _jsonable(result), "requests": client.requests}


async def capture() -> dict[str, Any]:
    scenarios = {
        "predict_context": predict_context,
        "filtered_events": filtered_events,
        "dynamic_failure": dynamic_failure,
        "skill_registry_context": skill_registry_context,
        "codeact_multiturn": codeact_multiturn,
        "budget_eviction": budget_eviction,
    }
    return {
        "schema_version": 1,
        "scenarios": {name: await scenario() for name, scenario in scenarios.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(capture())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
