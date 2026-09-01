# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM routing controls and generation observability."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest

from nooa import Agent, strategy
from nooa.runtime.hooks import set_hooks
from nooa.strategies import CurrentCall, GenerationStrategy, PredictStrategy, RuntimeServices
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _resp(value: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=f'{{"value": "{value}"}}',
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": f'{{"value": "{value}"}}'},
    )


def _llm(model: str, value: str) -> FakeLLMClient:
    client = FakeLLMClient(scripted_responses=[_resp(value)])
    client.model = model
    return client


class RoutingHooks:
    """Capture generation hook metadata for routing assertions."""

    def __init__(self) -> None:
        self.generations: list[dict[str, Any]] = []

    def before_generation(
        self,
        agent: Any,
        method_name: str,
        strategy: str,
        generation_id: str,
        parent_generation_id: str | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.generations.append(
            {
                "method_name": method_name,
                "strategy": strategy,
                "generation_id": generation_id,
                **kwargs,
            }
        )
        return {"generation_id": generation_id}

    def after_generation(
        self,
        agent: Any,
        method_name: str,
        result: Any,
        exception: Exception | None,
        context: Any,
        generation_id: str,
    ) -> None:
        pass

    def on_messages_built(
        self,
        agent: Any,
        method_name: str,
        messages: list[dict[str, Any]],
        generation_id: str,
        **kwargs: Any,
    ) -> None:
        pass

    def before_agent_call(
        self,
        agent: Any,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        call_id: str,
        parent_call_id: str | None,
        **extra: Any,
    ) -> dict[str, Any]:
        return {"call_id": call_id}

    def after_agent_call(
        self,
        agent: Any,
        method_name: str,
        result: Any,
        exception: Exception | None,
        context: Any,
        **kwargs: Any,
    ) -> None:
        pass

    def before_code_execution(
        self,
        agent: Any,
        code: str,
        execution_id: str,
        generation_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {"execution_id": execution_id}

    def after_code_execution(
        self,
        agent: Any,
        code: str,
        result: Any,
        exception: Exception | None,
        context: Any,
        execution_id: str,
        **kwargs: Any,
    ) -> None:
        pass

    def before_method_invocation(
        self,
        agent: Any,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        invocation_id: str,
        **extra: Any,
    ) -> dict[str, Any]:
        return {"invocation_id": invocation_id}

    def after_method_invocation(
        self,
        agent: Any,
        method_name: str,
        result: Any,
        exception: Exception | None,
        context: Any,
        invocation_id: str,
        **kwargs: Any,
    ) -> None:
        pass

    def before_tool_execution(
        self,
        agent: Any,
        tool_name: str,
        arguments: dict[str, Any],
        execution_id: str,
        generation_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {"execution_id": execution_id}

    def after_tool_execution(
        self,
        agent: Any,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        exception: Exception | None,
        context: Any,
        execution_id: str,
        **kwargs: Any,
    ) -> None:
        pass


@pytest.fixture
def routing_hooks() -> Iterator[RoutingHooks]:
    hooks = RoutingHooks()
    try:
        yield hooks
    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_call_site_llm_override_wins_and_is_observable(routing_hooks: RoutingHooks) -> None:
    default_llm = _llm("default-model", "default")
    decorator_llm = _llm("decorator-model", "decorator")
    override_llm = _llm("override-model", "override")

    class RoutingAgent(Agent, llm=default_llm):
        @strategy(PredictStrategy(), llm=decorator_llm)
        async def summarize(self, text: str) -> str:
            """Summarize text."""
            ...

    agent = RoutingAgent()
    set_hooks(cast(Any, routing_hooks))

    assert await agent.summarize("hello", llm=override_llm) == "override"  # pyright: ignore[reportCallIssue]

    assert default_llm.call_count == 0
    assert decorator_llm.call_count == 0
    assert override_llm.call_count == 1
    assert routing_hooks.generations[-1]["llm.model_name"] == "override-model"
    assert routing_hooks.generations[-1]["llm.selection_source"] == "call_site"


@pytest.mark.asyncio
async def test_nested_generation_inherits_routing_metadata(
    routing_hooks: RoutingHooks,
) -> None:
    default_llm = _llm("default-model", "default")
    override_llm = _llm("override-model", "nested")

    class NestedPredictStrategy(GenerationStrategy):
        @property
        def name(self) -> str:
            return "NESTED"

        async def execute(self, runtime: RuntimeServices, call: CurrentCall) -> Any:
            return await runtime.execute_nested(PredictStrategy(), call)

    class RoutingAgent(Agent, llm=default_llm):
        @strategy(NestedPredictStrategy())
        async def summarize(self, text: str) -> str:
            """Summarize text."""
            ...

    agent = RoutingAgent()
    set_hooks(cast(Any, routing_hooks))

    assert await agent.summarize("hello", llm=override_llm) == "nested"  # pyright: ignore[reportCallIssue]

    assert default_llm.call_count == 0
    assert override_llm.call_count == 1
    assert [g["strategy"] for g in routing_hooks.generations] == ["NESTED", "PREDICT"]
    for generation in routing_hooks.generations:
        assert generation["llm.model_name"] == "override-model"
        assert generation["llm.selection_source"] == "call_site"


@pytest.mark.asyncio
async def test_decorator_llm_selection_is_observable(routing_hooks: RoutingHooks) -> None:
    default_llm = _llm("default-model", "default")
    decorator_llm = _llm("decorator-model", "decorator")

    class RoutingAgent(Agent, llm=default_llm):
        @strategy(PredictStrategy(), llm=decorator_llm)
        async def summarize(self, text: str) -> str:
            """Summarize text."""
            ...

    agent = RoutingAgent()
    set_hooks(cast(Any, routing_hooks))

    assert await agent.summarize("hello") == "decorator"

    assert default_llm.call_count == 0
    assert decorator_llm.call_count == 1
    assert routing_hooks.generations[-1]["llm.model_name"] == "decorator-model"
    assert routing_hooks.generations[-1]["llm.selection_source"] == "decorator"


@pytest.mark.asyncio
async def test_agent_default_llm_selection_is_observable(routing_hooks: RoutingHooks) -> None:
    default_llm = _llm("default-model", "default")

    class RoutingAgent(Agent, llm=default_llm):
        @strategy(PredictStrategy())
        async def summarize(self, text: str) -> str:
            """Summarize text."""
            ...

    agent = RoutingAgent()
    set_hooks(cast(Any, routing_hooks))

    assert await agent.summarize("hello") == "default"

    assert default_llm.call_count == 1
    assert routing_hooks.generations[-1]["llm.model_name"] == "default-model"
    assert routing_hooks.generations[-1]["llm.selection_source"] == "agent_default"


@pytest.mark.asyncio
async def test_user_parameter_named_llm_is_not_consumed_as_framework_override(
    routing_hooks: RoutingHooks,
) -> None:
    default_llm = _llm("default-model", "ok")

    class RoutingAgent(Agent, llm=default_llm):
        @strategy(PredictStrategy())
        async def summarize(self, llm: str) -> str:
            """Summarize using the user-supplied label."""
            ...

    agent = RoutingAgent()
    set_hooks(cast(Any, routing_hooks))

    assert await agent.summarize(llm="customer-visible-label") == "ok"

    assert default_llm.call_count == 1
    assert routing_hooks.generations[-1]["llm.model_name"] == "default-model"
    assert routing_hooks.generations[-1]["llm.selection_source"] == "agent_default"
