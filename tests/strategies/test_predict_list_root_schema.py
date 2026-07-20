# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue 232.

PredictStrategy must not emit a top-level ``{"type": "array"}`` schema for ``list``
return types. The OpenAI / Azure Responses API rejects ``response_format`` schemas
whose root is not ``type: "object"``, so a generation method annotated ``-> list[T]``
used to fail at request time. The schema must be object-rooted (the list is wrapped
under a ``value`` property and unwrapped before returning to the caller).
"""

import json
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from nooa import Agent
from nooa.config.strategy_config import PredictConfig
from nooa.decorators import strategy
from nooa.strategies import PredictStrategy
from nooa.strategies.predict import PredictStrategy as _PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse


@dataclass
class Cluster:
    theme: str


class Topic(BaseModel):
    name: str
    weight: float


_TEST_LLM = FakeLLMClient()


def _llm_resp(content: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
    )


class TestPredictListRootSchema:
    """The generated response schema for list return types must be object-rooted."""

    @pytest.mark.parametrize(
        "return_type",
        [list, list[str], list[int], list[Cluster], list[Topic]],
        ids=["bare_list", "list_str", "list_int", "list_dataclass", "list_pydantic"],
    )
    def test_schema_root_is_object(self, return_type):
        """Schema root must be 'object', not 'array', for all list return type variants.

        The OpenAI/Azure Responses API rejects schemas whose root is ``type: array``,
        so every list variant must produce ``{"type": "object", ...}`` at the root.
        """
        s = _PredictStrategy()
        model = s._create_response_model(return_type, "propose_clusters")
        schema = model.model_json_schema()
        assert schema["type"] == "object", (
            f"Responses API requires an object-rooted schema for {return_type!r}, "
            f"got {schema['type']!r}"
        )


class TestPredictListReturnEndToEnd:
    """A ``-> list[T]`` method returns the unwrapped list to the caller."""

    @pytest.mark.asyncio
    async def test_returns_list_of_dataclass(self):
        class ClusterAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def propose_clusters(self, items: list[str]) -> list[Cluster]:
                """Group {items} into broad clusters."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[_llm_resp(json.dumps({"value": [{"theme": "a"}, {"theme": "b"}]}))]
        )
        agent = ClusterAgent(llm=fake_llm)
        result = await agent.propose_clusters(["x", "y"])
        assert result == [Cluster(theme="a"), Cluster(theme="b")]

    @pytest.mark.asyncio
    async def test_returns_list_from_bare_array(self):
        """A non-compliant model that emits a bare top-level array still validates.

        The schema asks for `{"value": [...]}`, but if the model returns `[...]`,
        `_parse_llm_response` wraps it into `{"value": [...]}` — this exercises that
        robustness path (the other end-to-end cases only use the wrapped payload).
        """

        class ClusterAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def propose_clusters(self, items: list[str]) -> list[Cluster]:
                """Group {items} into broad clusters."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[_llm_resp(json.dumps([{"theme": "a"}, {"theme": "b"}]))]
        )
        agent = ClusterAgent(llm=fake_llm)
        result = await agent.propose_clusters(["x", "y"])
        assert result == [Cluster(theme="a"), Cluster(theme="b")]

    @pytest.mark.asyncio
    async def test_returns_list_of_pydantic(self):
        class TopicAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def topics(self, text: str) -> list[Topic]:
                """Extract weighted topics from {text}."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _llm_resp(
                    json.dumps(
                        {"value": [{"name": "ai", "weight": 0.9}, {"name": "ml", "weight": 0.5}]}
                    )
                )
            ]
        )
        agent = TopicAgent(llm=fake_llm)
        result = await agent.topics("...")
        assert result == [Topic(name="ai", weight=0.9), Topic(name="ml", weight=0.5)]
