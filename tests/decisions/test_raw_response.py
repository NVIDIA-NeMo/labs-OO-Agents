# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from enum import Enum
from typing import Annotated

import httpx
import pytest
from pydantic import BaseModel

from nooa import (
    Agent,
    BooleanDecision,
    ChoiceDecision,
    DecideStrategy,
    DecisionClient,
    Instructions,
    strategy,
)
from nooa.decisions.client import ChoiceAnswer, DecisionResponse
from nooa.events import DecisionRecord
from nooa.runtime.middleware import DecisionCallContext
from nooa.unifiedllm import AssistantText, FakeLLMClient, LLMResponse

ENDPOINT = "https://decision.example/v1/decisions"
METADATA = {"probabilities": "one-hot", "calibration": "none"}


class Team(Enum):
    """Team that owns a message.

    Attributes:
        BILLING: Payments and refunds.
        TECHNICAL: Bugs and outages.
    """

    BILLING = "billing"
    TECHNICAL = "technical"


def _client(answers: dict) -> DecisionClient:
    body = {"id": "decision-1", "model": "loaded-model", "answers": answers, "metadata": METADATA}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    return DecisionClient(
        "requested-model", endpoint=ENDPOINT, client=httpx.AsyncClient(transport=transport)
    )


TEAM_ANSWER = {
    "choice": "BILLING",
    "probabilities": {"BILLING": 1.0, "TECHNICAL": 0.0},
    "confidence": 1.0,
}


class Triage(BaseModel):
    team: Annotated[ChoiceDecision[Team], Instructions("Which team owns it?")]
    urgent: Annotated[BooleanDecision, Instructions("Is it urgent?")]
    escalate: Annotated[bool, Instructions("Should it be escalated?")]


def _triage_client() -> DecisionClient:
    return _client({"team": TEAM_ANSWER, "urgent": {"noul": 0.9}, "escalate": {"noul": 0.1}})


def _record(agent: Agent) -> DecisionRecord:
    record = agent.event_manager.filter(type="DecisionRecord")[0]
    assert isinstance(record, DecisionRecord)
    return record


@pytest.mark.asyncio
async def test_raw_response_is_off_by_default() -> None:
    class Router(Agent, decision_model=_client({"result": TEAM_ANSWER})):
        @strategy(DecideStrategy())
        async def team(self, message: str) -> ChoiceDecision[Team]:
            """Which team owns the message?"""
            ...

    agent = Router()
    decision = await agent.team("Refund please")

    assert decision.raw_response is None
    assert _record(agent).raw_response is None


@pytest.mark.asyncio
async def test_opt_in_attaches_read_only_raw_response_and_records_it() -> None:
    class Router(Agent, decision_model=_client({"result": TEAM_ANSWER})):
        @strategy(DecideStrategy(include_raw_response=True))
        async def team(self, message: str) -> ChoiceDecision[Team]:
            """Which team owns the message?"""
            ...

    agent = Router()
    decision = await agent.team("Refund please")

    raw = decision.raw_response
    assert raw is not None
    assert dict(raw["metadata"]) == METADATA
    assert raw["model"] == "loaded-model"
    with pytest.raises(TypeError):
        raw["metadata"]["calibration"] = "temperature"  # type: ignore[index]

    record = _record(agent)
    assert record.raw_response is raw
    assert json.loads(record.model_dump_json())["raw_response"]["metadata"] == METADATA


@pytest.mark.asyncio
async def test_composite_decisions_share_one_raw_response() -> None:
    class Router(Agent, decision_model=_triage_client()):
        @strategy(DecideStrategy(include_raw_response=True))
        async def triage(self, message: str) -> Triage:
            """Triage the message."""
            ...

    result = await Router().triage("Refund please")

    assert result.team.raw_response is not None
    assert result.team.raw_response is result.urgent.raw_response
    assert result.escalate is False


@pytest.mark.asyncio
async def test_raw_response_is_excluded_from_equality_repr_and_dumps() -> None:
    class Router(Agent, decision_model=_triage_client()):
        @strategy(DecideStrategy(include_raw_response=True))
        async def triage(self, message: str) -> Triage:
            """Triage the message."""
            ...

    class PlainRouter(Agent, decision_model=_triage_client()):
        @strategy(DecideStrategy())
        async def triage(self, message: str) -> Triage:
            """Triage the message."""
            ...

    with_raw = await Router().triage("Refund please")
    without_raw = await PlainRouter().triage("Refund please")

    assert with_raw == without_raw
    assert "metadata" not in repr(with_raw)
    assert "raw_response" not in with_raw.model_dump()


@pytest.mark.asyncio
async def test_short_circuit_response_without_raw_body() -> None:
    class Router(Agent, decision_model=_client({"result": TEAM_ANSWER})):
        @strategy(DecideStrategy(include_raw_response=True))
        async def team(self, message: str) -> ChoiceDecision[Team]:
            """Which team owns the message?"""
            ...

    agent = Router()

    async def from_cache(ctx: DecisionCallContext, call_next):
        ctx.response = DecisionResponse(
            model="cache",
            answers={
                "result": ChoiceAnswer(
                    selected="BILLING",
                    probabilities={"BILLING": 0.9, "TECHNICAL": 0.1},
                    confidence=0.8,
                )
            },
        )
        return ctx

    agent.event_manager.intercept("decision_call", from_cache)

    decision = await agent.team("Refund please")

    assert decision.selected is Team.BILLING
    assert decision.raw_response is None
    assert _record(agent).raw_response is None


@pytest.mark.asyncio
async def test_llm_fallback_ignores_raw_response_option() -> None:
    llm = FakeLLMClient(
        [LLMResponse(parts=(AssistantText(text='{"value": true}'),), finish_reason="stop")]
    )

    class Router(Agent, llm=llm):
        @strategy(DecideStrategy(include_raw_response=True))
        async def urgent(self, message: str) -> bool:
            """Is it urgent?"""
            ...

    agent = Router()

    assert await agent.urgent("Production is down") is True
    assert _record(agent).raw_response is None
