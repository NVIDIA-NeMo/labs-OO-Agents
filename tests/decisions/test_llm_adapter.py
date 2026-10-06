# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from enum import Enum
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel

from nooa import (
    Agent,
    BooleanDecision,
    Criteria,
    DecideStrategy,
    DecisionClient,
    DecisionModel,
    DecisionModelRequiredError,
    Instructions,
    Threshold,
    UnifiedDecisionModel,
    strategy,
)
from nooa.decisions.client import InvalidDecisionResponseError
from nooa.events import DecisionRecord
from nooa.unifiedllm import AssistantText, FakeLLMClient, LLMResponse


class Team(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"


TEAM_CRITERIA = Criteria(
    by_value={
        Team.BILLING: "Payments and refunds.",
        Team.TECHNICAL: "Bugs and outages.",
    }
)
TeamChoice = Annotated[Team, TEAM_CRITERIA]


Severity = Annotated[float, Criteria("Cosmetic", "Degraded", "Outage")]


class Triage(BaseModel):
    team: Annotated[TeamChoice, Instructions("Which team owns it?")]
    urgent: Annotated[bool, Instructions("Is it urgent?")]
    severity: Annotated[Severity, Instructions("How severe is it?")]


def _reply(content: str) -> LLMResponse:
    return LLMResponse(parts=(AssistantText(text=content),), finish_reason="stop")


def _llm(*contents: str) -> FakeLLMClient:
    llm = FakeLLMClient([_reply(content) for content in contents])
    llm.model = "chat-model"
    return llm


def _user_payload(llm: FakeLLMClient) -> dict:
    return json.loads(llm.calls[0].messages[-1]["content"])


def test_from_llm_declares_capabilities() -> None:
    model = DecisionModel.from_llm(_llm())
    client = DecisionClient("decision-model", endpoint="https://decision.example/v1")

    assert isinstance(model, DecisionModel)
    assert isinstance(model, UnifiedDecisionModel)
    assert model.model == "chat-model"
    assert model.provides_probabilities is False
    assert model.decision_source == "llm"
    assert isinstance(client, DecisionModel)
    assert client.provides_probabilities is True
    assert client.decision_source == "native"


def test_from_llm_resolves_an_alias(monkeypatch) -> None:
    llm = _llm()
    requested: list[str] = []

    def resolve(alias: str):
        requested.append(alias)
        return llm

    monkeypatch.setattr("nooa.unifiedllm.registry.get_llm_client", resolve)

    model = DecisionModel.from_llm("chat")

    assert requested == ["chat"]
    assert model.model == "chat-model"


@pytest.mark.asyncio
async def test_chat_backed_agent_answers_primitive_results() -> None:
    llm = _llm(
        '{"result": "TECHNICAL"}', '{"result": true}', '{"result": 2}', '{"result": "choice_1"}'
    )

    class Router(Agent, decision_model=DecisionModel.from_llm(llm)):
        @strategy(DecideStrategy())
        async def team(self, message: str) -> TeamChoice:
            """Which team owns the message?"""
            ...

        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Is the message urgent?"""
            ...

        @strategy(DecideStrategy())
        async def severity(self, message: str) -> Severity:
            """How severe is the reported problem?"""
            ...

        @strategy(DecideStrategy())
        async def label(self, message: str) -> Literal["a", "b"]:
            """Pick a label."""
            ...

    agent = Router()

    assert await agent.team("The API is down") is Team.TECHNICAL
    assert await agent.urgent("The API is down") is True
    assert await agent.severity("The API is down") == 2.0
    assert await agent.label("The API is down") == "b"

    payload = _user_payload(llm)
    assert payload["state"] == {"inputs": {"message": "The API is down"}}
    assert payload["questions"]["result"]["criteria"] == {
        "BILLING": "Payments and refunds.",
        "TECHNICAL": "Bugs and outages.",
    }
    assert llm.calls[0].output_model is not None


@pytest.mark.asyncio
async def test_chat_backed_composite_and_record() -> None:
    llm = _llm('{"team": "BILLING", "urgent": false, "severity": 0}')

    class Router(Agent, decision_model=DecisionModel.from_llm(llm)):
        @strategy(DecideStrategy())
        async def triage(self, message: str) -> Triage:
            """Triage the message."""
            ...

    agent = Router()

    assert await agent.triage("Refund please") == Triage(
        team=Team.BILLING, urgent=False, severity=0.0
    )
    assert len(llm.calls) == 1
    record = agent.event_manager.filter(type="DecisionRecord")[0]
    assert isinstance(record, DecisionRecord)
    assert record.decision_source == "llm"
    assert record.requested_model == "chat-model"
    assert record.success is True


@pytest.mark.asyncio
async def test_chat_backed_model_rejects_results_that_need_probabilities() -> None:
    llm = _llm()

    class Router(Agent, decision_model=DecisionModel.from_llm(llm)):
        @strategy(DecideStrategy())
        async def detailed(self, message: str) -> BooleanDecision:
            """Is the message urgent?"""
            ...

        @strategy(DecideStrategy())
        async def thresholded(self, message: str) -> Annotated[bool, Threshold(0.8)]:
            """Is the message urgent?"""
            ...

    agent = Router()
    for method in (agent.detailed, agent.thresholded):
        with pytest.raises(DecisionModelRequiredError, match="needs probabilities"):
            await method("The API is down")
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_chat_backed_model_retries_an_invalid_reply() -> None:
    llm = _llm('{"result": "SALES"}', '{"result": "BILLING"}')

    class Router(Agent, decision_model=DecisionModel.from_llm(llm)):
        @strategy(DecideStrategy())
        async def team(self, message: str) -> TeamChoice:
            """Which team owns the message?"""
            ...

    assert await Router().team("Refund please") is Team.BILLING
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_chat_backed_model_fails_after_repeated_invalid_replies() -> None:
    llm = _llm('{"result": 7}', "not json")

    class Router(Agent, decision_model=DecisionModel.from_llm(llm)):
        @strategy(DecideStrategy())
        async def severity(self, message: str) -> Severity:
            """How severe is the reported problem?"""
            ...

    with pytest.raises(InvalidDecisionResponseError, match="after 2 attempts"):
        await Router().severity("The API is down")


@pytest.mark.asyncio
async def test_chat_backed_model_as_call_site_override() -> None:
    llm = _llm('{"result": true}')

    @strategy(DecideStrategy())
    async def urgent(message: str) -> bool:
        """Is the message urgent?"""
        ...

    assert await urgent("The API is down", decision_model=DecisionModel.from_llm(llm)) is True
