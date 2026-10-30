# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DecisionClient(api_style="openai") speaks OpenAI's /v1/decisions wire format."""

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
    Criteria,
    DecideStrategy,
    DecisionClient,
    Instructions,
    ScoreDecision,
    Threshold,
    strategy,
)
from nooa.decisions import DecisionRefusedError, InvalidDecisionResponseError
from nooa.events import DecisionRecord

ENDPOINT = "https://api.openai.com/v1/decisions"


class Team(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"


TeamDecision = Annotated[
    ChoiceDecision[Team],
    Criteria(
        by_value={
            Team.BILLING: {"what": "Payments and refunds", "not_for": ["outages"]},
            Team.TECHNICAL: "Bugs and outages.",
        }
    ),
    Threshold(0.8),
]


class Triage(BaseModel):
    team: Annotated[TeamDecision, Instructions("Which team owns it?")]
    urgent: Annotated[
        BooleanDecision,
        Instructions("Is it urgent?"),
        Criteria(by_value={True: "Customers are blocked.", False: "It can wait."}),
        Threshold(0.7),
    ]
    severity: Annotated[
        ScoreDecision,
        Instructions("How severe is it?"),
        Criteria("Cosmetic", "Degraded", "Outage"),
    ]


def _answers() -> list[dict]:
    return [
        {
            "type": "choice",
            "name": "team",
            "choice": "BILLING",
            "confidence": 0.4,
            "probabilities": [
                {"value": "BILLING", "probability": 0.7},
                {"value": "TECHNICAL", "probability": 0.3},
            ],
        },
        {"type": "predicate", "name": "urgent", "probability": 0.9},
        {
            "type": "score",
            "name": "severity",
            "score": 1.2,
            "confidence": 0.5,
            "probabilities": [
                {"label": "0", "value": 0, "probability": 0.1},
                {"label": "1", "value": 1, "probability": 0.6},
                {"label": "2", "value": 2, "probability": 0.3},
            ],
        },
    ]


def _body(answers: list[dict]) -> dict:
    return {
        "model": "gpt-6-luna-2026-10-01",
        "answers": answers,
        "usage": {
            "input_tokens": 120,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 120,
        },
    }


def _client(answers: list[dict], seen: list[dict] | None = None) -> DecisionClient:
    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == ENDPOINT
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(200, json=_body(answers), headers={"x-request-id": "req_123"})

    return DecisionClient(
        "gpt-6-luna",
        endpoint=ENDPOINT,
        api_style="openai",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )


def _router(client: DecisionClient) -> Agent:
    class Router(Agent, decision_model=client):
        @strategy(DecideStrategy(include_raw_response=True))
        async def triage(self, message: str) -> Triage:
            """Use only the message."""
            ...

    return Router()


def test_unknown_api_style_is_rejected() -> None:
    with pytest.raises(ValueError, match="api_style"):
        DecisionClient("m", endpoint=ENDPOINT, api_style="decisions")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_request_uses_openai_shapes() -> None:
    seen: list[dict] = []

    await _router(_client(_answers(), seen)).triage("Refund please")

    body = seen[0]
    assert body["model"] == "gpt-6-luna"
    assert "state" not in body
    assert json.loads(body["input"]) == {
        "inputs": {"message": "Refund please"},
        "shared_guidance": "Use only the message.",
    }
    team, urgent, severity = body["questions"]
    assert team == {
        "type": "choice",
        "name": "team",
        "instructions": "Which team owns it?",
        "choices": [
            {
                "value": "BILLING",
                "description": '{"what": "Payments and refunds", "not_for": ["outages"]}',
            },
            {"value": "TECHNICAL", "description": "Bugs and outages."},
        ],
    }
    assert urgent == {
        "type": "predicate",
        "name": "urgent",
        "instructions": "Is it urgent?\n\nTrue: Customers are blocked.\nFalse: It can wait.",
    }
    assert severity == {
        "type": "score",
        "name": "severity",
        "instructions": "How severe is it?",
        "levels": [
            {"label": "0", "description": "Cosmetic"},
            {"label": "1", "description": "Degraded"},
            {"label": "2", "description": "Outage"},
        ],
    }


@pytest.mark.asyncio
async def test_response_supports_detailed_results_thresholds_and_provenance() -> None:
    agent = _router(_client(_answers()))

    result = await agent.triage("Refund please")

    assert result.team.selected is Team.BILLING
    assert result.team.value is None  # 0.7 is below the 0.8 threshold
    assert result.team.probabilities == {Team.BILLING: 0.7, Team.TECHNICAL: 0.3}
    assert result.team.confidence == 0.4
    assert result.urgent.value is True
    assert result.urgent.probability_true == 0.9
    assert result.severity.value == 1.2
    assert result.severity.legend == {0: "Cosmetic", 1: "Degraded", 2: "Outage"}
    assert result.team.raw_response is result.urgent.raw_response
    assert result.team.raw_response is not None
    assert result.team.raw_response["model"] == "gpt-6-luna-2026-10-01"

    record = agent.event_manager.filter(type="DecisionRecord")[0]
    assert isinstance(record, DecisionRecord)
    assert record.decision_source == "native"
    assert record.requested_model == "gpt-6-luna"
    assert record.resolved_model == "gpt-6-luna-2026-10-01"
    assert record.response_id == "req_123"
    assert record.usage is not None and record.usage.input_tokens == 120
    assert record.raw_response is not None


@pytest.mark.asyncio
async def test_refusal_raises_with_the_refused_questions() -> None:
    answers = _answers()
    answers[1] = {"type": "refusal", "name": "urgent"}
    agent = _router(_client(answers))

    with pytest.raises(DecisionRefusedError, match="'urgent'") as raised:
        await agent.triage("Refund please")

    assert raised.value.questions == ("urgent",)
    assert raised.value.raw is not None
    assert raised.value.raw["answers"][0]["choice"] == "BILLING"
    record = agent.event_manager.filter(type="DecisionRecord")[0]
    assert isinstance(record, DecisionRecord)
    assert record.success is False
    assert record.exception_type == "DecisionRefusedError"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answers",
    [
        _answers()[:2],
        [_answers()[1], _answers()[0], _answers()[2]],
        [{**_answers()[0], "type": "score"}, *_answers()[1:]],
        [_answers()[0], {**_answers()[1], "probability": True}, _answers()[2]],
    ],
    ids=["missing-answer", "out-of-order", "wrong-type", "boolean-probability"],
)
async def test_malformed_responses_are_rejected(answers: list[dict]) -> None:
    with pytest.raises(InvalidDecisionResponseError):
        await _router(_client(answers)).triage("Refund please")


def test_registry_builds_an_openai_decision_client(monkeypatch) -> None:
    from nooa.unifiedllm.registry import client_from_config

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = client_from_config(
        "decisions",
        {
            "model_name": "gpt-6-luna",
            "client_type": "decision",
            "api_style": "openai",
            "endpoint": ENDPOINT,
            "api_key_env": "OPENAI_API_KEY",
        },
    )

    assert isinstance(client, DecisionClient)
    assert client.api_style == "openai"
    assert client.endpoint == ENDPOINT
