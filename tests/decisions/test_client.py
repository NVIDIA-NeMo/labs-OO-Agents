# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest

from nooa.decisions.client import (
    BooleanAnswer,
    BooleanQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAuthenticationError,
    DecisionClient,
    DecisionRequest,
    InvalidDecisionResponseError,
    ScoreAnswer,
    ScoreQuestion,
)
from nooa.unifiedllm import RetryConfig

TEST_ENDPOINT = "https://decision.example/v1/decisions"


def test_decision_client_requires_non_empty_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint must not be empty"):
        DecisionClient("decision-model", endpoint="")


@pytest.mark.asyncio
async def test_decision_client_serializes_and_normalizes_all_types() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "decision-1",
                "model": "decision-model",
                "usage": {"cost": 0},
                "answers": {
                    "urgent": {"noul": 0.9},
                    "team": {
                        "choice": "billing",
                        "probabilities": {"billing": 0.75, "technical": 0.25},
                        "confidence": 0.5,
                    },
                    "frustration": {
                        "score": 1.25,
                        "probabilities": {"0": 0.0, "1": 0.75, "2": 0.25},
                        "legend": {"0": "calm", "1": "frustrated", "2": "angry"},
                        "confidence": 0.5,
                    },
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DecisionClient(
        "decision-model",
        endpoint=TEST_ENDPOINT,
        client=http_client,
    )
    request = DecisionRequest(
        state={"message": "Payouts have failed.", "history": ["first", "second"]},
        questions={
            "urgent": BooleanQuestion("Urgent?"),
            "team": ChoiceQuestion("Which team?", {"billing": "payments", "technical": "bugs"}),
            "frustration": ScoreQuestion("How frustrated?", ["calm", "frustrated", "angry"]),
        },
    )

    response = await client.adecide(request)

    assert seen["model"] == "decision-model"
    assert seen["state"] == {
        "message": "Payouts have failed.",
        "history": ["first", "second"],
    }
    assert seen["questions"]["urgent"] == {"type": "noul", "instructions": "Urgent?"}
    assert response.id == "decision-1"
    assert response.answers["urgent"] == BooleanAnswer(0.9)
    assert response.answers["team"] == ChoiceAnswer(
        selected="billing",
        probabilities={"billing": 0.75, "technical": 0.25},
        confidence=0.5,
    )
    assert response.answers["frustration"] == ScoreAnswer(
        score=1.25,
        probabilities={0: 0.0, 1: 0.75, 2: 0.25},
        legend={0: "calm", 1: "frustrated", 2: "angry"},
        confidence=0.5,
    )
    await http_client.aclose()


@pytest.mark.asyncio
async def test_decision_client_retries_transient_status() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "model": "test",
                "answers": {"urgent": {"noul": 0.7}},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DecisionClient(
        "decision-model",
        endpoint=TEST_ENDPOINT,
        client=http_client,
        retry_config=RetryConfig(
            max_retries=1,
            base_delay=0.0,
            jitter_factor=0.0,
            rate_limit_extra_retries=0,
        ),
    )
    response = await client.adecide(
        DecisionRequest(state="state", questions={"urgent": BooleanQuestion("Urgent?")})
    )
    assert attempts == 2
    assert response.answers["urgent"] == BooleanAnswer(0.7)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_decision_client_distinguishes_authentication_failure() -> None:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(401))
    )
    client = DecisionClient("decision-model", endpoint=TEST_ENDPOINT, client=http_client)
    with pytest.raises(DecisionAuthenticationError, match="HTTP 401"):
        await client.adecide(
            DecisionRequest(state="state", questions={"urgent": BooleanQuestion("Urgent?")})
        )
    await http_client.aclose()


@pytest.mark.asyncio
async def test_decision_client_normalizes_malformed_nested_response() -> None:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "answers": {
                        "team": {
                            "choice": "billing",
                            "probabilities": [],
                            "confidence": 0.5,
                        }
                    }
                },
            )
        )
    )
    client = DecisionClient("decision-model", endpoint=TEST_ENDPOINT, client=http_client)
    with pytest.raises(InvalidDecisionResponseError, match="Invalid decision response"):
        await client.adecide(
            DecisionRequest(
                state="state",
                questions={"team": ChoiceQuestion("Which team?", {"billing": "payments"})},
            )
        )
    await http_client.aclose()


@pytest.mark.asyncio
async def test_decision_client_preserves_array_state() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "decision-model",
                "answers": {"urgent": {"noul": 0.2}},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DecisionClient("decision-model", endpoint=TEST_ENDPOINT, client=http_client)

    await client.adecide(
        DecisionRequest(
            state=["First message", "Second message"],
            questions={"urgent": BooleanQuestion("Urgent?")},
        )
    )

    assert seen["state"] == ["First message", "Second message"]
    await http_client.aclose()
