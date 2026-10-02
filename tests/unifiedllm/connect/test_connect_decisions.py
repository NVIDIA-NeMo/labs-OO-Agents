# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision-model onboarding uses the flat model registry and System One wire API."""

import json

import httpx
import pytest

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http


def test_decision_plan_has_no_chat_only_configuration() -> None:
    proposal = connect.plan(
        "decisions",
        "decision-model",
        "systemone",
        "https://decision.example/v1/systemone",
        "DECISION_API_KEY",
    )

    assert proposal.entry == {
        "model_name": "decision-model",
        "client_type": "decision",
        "api_style": "systemone",
        "endpoint": "https://decision.example/v1/systemone",
        "api_key_env": "DECISION_API_KEY",
        "transport": "direct",
        "provenance": {
            "probes": {},
            "requests_accepted": [],
            "not_probed": [
                "context_window",
                "reasoning",
                "tools",
                "sessions",
                "reply_limit",
            ],
        },
    }
    assert len(proposal.probes) == 1
    assert "max_tokens" not in proposal.entry


@pytest.mark.asyncio
async def test_decision_probe_uses_configured_endpoint_and_normalized_client(
    monkeypatch,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        assert str(request.url) == "https://decision.example/v1/systemone"
        assert request.headers["authorization"] == "Bearer secret"
        assert body["model"] == "decision-model"
        assert body["questions"]["supported"]["type"] == "noul"
        return httpx.Response(
            200,
            json={
                "model": "resolved-decision-model",
                "answers": {"supported": {"noul": 0.91}},
                "usage": {"input_tokens": 14, "output_tokens": 0},
            },
        )

    mock_http(monkeypatch, handle)
    proposal = connect.plan(
        "decisions",
        "decision-model",
        "systemone",
        "https://decision.example/v1/systemone",
        "DECISION_API_KEY",
    )

    result = await connect.run(proposal, approved="minimal", api_key="secret")

    assert len(requests) == 1
    record = result.entry["provenance"]["probes"]["routing"]
    assert record["outcome"] == "accepted"
    assert record["resolved_model"] == "resolved-decision-model"
    assert record["probability_true"] == pytest.approx(0.91)
    assert record["input_tokens"] == 14
    assert record["status_code"] == 200
    assert result.entry["provenance"]["warnings"] == []


@pytest.mark.asyncio
async def test_unapproved_decision_plan_does_not_send_http(monkeypatch) -> None:
    mock_http(monkeypatch, lambda request: pytest.fail("No HTTP request expected"))
    proposal = connect.plan(
        "decisions",
        "decision-model",
        "systemone",
        "https://decision.example/v1/systemone",
        "",
    )

    result = await connect.run(proposal, approved="none")

    assert result.entry["provenance"]["probes"]["routing"] == {
        "outcome": "not_probed",
        "reason": "not approved",
    }


@pytest.mark.asyncio
async def test_decision_discovery_is_rejected_without_http(monkeypatch) -> None:
    mock_http(monkeypatch, lambda request: pytest.fail("No HTTP request expected"))

    with pytest.raises(ValueError, match="do not list models"):
        await connect.discover("https://decision.example/v1/systemone", api_style="systemone")


@pytest.mark.asyncio
async def test_decision_interface_check_sends_one_decision_probe(monkeypatch) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        assert str(request.url) == "https://decision.example/v1/systemone"
        assert "max_tokens" not in body
        assert "max_output_tokens" not in body
        return httpx.Response(
            200,
            json={"model": "decision-model", "answers": {"supported": {"noul": 0.8}}},
        )

    mock_http(monkeypatch, handle)
    events = [
        event
        async for event in connect.check_interfaces(
            "decisions",
            "decision-model",
            "https://decision.example/v1/systemone",
            "",
            styles=("systemone",),
        )
    ]

    result = events[-1]
    assert isinstance(result, connect.InterfaceResult)
    assert len(requests) == 1
    assert result.accepted == ("systemone",)
    assert result.results["systemone"].entry["client_type"] == "decision"


@pytest.mark.asyncio
async def test_decision_interface_cannot_be_mixed_with_chat_styles(monkeypatch) -> None:
    mock_http(monkeypatch, lambda request: pytest.fail("No HTTP request expected"))

    with pytest.raises(ValueError, match="cannot be mixed"):
        async for _ in connect.check_interfaces(
            "decisions",
            "decision-model",
            "https://decision.example/v1/systemone",
            "",
            styles=("chat", "systemone"),
        ):
            pass


@pytest.mark.asyncio
async def test_decision_interface_rejects_reasoning_template(monkeypatch) -> None:
    mock_http(monkeypatch, lambda request: pytest.fail("No HTTP request expected"))

    with pytest.raises(ValueError, match="no reasoning levels"):
        async for _ in connect.check_interfaces(
            "decisions",
            "decision-model",
            "https://decision.example/v1/systemone",
            "",
            styles=("systemone",),
            reasoning_template="effort",
        ):
            pass
