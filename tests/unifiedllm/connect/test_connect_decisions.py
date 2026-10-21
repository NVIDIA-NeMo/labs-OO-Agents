# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision-model onboarding uses the flat model registry and System One wire API."""

import asyncio
import json

import httpx
import pytest

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body


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
        "media_inputs": [],
        "decision_image_transport": "json",
        "typesafe_danger_zone_images": False,
        "api_style": "systemone",
        "endpoint": "https://decision.example/v1/systemone",
        "api_key_env": "DECISION_API_KEY",
        "transport": "direct",
        "provenance": {
            "media_suggestions": connect.advertised_media_suggestions(api_style="systemone"),
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
    assert len(proposal.probes) == 4
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

    with pytest.raises(ValueError, match="do not list decision models"):
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
async def test_mixed_decision_interface_requires_explicit_endpoint(monkeypatch) -> None:
    mock_http(monkeypatch, lambda request: pytest.fail("No HTTP request expected"))

    with pytest.raises(ValueError, match="explicit endpoints entry"):
        async for _ in connect.check_interfaces(
            "decisions",
            "decision-model",
            "https://decision.example/v1/systemone",
            "",
            styles=("chat", "systemone"),
        ):
            pass


@pytest.mark.asyncio
async def test_decision_interface_ignores_reasoning_template(monkeypatch) -> None:
    requests = []

    def handle(request):
        requests.append(request)
        body = json.loads(request.content)
        assert set(body) == {"model", "state", "questions"}
        return httpx.Response(200, json={"answers": {"supported": {"noul": 0.8}}})

    mock_http(monkeypatch, handle)
    events = [
        event
        async for event in connect.check_interfaces(
            "decisions",
            "decision-model",
            "https://decision.example/v1/systemone",
            "",
            styles=("systemone",),
            reasoning_template="not-a-chat-template",
        )
    ]
    result = events[-1]
    assert result.accepted == ("systemone",)
    assert result.accepted_client_types == ("decision",)
    assert result.reasoning_observed == {"systemone": False}
    assert "reasoning_levels" not in result.results["systemone"].entry
    assert len(requests) == 1


OPENAI_ENDPOINT = "https://api.openai.com/v1/decisions"


def test_openai_decision_plan_saves_openai_api_style() -> None:
    proposal = connect.plan("decisions", "gpt-6-luna", "openai", OPENAI_ENDPOINT, "OPENAI_API_KEY")

    assert proposal.entry["client_type"] == "decision"
    assert proposal.entry["api_style"] == "openai"
    assert proposal.entry["endpoint"] == OPENAI_ENDPOINT
    assert "max_tokens" not in proposal.entry


@pytest.mark.asyncio
async def test_openai_interface_check_sends_one_openai_decision_probe(monkeypatch) -> None:
    bodies: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == OPENAI_ENDPOINT
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gpt-6-luna",
                "answers": [{"type": "predicate", "name": "supported", "probability": 0.8}],
            },
        )

    mock_http(monkeypatch, handle)
    events = [
        event
        async for event in connect.check_interfaces(
            "decisions", "gpt-6-luna", OPENAI_ENDPOINT, "", styles=("openai",)
        )
    ]

    result = events[-1]
    assert isinstance(result, connect.InterfaceResult)
    assert result.accepted == ("openai",)
    assert result.results["openai"].entry["api_style"] == "openai"
    assert len(bodies) == 1
    assert isinstance(bodies[0]["input"], str)
    assert bodies[0]["questions"][0]["type"] == "predicate"
    assert bodies[0]["questions"][0]["name"] == "supported"


@pytest.mark.asyncio
async def test_mixed_decision_styles_require_explicit_endpoints(monkeypatch) -> None:
    mock_http(monkeypatch, lambda request: pytest.fail("No HTTP request expected"))

    with pytest.raises(ValueError, match="explicit endpoints entry"):
        async for _ in connect.check_interfaces(
            "decisions", "gpt-6-luna", OPENAI_ENDPOINT, "", styles=("openai", "systemone")
        ):
            pass


ENDPOINTS = {
    "chat": "https://chat.example/v1",
    "responses": "https://responses.example/v1",
    "anthropic": "https://anthropic.example",
    "systemone": "https://hub.example/v1/systemone",
    "openai": "https://decision.example/custom/predicate",
}


def mixed_reply(style):
    if style == "systemone":
        return {"model": "exact-model", "answers": {"supported": {"noul": 0.8}}}
    if style == "openai":
        return {
            "model": "exact-model",
            "answers": [{"type": "predicate", "name": "supported", "probability": 0.8}],
        }
    return response_body(style)


async def compare(**kwargs):
    # These multi-origin transport tests deliberately approve sharing credentials.
    kwargs.setdefault("allow_cross_origin_credentials", True)
    events = [
        event
        async for event in connect.check_interfaces(
            "mixed", "exact-model", "https://fallback.example/v1", "", **kwargs
        )
    ]
    assert isinstance(events[-1], connect.InterfaceResult)
    return events[-1]


@pytest.mark.asyncio
async def test_all_styles_dispatch_exact_model_endpoint_auth_and_deadline(monkeypatch):
    requests = []
    urls = {
        "https://chat.example/v1/chat/completions": "chat",
        "https://responses.example/v1/responses": "responses",
        "https://anthropic.example/v1/messages": "anthropic",
        ENDPOINTS["systemone"]: "systemone",
        ENDPOINTS["openai"]: "openai",
    }

    def handle(request):
        requests.append(request)
        style = urls[str(request.url)]
        body = json.loads(request.content)
        assert body["model"] == "exact-model"
        if style == "anthropic":
            assert request.headers["x-api-key"] == "shared-secret"
        else:
            assert request.headers["authorization"] == "Bearer shared-secret"
        assert request.extensions["timeout"]["read"] == 7
        if style in connect.DECISION_API_STYLES:
            assert "max_tokens" not in body and "max_output_tokens" not in body
        return httpx.Response(200, json=mixed_reply(style))

    mock_http(monkeypatch, handle)
    result = await compare(
        styles=tuple(ENDPOINTS),
        endpoints=ENDPOINTS,
        api_key="shared-secret",
        timeout_seconds=7,
        output_tokens=20,
    )
    assert len(requests) == 5
    assert result.accepted == tuple(ENDPOINTS)
    assert result.accepted_client_types == ("completion", "responses", "decision")
    assert result.tokens_charged_to_budget == 3 * (512 + 20) + 2 * 512
    for style, checked in result.results.items():
        entry = checked.entry
        assert entry.get("endpoint", entry.get("api_base")) == ENDPOINTS[style]
        assert entry["provenance"]["interface_endpoint"] == {
            "api_style": style,
            "endpoint": ENDPOINTS[style],
            "source": "endpoints",
        }
    assert "shared-secret" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        "http://remote.example/v1/systemone",
        "https://user:secret@remote.example/v1/systemone",
        "https://remote.example/v1/systemone?token=secret",
        "https://remote.example/v1/systemone#fragment",
        "not-a-url",
    ],
)
@pytest.mark.parametrize("style", ["systemone", "openai", "anthropic"])
async def test_all_endpoint_values_validated_before_any_network(monkeypatch, bad, style):
    mock_http(monkeypatch, lambda request: pytest.fail("No network before validation"))
    endpoints = {**ENDPOINTS, style: bad}
    with pytest.raises(ValueError):
        await compare(styles=tuple(ENDPOINTS), endpoints=endpoints)
    # Even a supplied override not selected in styles must be safe.
    with pytest.raises(ValueError):
        await compare(styles=("chat",), endpoints=endpoints)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoints",
    [
        {"typo": "https://safe.example/v1"},
        {"systemone": None},
        [],
    ],
)
async def test_invalid_endpoint_mapping_rejected_before_network(monkeypatch, endpoints):
    mock_http(monkeypatch, lambda request: pytest.fail("No HTTP expected"))
    with pytest.raises(ValueError, match="endpoints must map"):
        await compare(styles=("chat",), endpoints=endpoints)


@pytest.mark.asyncio
async def test_each_mixed_decision_style_needs_its_own_mapping_entry(monkeypatch):
    mock_http(monkeypatch, lambda request: pytest.fail("No HTTP expected"))
    with pytest.raises(ValueError, match="entry for systemone"):
        await compare(styles=("openai", "systemone"), endpoints={"openai": ENDPOINTS["openai"]})
    with pytest.raises(ValueError, match="entry for openai"):
        await compare(
            styles=("systemone", "openai"), endpoints={"systemone": ENDPOINTS["systemone"]}
        )


@pytest.mark.asyncio
async def test_mapping_can_leave_chat_on_api_base(monkeypatch):
    urls = []

    def handle(request):
        urls.append(str(request.url))
        style = "systemone" if request.url.host == "hub.example" else "chat"
        return httpx.Response(200, json=mixed_reply(style))

    mock_http(monkeypatch, handle)
    result = await compare(
        styles=("chat", "systemone"),
        endpoints={"systemone": ENDPOINTS["systemone"]},
        api_key="secret",
    )
    assert urls == ["https://fallback.example/v1/chat/completions", ENDPOINTS["systemone"]]
    assert result.results["chat"].entry["provenance"]["interface_endpoint"]["source"] == "api_base"


@pytest.mark.asyncio
@pytest.mark.parametrize("styles", [("systemone", "chat"), ("chat", "systemone")])
async def test_mixed_reasoning_probes_only_chat_and_recommends_accepted_order(monkeypatch, styles):
    bodies = []

    def handle(request):
        style = "systemone" if request.url.host == "hub.example" else "chat"
        body = json.loads(request.content)
        bodies.append((style, body))
        reply = mixed_reply(style)
        if style == "chat":
            reply["choices"][0]["message"]["reasoning_content"] = "Think it through."
        else:
            assert set(body) == {"model", "state", "questions"}
        return httpx.Response(200, json=reply)

    mock_http(monkeypatch, handle)
    result = await compare(
        styles=styles,
        endpoints=ENDPOINTS,
        api_key="secret",
        reasoning_template="effort",
        reasoning_level="on",
        reasoning_output_tokens=64,
    )
    assert [style for style, _ in bodies].count("systemone") == 1
    assert [style for style, _ in bodies].count("chat") == 2
    assert result.accepted == styles
    assert result.reasoning_observed == {"systemone": False, "chat": True}
    assert result.recommended_style == styles[0]
    assert "reasoning_levels" not in result.results["systemone"].entry


@pytest.mark.asyncio
async def test_rejected_decision_does_not_change_chat_reasoning_preference(monkeypatch):
    def handle(request):
        if request.url.host == "hub.example":
            return httpx.Response(422, json={"error": "wrong model"})
        style = "responses" if request.url.host == "responses.example" else "chat"
        reply = mixed_reply(style)
        if style == "chat":
            reply["choices"][0]["message"]["reasoning_content"] = "Reasoning."
        return httpx.Response(200, json=reply)

    mock_http(monkeypatch, handle)
    result = await compare(
        styles=("systemone", "responses", "chat"),
        endpoints=ENDPOINTS,
        api_key="secret",
        reasoning_template="effort",
        reasoning_level="on",
        reasoning_output_tokens=64,
    )
    assert result.accepted == ("responses", "chat")
    assert result.accepted_client_types == ("responses", "completion")
    assert result.recommended_style == "chat"


@pytest.mark.asyncio
async def test_mixed_styles_share_one_budget_without_independent_reset(monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        assert str(request.url) == ENDPOINTS["systemone"]
        reply = mixed_reply("systemone")
        reply["usage"] = {"input_tokens": 700, "output_tokens": 0}
        return httpx.Response(200, json=reply)

    mock_http(monkeypatch, handle)
    result = await compare(
        styles=("systemone", "openai", "chat"),
        endpoints=ENDPOINTS,
        budget_tokens=1100,
        output_tokens=20,
        api_key="secret",
    )
    assert len(requests) == 1
    assert result.tokens_charged_to_budget == 700
    assert result.accepted == ("systemone",)
    for style in ("openai", "chat"):
        record = result.results[style].entry["provenance"]["probes"]["routing"]
        assert record["outcome"] == "not_probed"
        assert record["reason"] == "budget exhausted"
    assert result.recommended_style == "systemone"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [401, 429, 500, "timeout"])
async def test_failed_decision_does_not_retry_or_prevent_next_style(monkeypatch, failure):
    sent = []

    def handle(request):
        sent.append(str(request.url))
        if request.url.host == "hub.example":
            if failure == "timeout":
                raise httpx.ReadTimeout("slow", request=request)
            return httpx.Response(failure, json={"error": "failure"})
        return httpx.Response(200, json=mixed_reply("chat"))

    mock_http(monkeypatch, handle)
    result = await compare(
        styles=("systemone", "chat"), endpoints=ENDPOINTS, api_key="secret", timeout_seconds=3
    )
    assert sent == [ENDPOINTS["systemone"], "https://chat.example/v1/chat/completions"]
    assert result.accepted == ("chat",)
    record = result.results["systemone"].entry["provenance"]["probes"]["routing"]
    assert record["request_shape"]["timeout_seconds"] == 3
    if failure == "timeout":
        assert record["outcome"] == "not_confirmed"
        assert record["timeout_kind"] == "ReadTimeout"
    else:
        assert record["outcome"] == "not_probed"
        assert record["status_code"] == failure
    assert result.tokens_charged_to_budget == 512 + 712


@pytest.mark.asyncio
async def test_reused_decision_evidence_charges_zero_for_current_run(monkeypatch):
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=mixed_reply("systemone")))
    proposal = connect.plan("decisions", "exact-model", "systemone", ENDPOINTS["systemone"], "")
    previous = await connect.run(proposal, approved="minimal")
    assert previous.entry["provenance"]["tokens_charged_to_budget"] == 512
    proposal = connect.plan(
        "decisions",
        "exact-model",
        "systemone",
        ENDPOINTS["systemone"],
        "",
        existing_entry=previous.entry,
    )
    # Exercise stale prior aggregate too, rather than merely an absent aggregate.
    proposal.entry["provenance"]["tokens_charged_to_budget"] = 512
    proposal.entry["provenance"]["requests_accepted"] = []
    mock_http(monkeypatch, lambda request: pytest.fail("Reused evidence must not send"))
    result = await connect.run(proposal, approved="minimal")
    assert result.entry["provenance"]["tokens_charged_to_budget"] == 0
    assert result.entry["provenance"]["requests_accepted"] == ["routing"]
    assert (
        result.entry["provenance"]["probes"]["routing"]
        == previous.entry["provenance"]["probes"]["routing"]
    )


@pytest.mark.parametrize("style", connect.DECISION_API_STYLES)
@pytest.mark.parametrize("field", ["media_inputs", "test_media_inputs"])
def test_decision_plan_rejects_unverified_non_image_media(style, field):
    with pytest.raises(ValueError, match="only media_inputs"):
        connect.plan("decisions", "exact-model", style, ENDPOINTS[style], "", **{field: ["audio"]})


@pytest.mark.parametrize("style", connect.DECISION_API_STYLES)
def test_decision_plan_allows_empty_media_declarations(style):
    proposal = connect.plan(
        "decisions",
        "exact-model",
        style,
        ENDPOINTS[style],
        "",
        media_inputs=[],
        test_media_inputs=[],
    )
    assert proposal.entry["media_inputs"] == []


@pytest.mark.asyncio
async def test_two_decision_styles_ignore_reasoning_and_deduplicate_client_types(monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        style = "systemone" if request.url.host == "hub.example" else "openai"
        return httpx.Response(200, json=mixed_reply(style))

    mock_http(monkeypatch, handle)
    result = await compare(
        styles=("openai", "systemone"),
        endpoints=ENDPOINTS,
        reasoning_template="invalid-chat-template",
    )
    assert len(requests) == 2
    assert result.accepted == ("openai", "systemone")
    assert result.accepted_client_types == ("decision",)
    assert result.recommended_style == "openai"
    assert result.reasoning_observed == {"openai": False, "systemone": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("style", connect.DECISION_API_STYLES)
async def test_decision_deadline_cancels_pending_http_without_retry(monkeypatch, style):
    requests = []
    cancelled = []
    blocked = asyncio.Event()

    async def handle(request):
        requests.append(str(request.url))
        if str(request.url) == ENDPOINTS[style]:
            try:
                await blocked.wait()
            except asyncio.CancelledError:
                cancelled.append(str(request.url))
                raise
        return httpx.Response(200, json=mixed_reply("chat"))

    mock_http(monkeypatch, handle)
    result = await compare(
        styles=(style, "chat"), endpoints=ENDPOINTS, api_key="secret", timeout_seconds=1
    )
    assert requests == [ENDPOINTS[style], "https://chat.example/v1/chat/completions"]
    assert cancelled == [ENDPOINTS[style]]
    assert result.accepted == ("chat",)
    record = result.results[style].entry["provenance"]["probes"]["routing"]
    assert record["outcome"] == "not_confirmed"
    assert record["timeout_kind"] == "probe_deadline"
    assert 0.8 <= record["elapsed_seconds"] < 5


@pytest.mark.asyncio
async def test_no_accepted_styles_has_no_recommendation(monkeypatch):
    mock_http(monkeypatch, lambda request: httpx.Response(401, json={"error": "auth"}))
    result = await compare(styles=("systemone", "chat"), endpoints=ENDPOINTS, api_key="secret")
    assert result.accepted == ()
    assert result.accepted_client_types == ()
    assert result.recommended_style is None


@pytest.mark.asyncio
@pytest.mark.parametrize("credential", [{"api_key": "secret-canary"}, {"api_key_env": "UNSET_KEY"}])
@pytest.mark.parametrize(
    "target", ["https://other.example/full", "https://fallback.example:444/full"]
)
async def test_cross_origin_credentials_default_refuses_before_any_request(
    monkeypatch, credential, target
):
    mock_http(monkeypatch, lambda request: pytest.fail("Must validate before any request"))
    key_env = credential.get("api_key_env", "")
    with pytest.raises(
        ValueError, match="allow_cross_origin_credentials=True.*--share-interface-key"
    ) as error:
        _ = [
            event
            async for event in connect.check_interfaces(
                "local",
                "exact-model",
                "https://fallback.example/v1",
                key_env,
                styles=("chat", "systemone"),
                endpoints={"systemone": target},
                **({"api_key": credential["api_key"]} if "api_key" in credential else {}),
            )
        ]
    assert "secret-canary" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target, key",
    [
        ("https://FALLBACK.example:443/full/path", "secret"),
        ("https://other.example/full", None),
    ],
)
async def test_same_origin_paths_and_unauthenticated_cross_origin_allowed(monkeypatch, target, key):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=mixed_reply("systemone"))

    mock_http(monkeypatch, handle)
    events = [
        event
        async for event in connect.check_interfaces(
            "local",
            "exact-model",
            "https://fallback.example/v1",
            "",
            styles=("systemone",),
            endpoints={"systemone": target},
            api_key=key,
        )
    ]
    assert events[-1].accepted == ("systemone",)
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_local_scheme_change_requires_credential_opt_in(monkeypatch):
    mock_http(monkeypatch, lambda request: pytest.fail("No cross-scheme request"))
    with pytest.raises(ValueError, match="--share-interface-key"):
        _ = [
            event
            async for event in connect.check_interfaces(
                "local",
                "exact-model",
                "https://localhost:1234/v1",
                "",
                api_key="secret",
                styles=("chat", "systemone"),
                endpoints={"systemone": "http://localhost:1234/full"},
            )
        ]
