# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persisted reply limits and encrypted reasoning reach the runtime's wire body."""

import json

import httpx
import pytest
import yaml

from nooa.unifiedllm import connect, registry
from tests.unifiedllm.connect.connect_http import mock_http, response_body


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_saved_defaults_reach_wire(tmp_path, monkeypatch, style, asynchronous):
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response_body(style))

    mock_http(monkeypatch, handle)
    original = httpx.Client.__init__

    def init(self, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handle)
        original(self, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", init)
    monkeypatch.setattr(registry, "MODELS", {})
    model = "claude-sonnet-4-6" if style == "anthropic" else "gpt-5.1"
    proposal = connect.plan("test", model, style, "https://api.test/v1", "", reply_tokens=1536)
    path = tmp_path / "models.yaml"
    connect.write(proposal.entry, path, alias="test")
    registry.reload_registry(path)
    client = registry.get_llm_client("test", api_key="test-key")
    try:
        messages = [{"role": "user", "content": "Hello"}]
        if asynchronous:
            await client.acall(messages)
        else:
            client.call(messages)
    finally:
        await client.aclose()
    assert len(sent) == 1
    body = sent[0]
    caps = {
        k: body[k]
        for k in ("max_tokens", "max_output_tokens", "max_completion_tokens")
        if k in body
    }
    assert len(caps) == 1 and list(caps.values()) == [1536]
    if style == "responses":
        assert body["max_output_tokens"] == 1536
        assert body["store"] is False
        assert body["include"] == ["reasoning.encrypted_content"]


@pytest.mark.parametrize(
    "ceiling,recommended,expected", [(None, None, 32768), (1024, None, 1024), (16384, 4096, 4096)]
)
def test_plan_separates_recommendation_from_ceiling(ceiling, recommended, expected):
    proposal = connect.plan(
        "test",
        "model",
        "chat",
        "https://api.test/v1",
        "",
        catalogue={
            "id": "model",
            "top_provider": {"max_completion_tokens": ceiling},
            "default_parameters": {"max_tokens": recommended},
        },
    )
    assert proposal.entry["max_tokens"] == expected
    assert "max_output_tokens" not in proposal.entry


@pytest.mark.parametrize("cap", [0, -1, True, "100", 16385])
def test_invalid_reply_limit_cannot_be_saved(tmp_path, cap):
    path = tmp_path / "models.yaml"
    with pytest.raises(ValueError):
        connect.write(
            {"model_name": "openai/model", "max_tokens": cap, "context_window": 16384},
            path,
            alias="test",
        )
    assert not path.exists()


@pytest.mark.parametrize("include", [None, [], ["reasoning.encrypted_content"]])
def test_save_normalizes_legacy_stateless_entry_without_mutation(tmp_path, include):
    from copy import deepcopy

    entry = {"model_name": "openai/model", "client_type": "responses", "store": False}
    if include is not None:
        entry["include"] = include
    original = deepcopy(entry)
    path = tmp_path / "models.yaml"
    connect.write(entry, path, alias="test")
    saved = yaml.safe_load(path.read_text())["models"]["test"]
    assert saved["max_tokens"] == 32768
    assert saved["include"] == ([] if include == [] else ["reasoning.encrypted_content"])
    assert entry == original


def test_thinking_budget_raises_only_its_level_cap():
    entry = connect.configure_entry(
        {
            "max_tokens": 2048,
            "reasoning_levels": {
                "thinking": {"thinking": {"type": "enabled", "budget_tokens": 4096}},
                "plain": {"thinking": {"type": "disabled"}},
            },
        }
    )
    assert entry["max_tokens"] == 2048
    assert entry["reasoning_levels"]["thinking"]["max_tokens"] == 5120
    assert "max_tokens" not in entry["reasoning_levels"]["plain"]


@pytest.mark.parametrize(
    "extra", [{"max_tokens": 99999}, {"max_output_tokens": 99999}, {"include": []}, {"store": True}]
)
def test_extra_body_cannot_shadow_managed_defaults(extra):
    with pytest.raises(ValueError, match="not in extra_body"):
        connect.configure_entry(
            {"client_type": "responses", "max_tokens": 2048, "extra_body": extra}
        )


def test_preserved_cap_updates_stale_provenance():
    entry = connect.configure_entry(
        {
            "max_tokens": 1024,
            "provenance": {"reply_limit": {"source": "connect_default", "value": 8192}},
        }
    )
    assert entry["provenance"]["reply_limit"] == {"source": "entry", "value": 1024}


@pytest.mark.parametrize("base_key", ["max_tokens", "max_output_tokens"])
def test_responses_reply_alias_precedence(base_key):
    from nooa.unifiedllm import ResponsesClient

    with ResponsesClient(
        "openai/gpt-5.1",
        api_key="test-key",
        **{base_key: 8192},
        reasoning_levels={"short": {"max_tokens": 1024}, "long": {"max_output_tokens": 4096}},
    ) as client:
        for overrides, expected in [
            ({"max_tokens": 200}, 200),
            ({"max_output_tokens": 300}, 300),
            ({"reasoning_level": "short"}, 1024),
            ({"reasoning_level": "long"}, 4096),
        ]:
            config = client._prepare_call_config(overrides)
            assert config["max_output_tokens"] == expected
            assert "max_tokens" not in config
        with pytest.raises(ValueError, match="only one reply token limit"):
            client._prepare_call_config({"max_tokens": 100, "max_output_tokens": 200})


@pytest.mark.asyncio
@pytest.mark.parametrize("saved,tested", [(1024, 1024), (8192, 8192)])
async def test_session_reports_the_cap_actually_tested(monkeypatch, saved, tested):
    from nooa.unifiedllm.connect._session import TOKEN_RESERVATION, session_steps

    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        data = response_body("chat")
        data["choices"][0]["message"]["reasoning_content"] = "test reasoning"
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    proposal = connect.plan(
        "test", "gpt-5.1", "chat", "https://api.test/v1", "", reply_tokens=saved
    )
    events = [
        e
        async for e in session_steps(
            "test",
            proposal.entry,
            api_key="test-key",
            budget_tokens=max(TOKEN_RESERVATION, 3 * (8192 + 3 * saved)),
        )
    ]
    assert len(bodies) == 3
    assert all(b.get("max_completion_tokens", b.get("max_tokens")) == tested for b in bodies)
    seed = next(
        e.outcome for e in events if e.name == "session:seed" and e.outcome["outcome"] == "accepted"
    )
    assert seed["configured_reply_tokens"] == saved
    assert seed["tested_reply_tokens"] == tested
    assert seed["settings_sent"] is True
    assert saved == tested
