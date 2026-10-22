# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reasoning facts, active defaults, registry/Predict and session wire offline."""

import json
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest

from nooa.unifiedllm import connect, registry
from nooa.unifiedllm.connect._reasoning import native_levels, reasoning_metadata
from nooa.unifiedllm.connect._session import reply_budget, session_steps
from tests.unifiedllm.connect.connect_http import mock_http, response_body
from tests.unifiedllm.connect.reasoning_fixture import MODEL_ID, NATIVE_MODEL
from tests.unifiedllm.test_anthropic_native_profile import TypedAgent

BASE = "https://api.anthropic.com/v1"


def levels(model=NATIVE_MODEL, **kwargs):
    return native_levels(
        model,
        model_id=MODEL_ID,
        api_style="anthropic",
        source=BASE + "/models",
        endpoint=BASE,
        **kwargs,
    )


async def test_native_facts_survive_exact_listing_and_details(monkeypatch):
    raw = deepcopy(NATIVE_MODEL)
    raw["capabilities"]["effort"]["malicious"] = {"supported": True, "api_key": "BAD"}
    raw["capabilities"]["thinking"]["request"] = {"budget_tokens": 123}
    mock_http(monkeypatch, lambda request: httpx.Response(200, json={"data": [raw]}))
    listing = await connect.discover(BASE, api_style="anthropic", api_key="fake")
    details = await connect.server_details(BASE, MODEL_ID, api_style="anthropic", listing=listing)
    caps = details.model["capabilities"]
    assert caps == NATIVE_MODEL["capabilities"]
    assert details.model["max_input_tokens"] == 1000000
    assert details.model["max_output_tokens"] == 128000
    assert details.requests == 0
    assert details.records[0]["fields"]["capabilities.thinking.types.adaptive.supported"] == (
        "data[].capabilities.thinking.types.adaptive.supported"
    )
    assert details.records[0]["source"] == BASE + "/models"
    assert list(levels(details.model)) == ["low", "medium", "high", "xhigh", "max", "off"]


@pytest.mark.parametrize(
    "caps, expected",
    [
        ({}, {}),
        ({"thinking": {}}, {"capabilities": {"thinking": {}}}),
        ({"thinking": {"supported": False}}, {"capabilities": {"thinking": {"supported": False}}}),
        ({"effort": {"supported": True}}, {"capabilities": {"effort": {"supported": True}}}),
    ],
)
def test_unknown_empty_false_flags_are_distinct_not_controls(caps, expected):
    model = {"id": MODEL_ID, "capabilities": caps, "supports_reasoning": True}
    assert reasoning_metadata(model) == expected
    assert levels(model) == ({} if caps.get("thinking", {}).get("supported") is False else None)


@pytest.mark.parametrize("style", ["chat", "responses", "systemone", "openai"])
def test_native_controls_never_compile_for_foreign_interface(style):
    assert (
        native_levels(
            NATIVE_MODEL, model_id=MODEL_ID, api_style=style, source=BASE + "/models", endpoint=BASE
        )
        is None
    )
    assert (
        native_levels(
            NATIVE_MODEL,
            model_id="foreign",
            api_style="anthropic",
            source=BASE + "/models",
            endpoint=BASE,
        )
        is None
    )
    assert (
        native_levels(
            NATIVE_MODEL,
            model_id=MODEL_ID,
            api_style="anthropic",
            source="https://foreign.test/models",
            endpoint=BASE,
        )
        is None
    )


async def test_saved_active_native_effort_used_by_registry_predict_and_all_checks(
    monkeypatch, tmp_path
):
    bodies = []

    def handle(request):
        assert request.method == "POST" and request.url.path == "/v1/messages"
        body = json.loads(request.content)
        bodies.append(body)
        data = response_body("anthropic", '{"value":323}')
        data["content"].insert(
            0,
            {"type": "thinking", "thinking": "offline reasoning", "signature": "offline-signature"},
        )
        data["usage"]["cache_read_input_tokens"] = 15 if len(bodies) > 1 else 0
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    proposal = connect.plan(
        "local",
        MODEL_ID,
        "anthropic",
        BASE,
        "",
        reasoning_levels=levels(),
        reply_tokens=4096,
        structured_output_mode="anthropic",
        session_checks=True,
    )
    proposal.entry["reasoning_level"] = "high"
    proposal.entry["reasoning_default"] = "low"  # descriptive, NOT the active label
    proposal = connect.refresh_plan(proposal)
    before = deepcopy(proposal.entry)
    path = tmp_path / "models.yaml"
    connect.write(proposal.entry, path, alias="local")
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "_loaded", False)
    registry.reload_registry(path)
    c = registry.get_llm_client("local", api_key="fake")
    try:
        assert c.reasoning_level == "high" and c.reasoning_default == "low"
        await c.acall([{"role": "user", "content": "hello"}])
        result = await TypedAgent(llm=c).compute()
        assert result.value == 323
        assert bodies[-1]["output_config"]["format"]["type"] == "json_schema"
        for body in bodies:
            assert body["thinking"] == {"type": "adaptive"}
            assert body["output_config"]["effort"] == "high"
            assert body["max_tokens"] == 4096
            assert "reasoning_level" not in body and "extra_body" not in body
        await c.acall([{"role": "user", "content": "hello"}], reasoning_level="low")
        assert bodies[-1]["output_config"]["effort"] == "low"
        await c.acall(
            [{"role": "user", "content": "hello"}],
            reasoning_level=None,
            thinking={"type": "disabled"},
        )
        assert bodies[-1]["thinking"] == {"type": "disabled"}
        await c.acall([{"role": "user", "content": "hello"}])
        assert bodies[-1]["output_config"]["effort"] == "high"
    finally:
        await c.aclose()
    start = len(bodies)
    routing = replace(
        proposal,
        probes=tuple(p for p in proposal.probes if p.name == "routing"),
        session_checks=False,
    )
    result = await connect.run(routing, approved="minimal", api_key="fake")
    assert result.entry["provenance"]["probes"]["routing"]["outcome"] == "accepted"
    updates = [
        u
        async for u in session_steps("local", proposal.entry, api_key="fake", budget_tokens=200000)
    ]
    records = {u.name: u.outcome for u in updates if u.outcome["outcome"] != "running"}
    assert records["session"]["outcome"] == "completed"
    assert records["reasoning_retention"]["tested_reasoning_level"] == "high"
    assert all(
        b["thinking"] == {"type": "adaptive"}
        and b["output_config"]["effort"] == "high"
        and b["max_tokens"] == 4096
        for b in bodies[start:]
    )
    assert proposal.entry == before


def test_active_level_cap_and_selector_changes_invalidate_evidence():
    proposal = connect.plan(
        "local",
        MODEL_ID,
        "anthropic",
        BASE,
        "",
        reply_tokens=4096,
        reasoning_levels={
            "low": {**levels()["low"], "max_tokens": 2048},
            "high": {**levels()["high"], "max_tokens": 8192},
        },
    )
    proposal.entry["reasoning_level"] = "high"
    proposal.entry["reasoning_default"] = "low"
    entry = connect.configure_entry(proposal.entry)
    assert reply_budget(entry, 200000) == (8192, 8192, 8192 + 3 * 8192)
    entry["provenance"]["probes"] = {"routing": {"outcome": "accepted"}}
    entry["provenance"]["session_checks"] = {"outcome": "completed"}
    entry["reasoning_level"] = "low"
    updated = connect.configure_entry(entry)
    assert updated["provenance"]["probes"] == {}
    assert "session_checks" not in updated["provenance"]
    same = connect.plan(
        "local", MODEL_ID, "anthropic", BASE, "", existing_entry=updated, reply_tokens=4096
    )
    assert same.entry["reasoning_level"] == "low"
    assert same.entry["reasoning_levels"] == updated["reasoning_levels"]
    other = connect.plan("local", "other", "anthropic", BASE, "", existing_entry=updated)
    assert "reasoning_level" not in other.entry


def test_descriptive_default_does_not_select_session_or_runtime():
    entry = connect.plan(
        "local",
        MODEL_ID,
        "anthropic",
        BASE,
        "",
        reply_tokens=4096,
        reasoning_levels={"high": {**levels()["high"], "max_tokens": 8192}},
    ).entry
    entry["reasoning_default"] = "high"
    assert reply_budget(entry, 200000)[0] == 4096
    client = registry.client_from_config("local", entry, api_key="fake")
    assert client.reasoning_level is None
    assert "thinking" not in client._prepare_call_config({})


def test_invalid_active_selection_and_decision_fields_rejected_offline():
    entry = connect.plan("local", MODEL_ID, "anthropic", BASE, "", reply_tokens=4096).entry
    entry["reasoning_level"] = "high"
    with pytest.raises(ValueError):
        connect.configure_entry(entry)
    entry = connect.plan(
        "decision", "native-decision", "systemone", "https://api.test/decision", ""
    ).entry
    entry["reasoning_level"] = None
    with pytest.raises(ValueError, match="chat-only"):
        connect.configure_entry(entry)


async def test_refresh_off_removes_stale_adaptive_controls_before_wire(monkeypatch):
    bodies = []
    mock_http(
        monkeypatch,
        lambda request: (
            bodies.append(json.loads(request.content))
            or httpx.Response(200, json=response_body("anthropic", '{"value":323}'))
        ),
    )
    proposal = connect.plan(
        "local",
        MODEL_ID,
        "anthropic",
        BASE,
        "",
        reasoning_levels=levels(),
        reasoning_level="high",
        reply_tokens=4096,
    )
    assert proposal.probes[0].body["output_config"] == {"effort": "high"}
    proposal.entry["reasoning_level"] = "off"
    off = connect.refresh_plan(proposal)
    assert "output_config" not in off.probes[0].body
    assert off.probes[0].body["thinking"] == {"type": "disabled"}
    await connect.check_stage(off, "routing", api_key="fake")
    assert bodies and all(
        b["thinking"] == {"type": "disabled"} and "effort" not in b.get("output_config", {})
        for b in bodies
    )


def test_false_native_parent_cannot_authorize_off_and_explicit_clear():
    model = deepcopy(NATIVE_MODEL)
    model["capabilities"]["thinking"]["supported"] = False
    assert levels(model) == {}
    proposal = connect.plan(
        "local",
        MODEL_ID,
        "anthropic",
        BASE,
        "",
        reasoning_levels=levels(),
        reasoning_level="high",
        reply_tokens=4096,
    )
    clear = connect.plan(
        "local",
        MODEL_ID,
        "anthropic",
        BASE,
        "",
        existing_entry=proposal.entry,
        reasoning_level=None,
        reply_tokens=4096,
    )
    assert "reasoning_level" not in clear.entry
    assert clear.entry["reasoning_levels"] == proposal.entry["reasoning_levels"]


async def test_same_route_active_replan_retains_whitelist_and_actual_controls(monkeypatch):
    bodies = []
    mock_http(
        monkeypatch,
        lambda request: (
            bodies.append(json.loads(request.content))
            or httpx.Response(200, json=response_body("chat", '{"value":323}'))
        ),
    )
    monkeypatch.setenv("NOOA_LLM_TRANSPORT", "litellm")
    original = connect.plan(
        "local",
        "vendor/unlisted-model",
        "chat",
        "https://api.test/v1",
        "",
        reasoning_levels={"max": {"reasoning_effort": "max"}},
        reasoning_level="max",
        reply_tokens=4096,
    )
    original.entry["allowed_openai_params"].append("custom_control")
    inherited = connect.plan(
        "local",
        "vendor/unlisted-model",
        "chat",
        "https://api.test/v1",
        "",
        existing_entry=original.entry,
        reply_tokens=4096,
    )
    assert inherited.entry["reasoning_level"] == "max"
    assert inherited.entry["allowed_openai_params"] == ["custom_control", "reasoning_effort"]
    client = registry.client_from_config("local", inherited.entry, api_key="fake")
    try:
        await client.acall([{"role": "user", "content": "hello"}])
        assert bodies[-1]["reasoning_effort"] == "max" and bodies[-1]["max_tokens"] == 4096
    finally:
        await client.aclose()
    changed = connect.plan(
        "local", "different", "chat", "https://api.test/v1", "", existing_entry=original.entry
    )
    assert "reasoning_level" not in changed.entry
    assert "allowed_openai_params" not in changed.entry
