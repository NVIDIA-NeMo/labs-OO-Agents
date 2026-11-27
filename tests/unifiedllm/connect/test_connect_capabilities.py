# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connect checks for structured-output formats and the developer role (mocked HTTP only)."""

import json

import httpx
import pytest

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body


def plan(style="chat", **kwargs):
    return connect.plan(
        "local", "model", style, "https://api.test/v1", "", reply_tokens=1000, **kwargs
    )


def _format(body):
    fmt = body.get("response_format") or (body.get("text") or {}).get("format")
    return fmt if isinstance(fmt, dict) else None


def _probe_kind(body):
    """Which capability check a request is, or None for other checks."""
    fmt = _format(body)
    items = body.get("messages", body.get("input")) or []
    if fmt and "Reply with the JSON object" in json.dumps(items):
        return f"format:{fmt['type']}"
    if any(item.get("role") == "developer" for item in items):
        return "role:developer"
    return None


def serve(monkeypatch, style, rejections=None):
    """Answer every request; ``rejections`` maps a check name to (status, message)."""
    bodies = []
    rejections = rejections or {}

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        kind = _probe_kind(body)
        if kind in rejections:
            status, message = rejections[kind]
            return httpx.Response(
                status, json={"error": {"message": message, "type": "invalid_request_error"}}
            )
        text = '{"ok": true}' if kind else '{"value":323}'
        return httpx.Response(200, json=response_body(style, text))

    mock_http(monkeypatch, handle)
    return bodies


def test_capability_checks_are_opt_in():
    names = [probe.name for probe in plan().probes]
    assert not any(name.startswith(("format:", "role:")) for name in names)
    entry = plan("anthropic").entry
    assert "developer_role" not in entry


@pytest.mark.parametrize("style", ["chat", "responses"])
def test_probe_request_shapes(style):
    probes = {p.name: p for p in plan(style, capability_checks=True).probes}
    schema = probes["format:json_schema"].body
    obj = probes["format:json_object"].body
    role = probes["role:developer"].body
    if style == "responses":
        assert schema["text"]["format"]["type"] == "json_schema"
        assert schema["text"]["format"]["strict"] is True
        assert obj["text"] == {"format": {"type": "json_object"}}
        assert role["input"][0]["role"] == "developer"
    else:
        assert schema["response_format"]["type"] == "json_schema"
        assert schema["response_format"]["json_schema"]["schema"]["required"] == ["ok"]
        assert obj["response_format"] == {"type": "json_object"}
        assert role["messages"][0]["role"] == "developer"
    key = "input" if style == "responses" else "messages"
    assert "JSON" in obj[key][0]["content"]  # JSON mode requires the word in the prompt
    for probe in probes.values():
        assert probe.body.get("max_output_tokens", probe.body.get("max_tokens")) == 1000


async def test_responses_accepts_schema_and_developer(monkeypatch):
    bodies = serve(monkeypatch, "responses")
    result = await connect.check_stage(
        plan("responses", capability_checks=True), "capabilities", api_key="fake"
    )
    kinds = [_probe_kind(body) for body in bodies]
    assert kinds == [None, "format:json_schema", "role:developer"]  # json_object not needed
    entry = result.entry
    assert entry["structured_output_format"] == "json_schema"
    assert [k for k in entry if k.startswith("structured_output_format")] == [
        "structured_output_format"
    ]  # one key
    assert entry["developer_role"] is True
    assert entry["provenance"]["structured_output_format"]["checks"] == {
        "json_schema": "accepted",
        "json_object": "not_probed",
    }
    assert entry["provenance"]["structured_output_format"]["value"] == "json_schema"
    assert entry["provenance"]["developer_role"]["value"] is True
    assert "format:json_object" not in entry["provenance"]["probes"]
    verdict = connect.verdict(entry)
    assert {"format:json_schema", "role:developer"} <= set(verdict.passed)


async def test_schema_rejected_as_unsupported_probes_json_object(monkeypatch):
    # The typed routing baseline uses the configured format, so a DeepSeek-like
    # endpoint is connected with structured_output_format: json_object.
    bodies = serve(
        monkeypatch,
        "chat",
        {"format:json_schema": (400, "This response_format type is unavailable now")},
    )
    proposal = plan(
        "chat",
        capability_checks=True,
        existing_entry={**plan().entry, "structured_output_format": "json_object"},
    )
    assert proposal.entry["structured_output_format"] == "json_object"
    result = await connect.check_stage(proposal, "capabilities", api_key="fake")
    kinds = [_probe_kind(body) for body in bodies]
    # The developer check follows; LiteLLM Chat sends it as system (see below).
    assert kinds[1:3] == ["format:json_schema", "format:json_object"] and len(kinds) == 4
    assert bodies[0]["response_format"] == {"type": "json_object"}  # configured baseline
    probes = result.entry["provenance"]["probes"]
    assert probes["format:json_schema"]["outcome"] == "unsupported"
    assert probes["format:json_object"]["outcome"] == "accepted"
    assert result.entry["structured_output_format"] == "json_object"
    assert "unavailable now" not in repr(result)  # provider text is never stored


async def test_schema_rejected_without_explicit_setting_still_probes_json_object(monkeypatch):
    bodies = serve(
        monkeypatch,
        "responses",
        {
            "format:json_schema": (
                400,
                "Invalid parameter: text.format json_schema is not supported",
            )
        },
    )
    result = await connect.check_stage(
        plan("responses", capability_checks=True), "capabilities", api_key="fake"
    )
    assert "format:json_object" in [_probe_kind(body) for body in bodies]
    assert result.entry["structured_output_format"] == "json_object"
    assert result.entry["provenance"]["structured_output_format"]["checks"] == {
        "json_schema": "unsupported",
        "json_object": "accepted",
    }


async def test_both_formats_unsupported_writes_no_key(monkeypatch):
    unsupported = (400, "Invalid parameter: text.format json_schema is not supported")
    serve(
        monkeypatch,
        "responses",
        {
            "format:json_schema": unsupported,
            "format:json_object": (400, "response_format json_object is not supported"),
        },
    )
    result = await connect.check_stage(
        plan("responses", capability_checks=True), "capabilities", api_key="fake"
    )
    assert "structured_output_format" not in result.entry
    note = result.entry["provenance"]["structured_output_format"]
    assert note["value"] is None
    assert note["checks"] == {"json_schema": "unsupported", "json_object": "unsupported"}


async def test_conclusive_detection_overwrites_an_existing_value(monkeypatch):
    # Connect writes what it detected, whoever set the existing value.
    serve(monkeypatch, "responses")
    existing = {**plan("responses").entry, "structured_output_format": "json_object"}
    proposal = plan("responses", capability_checks=True, existing_entry=existing)
    result = await connect.check_stage(proposal, "capabilities", api_key="fake")
    assert result.entry["structured_output_format"] == "json_schema"
    note = result.entry["provenance"]["structured_output_format"]
    assert note["value"] == "json_schema"
    assert set(note) == {"source", "checks", "value", "checked_at"}


async def test_inconclusive_detection_leaves_an_existing_value(monkeypatch):
    serve(
        monkeypatch,
        "responses",
        {
            "format:json_schema": (500, "internal error"),
            "format:json_object": (500, "internal error"),
        },
    )
    existing = {**plan("responses").entry, "structured_output_format": "json_object"}
    proposal = plan("responses", capability_checks=True, existing_entry=existing)
    result = await connect.check_stage(proposal, "capabilities", api_key="fake")
    assert result.entry["structured_output_format"] == "json_object"
    assert result.entry["provenance"]["structured_output_format"]["value"] is None


@pytest.mark.parametrize(
    "status,message",
    [
        (500, "internal error"),
        (400, "invalid model"),
        (401, "response_format not supported"),
        (429, "rate limit"),
    ],
)
async def test_other_failures_leave_formats_unknown(monkeypatch, status, message):
    bodies = serve(monkeypatch, "responses", {"format:json_schema": (status, message)})
    result = await connect.check_stage(
        plan("responses", capability_checks=True), "capabilities", api_key="fake"
    )
    assert "format:json_object" not in [_probe_kind(body) for body in bodies]
    assert "structured_output_format" not in result.entry
    assert result.entry["provenance"]["structured_output_format"]["value"] is None


@pytest.mark.parametrize(
    "status,message,expected",
    [
        (400, "Invalid value: 'developer'. Supported values are: 'system', 'user'", False),
        (422, "Unexpected message role developer", False),
        (400, "invalid model", None),
        (503, "developer role unavailable", None),
    ],
)
async def test_developer_rejection_detection(monkeypatch, status, message, expected):
    serve(monkeypatch, "responses", {"role:developer": (status, message)})
    result = await connect.check_stage(
        plan("responses", capability_checks=True), "capabilities", api_key="fake"
    )
    assert result.entry.get("developer_role") is expected
    assert result.entry["provenance"]["developer_role"]["value"] is expected


async def test_litellm_chat_developer_check_sends_developer_on_the_wire(monkeypatch):
    # LiteLLM sends developer messages as system on ordinary Chat routes; the
    # role check keeps the developer role for its own request only.
    bodies = serve(monkeypatch, "chat")
    result = await connect.check_stage(
        plan("chat", capability_checks=True), "capabilities", api_key="fake"
    )
    record = result.entry["provenance"]["probes"]["role:developer"]
    assert record["outcome"] == "accepted"
    assert result.entry["developer_role"] is True
    assert result.entry["structured_output_format"] == "json_schema"
    assert len(bodies) == 3
    assert bodies[-1]["messages"][0]["role"] == "developer"


def test_developer_role_kept_only_inside_the_check():
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    from nooa.unifiedllm.adapters.litellm_chat import keep_developer_role

    config = ProviderConfigManager.get_provider_chat_config(
        model="model", provider=LlmProviders.OPENAI
    )
    messages = [{"role": "developer", "content": "x"}]
    with keep_developer_role():
        assert config.translate_developer_role_to_system_role(messages)[0]["role"] == "developer"
    # Normal requests keep LiteLLM's behavior.
    assert config.translate_developer_role_to_system_role(messages)[0]["role"] == "system"


async def test_anthropic_records_no_developer_role_without_probing(monkeypatch):
    bodies = serve(monkeypatch, "anthropic")
    proposal = plan("anthropic", capability_checks=True)
    assert not any(p.name.startswith(("format:", "role:")) for p in proposal.probes)
    assert proposal.entry["developer_role"] is False
    assert proposal.entry["provenance"]["developer_role"]["check"] == "not_applicable"
    assert "structured_output_format" not in proposal.entry
    result = await connect.run(proposal, approved="minimal", api_key="fake")
    assert len(bodies) == 1
    assert result.entry["developer_role"] is False


async def test_capability_probe_ignores_saved_facts(monkeypatch):
    # A saved developer_role: false must not turn the check's developer
    # message into a system message.
    bodies = serve(monkeypatch, "responses")
    existing = {**plan("responses").entry, "developer_role": False}
    proposal = plan("responses", capability_checks=True, existing_entry=existing)
    assert proposal.entry["developer_role"] is False
    result = await connect.check_stage(proposal, "capabilities", api_key="fake")
    assert "role:developer" in [_probe_kind(body) for body in bodies]
    assert result.entry["developer_role"] is True


def test_capabilities_stage_requires_planned_checks():
    with pytest.raises(ValueError, match="capability_checks"):
        import asyncio

        asyncio.run(connect.check_stage(plan(), "capabilities", api_key="fake"))


@pytest.mark.parametrize(
    "record,status",
    [
        ({"outcome": "unsupported"}, "passed"),
        # The endpoint answered but did not settle the fact: unknown, not a failure.
        ({"outcome": "not_confirmed", "settings_sent": False}, "passed"),
        ({"outcome": "rejected", "status_code": 400}, "passed"),
        # No answer at all (for example HTTP 503) still needs attention.
        ({"outcome": "not_probed", "error": "ServiceUnavailableError"}, "attention"),
        ({"outcome": "not_probed", "reason": "budget exhausted"}, "skipped"),
    ],
)
@pytest.mark.parametrize("name", ["format:json_schema", "role:developer"])
def test_inconclusive_capability_check_is_not_a_failure(name, record, status):
    from nooa.unifiedllm.connect._records import check_status

    if record["outcome"] != "not_probed":
        record = {**record, "verification_fingerprint": "f", "verification_current": True}
    assert check_status(name, record) == status
