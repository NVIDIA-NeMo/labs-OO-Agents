# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit cache opt-out uses the native Predict baseline, never a schema fallback."""

import json

import httpx
import pytest

from nooa.unifiedllm import connect, registry
from nooa.unifiedllm.connect._baseline import typed_baseline
from nooa.unifiedllm.connect._records import text_baseline_passed
from tests.unifiedllm.connect.connect_http import mock_http, response_body


def plan(**kwargs):
    return connect.plan(
        "local",
        "claude-sonnet-4-6",
        "responses",
        "https://api.test/v1",
        "",
        reply_tokens=1000,
        **kwargs,
    )


def assert_cache_off_schema(body, *, cache_off=True):
    def check(value):
        if isinstance(value, dict):
            assert (
                not {
                    "prompt_cache_options",
                    "prompt_cache_breakpoint",
                    "cache_control",
                    "nooa_cache_boundary",
                }
                & value.keys()
            )
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    if cache_off:
        check(body)
    assert body["max_output_tokens"] == 1000
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]
    format_ = body["text"]["format"]
    assert format_["type"] == "json_schema" and format_["strict"] is True
    schema = format_["schema"]
    assert schema["type"] == "object" and schema["required"] == ["value"]
    assert schema["properties"]["value"]["type"] == "integer"
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "case", ["success", "malformed", "missing", "wrong", "reasoning", "truncated", "400"]
)
async def test_actual_predict_responses_cache_off(monkeypatch, case):
    sent = []

    def handle(request):
        assert request.url.path == "/v1/responses"
        body = json.loads(request.content)
        sent.append(body)
        assert_cache_off_schema(body)
        if case == "400":
            return httpx.Response(
                400, json={"error": {"message": "rejected", "type": "invalid_request_error"}}
            )
        text = {"malformed": "{", "missing": "{}", "wrong": '{"value":322}'}.get(
            case, '{"value":323}'
        )
        response = response_body("responses", text)
        if case == "truncated":
            response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        if case == "reasoning":
            response["output"] = [
                {
                    "type": "reasoning",
                    "id": "rs-1",
                    "summary": [{"type": "summary_text", "text": '{"value":323}'}],
                }
            ]
        return httpx.Response(200, json=response)

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(plan(cache_enabled=False), "routing", api_key="fake")
    record = result.entry["provenance"]["probes"]["routing"]
    assert text_baseline_passed(record) is (case == "success"), record
    assert len(sent) == 1  # No retries, cache fallback or candidate switch.
    if case == "success":
        assert record["typed_result_validated"] and record["answer_correct"]
        assert record["final_answer_observed"] and record["settings_sent"]
        assert record["finish_reason"] == "stop"


async def test_save_reload_native_scalar_and_known_settings(tmp_path, monkeypatch):
    sent = []
    mock_http(
        monkeypatch,
        lambda request: (
            sent.append(json.loads(request.content))
            or httpx.Response(200, json=response_body("responses"))
        ),
    )
    monkeypatch.setattr(registry, "MODELS", {})
    path = tmp_path / "models.yaml"
    connect.write(plan(cache_enabled=False).entry, path, alias="local")
    assert "cache_breakpoint: null" in path.read_text()
    registry.reload_registry(path)
    client = registry.get_llm_client("local", api_key="fake")
    try:
        assert client.cache_breakpoint is None
        _, value = await typed_baseline(client)
        assert type(value) is int and value == 323
    finally:
        await client.aclose()
    assert len(sent) == 1
    assert_cache_off_schema(sent[0])


async def test_cache_change_invalidates_routing_evidence(monkeypatch):
    sent = []
    mock_http(
        monkeypatch,
        lambda request: (
            sent.append(request) or httpx.Response(200, json=response_body("responses"))
        ),
    )
    original = await connect.check_stage(plan(cache_enabled=False), "routing", api_key="fake")
    assert text_baseline_passed(original.entry["provenance"]["probes"]["routing"]), original
    unchanged = plan(existing_entry=original.entry)
    assert unchanged.entry["cache_breakpoint"] is None
    assert unchanged.entry["provenance"]["probes"], unchanged.entry
    await connect.run(unchanged, approved="minimal", api_key="fake")
    assert len(sent) == 1
    changed = plan(existing_entry=original.entry, cache_enabled=True)
    assert changed.entry["provenance"]["probes"] == {}
    await connect.check_stage(changed, "routing", api_key="fake")
    assert len(sent) == 2
    patched = connect.configure_entry(original.entry, cache_enabled=True)
    assert not connect.verification_is_current(patched["provenance"]["probes"]["routing"], patched)
    assert "routing" in original.entry["provenance"]["probes"]  # Detached edit.
    off = plan(existing_entry=original.entry)
    assert off.entry["cache_breakpoint"] is None
    on = plan(existing_entry=changed.entry, cache_enabled=True)
    assert on.entry["cache_breakpoint"] == "auto"


def test_unspecified_default_and_decision_rejection():
    entry = connect.plan("gpt", "gpt-5.1", "responses", "https://api.test/v1", "").entry
    assert "cache_breakpoint" not in entry
    client = registry.client_from_config("gpt", entry, api_key="fake")
    assert client.cache_breakpoint == "auto"
    for style in connect.DECISION_API_STYLES:
        with pytest.raises(ValueError, match="do not configure chat caching"):
            connect.plan("d", "d", style, "https://api.test/decisions", "", cache_enabled=False)
        decision = connect.plan("d", "d", style, "https://api.test/decisions", "").entry
        with pytest.raises(ValueError, match="do not configure chat caching"):
            connect.configure_entry(decision, cache_enabled=False)
        with pytest.raises(ValueError, match="chat-only fields: cache_breakpoint"):
            connect.configure_entry({**decision, "cache_breakpoint": None})
    with pytest.raises(ValueError, match="boolean"):
        plan(cache_enabled="off")


@pytest.mark.parametrize("bad", [None, [], "bad"])
@pytest.mark.parametrize("control", [None, False, True])
def test_malformed_provenance_fails_locally(bad, control):
    with pytest.raises(ValueError, match="provenance must be a mapping"):
        connect.configure_entry({**plan().entry, "provenance": bad}, cache_enabled=control)


@pytest.mark.parametrize("level", [False, True])
def test_explicit_custom_cache_options_conflict_without_rewriting(level):
    entry = plan().entry
    custom = {"extra_body": {"prompt_cache_options": {"mode": "explicit"}, "custom": 123}}
    if level:
        entry["reasoning_levels"] = {"high": custom}
    else:
        entry.update(custom)
    with pytest.raises(ValueError, match="conflicts with explicit cache options/markers"):
        connect.configure_entry(entry, cache_enabled=False)
    assert (entry["reasoning_levels"]["high"] if level else entry)["extra_body"] == custom[
        "extra_body"
    ]


def test_cache_change_discards_old_session_verdict_and_retains_known_settings():
    entry = plan().entry
    entry.update(store=True, include=[], max_tokens=1500)
    entry["provenance"].update(
        session_checks={"cache": {"outcome": "confirmed", "explicit_mode": True}},
        requests_accepted=["routing"],
        reasoning_observed=["high"],
    )
    connect.stamp_verification(entry["provenance"]["session_checks"]["cache"], entry, "cache")
    assert "cache" in connect.verdict(entry).passed
    changed = connect.configure_entry(entry, cache_enabled=False)
    assert "cache" not in connect.verdict(changed).passed
    assert changed["provenance"]["session_checks"]["cache"]["verification_current"] is False
    assert changed["provenance"]["requests_accepted"] == []
    assert changed["provenance"]["reasoning_observed"] == []
    assert changed["store"] is True and changed["include"] == [] and changed["max_tokens"] == 1500


async def test_default_gpt_predict_wire_policy_is_unchanged(monkeypatch):
    sent = []
    mock_http(
        monkeypatch,
        lambda request: (
            sent.append(json.loads(request.content))
            or httpx.Response(200, json=response_body("responses"))
        ),
    )
    proposal = connect.plan(
        "gpt", "gpt-5.1", "responses", "https://api.test/v1", "", reply_tokens=1000
    )
    result = await connect.check_stage(proposal, "routing", api_key="fake")
    assert text_baseline_passed(result.entry["provenance"]["probes"]["routing"])
    assert len(sent) == 1
    # GPT auto retains its existing explicit-mode default, independently of opt-out.
    assert sent[0]["prompt_cache_options"] == {"mode": "explicit"}
    assert '"prompt_cache_breakpoint": {"mode": "explicit"}' in json.dumps(sent[0])
    assert_cache_off_schema(sent[0], cache_off=False)
    assert "cache_breakpoint" not in result.entry
