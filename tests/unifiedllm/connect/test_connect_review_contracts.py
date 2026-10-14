# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Review regressions through native Predict/SDK; 9/10 remain compatibility controls."""

import json

import httpx
import pytest

from nooa.unifiedllm import connect
from nooa.unifiedllm.connect._records import text_baseline_passed
from tests.unifiedllm.connect.connect_http import mock_http, response_body


def plan(style="chat", **kw):
    return connect.plan("local", "model", style, "https://api.test/v1", "", reply_tokens=1000, **kw)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "pdf"])
@pytest.mark.parametrize("value", [True, False])
async def test_schema_tool(monkeypatch, kind, value):
    sent = []

    def handle(req):
        sent.append(json.loads(req.content))
        body = response_body("anthropic")
        body["stop_reason"] = "tool_use"
        body["content"] = [
            {
                "type": "tool_use",
                "id": "schema",
                "name": "json_tool_call",
                "input": {"value": 323 if len(sent) == 1 else value},
            }
        ]
        return httpx.Response(200, json=body)

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(
        plan("anthropic", test_media_inputs=[kind]), "media", api_key="fake"
    )
    records = result.entry["provenance"]["probes"]
    r = records["media:" + kind]
    print("SCHEMA", kind, value, json.dumps(connect.public_record(r)))
    assert text_baseline_passed(records["routing"]) and len(sent) == 2
    assert r["typed_result_validated"] is True and r["access_reported"] is value
    assert r["outcome"] == ("passed" if value else "unavailable")
    assert r["finish_reason"] == "stop" and r["final_answer_observed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["323", 323.0])
async def test_coercion(monkeypatch, value):
    mock_http(
        monkeypatch,
        lambda req: httpx.Response(200, json=response_body("chat", json.dumps({"value": value}))),
    )
    result = await connect.check_stage(plan(), "routing", api_key="fake")
    r = result.entry["provenance"]["probes"]["routing"]
    print("COERCION", repr(value), text_baseline_passed(r))
    assert text_baseline_passed(r)


@pytest.mark.asyncio
async def test_selected_cap(monkeypatch):
    sent = []
    mock_http(
        monkeypatch, lambda req: sent.append(req) or httpx.Response(200, json=response_body("chat"))
    )
    p = plan(reasoning_levels={"high": {"reasoning_effort": "high", "max_tokens": 2000}})
    p.entry["reasoning_level"] = "high"
    result = await connect.check_stage(p, "routing", api_key="fake")
    r = result.entry["provenance"]["probes"]["routing"]
    print("CAP", json.dumps(connect.public_record(r)), "sends", len(sent))
    assert len(sent) == 1 and text_baseline_passed(r)
    assert json.loads(sent[0].content)["max_tokens"] == 2000
    assert r["configured_reply_tokens"] == r["tested_reply_tokens"] == 2000
    assert result.entry["provenance"]["tokens_charged_to_budget"] == 2512


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", [None, "unrecognized"])
async def test_missing_finish(monkeypatch, finish):
    def handle(req):
        body = response_body("chat")
        body["choices"][0]["finish_reason"] = finish
        return httpx.Response(200, json=body)

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(plan(), "routing", api_key="fake")
    r = result.entry["provenance"]["probes"]["routing"]
    print("FINISH", repr(finish), json.dumps(connect.public_record(r)))
    assert text_baseline_passed(r)


@pytest.mark.parametrize(
    "mode", ["unrelated", "multiple", "length", "reasoning_only", "non_boolean"]
)
async def test_anthropic_schema_carrier_is_not_arbitrary_tool_completion(monkeypatch, mode):
    sent = []

    def handle(req):
        sent.append(json.loads(req.content))
        body = response_body("anthropic")
        if len(sent) == 1:
            return httpx.Response(200, json=body)
        body["stop_reason"] = "tool_use"
        carrier = {
            "type": "tool_use",
            "id": "schema",
            "name": "json_tool_call",
            "input": {"value": True},
        }
        body["content"] = [carrier]
        if mode == "unrelated":
            carrier["name"] = "user_tool"
        elif mode == "multiple":
            body["content"].append({**carrier, "id": "other", "name": "user_tool"})
        elif mode == "length":
            body["stop_reason"] = "max_tokens"
        elif mode == "reasoning_only":
            body["stop_reason"] = "end_turn"
            body["content"] = [
                {"type": "thinking", "thinking": '{"value":true}', "signature": "sig"}
            ]
        else:
            carrier["input"]["value"] = "true"
        return httpx.Response(200, json=body)

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(
        plan("anthropic", test_media_inputs=["image"]), "media", api_key="fake"
    )
    assert len(sent) == 2
    assert result.entry["provenance"]["probes"]["media:image"]["outcome"] == "inconclusive"


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("cap", [500, 2000])
async def test_selected_cap_media_budget_and_actual_saved_config(monkeypatch, style, cap):
    sent = []

    def handle(req):
        sent.append(json.loads(req.content))
        return httpx.Response(
            200, json=response_body(style, '{"value":323}' if len(sent) == 1 else '{"value":true}')
        )

    mock_http(monkeypatch, handle)
    cap_key = "max_output_tokens" if style == "responses" else "max_tokens"
    proposal = plan(
        style, test_media_inputs=["image"], reasoning_levels={"selected": {cap_key: cap}}
    )
    proposal.entry["reasoning_level"] = "selected"
    approved = 512 + cap + 4096 + cap
    from dataclasses import replace

    refreshed = connect.refresh_plan(replace(proposal, budget_tokens=approved))
    assert all(
        p.body[cap_key] == cap for p in refreshed.probes if p.name in {"routing", "media:image"}
    )
    result = await connect.check_stage(refreshed, "media", api_key="fake")
    records = result.entry["provenance"]["probes"]
    assert len(sent) == 2 and all(w[cap_key] == cap for w in sent)
    assert text_baseline_passed(records["routing"])
    assert records["media:image"]["outcome"] == "passed"
    assert result.entry["provenance"]["tokens_charged_to_budget"] == approved
    assert result.entry["max_tokens"] == 1000 and result.entry["reasoning_level"] == "selected"
    assert records["routing"]["baseline_config"]["reasoning_levels"]["selected"][cap_key] == cap
    sent.clear()
    too_small = await connect.check_stage(
        replace(refreshed, budget_tokens=approved - 1), "media", api_key="fake"
    )
    assert len(sent) == 1
    assert too_small.entry["provenance"]["probes"]["media:image"]["reason"] == "budget exhausted"


async def test_responses_pdf_sdk_projects_only_supported_file_keys(monkeypatch):
    import base64

    from nooa.media import File
    from nooa.runtime.media_capture import media_to_content_block
    from nooa.unifiedllm import ResponsesClient

    payload = b"%PDF-1.4 original bytes"
    block = media_to_content_block(File.from_bytes(payload, media_type="application/pdf"))
    assert block["file"]["format"] == "application/pdf"
    sent = []

    def handle(req):
        wire = json.loads(req.content)
        sent.append(wire)
        files = [
            b for m in wire["input"] for b in m.get("content", []) if b.get("type") == "input_file"
        ]
        assert len(files) == 1
        assert set(files[0]) <= {"type", "detail", "file_data", "file_id", "file_url", "filename"}
        assert "format" not in files[0]
        assert base64.b64decode(files[0]["file_data"].split(",", 1)[1]) == payload
        return httpx.Response(200, json=response_body("responses"))

    mock_http(monkeypatch, handle)
    client = ResponsesClient(
        "openai/model", api_base="https://api.test/v1", api_key="fake", media_inputs=["pdf"]
    )
    try:
        await client.acall([{"role": "user", "content": [block]}])
    finally:
        await client.aclose()
    assert len(sent) == 1 and block["file"]["format"] == "application/pdf"
    from nooa.unifiedllm.media_inputs import filter_media_messages

    generic = media_to_content_block(File.from_bytes(b"not a PDF", media_type="application/msword"))
    filtered = filter_media_messages([{"role": "user", "content": [generic]}], ("pdf",))
    assert filtered[0]["content"][0]["type"] == "text"


async def test_stage_does_not_reapprove_mutated_selected_cap(monkeypatch):
    sent = []
    mock_http(
        monkeypatch, lambda req: sent.append(req) or httpx.Response(200, json=response_body("chat"))
    )
    proposal = plan(reasoning_levels={"high": {"max_tokens": 2000}})
    proposal.entry["reasoning_level"] = "high"
    proposal.entry["reasoning_levels"]["high"]["max_tokens"] = 100000
    result = await connect.check_stage(proposal, "routing", api_key="fake")
    assert not sent
    assert not text_baseline_passed(result.entry["provenance"]["probes"]["routing"])


async def test_selected_alternate_chat_cap_spelling(monkeypatch):
    sent = []

    def handle(req):
        sent.append(json.loads(req.content))
        return httpx.Response(
            200, json=response_body("chat", '{"value":323}' if len(sent) == 1 else '{"value":true}')
        )

    mock_http(monkeypatch, handle)
    proposal = plan(
        test_media_inputs=["image"], reasoning_levels={"selected": {"max_completion_tokens": 2000}}
    )
    proposal.entry["reasoning_level"] = "selected"
    result = await connect.check_stage(proposal, "media", api_key="fake")
    assert len(sent) == 2
    assert all(w["max_completion_tokens"] == 2000 and "max_tokens" not in w for w in sent)
    assert text_baseline_passed(result.entry["provenance"]["probes"]["routing"])
    assert result.entry["provenance"]["probes"]["media:image"]["outcome"] == "passed"
