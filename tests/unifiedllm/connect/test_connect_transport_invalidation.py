# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detached edits cannot certify a different transport using prior SDK evidence."""

from copy import deepcopy
from dataclasses import replace

import httpx
import pytest

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.asyncio
async def test_configured_conversion_can_run_none_and_check_routing(monkeypatch, direct):
    sent = []

    def handle(request):
        sent.append(request)
        return httpx.Response(200, json=response_body("chat"))

    mock_http(monkeypatch, handle)
    proposal = connect.plan("test", "model", "chat", "https://api.test/v1", "", direct=direct)
    checked = await connect.run(proposal, approved="minimal", api_key="key")
    converted = connect.configure_entry(checked.entry, direct=not direct)
    assert converted["provenance"]["probes"] == {}
    changed = replace(proposal, entry=converted)
    skipped = await connect.run(changed, approved="none")
    assert len(sent) == 1
    assert not connect.verdict(skipped.entry).passed
    routed = await connect.check_stage(changed, "routing", api_key="key")
    assert len(sent) == 2
    record = routed.entry["provenance"]["probes"]["routing"]
    assert record["outcome"] == "accepted"
    assert record["transport"] == ("litellm" if direct else "direct")
    assert record["settings_sent"] is True


@pytest.fixture
async def checked_session(monkeypatch, request):
    style, direct = request.param

    def handle(request):
        data = response_body(style)
        if style == "chat":
            data["choices"][0]["message"]["reasoning_content"] = "Replay this reasoning"
            data["usage"]["prompt_tokens_details"] = {"cached_tokens": 12}
        else:
            data["output"].insert(
                0,
                {
                    "type": "reasoning",
                    "id": "r1",
                    "summary": [],
                    "encrypted_content": "opaque-replay-state",
                },
            )
            data["usage"]["input_tokens_details"] = {"cached_tokens": 12}
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    from nooa.unifiedllm.connect import _session

    monkeypatch.setattr(_session, "SESSION_CALL_PACING_SECONDS", 0)
    proposal = connect.plan(
        "test",
        "gpt-5.1",
        style,
        "https://api.test/v1",
        "",
        direct=direct,
        reply_tokens=2048,
        budget_tokens=65536,
        session_checks=True,
    )
    checked = await connect.run(proposal, approved="all", api_key="key")
    provenance = checked.entry["provenance"]
    assert {"session", "cache", "reasoning_retention"} <= set(connect.verdict(checked.entry).passed)
    assert not any("Reply limit not tested" in w for w in connect.entry_warnings(checked.entry))
    provenance["interfaces"] = {style: deepcopy(provenance["probes"]["routing"])}
    provenance["catalogue"] = {"id": "fixture-model"}
    provenance["custom_note"] = {"source": "user", "value": "retain me"}
    return replace(proposal, entry=checked.entry)


@pytest.mark.parametrize(
    "checked_session", [(s, d) for s in ("chat", "responses") for d in (False, True)], indirect=True
)
@pytest.mark.parametrize("path", ["refresh", "run", "stage", "replan"])
@pytest.mark.parametrize("relabel_metadata", [False, True])
@pytest.mark.asyncio
async def test_detached_switch_discards_all_certification(
    monkeypatch, checked_session, path, relabel_metadata
):
    proposal = checked_session
    original = deepcopy(proposal.entry)
    selected = not original["direct"]
    edited = original | {"direct": selected}
    if relabel_metadata:
        # Even changing both declarations cannot relabel actual recorded evidence.
        edited["transport"] = "direct" if selected else "litellm"
    changed = replace(proposal, entry=edited, session_checks=False)

    def forbidden(request):
        raise AssertionError("No HTTP is approved")

    mock_http(monkeypatch, forbidden)
    assert not connect.verdict(edited).passed
    assert any("Reply limit not tested" in w for w in connect.entry_warnings(edited))
    if path == "refresh":
        entry = connect.refresh_plan(changed).entry
    elif path == "run":
        entry = (await connect.run(changed, approved="none")).entry
    elif path == "stage":
        # Invalidating the session must also happen when only another stage runs.
        monkeypatch.undo()
        mock_http(
            monkeypatch, lambda r: httpx.Response(200, json=response_body(original["api_style"]))
        )
        entry = (await connect.check_stage(changed, "routing", api_key="key")).entry
    else:
        entry = connect.plan(
            "test",
            "gpt-5.1",
            original["api_style"],
            "https://api.test/v1",
            "",
            existing_entry=edited,
            reply_tokens=2048,
        ).entry
    evidence = entry["provenance"]
    assert "session_checks" not in evidence
    assert "interfaces" not in evidence
    if path != "stage":
        assert (
            "encrypted_reasoning" not in evidence
            or evidence["encrypted_reasoning"]["outcome"] == "not_probed"
        )
    elif original["api_style"] == "responses":
        assert evidence["encrypted_reasoning"]["transport"] == entry["transport"]
    assert not {"session", "cache", "reasoning_retention"} & set(connect.verdict(entry).passed)
    if path != "stage":
        assert not connect.verdict(entry).passed
        assert any("Reply limit not tested" in w for w in connect.entry_warnings(entry))
        assert all(r["outcome"] != "accepted" for r in evidence["probes"].values())
    if path != "replan":
        assert evidence["catalogue"] == {"id": "fixture-model"}
        assert evidence["custom_note"] == {"source": "user", "value": "retain me"}
    assert proposal.entry == original


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("transport", ["direct", "litellm", None])
def test_booleanless_legacy_evidence_is_not_certification(direct, transport):
    entry = connect.plan("test", "model", "chat", "https://api.test/v1", "").entry
    entry.pop("direct")
    if transport is None:
        entry.pop("transport")
    else:
        entry["transport"] = transport
    entry["provenance"]["probes"] = {
        "routing": {
            "outcome": "accepted",
            "settings_sent": True,
            "tested_reply_tokens": entry["max_tokens"],
            "transport": "litellm",
        }
    }
    entry["provenance"]["session_checks"] = {"cache": {"outcome": "confirmed"}}
    assert not connect.verdict(entry).passed
    configured = connect.configure_entry(entry, direct=direct)
    assert configured["direct"] is direct
    # Records may stay as history, but none certifies the selected transport:
    # they carry no current verification fingerprint.
    provenance = configured["provenance"]
    for record in (*provenance["probes"].values(), *provenance.get("session_checks", {}).values()):
        assert record["verification_current"] is False
    assert not connect.verdict(configured).passed


def test_isolated_mismatched_interface_record_invalidates_untagged_summaries():
    entry = connect.plan("test", "model", "chat", "https://api.test/v1", "").entry
    entry["provenance"].update(
        interfaces={"responses": {"outcome": "accepted", "transport": "litellm"}},
        session_checks={"cache": {"outcome": "confirmed"}},
        requests_accepted=["routing"],
        reasoning_observed=["level:high"],
    )
    configured = connect.configure_entry(entry)
    assert "session_checks" not in configured["provenance"]
    assert "requests_accepted" not in configured["provenance"]
    assert "reasoning_observed" not in configured["provenance"]


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("tagged", [False, True])
def test_aggregate_only_evidence_cannot_be_relabeled(direct, tagged):
    entry = connect.plan(
        "test", "model", "responses", "https://api.test/v1", "", direct=direct
    ).entry
    transport = "direct" if direct else "litellm"
    tag = {"transport": transport} if tagged else {}
    entry["provenance"].update(
        session_checks={
            name: {"outcome": outcome, **tag}
            for name, outcome in (
                ("cache", "confirmed"),
                ("reasoning_retention", "confirmed"),
                ("session", "completed"),
            )
        },
        encrypted_reasoning={"source": "connect", "outcome": "rejected", **tag},
    )
    entry.update(direct=not direct, transport="litellm" if direct else "direct")
    assert not connect.verdict(entry).passed
    configured = connect.configure_entry(entry)
    if tagged:
        assert "session_checks" not in configured["provenance"]
    else:
        # Untagged records name no transport; they are kept as history but carry
        # no current verification fingerprint, so they certify nothing.
        records = configured["provenance"].get("session_checks", {}).values()
        assert all(record["verification_current"] is False for record in records)
    assert not connect.verdict(configured).passed
    assert "encrypted_reasoning" not in configured["provenance"]


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("learned", [False, True])
def test_transport_switch_resets_only_learned_encrypted_opt_out(direct, learned):
    entry = connect.plan(
        "test", "model", "responses", "https://api.test/v1", "", direct=direct
    ).entry
    entry["include"] = []
    if learned:
        connect._disable_encrypted_reasoning(entry, 400)
    configured = connect.configure_entry(entry, direct=not direct)
    assert configured["include"] == (["reasoning.encrypted_content"] if learned else [])
    assert "encrypted_reasoning" not in configured["provenance"]


@pytest.mark.parametrize("model", ["bedrock/model", "vertex_ai/model"])
@pytest.mark.asyncio
async def test_connect_route_without_direct_sdk_fails_check_without_fallback(monkeypatch, model):
    """Connect saves its direct default for any route; one with no direct SDK fails its check."""
    sent = []
    mock_http(
        monkeypatch,
        lambda request: sent.append(request) or httpx.Response(200, json=response_body("chat")),
    )
    proposal = connect.plan("test", "model", "chat", "https://api.test/v1", "")
    entry = connect.configure_entry({**proposal.entry, "model_name": model})
    assert entry["direct"] is True and entry["transport"] == "direct"
    result = await connect.check_stage(replace(proposal, entry=entry), "routing", api_key="key")
    record = result.entry["provenance"]["probes"]["routing"]
    assert sent == []  # no HTTP, and no silent LiteLLM fallback
    assert record["outcome"] != "accepted" and record["error"] == "ValueError"
    assert not connect.routing_allows_save(result.entry, result.entry)
    # --litellm (direct=False) is the explicit way to check such a route.
    litellm_entry = connect.configure_entry(entry, direct=False)
    assert litellm_entry["direct"] is False and litellm_entry["transport"] == "litellm"


@pytest.mark.asyncio
async def test_connect_gemini_prefix_with_endpoint_uses_openai_sdk(monkeypatch):
    """gemini/ has no native SDK here; with an api_base it goes over the OpenAI SDK."""
    sent = []
    mock_http(
        monkeypatch,
        lambda request: sent.append(request) or httpx.Response(200, json=response_body("chat")),
    )
    proposal = connect.plan("test", "model", "chat", "https://api.test/v1", "")
    entry = connect.configure_entry({**proposal.entry, "model_name": "gemini/model"})
    result = await connect.check_stage(replace(proposal, entry=entry), "routing", api_key="key")
    assert [request.url.path for request in sent] == ["/v1/chat/completions"]
    assert result.entry["provenance"]["probes"]["routing"]["transport"] == "direct"
