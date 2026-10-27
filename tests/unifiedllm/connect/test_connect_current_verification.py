# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current dependency trust: real Agent/Predict over MockTransport below SDKs."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest

from nooa.unifiedllm import connect
from nooa.unifiedllm.connect._records import public_record, text_baseline_passed
from tests.unifiedllm.connect.connect_http import mock_http, response_body


def plan(style="chat", **kwargs):
    return connect.plan(
        "local", "model", style, "https://api.test/v1", "", reply_tokens=1000, **kwargs
    )


@pytest.fixture
async def fresh_media(monkeypatch):
    sent = []

    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        media = any(
            block.get("type") in {"image_url", "file"}
            for message in body["messages"]
            if isinstance(message.get("content"), list)
            for block in message["content"]
        )
        # Image access succeeds; independent PDF access is genuinely false.
        pdf = any(
            block.get("type") == "file"
            for message in body["messages"]
            if isinstance(message.get("content"), list)
            for block in message["content"]
        )
        value = (not pdf and body.get("temperature") != 0.4) if media else 323
        return httpx.Response(200, json=response_body("chat", json.dumps({"value": value})))

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(
        plan(test_media_inputs=["image", "pdf"]), "media", api_key="transient-secret"
    )
    records = result.entry["provenance"]["probes"]
    assert len(sent) == 3
    assert text_baseline_passed(records["routing"])
    assert records["media:image"]["outcome"] == "passed"
    assert records["media:pdf"]["outcome"] == "unavailable"
    assert records["media:pdf"]["access_reported"] is False
    return result, sent


MUTATIONS = [
    {"timeout": 30},
    {"retry_config": False},
    {"retry_config": {"max_retries": 5, "base_delay": 2.0}},
    {"temperature": 0.7},
    {"top_p": 0.8},
    {"thinking": {"type": "adaptive"}},
    {"reasoning": {"effort": "high"}},
    {"reasoning_effort": "high"},
    {"output_config": {"effort": "high"}},
    {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    {"chat_template_kwargs": {"enable_thinking": True}},
    {"context_window": 64000},
    {"max_tokens": 2000},
    {"api_base": "https://other.test/v1"},
    {"api_key_env": "OTHER_KEY_SCOPE"},
    {"model_name": "openai/other"},
    {"api_style": "anthropic"},
    {"client_type": "responses", "api_style": "responses"},
    {"transport": "litellm"},
    {"drop_params": False},
    {"allowed_openai_params": ["thinking"]},
    {"additional_drop_params": ["temperature"]},
    {"cache_breakpoint": None},
    {"structured_output_mode": "anthropic"},
    {"reasoning_level": "high", "reasoning_levels": {"high": {"reasoning_effort": "high"}}},
]


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda m: "+".join(m))
async def test_every_behavior_edit_removes_current_routing_media_and_session_trust(
    fresh_media, mutation
):
    result, sent = fresh_media
    entry = deepcopy(result.entry)
    # A separate fresh session fixture makes the per-check dependencies explicit.
    session = {"outcome": "completed"}
    connect.stamp_verification(session, entry, "session")
    entry["provenance"]["session_checks"] = {"session": session}
    before = deepcopy(entry)
    entry.update(mutation)
    # Even direct mutation, without configure_entry, cannot borrow a stale boolean.
    assert not connect.routing_allows_save(result.entry, entry)
    assert "routing" not in connect.verdict(entry).passed
    assert "media:image" not in connect.verdict(entry).passed
    assert "session" not in connect.verdict(entry).passed
    edited = connect.configure_entry(entry)
    records = edited["provenance"]["probes"]
    for name in ("routing", "media:image", "media:pdf"):
        assert records[name]["verification_current"] is False
        assert not connect.verification_is_current(records[name], edited, name)
    assert edited["provenance"]["session_checks"]["session"]["verification_current"] is False
    # Past access evidence remains honest, not turned into fresh failure/success.
    assert records["media:image"]["access_reported"] is True
    assert records["media:pdf"]["access_reported"] is False
    assert records["media:pdf"]["outcome"] == "unavailable"
    assert before["provenance"]["probes"]["routing"]["verification_current"] is True
    assert len(sent) == 3  # Edits and gates perform zero calls.


async def test_exact_restoration_uses_recomputed_currentness_for_routing_verdict(fresh_media):
    result, sent = fresh_media
    candidate = connect.configure_entry({**deepcopy(result.entry), "temperature": 0.5})
    record = candidate["provenance"]["probes"]["routing"]
    assert record["verification_current"] is False
    assert not connect.routing_allows_save(candidate, candidate)
    assert "routing" not in connect.verdict(candidate).passed
    # Direct restoration must agree with the exact save gate, without relying
    # on configure_entry to repair an old cached currentness flag.
    candidate.pop("temperature")
    assert connect.verification_is_current(record, candidate)
    assert connect.routing_allows_save(candidate, candidate)
    assert "routing" in connect.verdict(candidate).passed
    assert "media:image" in connect.verdict(candidate).passed
    assert len(sent) == 3


async def test_permission_only_changes_keep_baseline_and_independent_false_media_evidence(
    fresh_media,
):
    result, sent = fresh_media
    entry = deepcopy(result.entry)
    original = deepcopy(entry)
    # Passing the image check grants a permission, not a new request config.
    entry["media_inputs"] = ["image"]
    allowed = connect.configure_entry(entry)
    assert connect.routing_allows_save(result.entry, allowed)
    assert "routing" in connect.verdict(allowed).passed
    assert "media:image" in connect.verdict(allowed).passed
    assert (
        allowed["provenance"]["probes"]["media:pdf"]
        == original["provenance"]["probes"]["media:pdf"]
    )
    # Later manual grants don't manufacture evidence for the new format.
    allowed["media_inputs"] = ["image", "audio", "pdf"]
    manual = connect.configure_entry(allowed)
    assert connect.routing_allows_save(result.entry, manual)
    assert "media:audio" not in manual["provenance"]["probes"]
    assert manual["provenance"]["probes"]["media:pdf"]["access_reported"] is False
    assert "media:pdf" not in connect.verdict(manual).passed
    assert len(sent) == 3


async def test_success_then_changed_config_with_fresh_false_media_does_not_reuse_pass(fresh_media):
    result, sent = fresh_media
    changed = deepcopy(result.entry)
    changed["temperature"] = 0.4
    # The PDF was genuinely tested false already; stamping a permission cannot alter it.
    refreshed = connect.refresh_plan(replace(plan(test_media_inputs=["image"]), entry=changed))
    rerun = await connect.check_stage(refreshed, "media", api_key="transient-secret")
    assert len(sent) == 5  # New baseline and image, not borrowed old success.
    records = rerun.entry["provenance"]["probes"]
    assert records["routing"]["verification_config"]["params"]["temperature"] == 0.4
    assert records["media:image"]["outcome"] == "unavailable"
    assert records["media:image"]["access_reported"] is False
    assert "media:image" not in connect.verdict(rerun.entry).passed
    assert records["media:pdf"]["access_reported"] is False
    assert records["media:pdf"]["verification_current"] is False
    assert "media:pdf" not in connect.verdict(rerun.entry).passed


@pytest.mark.parametrize(
    "mutation",
    [{"store": True}, {"include": []}, {"include": ["reasoning.encrypted_content", "other"]}],
)
async def test_responses_retention_edits_invalidate_actual_baseline(monkeypatch, mutation):
    sent = []
    mock_http(
        monkeypatch,
        lambda request: (
            sent.append(request) or httpx.Response(200, json=response_body("responses"))
        ),
    )
    result = await connect.check_stage(plan("responses"), "routing", api_key="fake")
    entry = deepcopy(result.entry)
    entry.update(mutation)
    assert not connect.routing_allows_save(result.entry, connect.configure_entry(entry))
    assert "routing" not in connect.verdict(entry).passed
    assert len(sent) == 1


def test_canonical_snapshot_resolves_selected_patch_shadowed_defaults_and_provenance():
    entry = connect.configure_entry(
        {
            "model_name": "openai/model",
            "api_base": "https://API.TEST/v1/",
            "api_key_env": "MODEL_KEY",
            "temperature": 0.9,
            "extra_body": {"temperature": 0.5, "thinking": {"type": "disabled"}},
            "reasoning_level": "high",
            "reasoning_default": "low",
            "reasoning_levels": {
                "high": {"temperature": 0.2, "thinking": {"type": "adaptive"}},
                "low": {"temperature": 0.7},
            },
        }
    )
    snapshot = connect.effective_config_snapshot(entry)
    assert snapshot["level"] == "high"
    assert snapshot["params"]["temperature"] == 0.2
    assert snapshot["params"]["thinking"] == {"type": "adaptive"}
    assert "extra_body" not in snapshot["params"]
    assert snapshot["route"] == "https://api.test/v1"
    assert snapshot["credential_scope"] == {"api_key_env": "MODEL_KEY"}
    changed = deepcopy(entry)
    changed.update(temperature=0.1, media_inputs=["image"])
    changed["extra_body"].update(temperature=0.1, thinking={"type": "enabled"})
    changed["reasoning_levels"]["low"] = {"temperature": 0.4}
    changed["reasoning_default"] = "high"
    changed["provenance"]["unrelated"] = "description changed"
    assert connect.effective_config_snapshot(changed) == snapshot
    assert connect.effective_config_fingerprint(changed) == connect.effective_config_fingerprint(
        entry
    )
    # Without selection, SDK extra_body has final precedence; descriptive default never selects.
    changed.pop("reasoning_level")
    assert connect.effective_config_snapshot(changed)["level"] is None
    assert connect.effective_config_snapshot(changed)["params"]["temperature"] == 0.1
    assert connect.effective_config_fingerprint(changed) != connect.effective_config_fingerprint(
        entry
    )
    assert snapshot == json.loads(json.dumps(snapshot))


def test_level_checks_have_independent_dependencies():
    entry = connect.configure_entry(
        {
            "model_name": "openai/model",
            "reasoning_level": "low",
            "reasoning_levels": {
                "low": {"reasoning_effort": "low"},
                "high": {"reasoning_effort": "high"},
            },
        }
    )
    records = {
        name: {"outcome": "accepted", "reasoning_observed": True}
        for name in ("level:low", "level:high")
    }
    for name, record in records.items():
        connect.stamp_verification(record, entry, name)
    entry["provenance"]["probes"] = records
    entry["reasoning_levels"]["low"]["reasoning_effort"] = "medium"
    changed = connect.configure_entry(entry)
    assert "level:high" in connect.verdict(changed).passed
    assert "level:low" not in connect.verdict(changed).passed
    entry["reasoning_level"] = "high"
    assert connect.verification_is_current(records["level:high"], entry, "level:high")


@pytest.mark.parametrize("client_type", ["completion", "responses"])
def test_effective_route_alias_precedence_and_normalization(client_type):
    entry = connect.configure_entry(
        {
            "model_name": "openai/model",
            "client_type": client_type,
            "api_base": "https://API.TEST/v1/",
            "base_url": "https://other.test/v1",
        }
    )
    chosen = "base_url" if client_type == "responses" else "api_base"
    dormant = "api_base" if client_type == "responses" else "base_url"
    fingerprint = connect.effective_config_fingerprint(entry)
    entry[dormant] = "https://dormant.test/v1"
    assert connect.effective_config_fingerprint(entry) == fingerprint
    entry[chosen] = "https://new.test/v1"
    assert connect.effective_config_fingerprint(entry) != fingerprint


def test_credential_scope_contains_no_values_or_hashes_and_never_reads_environment(monkeypatch):
    entry = connect.configure_entry({"model_name": "openai/model", "api_key_env": "MODEL_KEY"})
    monkeypatch.setenv("MODEL_KEY", "secret-one")
    first = connect.effective_config_fingerprint(entry)
    monkeypatch.setenv("MODEL_KEY", "secret-two")
    assert connect.effective_config_fingerprint(entry) == first
    record = {"outcome": "accepted"}
    connect.stamp_verification(record, entry, "routing")
    encoded = json.dumps(record)
    for secret in ("secret-one", "secret-two"):
        assert secret not in encoded
        assert hashlib.sha256(secret.encode()).hexdigest() not in encoded
    assert "verification_config" not in public_record(record)
    entry["api_key_env"] = "OTHER_KEY"
    assert not connect.verification_is_current(record, entry)


@pytest.mark.parametrize(
    "patch",
    [
        {"api_key": "secret"},
        {"extra_body": {"Authorization": "secret"}},
        {"api_base": "https://user:secret@api.test/v1"},
        {"api_base": "https://api.test/v1?api_key=secret"},
    ],
)
def test_snapshots_reject_credentials_instead_of_hashing_them(patch):
    with pytest.raises(ValueError):
        connect.effective_config_fingerprint({"model_name": "openai/model", **patch})


async def test_unchanged_unstamped_complete_proof_stays_historical_and_is_rerun(fresh_media):
    result, sent = fresh_media
    entry = deepcopy(result.entry)
    records = entry["provenance"]["probes"]
    for record in records.values():
        for key in ("verification_config", "verification_fingerprint", "verification_current"):
            record.pop(key, None)
    unchanged = connect.configure_entry(entry)
    assert not connect.routing_allows_save(unchanged, unchanged)
    assert not text_baseline_passed(unchanged["provenance"]["probes"]["routing"])
    assert not connect.verdict(unchanged).passed
    assert records["media:pdf"]["access_reported"] is False
    proposal = connect.refresh_plan(replace(plan(test_media_inputs=["image"]), entry=unchanged))
    rerun = await connect.run(proposal, approved="minimal", api_key="fake")
    assert len(sent) == 4
    assert connect.routing_allows_save(rerun.entry, rerun.entry)


async def test_exact_savegate_compares_candidate_not_record_boolean(fresh_media):
    result, _ = fresh_media
    candidate = deepcopy(result.entry)
    candidate["extra_body"] = {"temperature": 0.8}
    assert result.entry["provenance"]["probes"]["routing"]["verification_current"] is True
    assert not connect.routing_allows_save(result.entry, candidate)
    candidate.pop("extra_body")
    assert connect.routing_allows_save(result.entry, candidate)
    tampered = deepcopy(result.entry)
    tampered["provenance"]["probes"]["routing"]["verification_fingerprint"] = "bad"
    assert not connect.routing_allows_save(tampered, candidate)


async def test_actual_session_execution_stamped_and_context_edit_cannot_pass(monkeypatch):
    sent = []
    mock_http(
        monkeypatch,
        lambda request: (
            sent.append(request) or httpx.Response(200, json=response_body("chat", "ok"))
        ),
    )
    result = await connect.check_stage(
        plan(session_checks=True, budget_tokens=100000), "session", api_key="fake"
    )
    assert len(sent) == 3
    checks = result.entry["provenance"]["session_checks"]
    assert checks["session"]["outcome"] == "completed"
    assert "session" in connect.verdict(result.entry).passed
    changed = deepcopy(result.entry)
    changed["context_window"] = 64000
    assert "session" not in connect.verdict(changed).passed
    assert not connect.verification_is_current(checks["session"], changed, "session")


async def test_no_approval_has_no_execution_stamp_or_requests(monkeypatch):
    mock_http(monkeypatch, lambda request: pytest.fail("No calls before consent"))
    result = await connect.run(
        plan(test_media_inputs=["image"], session_checks=True), approved="none"
    )
    records = {
        **result.entry["provenance"]["probes"],
        **result.entry["provenance"]["session_checks"],
    }
    assert all("verification_config" not in record for record in records.values())
    assert not connect.routing_allows_save(result.entry, result.entry)


@pytest.mark.parametrize(
    "patch",
    [
        {"temperature": 0.6},
        {"top_p": 0.8},
        {"reasoning_effort": "low"},
        {"thinking": {"type": "enabled", "budget_tokens": 100}},
        {"output_config": {"effort": "high"}},
        {"max_tokens": 2000},
    ],
)
async def test_active_level_parameter_mutations_use_executed_snapshot(monkeypatch, patch):
    sent = []
    mock_http(
        monkeypatch,
        lambda request: (
            sent.append(json.loads(request.content))
            or httpx.Response(200, json=response_body("chat"))
        ),
    )
    proposal = plan(
        reasoning_level="high",
        reasoning_levels={
            "high": {"temperature": 0.2, "reasoning_effort": "high"},
            "low": {"temperature": 0.1, "reasoning_effort": "low"},
        },
    )
    result = await connect.check_stage(proposal, "routing", api_key="fake")
    assert len(sent) == 1
    assert sent[0]["temperature"] == 0.2 and sent[0]["reasoning_effort"] == "high"
    record = result.entry["provenance"]["probes"]["routing"]
    assert connect.routing_allows_save(result.entry, result.entry)
    assert record["verification_config"]["level"] == "high"
    assert record["verification_config"]["params"]["temperature"] == sent[0]["temperature"]
    dormant = deepcopy(result.entry)
    dormant["reasoning_levels"]["low"].update(patch)
    assert connect.routing_allows_save(result.entry, connect.configure_entry(dormant))
    active = deepcopy(result.entry)
    active["reasoning_levels"]["high"].update(patch)
    active = connect.configure_entry(active)
    assert not connect.routing_allows_save(result.entry, active)
    assert record["verification_config"]["params"]["temperature"] == 0.2
    assert len(sent) == 1


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("transport", ["direct", "litellm"])
async def test_factory_backend_extra_body_precedence_matches_actual_http(
    monkeypatch, style, transport
):
    sent = []
    mock_http(
        monkeypatch,
        lambda request: (
            sent.append(json.loads(request.content))
            or httpx.Response(200, json=response_body(style))
        ),
    )
    proposal = plan(style)
    proposal.entry.update(
        transport=transport,
        temperature=0.9,
        top_p=0.8,
        extra_body={"temperature": 0.5, "top_p": 0.6},
    )
    result = await connect.check_stage(proposal, "routing", api_key="fake")
    snapshot = connect.effective_config_snapshot(result.entry)
    assert len(sent) == 1
    assert snapshot["params"]["temperature"] == sent[0]["temperature"]
    assert snapshot["params"]["top_p"] == sent[0]["top_p"]
    if style == "anthropic":
        assert snapshot["params"]["extra_body"] == sent[0]["extra_body"]
        # Group4's strict observation does not confirm conflicting nested native
        # controls. We preserve that fail-closed behavior rather than regrade it.
        assert not connect.routing_allows_save(result.entry, result.entry)
    else:
        assert "extra_body" not in snapshot["params"]
        assert connect.routing_allows_save(result.entry, result.entry)


def test_snapshot_drop_params_matches_factory_defaults_and_explicit_setting(monkeypatch):
    from nooa.unifiedllm import registry

    observed = []

    class CaptureClient:
        def __init__(self, **params):
            self.model = params["model"]
            self.config = params
            observed.append(params)

    monkeypatch.setattr("nooa.unifiedllm.CompletionClient", CaptureClient)
    for settings in ({}, {"drop_params": False}, {"drop_params": True}):
        entry = connect.configure_entry({"model_name": "openai/model", **settings})
        registry.client_from_config("local", entry, api_key="transient-secret")
        assert (
            connect.effective_config_snapshot(entry)["drop_params"] is observed[-1]["drop_params"]
        )


@pytest.mark.parametrize("style", ["openai", "systemone"])
async def test_actual_decision_baseline_exact_route_scope_and_permission_policy(monkeypatch, style):
    from tests.unifiedllm.connect.test_connect_decision_smoke import fixture_response, proposal

    sent = []
    mock_http(monkeypatch, lambda request: sent.append(request) or fixture_response(request))
    result = await connect.check_stage(proposal(style), "routing", api_key="fake")
    assert len(sent) == 1
    assert connect.routing_allows_save(result.entry, result.entry)
    assert result.entry["provenance"]["probes"]["routing"]["verification_current"] is True
    permission = deepcopy(result.entry)
    permission["media_inputs"] = ["image"]
    assert connect.routing_allows_save(result.entry, connect.configure_entry(permission))
    for patch in (
        {"endpoint": "https://other.test/v1/decisions"},
        {"api_key_env": "OTHER_KEY"},
        {"model_name": "other"},
        {"api_style": "systemone" if style == "openai" else "openai"},
    ):
        candidate = connect.configure_entry({**result.entry, **patch})
        assert not connect.routing_allows_save(result.entry, candidate)
        assert "routing" not in connect.verdict(candidate).passed
    assert len(sent) == 1


async def test_decision_image_encoding_dependency_does_not_erase_genuine_baseline(monkeypatch):
    from tests.unifiedllm.connect.test_connect_decision_smoke import fixture_response, proposal

    sent = []
    mock_http(monkeypatch, lambda request: sent.append(request) or fixture_response(request))
    result = await connect.check_stage(proposal("systemone", image=True), "media", api_key="fake")
    assert len(sent) == 5
    assert "media:image" in connect.verdict(result.entry).passed
    candidate = connect.configure_entry(
        {**result.entry, "decision_image_transport": "json", "typesafe_danger_zone_images": False}
    )
    assert connect.routing_allows_save(result.entry, candidate)
    assert "routing" in connect.verdict(candidate).passed
    assert "media:image" not in connect.verdict(candidate).passed
    assert candidate["provenance"]["probes"]["media:image"]["verification_current"] is False
    assert len(sent) == 5


async def test_configured_retry_identity_is_not_claimed_as_executed_retry_policy(monkeypatch):
    from nooa.unifiedllm import registry

    sent, clients = [], []
    mock_http(
        monkeypatch,
        lambda request: sent.append(request) or httpx.Response(200, json=response_body("chat")),
    )
    factory = registry.client_from_config

    def capture(*args, **kwargs):
        client = factory(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(registry, "client_from_config", capture)
    proposal = plan()
    proposal.entry.update(timeout=30, retry_config={"max_retries": 5, "base_delay": 2.0})
    result = await connect.check_stage(proposal, "routing", api_key="fake")
    record = result.entry["provenance"]["probes"]["routing"]
    assert record["verification_config"]["retry_policy"]["max_retries"] == 5
    assert record["verification_config"]["params"]["timeout"] == 30
    assert clients[0].retry_config.max_retries == 0
    assert clients[0].retry_config.rate_limit_extra_retries == 0
    assert record["verification_execution"] == {
        "endpoint_retries": 0,
        "timeout_seconds": 120,
        "configured_retry_policy_tested": False,
        "configured_timeout_policy_tested": False,
    }
    assert connect.routing_allows_save(result.entry, result.entry)
    changed = deepcopy(result.entry)
    changed["retry_config"]["max_retries"] = 6
    assert not connect.routing_allows_save(result.entry, changed)
    assert len(sent) == 1


def test_retry_policy_normalizes_factory_defaults_and_disabled_spellings():
    entry = connect.configure_entry({"model_name": "openai/model"})
    omitted = connect.effective_config_fingerprint(entry)
    assert connect.effective_config_fingerprint({**entry, "retry_config": {}}) == omitted
    disabled = connect.effective_config_fingerprint({**entry, "retry_config": False})
    assert connect.effective_config_fingerprint({**entry, "retry_config": None}) == disabled
    assert (
        connect.effective_config_fingerprint(
            {
                **entry,
                "retry_config": {
                    "max_retries": 0,
                    "rate_limit_extra_retries": 0,
                },
            }
        )
        == disabled
    )
    assert disabled != omitted
