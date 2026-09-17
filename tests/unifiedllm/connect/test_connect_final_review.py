# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Review regressions exercise persistence and actual configured requests."""

import json
from dataclasses import replace

import httpx
import pytest
import yaml

from nooa.secrets import write_secret_env
from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body


def proposal(style="chat", **kwargs):
    return connect.plan("test", "model", style, "https://api.test/v1", "", **kwargs)


def test_writers_follow_symlinks_and_registry_preserves_mode_and_newlines(tmp_path):
    target = tmp_path / "real.yaml"
    target.write_bytes(b"# preserved\r\nmodels: {}\r\n")
    target.chmod(0o640)
    link = tmp_path / "linked.yaml"
    link.symlink_to(target)
    connect.write(proposal().entry, link, alias="test")
    assert link.is_symlink()
    assert "test" in yaml.safe_load(target.read_text())["models"]
    assert target.stat().st_mode & 0o777 == 0o640
    assert target.read_bytes().startswith(b"# preserved\r\n")
    assert b"\n" not in target.read_bytes().replace(b"\r\n", b"")
    secret_target = tmp_path / "real-secrets.yaml"
    secret_target.write_text("env: {}\n")
    secret_link = tmp_path / "secrets.yaml"
    secret_link.symlink_to(secret_target)
    write_secret_env(secret_link, "TEST_KEY", "test-only-value")
    assert secret_link.is_symlink()
    assert yaml.safe_load(secret_target.read_text())["env"]["TEST_KEY"] == "test-only-value"
    assert secret_target.stat().st_mode & 0o777 == 0o600


def test_null_models_mapping_is_an_empty_registry(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("models:\n")
    connect.write(proposal().entry, path, alias="test")
    assert "test" in yaml.safe_load(path.read_text())["models"]


@pytest.mark.parametrize("cap", [32768, 65536])
def test_save_refuses_reply_cap_without_input_room(cap, tmp_path):
    entry = proposal(reply_tokens=cap).entry
    entry["context_window"] = 32768
    with pytest.raises(ValueError, match="room for input"):
        connect.write(entry, tmp_path / "config.yaml", alias="test")
    assert not (tmp_path / "config.yaml").exists()


def test_default_for_small_window_leaves_room_for_input():
    entry = proposal(endpoint_model={"context_window": 32768}).entry
    assert 0 < entry["max_tokens"] < entry["context_window"]


def test_adding_known_window_rebounds_only_an_automatic_default():
    entry = proposal().entry
    entry["context_window"] = 32768
    assert connect.configure_entry(entry)["max_tokens"] == 16384
    explicit = proposal(reply_tokens=32768).entry | {"context_window": 32768}
    with pytest.raises(ValueError, match="room for input"):
        connect.configure_entry(explicit)


def test_catalogue_recommendation_cannot_allocate_the_whole_window():
    entry = proposal(
        catalogue={
            "id": "model",
            "context_length": 32768,
            "default_parameters": {"max_tokens": 32768},
        }
    ).entry
    assert entry["max_tokens"] < 32768


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
async def test_each_probe_sends_configured_cap_through_runtime(monkeypatch, style):
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=response_body(style))

    mock_http(monkeypatch, handle)
    plan = proposal(
        style,
        reply_tokens=8192,
        reasoning_levels={"custom": {"temperature": 0.5, "max_tokens": 12288}},
    )
    result = await connect.run(plan, approved="all", api_key="test-key")
    key = "max_output_tokens" if style == "responses" else "max_tokens"
    assert [body[key] for body in bodies] == [8192, 8192, 12288]
    for record in result.entry["provenance"]["probes"].values():
        assert record["settings_sent"] is True
        assert record["tested_reply_tokens"] == record["configured_reply_tokens"]
        assert record["transport"] == "litellm"  # This runtime predates direct SDK support.
    assert result.entry["transport"] == "direct"  # Saved preference is forward-compatible.


async def test_insufficient_budget_never_substitutes_smaller_cap(monkeypatch):
    def unexpected(request):
        raise AssertionError("No request fits the approved budget")

    mock_http(monkeypatch, unexpected)
    plan = proposal(reply_tokens=32768, budget_tokens=4096, session_checks=True)
    result = await connect.run(plan, approved="all", api_key="test-key")
    assert result.entry["provenance"]["tokens_charged_to_budget"] == 0
    assert all(r["outcome"] == "not_probed" for r in result.entry["provenance"]["probes"].values())
    assert not connect.verdict(result.entry).ok
    assert any("unverified" in warning for warning in connect.entry_warnings(result.entry))


def test_refresh_after_edit_rebuilds_detached_requests_and_estimates():
    original = proposal(reply_tokens=8192)
    edited = connect.configure_entry(original.entry, reply_tokens=16384)
    refreshed = connect.refresh_plan(replace(original, entry=edited))
    assert all(p.body["max_tokens"] == 16384 for p in refreshed.probes)
    assert all(p.token_estimate == 16384 + 512 for p in refreshed.probes)
    assert original.probes[0].body["max_tokens"] == 8192


def test_diagnostic_scrubs_active_key_in_values_and_mapping_keys(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "private-test-key")
    entry = proposal().entry | {"model_name": "openai/private-test-key", "api_key_env": "TEST_KEY"}
    prompt = connect.diagnostic_prompt(
        "routing", entry, {"private-test-key": {"outcome": "rejected"}}
    )
    assert "private-test-key" not in prompt


def test_new_entries_select_future_direct_transport_without_overwriting_explicit_choice():
    assert proposal().entry["transport"] == "direct"
    assert connect.configure_entry({"model_name": "openai/model"})["transport"] == "direct"
    assert (
        connect.configure_entry({"model_name": "openai/model", "transport": "litellm"})["transport"]
        == "litellm"
    )
