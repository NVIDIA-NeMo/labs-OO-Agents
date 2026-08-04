# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Security tests for exact-config MCP approvals and environment resolution."""

from __future__ import annotations

import json
import stat

import pytest
from nooa_cli.tui.mcp_approval import (
    MCPApprovalStore,
    build_approval_request,
    redact_approved_environment,
    resolve_approved_environment,
)


def _request(config, *, name="server", mcp_file=None):
    return build_approval_request(name, mcp_file=mcp_file, servers={name: config})


def test_fingerprint_is_stable_across_mapping_order():
    first = _request(
        {
            "transport": "streamable-http",
            "url": "https://example.test/mcp",
            "headers": {"X-B": "2", "X-A": "1"},
        }
    )
    second = _request(
        {
            "headers": {"X-A": "1", "X-B": "2"},
            "url": "https://example.test/mcp",
            "transport": "streamable-http",
        }
    )
    assert first.fingerprint == second.fingerprint


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("url", "https://attacker.test/mcp"),
        ("command", "attacker-command"),
        ("args", ["--changed"]),
        ("headers", {"Authorization": "Bearer ${OTHER_TOKEN}"}),
        ("env", {"TOKEN": "${OTHER_TOKEN}"}),
        ("oauth_scope", "admin"),
    ],
)
def test_any_effective_config_change_invalidates_fingerprint(field, changed):
    base = {
        "transport": "streamable-http",
        "url": "https://trusted.test/mcp",
        "command": "python",
        "args": ["server.py"],
        "headers": {"Authorization": "Bearer ${TOKEN}"},
        "env": {"TOKEN": "${TOKEN}"},
        "oauth_scope": "read",
    }
    original = _request(base)
    modified = _request({**base, field: changed})
    assert original.fingerprint != modified.fingerprint


def test_inline_server_wholly_overrides_same_named_file_entry(tmp_path):
    mcp_file = tmp_path / ".mcp.json"
    mcp_file.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "server": {
                        "url": "https://file-attacker.test/${HOST_SECRET}",
                        "transport": "streamable-http",
                    }
                }
            }
        )
    )
    request = build_approval_request(
        "server",
        mcp_file=mcp_file,
        servers={"server": {"url": "https://inline.test/mcp"}},
    )
    assert request.target == "https://inline.test/mcp"
    assert request.variables == ()


@pytest.mark.parametrize(
    "config",
    [
        {"transport": "stdio"},
        {"transport": "streamable-http"},
        {"transport": "websocket", "url": "https://example.test"},
        {"command": "python", "args": "server.py"},
        {"url": "https://example.test", "headers": {"X-Test": 1}},
        {"command": "python", "env": {"TOKEN": 1}},
        {"command": ""},
        {"url": ""},
        {"url": "https://example.test", "oauth_manual": "false"},
        {"url": "https://example.test", "oauth_scope": ["read"]},
        {"url": "https://example.test", "expand_env_vars": True},
    ],
)
def test_invalid_config_is_rejected_before_approval(config):
    with pytest.raises(ValueError):
        _request(config)


def test_request_finds_nested_bindings_without_reading_secret(monkeypatch):
    secret = "must-not-appear-in-review"
    monkeypatch.setenv("HOST_SECRET", secret)
    request = _request(
        {
            "command": "python",
            "args": ["server.py", "${HOST_SECRET}"],
            "env": {"TOKEN": "prefix-${HOST_SECRET}"},
        }
    )
    assert request.bindings == (
        ("HOST_SECRET", "args[1]"),
        ("HOST_SECRET", "env.TOKEN"),
    )
    assert secret not in request.review_text()


def test_approved_resolution_handles_http_and_stdio_sinks(monkeypatch):
    monkeypatch.setenv("HOST_SECRET", "canary")
    request = _request(
        {
            "url": "https://example.test/${HOST_SECRET}",
            "transport": "streamable-http",
            "headers": {"Authorization": "Bearer ${HOST_SECRET}"},
            "args": ["--token", "${HOST_SECRET}"],
            "env": {"TOKEN": "${HOST_SECRET}"},
        }
    )
    resolved = resolve_approved_environment(request)
    assert resolved["url"] == "https://example.test/canary"
    assert resolved["headers"]["Authorization"] == "Bearer canary"
    assert resolved["args"] == ["--token", "canary"]
    assert resolved["env"] == {"TOKEN": "canary"}
    assert request.config["env"] == {"TOKEN": "${HOST_SECRET}"}


def test_resolution_fails_before_partial_output_when_variable_is_missing(monkeypatch):
    monkeypatch.setenv("PRESENT", "value")
    monkeypatch.delenv("MISSING", raising=False)
    request = _request(
        {
            "url": "https://example.test/${PRESENT}",
            "headers": {"Authorization": "Bearer ${MISSING}"},
        }
    )
    with pytest.raises(ValueError, match="MISSING"):
        resolve_approved_environment(request)
    assert request.config["url"].endswith("${PRESENT}")


def test_error_redaction_covers_literal_and_url_encoded_environment_values(monkeypatch):
    monkeypatch.setenv("HOST_SECRET", "canary / value")
    request = _request({"url": "https://example.test/${HOST_SECRET}"})

    redacted = redact_approved_environment(
        request,
        "literal canary / value; encoded canary%20%2F%20value; plus canary+%2F+value",
    )

    assert "canary" not in redacted
    assert redacted.count("${HOST_SECRET}") == 3


def test_store_contains_no_secret_and_is_user_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HOST_SECRET", "canary-secret")
    request = _request(
        {
            "url": "https://example.test/mcp",
            "headers": {"Authorization": "Bearer ${HOST_SECRET}"},
        }
    )
    path = tmp_path / "mcp_approvals.json"
    store = MCPApprovalStore(path)
    store.approve(request)

    content = path.read_text()
    assert store.is_approved(request) is True
    assert "canary-secret" not in content
    assert "HOST_SECRET" in content
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_corrupt_or_unknown_store_version_fails_closed(tmp_path):
    request = _request({"url": "https://example.test/mcp"})
    path = tmp_path / "mcp_approvals.json"
    path.write_text("not json")
    assert MCPApprovalStore(path).is_approved(request) is False
    path.write_text('{"version": 999, "approvals": {}}')
    assert MCPApprovalStore(path).is_approved(request) is False


def test_revoke_removes_all_historical_approvals_for_server(tmp_path):
    store = MCPApprovalStore(tmp_path / "mcp_approvals.json")
    first = _request({"url": "https://one.test/mcp"})
    second = _request({"url": "https://two.test/mcp"})
    other = _request({"url": "https://other.test/mcp"}, name="other")
    for request in (first, second, other):
        store.approve(request)

    assert store.revoke_server("server") is True
    assert store.is_approved(first) is False
    assert store.is_approved(second) is False
    assert store.is_approved(other) is True


def test_review_redacts_url_userinfo_query_and_fragment():
    request = _request(
        {
            "url": "https://user:password@example.test/mcp?token=secret#fragment",
            "transport": "streamable-http",
        }
    )
    review = request.review_text()
    assert "example.test/mcp?<redacted>" in review
    assert "password" not in review
    assert "token=secret" not in review
    assert "fragment" not in review


def test_review_lists_security_metadata_without_header_secret():
    request = _request(
        {
            "url": "https://example.test/mcp",
            "headers": {
                "Authorization": "hardcoded-secret",
                "Host": "virtual.example.test",
            },
            "oauth_redirect_uri": "https://callback.example.test/return?code=secret",
            "oauth_scope": "read admin",
            "oauth_client_id": "approved-client",
        }
    )

    review = request.review_text()

    assert "Header names: Authorization, Host" in review
    assert "hardcoded-secret" not in review
    assert "OAuth redirect: https://callback.example.test/return?<redacted>" in review
    assert "OAuth scope: read admin" in review
    assert "OAuth client ID: approved-client" in review


def test_review_escapes_terminal_control_characters():
    request = _request(
        {
            "command": "python\x1b[2J\u202espoof",
            "env": {"BAD\x07KEY": "${HOST_SECRET}"},
        }
    )
    review = request.review_text()
    assert "\x1b" not in review
    assert "\x07" not in review
    assert "\u202e" not in review
    assert r"\x1b" in review
    assert r"\x07" in review
    assert r"\u202e" in review


def test_stdio_review_shows_bounded_literal_invocation():
    request = _request(
        {
            "command": "sh",
            "args": ["-c", "touch /tmp/review-this-command", "x" * 300],
        }
    )

    review = request.review_text()

    assert "sh -c 'touch /tmp/review-this-command'" in review
    assert "more characters" in review
    assert len(request.target) < 300
