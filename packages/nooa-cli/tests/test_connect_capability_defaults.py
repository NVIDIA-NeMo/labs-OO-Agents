# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``nooa connect`` runs the structured-output and developer-role checks by default (fake HTTP only)."""

import json

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands.connect import command

from nooa import llm_config
from tests.unifiedllm.connect.connect_http import mock_http, response_body

FORMAT_PROMPT = 'Reply with the JSON object {"ok": true} and nothing else.'
DEVELOPER_TEXT = "Answer with the single word ok."
UNSUPPORTED = {
    "error": {
        "message": "This response_format type is unavailable now",
        "type": "invalid_request_error",
        "param": "response_format",
    }
}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("CAP_TEST_KEY", "offline-dummy")
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])


def items(body):
    return body.get("messages", body.get("input", []))


def text(message):
    """The text of a Chat or Responses message (string or content parts)."""
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content


def kind(body):
    """Which capability check a request body is, or None for other test calls."""
    # LiteLLM may send the developer message as system, so match its text.
    if any(text(m) == DEVELOPER_TEXT for m in items(body)):
        return "role:developer"
    if any(text(m) == FORMAT_PROMPT for m in items(body)):
        fmt = body.get("response_format") or body.get("text", {}).get("format") or {}
        return "format:" + fmt.get("type", "?")
    return None


def fake_http(monkeypatch, style, *, reject_json_schema=False):
    sent = []

    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        check = kind(body)
        if check == "format:json_schema" and reject_json_schema:
            return httpx.Response(400, json=UNSUPPORTED)
        if check and check.startswith("format:"):
            return httpx.Response(200, json=response_body(style, '{"ok": true}'))
        return httpx.Response(200, json=response_body(style))

    mock_http(monkeypatch, handle)
    return sent


def args(style, *extra):
    return [
        "model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        style,
        "--api-key-env",
        "CAP_TEST_KEY",
        "--max-tokens",
        "1000",
        *extra,
    ]


def test_wizard_runs_capability_checks_and_saves_the_facts(monkeypatch, tmp_path):
    sent = fake_http(monkeypatch, "chat", reject_json_schema=True)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, args("chat", "--no-catalogue", "--yes", "--output", str(path))
    )
    assert result.exit_code == 0, result.output
    checks = [kind(body) for body in sent if kind(body)]
    # json_schema was rejected as unsupported, so json_object is tried too.
    assert checks == ["format:json_schema", "format:json_object", "role:developer"]
    assert "Structured output (JSON schema)" in result.output
    assert "Structured output (JSON mode)" in result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["structured_output_format"] == "json_object"
    assert [k for k in entry if k.startswith("structured_output_format")] == [
        "structured_output_format"
    ]
    # LiteLLM sends the developer message as system on this Chat route, so
    # developer support stays unknown and is not saved; it is not a failure.
    assert "developer_role" not in entry
    assert "Developer instructions: Not available" in result.output


@pytest.mark.parametrize("stage", ["capabilities", "all"])
@pytest.mark.parametrize("style", ["chat", "responses"])
def test_stage_runs_capability_checks_and_reports_the_facts(monkeypatch, stage, style):
    sent = fake_http(monkeypatch, style)
    result = CliRunner().invoke(command, args(style, "--stage", stage))
    report = json.loads(result.stdout)
    if stage == "capabilities":
        # This fake server returns no reasoning or cache data, so only the
        # capabilities stage is fully green here; test_connect_stages checks
        # that "all" stays ok with the capability checks included.
        assert result.exit_code == 0, result.output
        assert report["ok"] is True
    # json_object is not needed once json_schema is accepted.
    assert [kind(body) for body in sent if kind(body)] == ["format:json_schema", "role:developer"]
    assert {"format:json_schema", "role:developer"} <= set(report["checks"])
    entry = report["data"]["entry"]
    assert entry["structured_output_format"] == "json_schema"
    assert [k for k in entry if k.startswith("structured_output_format")] == [
        "structured_output_format"
    ]
    assert entry["provenance"]["structured_output_format"]["source"] == "connect"
    assert entry["provenance"]["developer_role"]["source"] == "connect"
    # Responses sends the developer message as is, so support is detected;
    # LiteLLM sends it as system on this Chat route, so it stays unknown.
    if style == "responses":
        assert entry["developer_role"] is True
    else:
        assert "developer_role" not in entry
    if stage == "capabilities":
        # Only the baseline runs besides the capability checks.
        assert len(sent) == 3


@pytest.mark.parametrize("stage", ["routing", "tools", "session"])
def test_other_stages_do_not_run_capability_checks(monkeypatch, stage):
    sent = fake_http(monkeypatch, "chat")
    CliRunner().invoke(command, args("chat", "--stage", stage))
    assert sent
    assert not [body for body in sent if kind(body)]


def test_anthropic_route_sends_no_capability_check(monkeypatch):
    sent = fake_http(monkeypatch, "anthropic")
    # The fake server returns no cache data, so "ok" is not checked here.
    result = CliRunner().invoke(command, args("anthropic", "--stage", "all"))
    assert not [body for body in sent if kind(body)]
    entry = json.loads(result.stdout)["data"]["entry"]
    assert entry["developer_role"] is False
    assert entry["provenance"]["developer_role"]["check"] == "not_applicable"
