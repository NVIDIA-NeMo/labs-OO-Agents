# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-facing recovery and metadata paths use the shared client on the wire."""

import json

import click
import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_view as view
from nooa_cli.commands.connect import command

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body


def test_wizard_saves_endpoint_limits_not_catalogue(tmp_path, monkeypatch):
    async def catalogue():
        return [
            {
                "id": "model",
                "context_length": 200000,
                "top_provider": {"max_completion_tokens": 180000},
            }
        ]

    monkeypatch.setattr(connect, "catalogue", catalogue)

    def handle(request):
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={
                "data": [{"id": "model", "max_input_tokens": 100000, "max_output_tokens": 16000}]
            },
        )

    mock_http(monkeypatch, handle)
    target = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--as",
            "local",
            "--endpoint",
            "https://models.example/v1",
            "--api-style",
            "chat",
            "--api-key-env",
            "",
            "--no-probe",
            "--yes",
            "--output",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(target.read_text())["models"]["local"]
    assert entry["context_window"] == 100000
    assert entry["max_tokens"] == 16000
    assert "endpoint input limit" in result.output
    assert entry["provenance"]["endpoint_limits"]["max_input_tokens"] == 100000


@pytest.mark.parametrize("cached", [(12, 0), (0, 0)])
def test_stage_cache_miss_does_not_fail_working_entry(monkeypatch, cached):
    calls = []

    def handle(request):
        calls.append(request)
        data = response_body("chat")
        data["choices"][0]["message"]["reasoning_content"] = "private reasoning"
        data["usage"]["prompt_tokens_details"] = {
            "cached_tokens": cached[len(calls) - 2] if len(calls) > 1 else 0
        }
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--stage",
            "session",
            "--endpoint",
            "https://models.example/v1",
            "--api-style",
            "chat",
            "--prompt-key",
            "--max-tokens",
            "2048",
        ],
        input="test-key\n",
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["checks"]["cache"]["outcome"] == ("confirmed" if cached[0] else "warning")
    assert report["diagnostic_prompt"] is None
    assert "Context window unknown" in report["warnings"][0]


def test_stage_plan_reuses_discovery_without_http(tmp_path, monkeypatch):
    path = tmp_path / "discovery.json"
    path.write_text(
        json.dumps(
            {
                "data": {
                    "api_base": "https://models.example/v1",
                    "models": [
                        {"id": "model", "max_input_tokens": 100000, "max_output_tokens": 16000}
                    ],
                }
            }
        )
    )
    mock_http(monkeypatch, lambda request: pytest.fail("plan must stay offline"))
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--stage",
            "plan",
            "--endpoint",
            "https://models.example/v1",
            "--api-style",
            "chat",
            "--discovery-file",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.output
    entry = json.loads(result.stdout)["data"]["entry"]
    assert entry["context_window"] == 100000
    assert entry["max_tokens"] == 16000
    assert "unverified" in json.loads(result.stdout)["warnings"][0]
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--stage",
            "plan",
            "--endpoint",
            "https://other.example/v1",
            "--api-style",
            "chat",
            "--discovery-file",
            str(path),
        ],
    )
    assert result.exit_code != 0
    assert "different endpoint" in result.output


def test_wizard_retries_only_selected_interface_at_120_seconds(tmp_path, monkeypatch):
    posts = []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "model"}]})
        posts.append(request)
        if len(posts) <= 3:
            raise httpx.ReadTimeout("private server detail", request=request)
        assert request.url.path == "/v1/chat/completions"
        assert request.extensions["timeout"]["read"] == 120
        return httpx.Response(200, json=response_body("chat"))

    mock_http(monkeypatch, handle)
    monkeypatch.setenv("REVIEW_KEY", "test-key")
    target = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "--as",
            "local",
            "--endpoint",
            "https://models.example/v1",
            "--api-key-env",
            "REVIEW_KEY",
            "--no-catalogue",
            "--probe",
            "minimal",
            "--max-tokens",
            "8192",
            "--output",
            str(target),
        ],
        input="y\nmodel\nlonger\nchat\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(posts) == 5  # Configured-cap routing is distinct from interface discovery.
    assert result.output.count("Approve API checks") == 1
    assert result.output.count("Results ·") == 1
    assert "route may be slow" in result.output
    assert "private server detail" not in result.output
    entry = yaml.safe_load(target.read_text())["models"]["local"]
    assert entry["provenance"]["tokens_charged_to_budget"] == 4 * 712 + 8192 + 512


@pytest.mark.parametrize("correct", [True, False])
def test_progress_reports_puzzle_result_and_parse_failure(correct):
    @click.command()
    def display():
        progress = view.CheckProgress()
        progress.update(
            "level:high",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "answer_correct": correct,
                "reasoning_tokens": 4096,
            },
        )
        progress.update("anthropic", {"outcome": "not_confirmed", "error": "ReasoningReplayError"})

    result = CliRunner().invoke(display)
    normalized_output = " ".join(result.output.split())
    assert "4,096 reasoning tokens" in normalized_output
    assert ("answer correct" if correct else "answer incorrect") in normalized_output
    assert "Reply not understood (ReasoningReplayError)" in result.output
    assert "Not checked" not in result.output
