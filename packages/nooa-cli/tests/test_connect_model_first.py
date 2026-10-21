# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider → model → metadata → interface → review, and known OpenRouter routes."""

import json

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands.connect import command

from nooa import llm_config
from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body
from tests.unifiedllm.connect.test_connect_decision_smoke import fixture_response
from tests.unifiedllm.connect.test_server_details import GROUP, SCHEMA

ALPHA = "https://openrouter.ai/api/alpha/decisions"
ROUTER_MODEL = "typesafe/jev-router"


@pytest.fixture
def local(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("MODEL_KEY", "approved-work-key")
    return tmp_path / "models.yaml"


@pytest.mark.parametrize(
    "provider", ["hub", "openrouter", "anthropic", "openai", "nvidia", "custom"]
)
def test_provider_model_interface_order_and_listing_headers_before_choice(
    monkeypatch, local, provider
):
    calls, questions, reviews = [], [], []
    prompt = prompts.prompt
    choice = prompts.labelled_choice

    def ask(text, **kwargs):
        questions.append(text)
        if text == "Request interface":
            assert calls and all(request.method == "GET" for request in calls)
            assert (
                questions.index("Choose a provider")
                < questions.index("Search available models (type to search; Tab to select)")
                < questions.index(text)
            )
        return prompt(text, **kwargs)

    def review(text, options, **kwargs):
        if text == "Would you like to test these settings?":
            assert all(request.method == "GET" for request in calls)
            assert questions[-1] == "Request interface"
            reviews.append(options)
        return choice(text, options, **kwargs)

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        if request.url.host == "openrouter.ai":
            assert request.url.path == "/api/v1/models" and "authorization" not in request.headers
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "exact-model",
                            "context_length": 64000,
                            "top_provider": {"max_completion_tokens": 8000},
                        }
                    ]
                },
            )
        if request.url.path == "/openapi.json":
            assert "authorization" not in request.headers
            return httpx.Response(200, json=SCHEMA)
        if request.url.path == "/model_group/info":
            assert request.headers["authorization"] == "Bearer approved-work-key"
            return httpx.Response(200, json={"data": [{**GROUP, "model_group": "exact-model"}]})
        if provider == "anthropic":
            assert request.headers["x-api-key"] == "approved-work-key"
            assert request.headers["anthropic-version"] == "2023-06-01"
            assert "authorization" not in request.headers
            item = {
                "id": "exact-model",
                "max_input_tokens": 64000,
                "max_tokens": 8000,
                "capabilities": {"image_input": {"supported": True}},
            }
        else:
            assert request.headers["authorization"] == "Bearer approved-work-key"
            item = (
                {"id": "exact-model"}
                if provider == "openai"
                else {"id": "exact-model", "context_window": 64000, "max_output_tokens": 8000}
            )
        return httpx.Response(200, json={"data": [item], "has_more": False})

    monkeypatch.setattr(prompts, "prompt", ask)
    monkeypatch.setattr(prompts, "labelled_choice", review)
    mock_http(monkeypatch, handle)
    style = connect.PROVIDERS[provider].api_style if provider != "custom" else "chat"
    answers = (
        provider
        + "\n"
        + ("https://generic.test/v1\n" if provider == "custom" else "")
        + "exact-model\n"
        + style
        + "\nsave\ny\n"
    )
    result = CliRunner().invoke(
        command,
        [
            "--as",
            "local",
            "--api-key-env",
            "MODEL_KEY",
            "--no-probe",
            "--media-inputs",
            "",
            "--reply-tokens",
            "2000",
            "--output",
            str(local),
        ],
        input=answers,
    )
    assert result.exit_code == 0, result.output
    assert reviews and " ".join(result.output.split()).index("Model: exact-model") < " ".join(
        result.output.split()
    ).index("Request interface") < result.output.index("Proposed configuration")
    assert yaml.safe_load(local.read_text())["models"]["local"]["api_style"] == style
    assert "approved-work-key" not in result.output + local.read_text()
    assert sum(
        request.url.path.endswith("/models") and request.url.host != "openrouter.ai"
        for request in calls
    ) == (0 if provider == "openrouter" else 1)
    assert sum(request.url.host == "openrouter.ai" for request in calls) == 1


def test_no_metadata_exact_model_precedes_interface_without_key_or_http(monkeypatch, local):
    questions = []
    original = prompts.prompt

    def ask(text, **kwargs):
        questions.append(text)
        return original(text, **kwargs)

    monkeypatch.setattr(prompts, "prompt", ask)
    mock_http(
        monkeypatch, lambda request: pytest.fail("Complete offline setup must not fetch or infer")
    )
    result = CliRunner().invoke(
        command,
        [
            "--as",
            "local",
            "--api-key-env",
            "",
            "--no-metadata",
            "--no-probe",
            "--media-inputs",
            "",
            "--output",
            str(local),
        ],
        input="custom\nhttps://generic.test/v1\nexact/model\nchat\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    model_question = "Exact model ID (metadata is disabled; Ctrl-C to cancel)"
    assert (
        questions.index("Choose a provider")
        < questions.index(model_question)
        < questions.index("Request interface")
    )


def test_failed_metadata_model_then_interface_then_test_no_repeat(monkeypatch, local):
    calls, questions, reviews = [], [], []
    original_prompt, original_choice = prompts.prompt, prompts.labelled_choice

    def ask(text, **kwargs):
        questions.append(text)
        return original_prompt(text, **kwargs)

    def choose(text, options, **kwargs):
        if text == "Would you like to test these settings?":
            assert len(calls) == 2 and all(request.method == "GET" for request in calls)
            reviews.append(options)
        return original_choice(text, options, **kwargs)

    def handle(request):
        calls.append(request)
        if request.url.host == "openrouter.ai":
            return httpx.Response(200, json={"data": []})
        if request.method == "GET":
            return httpx.Response(403)
        assert reviews
        return httpx.Response(200, json=response_body("chat"))

    monkeypatch.setattr(prompts, "prompt", ask)
    monkeypatch.setattr(prompts, "labelled_choice", choose)
    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "--endpoint",
            "https://generic.test/v1",
            "--api-key-env",
            "",
            "--as",
            "local",
            "--media-inputs",
            "",
            "--probe",
            "minimal",
            "--output",
            str(local),
        ],
        input="exact/model\nchat\ntest\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert questions.index("Exact model ID (if known; Ctrl-C to cancel)") < questions.index(
        "Request interface"
    )
    assert [request.method for request in calls] == ["GET", "GET", "POST"]


@pytest.mark.parametrize("override", [None, "mapping", "endpoint"])
@pytest.mark.parametrize("yes", [False, True])
def test_known_openrouter_systemone_alpha_or_explicit_override(monkeypatch, local, override, yes):
    calls, reviews = [], []
    original = prompts.labelled_choice

    def choose(text, options, **kwargs):
        if text == "Would you like to test these settings?":
            assert calls == []
            reviews.append(options)
        return original(text, options, **kwargs)

    target = ALPHA if override is None else "https://openrouter.ai/exact/systemone"

    def handle(request):
        calls.append(request)
        assert yes or reviews
        assert request.method == "POST" and str(request.url) == target
        assert request.headers["authorization"] == "Bearer approved-work-key"
        body = json.loads(request.content)
        assert body["model"] == ROUTER_MODEL
        assert set(body) == {"model", "state", "questions"}
        return fixture_response(request)

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    mock_http(monkeypatch, handle)
    extra = (
        ["--systemone-endpoint" if override == "mapping" else "--endpoint", target]
        if override
        else []
    )
    result = CliRunner().invoke(
        command,
        [
            ROUTER_MODEL,
            "--provider",
            "openrouter",
            "--api-style",
            "systemone",
            "--as",
            "local",
            "--api-key-env",
            "MODEL_KEY",
            "--no-catalogue",
            "--media-inputs",
            "",
            "--probe",
            "minimal",
            "--output",
            str(local),
            *extra,
            *(["--yes"] if yes else []),
        ],
        input="" if yes else "test\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    entry = yaml.safe_load(local.read_text())["models"]["local"]
    assert entry["endpoint"] == target and entry["client_type"] == "decision"
    assert entry["model_name"] == ROUTER_MODEL and "max_tokens" not in entry
    assert "System One Decisions" in result.output


@pytest.mark.parametrize("budget", [512, 60000])
def test_openrouter_discovery_offers_chat_responses_and_decision_without_name_classification(
    monkeypatch, local, budget
):
    calls, reviews, entries = [], [], []
    original_choice, original_write = prompts.labelled_choice, connect.write

    def choose(text, options, **kwargs):
        if text == "Would you like to test these settings?":
            assert calls == []
            reviews.append(options)
        return original_choice(text, options, **kwargs)

    def write(entry, *args, **kwargs):
        entries.append(entry)
        return original_write(entry, *args, **kwargs)

    def handle(request):
        calls.append(request)
        assert reviews and request.method == "POST"
        assert request.headers["authorization"] == "Bearer approved-work-key"
        body = json.loads(request.content)
        assert body["model"] == ROUTER_MODEL
        if str(request.url) == ALPHA:
            assert set(body) == {"model", "state", "questions"}
            return fixture_response(request)
        if request.url.path.endswith("/messages"):
            return httpx.Response(404)
        style = "responses" if request.url.path.endswith("/responses") else "chat"
        return httpx.Response(200, json=response_body(style))

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    monkeypatch.setattr(connect, "write", write)
    mock_http(monkeypatch, handle)
    answers = "discover\ntest\n" + ("systemone\n" if budget > 512 else "") + "y\n"
    result = CliRunner().invoke(
        command,
        [
            ROUTER_MODEL,
            "--provider",
            "openrouter",
            "--as",
            "local",
            "--api-key-env",
            "MODEL_KEY",
            "--no-catalogue",
            "--media-inputs",
            "",
            "--probe",
            "minimal",
            "--budget-tokens",
            str(budget),
            "--output",
            str(local),
        ],
        input=answers,
    )
    assert result.exit_code == 0, result.output
    assert sum(str(request.url) == ALPHA for request in calls) == 1
    assert len(calls) == (1 if budget == 512 else 6)
    entry = yaml.safe_load(local.read_text())["models"]["local"]
    assert entry["endpoint"] == ALPHA and entry["client_type"] == "decision"
    assert entries[0]["provenance"]["tokens_charged_to_budget"] <= budget
    plain = " ".join(result.output.split())
    if budget > 512:
        assert "Accepted client types: completion, responses, decision" in plain
        assert (
            "Working API formats: OpenAI-compatible Chat, OpenAI Responses, System One Decisions"
            in plain
        )


def test_headless_interfaces_still_requires_explicit_decision_url(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "POST" and str(request.url) != ALPHA
        assert request.url.path in {
            "/api/v1/chat/completions",
            "/api/v1/responses",
            "/api/v1/messages",
        }
        return httpx.Response(404)

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            ROUTER_MODEL,
            "--stage",
            "interfaces",
            "--endpoint",
            "https://openrouter.ai/api/v1",
            "--api-key-env",
            "",
        ],
    )
    assert result.exit_code == 1 and len(calls) == 3


@pytest.mark.parametrize("share", [False, True])
def test_late_known_interface_never_forwards_key_to_another_origin_without_optin(
    monkeypatch, local, share
):
    calls = []
    target = "https://decision.test/exact"

    def handle(request):
        calls.append(request)
        assert share and request.method == "POST" and str(request.url) == target
        assert request.headers["authorization"] == "Bearer approved-work-key"
        return fixture_response(request)

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "exact/model",
            "--endpoint",
            "https://listing.test/v1",
            "--systemone-endpoint",
            target,
            "--as",
            "local",
            "--api-key-env",
            "MODEL_KEY",
            "--no-metadata",
            "--media-inputs",
            "",
            "--probe",
            "minimal",
            "--output",
            str(local),
            *(["--share-interface-key"] if share else []),
        ],
        input="systemone\ntest\ny\n",
    )
    assert result.exit_code == (0 if share else 2), result.output
    assert len(calls) == (1 if share else 0)
    if not share:
        assert "--share-interface-key" in result.output and not local.exists()
