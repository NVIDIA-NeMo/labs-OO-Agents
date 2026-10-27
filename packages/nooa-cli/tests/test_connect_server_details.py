# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server details → public gaps → review: inference and routing stay opt-in."""

import json

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands.connect import command

from nooa import llm_config
from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body
from tests.unifiedllm.connect.test_server_details import BASE, GROUP, MODEL, SCHEMA


@pytest.fixture
def local(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("WORK_KEY", "normal-work-key")
    return tmp_path / "models.yaml"


@pytest.mark.parametrize("action", ["save", "test"])
def test_litellm_details_first_review_zero_posts_then_optional_test(monkeypatch, local, action):
    calls, reviews = [], []
    original = prompts.labelled_choice

    def choose(label, options, **kwargs):
        if label == "Would you like to test these settings?":
            assert [r.method for r in calls] == ["GET"] * 4
            assert [r.url.path for r in calls[:3]] == [
                "/v1/models",
                "/openapi.json",
                "/model_group/info",
            ]
            reviews.append(options)
        return original(label, options, **kwargs)

    def handle(request):
        calls.append(request)
        if request.method == "POST":
            assert reviews  # Inference can happen only after Test now.
            assert request.url.path == "/v1/chat/completions"
            assert json.loads(request.content)["model"] == MODEL
            return httpx.Response(200, json=response_body("chat"))
        if request.url.host == "openrouter.ai":
            assert "authorization" not in request.headers
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": MODEL,
                            "context_length": 2000000,
                            "top_provider": {"max_completion_tokens": 256000},
                            "reasoning": {"supported_efforts": ["high"], "default_effort": "high"},
                        }
                    ]
                },
            )
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": MODEL, "max_input_tokens": 1048576, "max_output_tokens": 128000}
                    ]
                },
            )
        if request.url.path == "/openapi.json":
            assert "authorization" not in request.headers
            return httpx.Response(200, json=SCHEMA)
        assert request.url.path == "/model_group/info"
        assert request.url.params["model_group"] == MODEL
        return httpx.Response(
            200,
            json={
                "data": [{**GROUP, "max_input_tokens": 2000000.0, "max_output_tokens": 256000.0}]
            },
        )

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            MODEL,
            "--as",
            "local",
            "--endpoint",
            BASE,
            "--api-style",
            "chat",
            "--server-metadata",
            "litellm",
            "--api-key-env",
            "WORK_KEY",
            "--reply-tokens",
            "2000",
            "--context-window",
            "60000",
            "--media-inputs",
            "image",
            "--probe",
            "minimal",
            "--output",
            str(local),
        ],
        input=action + "\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == (5 if action == "test" else 4)
    entry = yaml.safe_load(local.read_text())["models"]["local"]
    assert entry["context_window"] == 60000 and entry["max_tokens"] == 2000
    assert entry["media_inputs"] == ["image"] and entry["api_style"] == "chat"
    assert not entry.get("reasoning_levels")
    assert "normal-work-key" not in result.output + local.read_text()
    summary = " ".join(result.output.split())
    assert "Context window: 60,000 tokens" in summary
    assert "Reply limit: 2,000 tokens" in summary
    assert "Reply maximum: 128,000 tokens" in summary
    assert "Server: LiteLLM" in summary


@pytest.mark.parametrize(
    "style,endpoint,item",
    [
        (
            "responses",
            "https://api.openai.com/v1",
            {"id": "openai-model", "owned_by": "openai", "created": 1},
        ),
        (
            "anthropic",
            "https://api.anthropic.com/v1",
            {
                "id": "claude-model",
                "max_input_tokens": 200000,
                "max_tokens": 64000,
                "capabilities": {
                    "image_input": {"supported": True},
                    "pdf_input": {"supported": True},
                },
            },
        ),
    ],
)
def test_native_review_server_then_exact_public_gaps_no_profiles(
    monkeypatch, local, style, endpoint, item
):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        if request.url.host == "openrouter.ai":
            assert "authorization" not in request.headers
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": item["id"],
                            "context_length": 300000,
                            "top_provider": {"max_completion_tokens": 8000},
                            "reasoning": {"supported_efforts": ["high"], "default_effort": "high"},
                        }
                    ]
                },
            )
        assert request.url.copy_with(query=None) == httpx.URL(endpoint + "/models")
        if style == "anthropic":
            assert request.headers["x-api-key"] == "normal-work-key"
        else:
            assert request.headers["authorization"] == "Bearer normal-work-key"
        return httpx.Response(200, json={"data": [item], "has_more": False})

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            item["id"],
            "--as",
            "local",
            "--endpoint",
            endpoint,
            "--api-style",
            style,
            "--api-key-env",
            "WORK_KEY",
            "--media-inputs",
            "",
            "--reply-tokens",
            "2000",
            "--output",
            str(local),
        ],
        input="save\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 2 and calls[0].url.host != "openrouter.ai"
    entry = yaml.safe_load(local.read_text())["models"]["local"]
    assert entry["context_window"] == (200000 if style == "anthropic" else 300000)
    assert not entry.get("reasoning_levels")
    assert "reasoning_default" not in entry and entry["media_inputs"] == []


@pytest.mark.parametrize("mapping", [None, "metadata-b"])
def test_openrouter_reuses_one_public_get_and_never_imports_different_profile(
    monkeypatch, local, mapping
):
    calls, entries = [], []
    original = connect.write

    def write(entry, *args, **kwargs):
        entries.append(entry)
        return original(entry, *args, **kwargs)

    def handle(request):
        calls.append(request)
        assert request.method == "GET" and (
            str(request.url).split("?")[0] == connect.CATALOGUE_URL
            and request.url.params.get("output_modalities") == "text,decisions"
        )
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "model-a",
                        "context_length": 64000,
                        "top_provider": {"max_completion_tokens": 8000},
                        "reasoning": {"supported_efforts": ["low"], "default_effort": "low"},
                    },
                    {
                        "id": "metadata-b",
                        "context_length": 128000,
                        "reasoning": {"supported_efforts": ["high"], "default_effort": "high"},
                    },
                ]
            },
        )

    monkeypatch.setattr(connect, "write", write)
    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "model-a",
            "--as",
            "local",
            "--endpoint",
            "https://openrouter.ai/api/v1",
            "--api-style",
            "chat",
            "--api-key-env",
            "WORK_KEY",
            "--media-inputs",
            "",
            "--reply-tokens",
            "2000",
            "--output",
            str(local),
            *(["--catalogue-model", mapping] if mapping else []),
        ],
        input=("default\n" if mapping is None else "") + "save\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    entry = yaml.safe_load(local.read_text())["models"]["local"]
    assert entry["context_window"] == 64000
    assert entry["model_name"] == "openai/model-a"
    if mapping:
        assert not entry.get("reasoning_levels")
        fallback = entries[0]["provenance"]["server_details"]["records"][-1]
        assert fallback["id"] == mapping and fallback["selected_id"] == "model-a"
    else:
        assert entry["reasoning_levels"] == {"low": {"reasoning_effort": "low"}}


@pytest.mark.parametrize("stage", ["plan", "routing"])
def test_no_metadata_headless_preserves_pure_routing(monkeypatch, stage):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "POST" and request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json=response_body("chat"))

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--stage",
            stage,
            "--endpoint",
            "https://generic.test/v1",
            "--api-style",
            "chat",
            "--api-key-env",
            "",
            "--no-metadata",
            "--server-metadata",
            "litellm",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == (0 if stage == "plan" else 1)


@pytest.mark.parametrize(
    "style,endpoint",
    [("chat", "https://generic.test/v1"), ("systemone", "https://generic.test/exact-decisions")],
)
def test_exact_wizard_bypass_all_metadata_then_direct_test(monkeypatch, local, style, endpoint):
    from tests.unifiedllm.connect.test_connect_decision_smoke import fixture_response

    calls, reviews = [], []
    original = prompts.labelled_choice

    def choose(label, options, **kwargs):
        if label == "Would you like to test these settings?":
            assert calls == []
            reviews.append(options)
        return original(label, options, **kwargs)

    def handle(request):
        calls.append(request)
        assert reviews and request.method == "POST"
        assert str(request.url) == (endpoint + "/chat/completions" if style == "chat" else endpoint)
        return (
            httpx.Response(200, json=response_body("chat"))
            if style == "chat"
            else fixture_response(request)
        )

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "fixture",
            "--as",
            "local",
            "--endpoint",
            endpoint,
            "--api-style",
            style,
            "--api-key-env",
            "",
            "--media-inputs",
            "",
            "--no-metadata",
            "--probe",
            "minimal",
            "--output",
            str(local),
        ],
        input=("262144\n262144\n" if style == "chat" else "") + "test\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1


def test_model_info_stage_uses_approved_work_key_and_is_get_only(monkeypatch):
    monkeypatch.setenv("WORK_KEY", "normal-work-key")
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        if request.url.path == "/openapi.json":
            assert "authorization" not in request.headers
            return httpx.Response(200, json=SCHEMA)
        assert request.url.path == "/model_group/info"
        assert request.headers["authorization"] == "Bearer normal-work-key"
        return httpx.Response(200, json={"data": [GROUP]})

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            MODEL,
            "--stage",
            "model-info",
            "--endpoint",
            BASE,
            "--api-key-env",
            "WORK_KEY",
            "--server-metadata",
            "litellm",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)["data"]
    assert data["model_id"] == MODEL and data["requests"] == len(calls) == 3
    assert "normal-work-key" not in result.output
    assert data["model"]["model_type"] == "decision"


@pytest.mark.parametrize("badflag", ["--no-metadata", "--no-catalogue"])
def test_metadata_stage_bypass_flags_cannot_authorize_a_get(monkeypatch, badflag):
    mock_http(monkeypatch, lambda request: pytest.fail("Metadata was disabled"))
    result = CliRunner().invoke(
        command, ["model", "--stage", "model-info", "--endpoint", BASE, badflag]
    )
    assert result.exit_code == 2


@pytest.mark.parametrize("status", [401, 403, 404])
def test_failed_picker_metadata_not_retried_and_test_still_available(monkeypatch, local, status):
    calls = []

    def handle(request):
        calls.append(request)
        if request.url.host == "openrouter.ai":
            return httpx.Response(200, json={"data": []})
        if request.method == "GET":
            return httpx.Response(status)
        return httpx.Response(200, json=response_body("chat"))

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "--endpoint",
            "https://generic.test/v1",
            "--api-style",
            "chat",
            "--as",
            "local",
            "--api-key-env",
            "",
            "--media-inputs",
            "",
            "--probe",
            "minimal",
            "--output",
            str(local),
        ],
        input="model\ntest\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert [r.method for r in calls] == ["GET", "GET", "POST"]
    assert len([r for r in calls if r.url.host == "generic.test" and r.method == "GET"]) == 1


def test_empty_fetched_listing_reused_without_get(monkeypatch):
    state = wizard.WizardState(
        endpoint="https://generic.test/v1",
        model="model",
        api_style="chat",
        data={},
        discovery_endpoint="https://generic.test/v1",
        endpoint_models=(),
        endpoint_models_url="https://generic.test/v1/models",
        yes=True,
    )
    calls = []

    def handle(request):
        calls.append(request)
        assert (
            str(request.url).split("?")[0] == connect.CATALOGUE_URL
            and request.url.params.get("output_modalities") == "text,decisions"
        )
        return httpx.Response(200, json={"data": []})

    mock_http(monkeypatch, handle)
    assert wizard.prepare_metadata(state)
    assert len(calls) == 1 and state.server_details.requests == 0


def test_model_info_reuses_saved_native_record_and_reports_original_field(monkeypatch, tmp_path):
    discovery = tmp_path / "discovery.json"
    discovery.write_text(
        json.dumps(
            {
                "api_base": "https://api.test/v1",
                "models_url": "https://api.test/v1/models",
                "models": [
                    {
                        "id": "model",
                        "max_output_tokens": 64000,
                        "metadata_fields": {"max_output_tokens": "data[].max_tokens"},
                        "api_key": "private-config",
                    }
                ],
            }
        )
    )
    mock_http(monkeypatch, lambda request: pytest.fail("Supplied listing must be reused"))
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--stage",
            "model-info",
            "--endpoint",
            "https://api.test",
            "--api-style",
            "anthropic",
            "--api-key-env",
            "",
            "--discovery-file",
            str(discovery),
        ],
    )
    assert result.exit_code == 0, result.output
    details = json.loads(result.output)["data"]
    assert details["requests"] == 0
    assert details["records"][0]["fields"]["max_output_tokens"] == "data[].max_tokens"
    assert "private-config" not in result.output


def test_openrouter_model_picker_reuses_public_metadata_without_second_get(monkeypatch, local):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET" and (
            str(request.url).split("?")[0] == connect.CATALOGUE_URL
            and request.url.params.get("output_modalities") == "text,decisions"
        )
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "model",
                        "context_length": 128000,
                        "top_provider": {"context_length": 64000, "max_completion_tokens": 8000},
                    }
                ]
            },
        )

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "--endpoint",
            "https://openrouter.ai/api/v1",
            "--api-style",
            "chat",
            "--as",
            "local",
            "--api-key-env",
            "",
            "--media-inputs",
            "",
            "--reply-tokens",
            "2000",
            "--output",
            str(local),
        ],
        input="model\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert yaml.safe_load(local.read_text())["models"]["local"]["context_window"] == 64000


@pytest.mark.parametrize("status", [401, 403, 404])
def test_openrouter_denied_metadata_not_retried_and_save_remains_available(
    monkeypatch, local, status
):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        return httpx.Response(status)

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--endpoint",
            "https://openrouter.ai/api/v1",
            "--as",
            "local",
            "--api-style",
            "chat",
            "--api-key-env",
            "",
            "--media-inputs",
            "",
            "--output",
            str(local),
        ],
        input="save\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and local.exists()


@pytest.mark.parametrize("style", ["chat", "systemone"])
def test_explicit_missing_catalogue_mapping_does_not_block_test(monkeypatch, local, style):
    from tests.unifiedllm.connect.test_connect_decision_smoke import fixture_response

    calls = []
    endpoint = "https://generic.test/v1" if style == "chat" else "https://generic.test/decision"

    def handle(request):
        calls.append(request)
        if request.url.host == "openrouter.ai":
            return httpx.Response(403)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "fixture", "context_length": 64000}]})
        return (
            httpx.Response(200, json=response_body("chat"))
            if style == "chat"
            else fixture_response(request)
        )

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "fixture",
            "--endpoint",
            endpoint,
            "--api-style",
            style,
            "--as",
            "local",
            "--api-key-env",
            "",
            "--media-inputs",
            "",
            "--catalogue-model",
            "absent",
            "--probe",
            "minimal",
            "--output",
            str(local),
        ],
        input="test\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len([r for r in calls if r.url.host == "openrouter.ai"]) == 1
    assert len([r for r in calls if r.method == "POST"]) == 1


def test_shared_public_listing_and_explicit_mapping_reused_once(monkeypatch, local):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET" and (
            str(request.url).split("?")[0] == connect.CATALOGUE_URL
            and request.url.params.get("output_modalities") == "text,decisions"
        )
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "wire-model", "context_length": 64000},
                    {
                        "id": "other/model",
                        "context_length": 128000,
                        "reasoning": {"supported_efforts": ["high"]},
                    },
                ]
            },
        )

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "--endpoint",
            "https://generic.test/v1",
            "--api-style",
            "chat",
            "--as",
            "local",
            "--models-url",
            connect.CATALOGUE_URL,
            "--discovery-api-key-env",
            "-",
            "--catalogue-model",
            "other/model",
            "--api-key-env",
            "",
            "--media-inputs",
            "",
            "--output",
            str(local),
        ],
        input="wire-model\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    entry = yaml.safe_load(local.read_text())["models"]["local"]
    assert entry["context_window"] == 64000 and not entry.get("reasoning_levels")


def test_stage_plan_validates_source_connection_and_filters_foreign_profiles(monkeypatch, tmp_path):
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(
        json.dumps(
            {
                "source": connect.CATALOGUE_URL,
                "models": [
                    {
                        "id": "other/model",
                        "context_length": 128000,
                        "reasoning": {"supported_efforts": ["high"], "default_effort": "high"},
                        "default_parameters": {"max_tokens": 6000},
                    }
                ],
            }
        )
    )
    mock_http(monkeypatch, lambda request: pytest.fail("Stage plan stays offline"))
    flags = [
        "wire-model",
        "--stage",
        "plan",
        "--endpoint",
        "https://api.anthropic.com/v1",
        "--api-style",
        "anthropic",
        "--catalogue-file",
        str(catalogue),
        "--catalogue-model",
        "other/model",
        "--api-key-env",
        "",
    ]
    result = CliRunner().invoke(command, flags)
    assert result.exit_code == 0, result.output
    entry = json.loads(result.output)["data"]["entry"]
    assert (
        entry["context_window"] == 128000
        and not entry.get("reasoning_levels")
        and entry["max_tokens"] != 6000
    )
    discovery = tmp_path / "discovery.json"
    discovery.write_text(
        json.dumps(
            {
                "api_base": "https://wrong.test/v1",
                "models_url": "https://shared.test/models",
                "models": [{"id": "wire-model", "max_input_tokens": 999}],
            }
        )
    )
    mismatch = CliRunner().invoke(
        command,
        [*flags, "--discovery-file", str(discovery), "--models-url", "https://shared.test/models"],
    )
    assert mismatch.exit_code == 2 and "different endpoint" in mismatch.output
