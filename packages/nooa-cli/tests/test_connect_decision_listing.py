# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision catalogue discovery is independent of POST endpoint and chat limits."""

import json
from dataclasses import asdict

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands._connect_stages import read_discovery
from nooa_cli.commands.connect import command

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http

ENDPOINT = "https://hub.test/v1/decisions"
MODELS_URL = "https://hub.test/v1/models"
MODEL = {
    "id": "decision/model",
    "context_window": 8192,
    "max_input_tokens": 4096,
    "max_output_tokens": 1024,
    "input_modalities": ["image"],
    "supported_interfaces": ["systemone", "openai"],
}


def stage_options(style="systemone"):
    return ["--endpoint", ENDPOINT, "--api-style", style, "--models-url", MODELS_URL]


@pytest.mark.parametrize("style", connect.DECISION_API_STYLES)
def test_decision_discover_and_offline_plan_keep_endpoint_and_hints(monkeypatch, tmp_path, style):
    requests = []
    monkeypatch.setenv("LISTING_KEY", "listing-only-key")
    monkeypatch.delenv("MISSING_INFERENCE_KEY", raising=False)

    def handle(request):
        requests.append(request)
        assert request.method == "GET"
        assert str(request.url) == MODELS_URL
        assert request.headers["authorization"] == "Bearer listing-only-key"
        return httpx.Response(200, json={"data": [MODEL]})

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            *stage_options(style),
            "--stage",
            "discover",
            "--api-key-env",
            "MISSING_INFERENCE_KEY",
            "--discovery-api-key-env",
            "LISTING_KEY",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)["data"]
    assert data["api_base"] == ENDPOINT
    assert data["models_url"] == MODELS_URL
    assert data["models"] == [{**MODEL, "media_source": MODELS_URL}]
    assert "listing-only-key" not in result.output
    discovery = tmp_path / "discovery.json"
    discovery.write_text(result.stdout)
    mock_http(monkeypatch, lambda request: pytest.fail("Offline plan must not send HTTP"))
    result = CliRunner().invoke(
        command,
        [
            MODEL["id"],
            *stage_options(style),
            "--as",
            "local",
            "--stage",
            "plan",
            "--discovery-file",
            str(discovery),
        ],
    )
    assert result.exit_code == 0, result.output
    entry = json.loads(result.stdout)["data"]["entry"]
    assert entry["endpoint"] == ENDPOINT
    assert entry["model_name"] == MODEL["id"]
    assert entry["provenance"]["advertised_model"] == {**MODEL, "media_source": MODELS_URL}
    assert entry["provenance"]["models_url"] == MODELS_URL
    assert entry["provenance"]["probes"] == {}
    assert entry["media_inputs"] == []
    assert not {
        "max_tokens",
        "context_window",
        "max_input_tokens",
        "max_output_tokens",
        "reasoning_levels",
    }.intersection(entry)
    assert len(requests) == 1


def test_discovery_files_match_catalogue_source_not_decision_url(tmp_path):
    path = tmp_path / "discovery.json"
    path.write_text(
        json.dumps(asdict(connect.Discovery("https://hub.test/v1", (MODEL,), MODELS_URL)))
    )
    assert read_discovery(path, "https://hub.test/v1/systemone", models_url=MODELS_URL) == [
        {**MODEL, "media_source": MODELS_URL}
    ]
    with pytest.raises(ValueError, match="different endpoint"):
        read_discovery(path, ENDPOINT, models_url="https://other.test/v1/models")
    path.write_text(json.dumps({"api_base": ENDPOINT, "models": [MODEL]}))
    with pytest.raises(ValueError, match="different endpoint"):
        read_discovery(path, ENDPOINT, models_url=MODELS_URL)


@pytest.mark.parametrize("explicit_auth", [None, "-", "CATALOGUE_KEY"])
def test_cross_origin_listing_requires_explicit_auth_choice(monkeypatch, explicit_auth):
    monkeypatch.setenv("INFERENCE_KEY", "inference-only-key")
    monkeypatch.setenv("CATALOGUE_KEY", "catalogue-only-key")
    requests = []

    def handle(request):
        requests.append(request)
        assert str(request.url) == "https://catalogue.test/list"
        assert "inference-only-key" not in str(request.headers)
        if explicit_auth == "-":
            assert "authorization" not in request.headers
        else:
            assert request.headers["authorization"] == "Bearer catalogue-only-key"
        return httpx.Response(200, json={"data": [MODEL]})

    mock_http(monkeypatch, handle)
    options = [
        "--endpoint",
        ENDPOINT,
        "--api-style",
        "openai",
        "--stage",
        "discover",
        "--models-url",
        "https://catalogue.test/list",
        "--api-key-env",
        "INFERENCE_KEY",
    ]
    if explicit_auth is not None:
        options += ["--discovery-api-key-env", explicit_auth]
    result = CliRunner().invoke(command, options)
    if explicit_auth is None:
        assert result.exit_code != 0
        assert "--discovery-api-key-env" in result.stdout
        assert not requests
    else:
        assert result.exit_code == 0, result.output
        assert len(requests) == 1
    assert "inference-only-key" not in result.output
    assert "catalogue-only-key" not in result.output


@pytest.mark.parametrize("style", connect.DECISION_API_STYLES)
@pytest.mark.parametrize("offline", [False, True])
def test_wizard_lists_decisions_and_saves_unverified_hints(monkeypatch, tmp_path, style, offline):
    path = tmp_path / "models.yaml"
    reviewed_entries = []
    original_write = connect.write

    def capture_write(entry, target, *, alias):
        reviewed_entries.append(entry)
        original_write(entry, target, alias=alias)

    monkeypatch.setattr(connect, "write", capture_write)
    monkeypatch.setattr(wizard, "entries", lambda *args, **kwargs: {})
    monkeypatch.setattr(wizard, "shadowing_source", lambda *args, **kwargs: None)
    requests = []

    def handle(request):
        requests.append(request)
        assert request.method == "GET"  # --no-probe means no inference at all.
        assert str(request.url) == MODELS_URL
        return httpx.Response(200, json={"data": [MODEL]})

    mock_http(monkeypatch, handle)
    options = [
        *stage_options(style),
        "--as",
        "local",
        "--api-key-env",
        "",
        "--no-probe",
        "--no-catalogue",
        "--output",
        str(path),
    ]
    if offline:
        discovery = tmp_path / "discovery.json"
        discovery.write_text(json.dumps(asdict(connect.Discovery(ENDPOINT, (MODEL,), MODELS_URL))))
        options += ["--discovery-file", str(discovery)]
        mock_http(monkeypatch, lambda request: pytest.fail("File picker must not fetch again"))
    result = CliRunner().invoke(command, options, input=MODEL["id"] + "\nkeep\nsave\ny\n")
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["endpoint"] == ENDPOINT
    assert entry["model_name"] == MODEL["id"]
    assert entry["api_style"] == style
    assert reviewed_entries[0]["provenance"]["advertised_model"] == {
        **MODEL,
        "media_source": MODELS_URL,
    }
    assert reviewed_entries[0]["provenance"]["models_url"] == MODELS_URL
    assert "provenance" not in entry  # Keep the established runtime-only registry contract.
    assert entry["media_inputs"] == []
    assert not {"max_tokens", "context_window", "reasoning_levels"}.intersection(entry)
    assert len(requests) == (0 if offline else 1)


@pytest.mark.parametrize("failed_listing", [False, True])
def test_wizard_exact_id_fallback_does_not_guess_models_url(monkeypatch, tmp_path, failed_listing):
    monkeypatch.setattr(wizard, "entries", lambda *args, **kwargs: {})
    monkeypatch.setattr(wizard, "shadowing_source", lambda *args, **kwargs: None)
    path = tmp_path / "models.yaml"
    requests = []

    def handle(request):
        requests.append(request)
        assert str(request.url) == MODELS_URL
        return httpx.Response(404)

    mock_http(
        monkeypatch, handle if failed_listing else lambda request: pytest.fail("No guessed URL")
    )
    options = [
        "--endpoint",
        ENDPOINT,
        "--api-style",
        "systemone",
        "--as",
        "local",
        "--api-key-env",
        "",
        "--no-probe",
        "--no-catalogue",
        "--output",
        str(path),
    ]
    if failed_listing:
        options += ["--models-url", MODELS_URL]
    result = CliRunner().invoke(command, options, input="exact/decision-id\nsave\ny\n")
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["model_name"] == "exact/decision-id"
    assert entry["endpoint"] == ENDPOINT
    assert "provenance" not in entry
    assert len(requests) == (1 if failed_listing else 0)


@pytest.mark.parametrize("style", connect.DECISION_API_STYLES)
def test_hub_preset_uses_shared_listing_for_decision_route(monkeypatch, tmp_path, style):
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "hub-key")
    monkeypatch.setattr(wizard, "entries", lambda *args, **kwargs: {})
    monkeypatch.setattr(wizard, "shadowing_source", lambda *args, **kwargs: None)
    path = tmp_path / "models.yaml"
    endpoint = "https://inference-api.nvidia.com/v1/decisions"
    requests = []

    def handle(request):
        requests.append(request)
        assert request.method == "GET"
        assert str(request.url) == "https://inference-api.nvidia.com/v1/models"
        assert request.headers["authorization"] == "Bearer hub-key"
        return httpx.Response(200, json={"data": [MODEL, {"id": "chat/model"}]})

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "--provider",
            "hub",
            "--endpoint",
            endpoint,
            "--api-style",
            style,
            "--as",
            "local",
            "--no-probe",
            "--no-catalogue",
            "--output",
            str(path),
        ],
        input=MODEL["id"] + "\nkeep\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert "Found 2 models" in result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["endpoint"] == endpoint
    assert entry["api_key_env"] == "NVIDIA_INFERENCE_API_KEY"
    assert len(requests) == 1


def test_listing_cancel_does_not_write_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(wizard, "entries", lambda *args, **kwargs: {})
    path = tmp_path / "models.yaml"
    mock_http(monkeypatch, lambda request: httpx.Response(200, json={"data": [MODEL]}))
    result = CliRunner().invoke(
        command, [*stage_options(), "--api-key-env", "", "--no-probe", "--output", str(path)]
    )
    assert result.exit_code != 0
    assert not path.exists()


@pytest.mark.parametrize("status", [200, 401])
def test_separate_listing_and_inference_credentials_are_both_redacted(monkeypatch, status):
    inference_key = "inference-secret-canary"
    catalogue_key = "catalogue-secret-canary"
    monkeypatch.setenv("INFERENCE_KEY", inference_key)
    monkeypatch.setenv("CATALOGUE_KEY", catalogue_key)

    def handle(request):
        assert request.method == "GET"
        assert request.headers["authorization"] == "Bearer " + catalogue_key
        return httpx.Response(
            status,
            json={
                "data": [
                    {
                        "id": "decision/model",
                        "description": f"echo {catalogue_key} {inference_key}",
                    }
                ]
            },
        )

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            catalogue_key,
            *stage_options(),
            "--stage",
            "discover",
            "--api-key-env",
            "INFERENCE_KEY",
            "--discovery-api-key-env",
            "CATALOGUE_KEY",
        ],
    )
    assert result.exit_code == (0 if status == 200 else 1), result.output
    assert inference_key not in result.output
    assert catalogue_key not in result.output  # Includes data, run context, and diagnostic prompt.
    if status == 200:
        assert (
            json.loads(result.stdout)["data"]["models"][0]["description"]
            == "echo [redacted] [redacted]"
        )


def test_wizard_preview_redacts_echoed_catalogue_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("INFERENCE_KEY", "inference-secret-canary")
    monkeypatch.setenv("CATALOGUE_KEY", "catalogue-secret-canary")
    monkeypatch.setattr(wizard, "entries", lambda *args, **kwargs: {})
    monkeypatch.setattr(wizard, "shadowing_source", lambda *args, **kwargs: None)
    path = tmp_path / "models.yaml"

    def handle(request):
        assert request.method == "GET"
        assert request.headers["authorization"] == "Bearer catalogue-secret-canary"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        **MODEL,
                        "description": "echo inference-secret-canary catalogue-secret-canary",
                    }
                ]
            },
        )

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            *stage_options(),
            "--as",
            "local",
            "--no-probe",
            "--no-catalogue",
            "--api-key-env",
            "INFERENCE_KEY",
            "--discovery-api-key-env",
            "CATALOGUE_KEY",
            "--show-config",
            "--output",
            str(path),
        ],
        input=MODEL["id"] + "\nkeep\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert "inference-secret-canary" not in result.output
    assert "catalogue-secret-canary" not in result.output
    assert "echo [redacted] [redacted]" in result.output
    assert "secret-canary" not in path.read_text()


def test_wizard_diagnostic_redacts_both_active_credentials(monkeypatch):
    monkeypatch.setenv("CATALOGUE_KEY", "catalogue-secret-canary")
    state = wizard.WizardState(
        api_key="inference-secret-canary", discovery_api_key_env="CATALOGUE_KEY"
    )
    entry = connect.plan("local", MODEL["id"], "systemone", ENDPOINT, "").entry
    entry["provenance"]["advertised_model"] = {
        "description": "echo inference-secret-canary catalogue-secret-canary"
    }
    diagnostic = wizard.setup_diagnostic(state, "checks", entry, {}, api_key=state.api_key)
    assert "inference-secret-canary" not in diagnostic
    assert "catalogue-secret-canary" not in diagnostic


@pytest.mark.parametrize("credential", ["CATALOGUE_KEY", "-", "UNSET_KEY", "bad-name", None])
def test_explicit_chat_metadata_respects_listing_url_style_and_credential(monkeypatch, credential):
    monkeypatch.setenv("CATALOGUE_KEY", "catalogue-secret-canary")
    monkeypatch.delenv("UNSET_KEY", raising=False)
    endpoint = "https://inference.test/v1"
    listing_url = "https://catalogue.test/custom/list"
    requests = []
    candidate = {"id": "chat/model", "context_length": 4096}

    def handle(request):
        requests.append(request)
        assert request.method == "GET"
        assert str(request.url) == listing_url + "?limit=100"
        assert "authorization" not in request.headers
        if credential == "CATALOGUE_KEY":
            assert request.headers["x-api-key"] == "catalogue-secret-canary"
        else:
            assert "x-api-key" not in request.headers
        assert request.headers["anthropic-version"] == "2023-06-01"
        return httpx.Response(
            200, json={"data": [{"id": "chat/model", "max_input_tokens": 2048}], "has_more": False}
        )

    async def catalogue():
        assert credential in {"CATALOGUE_KEY", "-"}, (
            "Validate explicit config before even public catalogue HTTP"
        )
        return [candidate]

    monkeypatch.setattr(connect, "catalogue", catalogue)
    mock_http(monkeypatch, handle)
    state = wizard.WizardState(
        data={},
        model="chat/model",
        endpoint=endpoint,
        api_style="chat",
        api_key="inference-secret-canary",
        models_url=listing_url,
        discovery_style="anthropic",
        discovery_api_key_env=credential,
        endpoint_models=[],
        yes=True,
        output_tokens=200,
    )
    if credential not in {"CATALOGUE_KEY", "-"}:
        with pytest.raises(ValueError):
            wizard.configure_metadata(state)
        assert requests == []
    else:
        assert wizard.configure_metadata(state)
        assert len(requests) == 1
        assert state.candidate["endpoint_limits"]["max_input_tokens"] == 2048
        assert state.discovery_api_key == (
            "catalogue-secret-canary" if credential == "CATALOGUE_KEY" else None
        )
