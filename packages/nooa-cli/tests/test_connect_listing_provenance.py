# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discovery-file parser bounds and offline listing-source propagation."""

import json

import httpx
import pytest
from click.testing import CliRunner
from nooa_cli.commands._connect_stages import read_discovery_listing
from nooa_cli.commands._connect_wizard import (
    WizardState,
    configure_checks,
    prepare_metadata,
    select_model,
)
from nooa_cli.commands.connect import command

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http

SOURCE = "https://catalogue.example/available"
ENDPOINT = "https://inference.example/v1"


@pytest.fixture
def no_http(monkeypatch):
    sent = []

    def forbidden(request):
        sent.append(request)
        raise AssertionError("Offline parser/plan must not use HTTP")

    mock_http(monkeypatch, forbidden)
    yield
    assert not sent


def write_listing(tmp_path, models, **extra):
    path = tmp_path / "discovery.json"
    path.write_text(json.dumps({"api_base": ENDPOINT, "models": models, **extra}))
    return path


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("media", [False, True])
def test_stage_file_source_is_independent_of_media(tmp_path, no_http, style, media):
    record = {
        "id": "model",
        "max_input_tokens": 12345,
        "media_source": "https://forged.example/models",
        "endpoint_limits": {"url": "forged"},
    }
    if media:
        record["input_modalities"] = ["image"]
    path = write_listing(tmp_path, [record], models_url=SOURCE)
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--stage",
            "plan",
            "--as",
            "local",
            "--api-style",
            style,
            "--endpoint",
            ENDPOINT,
            "--api-key-env",
            "",
            "--discovery-file",
            str(path),
            "--models-url",
            SOURCE,
        ],
    )
    assert result.exit_code == 0, result.output
    entry = json.loads(result.output)["data"]["entry"]
    assert entry["provenance"]["endpoint_limits"] == {"url": SOURCE, "max_input_tokens": 12345}


@pytest.mark.parametrize(
    "models",
    [
        [{"id": ""}],
        [{"id": "  "}],
        [{"id": "bad\nID"}],
        [{"id": "x" * 501}],
        [{"id": "same"}, {"id": "same"}],
        [{"id": 123}],
        [None],
        [{"id": str(i)} for i in range(5001)],
    ],
)
def test_invalid_file_ids_and_count_rejected_before_http(tmp_path, no_http, models):
    path = write_listing(tmp_path, models)
    with pytest.raises(ValueError, match="unique valid IDs"):
        read_discovery_listing(path, ENDPOINT)
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--stage",
            "plan",
            "--as",
            "local",
            "--api-style",
            "chat",
            "--endpoint",
            ENDPOINT,
            "--api-key-env",
            "",
            "--discovery-file",
            str(path),
        ],
    )
    assert result.exit_code == 2, result.output


@pytest.mark.parametrize("document", [[], None, "bad", 1, {"data": []}])
def test_bad_document_shapes_are_safe_usage_errors(tmp_path, no_http, document):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document))
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--stage",
            "plan",
            "--as",
            "local",
            "--api-style",
            "chat",
            "--endpoint",
            ENDPOINT,
            "--api-key-env",
            "",
            "--discovery-file",
            str(path),
        ],
    )
    assert result.exit_code == 2, result.output


def test_file_byte_and_model_count_boundaries(tmp_path, no_http):
    path = write_listing(tmp_path, [{"id": str(i)} for i in range(5000)])
    assert len(read_discovery_listing(path, ENDPOINT).models) == 5000
    raw = path.read_bytes()
    limit = 5 * 1024 * 1024
    path.write_bytes(raw + b" " * (limit - len(raw)))
    assert len(read_discovery_listing(path, ENDPOINT).models) == 5000
    with path.open("ab") as output:
        output.write(b" ")
    with pytest.raises(ValueError, match="5 MiB"):
        read_discovery_listing(path, ENDPOINT)


def test_file_unknown_model_does_not_invent_limits(tmp_path, no_http):
    path = write_listing(
        tmp_path, [{"id": "neighbour", "context_window": 99999}], models_url=SOURCE
    )
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--stage",
            "plan",
            "--as",
            "local",
            "--api-style",
            "openai",
            "--endpoint",
            ENDPOINT,
            "--api-key-env",
            "",
            "--discovery-file",
            str(path),
            "--models-url",
            SOURCE,
        ],
    )
    assert result.exit_code == 0, result.output
    entry = json.loads(result.output)["data"]["entry"]
    assert "endpoint_limits" not in entry["provenance"]
    assert "context_window" not in entry


def test_wizard_file_limits_source_reaches_plan(tmp_path, no_http):
    path = write_listing(tmp_path, [{"id": "model", "context_window": 12345}], models_url=SOURCE)
    state = WizardState(
        model="model",
        endpoint=ENDPOINT,
        api_style="chat",
        api_key_env="",
        discovery_file=path,
        models_url=SOURCE,
        no_catalogue=True,
        budget_tokens=10000,
        output_tokens=200,
        reasoning_output_tokens=1024,
        approval="none",
        alias="local",
        data={},
        yes=True,
        discovery_style="chat",
    )
    assert select_model(state)
    assert prepare_metadata(state)
    assert configure_checks(state)
    assert state.proposal.entry["provenance"]["endpoint_limits"]["url"] == SOURCE


def test_wizard_metadata_get_limits_source_reaches_plan(monkeypatch):
    sent = []

    def handle(request):
        sent.append(request)
        assert request.method == "GET"
        assert str(request.url) == SOURCE
        return httpx.Response(200, json={"data": [{"id": "model", "context_window": 12345}]})

    mock_http(monkeypatch, handle)

    async def catalogue():
        return [{"id": "model", "context_length": 54321}]

    monkeypatch.setattr(connect, "catalogue", catalogue)
    state = WizardState(
        model="model",
        endpoint=ENDPOINT,
        api_style="chat",
        api_key_env="",
        models_url=SOURCE,
        no_catalogue=False,
        discovery_api_key_env="-",
        candidate={"id": "model", "context_length": 54321},
        budget_tokens=10000,
        output_tokens=200,
        reasoning_output_tokens=1024,
        approval="none",
        alias="local",
        data={},
        yes=True,
        discovery_style="chat",
    )
    assert prepare_metadata(state)
    assert configure_checks(state)
    assert state.proposal.entry["provenance"]["endpoint_limits"]["url"] == SOURCE
    assert len(sent) == 1


@pytest.mark.parametrize(
    "bad_source",
    [
        123,
        [],
        {},
        True,
        "https://user:password@example.test/models",
        "https://example.test/models?key=secret",
    ],
)
def test_file_listing_source_validation_before_http(tmp_path, no_http, bad_source):
    path = write_listing(
        tmp_path, [{"id": "model", "context_window": 12345}], models_url=bad_source
    )
    with pytest.raises(ValueError):
        read_discovery_listing(path, ENDPOINT)


def test_file_saved_source_without_models_url_option(tmp_path, no_http):
    path = write_listing(
        tmp_path,
        [
            {
                "id": "model",
                "context_window": 12345,
                "source": "https://forged.example/models",
                "models_url": "https://forged.example/models",
            }
        ],
        models_url=SOURCE,
    )
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--stage",
            "plan",
            "--as",
            "local",
            "--api-style",
            "chat",
            "--endpoint",
            ENDPOINT,
            "--api-key-env",
            "",
            "--discovery-file",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.output
    entry = json.loads(result.output)["data"]["entry"]
    assert entry["provenance"]["endpoint_limits"]["url"] == SOURCE


def test_wizard_live_picker_keeps_limits_only_listing_source(monkeypatch):
    from nooa_cli.commands import _connect_prompts as prompts

    sent = []

    def handle(request):
        sent.append(request)
        assert request.method == "GET"
        assert str(request.url) == SOURCE
        return httpx.Response(200, json={"data": [{"id": "model", "context_window": 12345}]})

    mock_http(monkeypatch, handle)
    monkeypatch.setattr(prompts, "prompt", lambda *a, **kw: "model")
    state = WizardState(
        endpoint=ENDPOINT,
        api_style="chat",
        api_key_env="",
        models_url=SOURCE,
        no_catalogue=True,
        discovery_api_key_env="-",
        budget_tokens=10000,
        output_tokens=200,
        reasoning_output_tokens=1024,
        approval="none",
        alias="local",
        data={},
        yes=True,
        discovery_style="chat",
    )
    assert select_model(state)
    assert prepare_metadata(state)
    assert configure_checks(state)
    assert state.proposal.entry["provenance"]["endpoint_limits"]["url"] == SOURCE
    assert len(sent) == 1
