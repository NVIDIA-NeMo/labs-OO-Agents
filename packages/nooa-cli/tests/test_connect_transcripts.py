# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete offline human transcripts, including narrow terminals and source precedence."""

import os
from copy import deepcopy

import click
import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_view as view
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands.connect import command

from nooa import llm_config
from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body


def plain(output):
    return " ".join(click.unstyle(output).split())


@pytest.mark.parametrize("columns", [24, 40, 80, 120])
@pytest.mark.parametrize("color", [False, True, "no_color"])
@pytest.mark.parametrize("media", [None, [], ["text"], ["image"], ["file"]])
def test_full_manual_transcript_sources_media_pending_save(
    monkeypatch, tmp_path, columns, color, media
):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setattr(
        view.shutil, "get_terminal_size", lambda *a, **k: os.terminal_size((columns, 24))
    )
    if color == "no_color":
        monkeypatch.setenv("NO_COLOR", "")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET"  # Save-without-test never sends selected samples.
        public = request.url.host == "openrouter.ai"
        model = (
            {"id": "model", "context_window": 64000}
            if not public
            else {
                "id": "model",
                "context_length": 128000,
                "architecture": {"input_modalities": ["image", "pdf"]},
            }
        )
        if not public and media is not None:
            model["input_modalities"] = media
        return httpx.Response(200, json={"data": [model]})

    mock_http(monkeypatch, handle)
    target = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--as",
            "local",
            "--endpoint",
            "https://server.test/v1",
            "--api-style",
            "chat",
            "--api-key-env",
            "",
            "--test-media-inputs",
            "image,pdf",
            "--reply-tokens",
            "2000",
            "--output",
            str(target),
        ],
        input="save\ny\n",
        color=bool(color),
    )
    assert result.exit_code == 0, result.output
    output = plain(result.output)
    assert output.count("Model details") == 2  # One section + compact proposal source row.
    assert "Reading model details" not in output and "Looking up" not in output
    assert "Source: This server" in output
    if media is None:
        assert "Additional details: OpenRouter" in output
        assert "Media types: Images, PDFs" in output
    else:
        assert "Additional details: OpenRouter" not in output
        if media == ["image"]:
            assert "Media types: Images" in output and "Media types: Images, PDFs" not in output
        else:
            assert output.count("No media is listed.") == 1
            assert "Media types:" not in output
    if media == ["file"]:
        assert "PDF support isn't listed" in output
    assert (
        output.index("Model details")
        < output.index("Proposed configuration")
        < output.index("Would you like to test")
    )
    assert "Media samples: Images, PDFs" in output
    assert "Saved model" in output and "get_llm_client" in "".join(output.split())
    assert "not tested (manual setup)" in output
    assert yaml.safe_load(target.read_text())["models"]["local"]["media_inputs"] == []
    assert len(calls) == 2
    if color == "no_color" or color is False:
        assert "\x1b[" not in result.output
    # Paths/long tokens are reflowed, never silently cropped.
    assert str(target) in "".join(click.unstyle(result.output).split())
    assert all(len(row) <= columns for row in click.unstyle(result.output).splitlines())


@pytest.mark.parametrize("columns", [24, 40, 80, 120])
@pytest.mark.parametrize("color", [False, True, "no_color"])
@pytest.mark.parametrize("failure", [False, True])
def test_full_check_transcript_and_failure_no_save(monkeypatch, tmp_path, columns, color, failure):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setattr(
        view.shutil, "get_terminal_size", lambda *a, **k: os.terminal_size((columns, 24))
    )
    if color == "no_color":
        monkeypatch.setenv("NO_COLOR", "")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "POST"
        return (
            httpx.Response(401, json={"error": {"message": "PRIVATE_PROVIDER_BODY"}})
            if failure
            else httpx.Response(200, json=response_body("chat", '{"value":323}'))
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
            "https://server.test/v1",
            "--api-style",
            "chat",
            "--api-key-env",
            "",
            "--no-metadata",
            "--media-inputs",
            "",
            "--probe",
            "minimal",
            "--reply-tokens",
            "2000",
            "--output",
            str(target),
        ],
        input="262144\n262144\ntest\n" + ("cancel\n" if failure else "y\n"),
        color=bool(color),
    )
    assert result.exit_code == (1 if failure else 0), result.output
    output = plain(result.output)
    assert "Connection: Failed" in output if failure else "Connection: Passed" in output
    assert "PRIVATE_PROVIDER_BODY" not in output
    assert target.exists() is not failure
    assert len(calls) == 1
    assert "No media is listed." in output
    assert all(len(row) <= columns for row in click.unstyle(result.output).splitlines())
    if color == "no_color" or color is False:
        assert "\x1b[" not in result.output


@pytest.mark.parametrize("columns", [24, 40, 80, 120])
@pytest.mark.parametrize("color", [False, True, "no_color"])
def test_model_details_long_text_scrub_retry_and_help(monkeypatch, columns, color):
    monkeypatch.setattr(
        view.shutil, "get_terminal_size", lambda *a, **k: os.terminal_size((columns, 24))
    )
    if color == "no_color":
        monkeypatch.setenv("NO_COLOR", "")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    metadata = {
        "id": "model",
        "description": "Long description " * 12 + "SECRET_KEY",
        "model_type": "decision",
        "capabilities": {"image_input": {"supported": False}},
    }
    before = deepcopy(metadata)

    @click.command()
    def transcript():
        state = wizard.WizardState(
            model="model",
            api_key="SECRET_KEY",
            selected_endpoint_model=metadata,
            media_hints=metadata,
            server_details=connect.ModelDetails("model", metadata),
        )
        wizard.show_model_metadata(state)
        progress = view.CheckProgress()
        progress.update(
            "level:very-long-custom-name", {"outcome": "retrying", "tested_reply_tokens": 8192}
        )
        progress.update("media:image", {"outcome": "not_probed", "reason": "budget exhausted"})
        progress.finish()

    result = CliRunner().invoke(transcript, color=bool(color))
    output = plain(result.output)
    assert "No media is listed." in output and "Model type: Decision" in output
    assert "SECRET_KEY" not in output and "[redacted]" in output
    assert "retrying with 8,192 tokens" in output and "check budget used up" in output
    assert all(len(row) <= columns for row in click.unstyle(result.output).splitlines())
    assert metadata == before
    help_result = CliRunner().invoke(command, ["--help"], color=bool(color))
    assert help_result.exit_code == 0
    assert "--stage" in help_result.output
    assert all(len(row) <= columns for row in click.unstyle(help_result.output).splitlines())
    if color == "no_color" or color is False:
        assert "\x1b[" not in result.output + help_result.output
