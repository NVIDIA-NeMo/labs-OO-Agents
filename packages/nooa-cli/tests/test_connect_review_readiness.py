# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions for outstanding human/catalogue review findings."""

import httpx
import pytest
import yaml

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http


@pytest.mark.parametrize("indent", [2, 4, 6])
@pytest.mark.parametrize("alias", ["local", "new"])
def test_save_preserves_indentation_and_neighbor_comments(tmp_path, indent, alias):
    path = tmp_path / "models.yaml"
    pad = " " * indent
    neighbor = f"{pad}# Important description for other\n{pad}other: {{model_name: openai/other}}\n"
    source = (
        f"models:\n{pad}local:\n{pad}{pad}model_name: openai/old\n"
        + neighbor
        + "# Unrelated configuration\nsettings: true\n"
    )
    path.write_text(source)
    connect.write({"model_name": "openai/new"}, path, alias=alias)
    text = path.read_text()
    expected = yaml.safe_load(source)
    expected["models"][alias] = connect.configure_entry({"model_name": "openai/new"})
    assert yaml.safe_load(text) == expected
    assert neighbor in text
    assert "# Unrelated configuration\nsettings: true\n" in text
    assert f"\n{pad}{alias}:\n" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("efforts", [42, "high", ["high", "", 3, None, "low"]])
async def test_catalogue_normalizes_before_frontend_use(monkeypatch, efforts):
    mock_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "model",
                        "context_length": 10000,
                        "top_provider": "bad",
                        "reasoning": {"supported_efforts": efforts, "default_effort": []},
                    }
                ]
            },
        ),
    )
    (record,) = await connect.catalogue()
    assert record["context_length"] == 10000
    assert record["top_provider"] == {}
    assert record["reasoning"]["supported_efforts"] == (
        ["high", "low"] if isinstance(efforts, list) else []
    )
    assert "default_effort" not in record["reasoning"]


@pytest.mark.asyncio
@pytest.mark.parametrize("identifier", [None, 4, ""])
async def test_catalogue_rejects_invalid_ids(monkeypatch, identifier):
    mock_http(monkeypatch, lambda request: httpx.Response(200, json={"data": [{"id": identifier}]}))
    with pytest.raises(ValueError, match="string id"):
        await connect.catalogue()


def test_no_probe_can_authenticate_model_discovery(monkeypatch, tmp_path):
    import click
    from click.testing import CliRunner
    from nooa_cli.commands import _connect_prompts as _connect_prompts
    from nooa_cli.commands import connect as cli

    monkeypatch.delenv("CONNECT_DISCOVERY_KEY", raising=False)
    sent = []

    def handle(request):
        sent.append(request)
        assert request.method == "GET"
        assert request.headers["authorization"] == "Bearer discovery-secret"
        return httpx.Response(200, json={"data": [{"id": "model"}]})

    def prompt(label, **kwargs):
        if label.startswith("API key"):
            assert kwargs["hide_input"]
            return "discovery-secret"
        assert label == "Model"
        raise click.Abort()

    mock_http(monkeypatch, handle)
    monkeypatch.setattr(_connect_prompts, "prompt", prompt)
    target = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        cli.command,
        [
            "--endpoint",
            "https://models.example/v1",
            "--api-key-env",
            "CONNECT_DISCOVERY_KEY",
            "--no-probe",
            "--no-catalogue",
            "--output",
            str(target),
        ],
    )
    assert len(sent) == 1
    assert result.exit_code == 1
    assert "discovery-secret" not in result.output
    assert not target.exists()
