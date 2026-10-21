# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connect documentation's offline catalogue reuse keeps explicit credentials."""

import json
import re
import shlex
from pathlib import Path

import pytest
from click.testing import CliRunner
from nooa_cli.commands.connect import command

from tests.unifiedllm.connect.connect_http import mock_http

ROOT = Path(__file__).parents[3]


def test_documented_decision_catalogue_reuse_keeps_key_source(monkeypatch, tmp_path):
    text = (ROOT / "docs/model-connect.md").read_text()
    snippet = (
        re.search(
            r"uv run nooa connect decision-model --stage plan --as decisions(.*?)(?:\n```)",
            text,
            re.S,
        )
        .group(0)
        .removesuffix("\n```")
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HUB_KEY", "not-saved-secret")
    Path("models.json").write_text(
        json.dumps(
            {
                "data": {
                    "api_base": "https://hub.example/v1/decisions",
                    "models_url": "https://hub.example/v1/models",
                    "models": [{"id": "decision-model", "context_window": 4096}],
                }
            }
        )
    )
    mock_http(monkeypatch, lambda request: pytest.fail("Documented plan must be offline"))
    args = shlex.split(snippet.replace("\\\n", " "))[4:]
    result = CliRunner().invoke(command, args)
    assert result.exit_code == 0, result.output
    entry = json.loads(result.stdout)["data"]["entry"]
    assert entry["api_key_env"] == "HUB_KEY"
    assert entry["provenance"]["advertised_model"]["id"] == "decision-model"
    assert "not-saved-secret" not in result.output
    assert "max_tokens" not in entry


def test_connect_help_distinguishes_native_decision_limits():
    result = CliRunner().invoke(command, ["--help"])
    assert result.exit_code == 0
    text = " ".join(result.output.split())
    assert "Decision models accept images only" in text
    assert "Decision calls have no reply limit" in text
