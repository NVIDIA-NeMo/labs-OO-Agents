# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TUI model-registry command-line and bootstrap coverage."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from nooa_cli.commands.tui import command
from nooa_cli.tui.bootstrap import _load_llm_registry
from nooa_cli.tui.config import Config


def test_tui_help_documents_explicit_llm_config() -> None:
    result = CliRunner().invoke(command, ["--help"])

    assert result.exit_code == 0
    assert "--llm-config FILE" in result.output
    assert "highest precedence" in result.output


def test_config_accepts_repeated_explicit_registry_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"

    config = Config.load(llm_config=[first, second])

    assert config.llm_config_paths == [first, second]


def test_explicit_registry_paths_load_after_discovered_chain(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled.yaml"
    private = tmp_path / "private.yaml"
    messages = []

    with (
        patch("nooa.llm_config.llm_config_chain", return_value=[bundled]),
        patch("nooa.secrets.load_secrets_into_env") as load_secrets,
        patch("nooa.unifiedllm.reload_registry") as reload_registry,
    ):
        _load_llm_registry(messages, [private])

    load_secrets.assert_called_once_with()
    reload_registry.assert_called_once_with(bundled, private)
    assert messages == []
