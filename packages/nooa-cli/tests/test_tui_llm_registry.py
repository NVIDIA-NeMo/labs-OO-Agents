# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TUI model-registry command-line and bootstrap coverage."""

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from nooa_cli.commands.tui import command
from nooa_cli.registry_sources import RegistrySourceError, resolve_registry_source
from nooa_cli.tui.bootstrap import _load_llm_registry
from nooa_cli.tui.config import Config


def test_tui_help_documents_explicit_llm_config() -> None:
    result = CliRunner().invoke(command, ["--help"])

    assert result.exit_code == 0
    assert "--llm-config FILE" in result.output
    assert "--registry SOURCE" in result.output
    assert "highest precedence" in result.output


def test_config_accepts_repeated_explicit_registry_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"

    config = Config.load(llm_config=[first, second])

    assert config.llm_config_paths == [first, second]


def test_tui_registry_option_passes_resolved_yaml_to_config(tmp_path: Path) -> None:
    resolved = tmp_path / "resolved.yaml"
    config = MagicMock()
    tui_main = AsyncMock(return_value=None)

    with (
        patch(
            "nooa_cli.registry_sources.resolve_registry_sources",
            return_value=[resolved],
        ) as resolve_sources,
        patch("nooa_cli.tui.config.Config.load", return_value=config) as load_config,
        patch("nooa_cli.tui.main.main", tui_main),
    ):
        result = CliRunner().invoke(
            command,
            ["--registry", "https://git.example.com/team/registry.git", "--no-splash"],
        )

    assert result.exit_code == 0
    resolve_sources.assert_called_once_with(["https://git.example.com/team/registry.git"])
    assert load_config.call_args.kwargs["llm_config"] == [resolved]
    tui_main.assert_awaited_once()


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


def test_local_registry_directory_uses_manifest(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    config = registry / "config" / "models.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("models: {}\n")
    (registry / "nooa-registry.yaml").write_text(
        "version: 1\nllm_config: config/models.yaml\n"
    )

    assert resolve_registry_source(registry) == config.resolve()


def test_registry_manifest_cannot_escape_source_directory(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    (tmp_path / "outside.yaml").write_text("models: {}\n")
    (registry / "nooa-registry.yaml").write_text("llm_config: ../outside.yaml\n")

    with pytest.raises(RegistrySourceError, match="stay inside"):
        resolve_registry_source(registry)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


def test_git_registry_is_cached_and_refreshed(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-b", "main")
    config = source / "private" / "models.yaml"
    config.parent.mkdir()
    config.write_text("models:\n  first: {model_name: openai/first}\n")
    (source / "nooa-registry.yaml").write_text(
        "version: 1\nllm_config: private/models.yaml\n"
    )
    _git(source, "add", ".")
    _git(
        source,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "initial",
    )
    monkeypatch.setenv("NEMO_OO_REGISTRY_CACHE", str(tmp_path / "cache"))

    materialized = resolve_registry_source(source.as_uri())
    assert "first" in materialized.read_text()

    config.write_text("models:\n  second: {model_name: openai/second}\n")
    _git(source, "add", ".")
    _git(
        source,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "update",
    )

    refreshed = resolve_registry_source(source.as_uri())
    assert refreshed == materialized
    assert "second" in refreshed.read_text()
    assert "first" not in refreshed.read_text()
