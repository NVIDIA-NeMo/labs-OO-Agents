# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TUI model-registry command-line and bootstrap coverage."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
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
            ["--registry", "https://git.example.com/team/models.yaml", "--no-splash"],
        )

    assert result.exit_code == 0
    resolve_sources.assert_called_once_with(["https://git.example.com/team/models.yaml"])
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


class _RegistryHandler(BaseHTTPRequestHandler):
    content = b"models:\n  first: {model_name: openai/first}\n"
    request_paths: list[str] = []

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        type(self).request_paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(type(self).content)))
        self.end_headers()
        self.wfile.write(type(self).content)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def registry_url():
    _RegistryHandler.content = b"models:\n  first: {model_name: openai/first}\n"
    _RegistryHandler.request_paths = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RegistryHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/models.yaml?ref=main"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_url_registry_fetches_one_file_and_refreshes(
    tmp_path: Path, monkeypatch, registry_url: str
) -> None:
    monkeypatch.setenv("NEMO_OO_REGISTRY_CACHE", str(tmp_path / "cache"))

    materialized = resolve_registry_source(registry_url)
    assert "first" in materialized.read_text()
    assert _RegistryHandler.request_paths == ["/models.yaml?ref=main"]
    assert materialized.stat().st_mode & 0o077 == 0

    _RegistryHandler.content = b"models:\n  second: {model_name: openai/second}\n"

    refreshed = resolve_registry_source(registry_url)
    assert refreshed == materialized
    assert "second" in refreshed.read_text()
    assert "first" not in refreshed.read_text()


def test_url_registry_rejects_credentials() -> None:
    with pytest.raises(RegistrySourceError, match="must not contain credentials"):
        resolve_registry_source("https://user:secret@git.example.com/models.yaml")


def test_url_registry_rejects_non_registry_yaml(
    tmp_path: Path, monkeypatch, registry_url: str
) -> None:
    monkeypatch.setenv("NEMO_OO_REGISTRY_CACHE", str(tmp_path / "cache"))
    _RegistryHandler.content = b"<html>sign in</html>\n"

    with pytest.raises(RegistrySourceError, match="'models' mapping"):
        resolve_registry_source(registry_url)
