# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TUI model-registry command-line and bootstrap coverage."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner
from nooa_cli.commands.tui import command
from nooa_cli.tui.bootstrap import _load_llm_registry
from nooa_cli.tui.config import Config, get_llm

import nooa.unifiedllm as unifiedllm


def test_tui_help_documents_explicit_llm_config() -> None:
    result = CliRunner().invoke(command, ["--help"])

    assert result.exit_code == 0
    assert "--api-base TEXT" in result.output
    assert "--api-key-env ENVVAR" in result.output
    assert "Advanced: custom LLM API base URL" in result.output
    assert "--model." in result.output
    assert "--llm-config FILE" in result.output
    assert "--registry" not in result.output
    assert "highest precedence" in result.output


def test_tui_api_base_requires_model() -> None:
    result = CliRunner().invoke(command, ["--api-base", "http://localhost:8000/v1"])

    assert result.exit_code != 0
    assert "--api-base requires --model" in result.output
    assert "Use /connect for interactive setup" in result.output


def test_tui_api_key_env_requires_api_base() -> None:
    result = CliRunner().invoke(command, ["--model", "openai/custom", "--api-key-env", "KEY"])

    assert result.exit_code != 0
    assert "--api-key-env requires --api-base" in result.output


def test_config_accepts_direct_endpoint_overrides() -> None:
    config = Config.load(
        model="hosted_vllm/Qwen/Qwen3-1.7B",
        api_base="http://localhost:8000/v1",
        api_key_env="MY_API_KEY",
    )

    assert config.tui.default_model == "hosted_vllm/Qwen/Qwen3-1.7B"
    assert config.tui.api_base == "http://localhost:8000/v1"
    assert config.tui.api_key_env == "MY_API_KEY"


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


def test_working_dir_scopes_project_settings(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    project_config = workspace / ".nooa"
    project_config.mkdir(parents=True)
    (project_config / "settings.yaml").write_text(
        "tui:\n  default_model: workspace-model\n"
    )
    monkeypatch.delenv("NEMO_OO_PROJECT_DIR", raising=False)
    seen = {}

    async def fake_main(*, config, **_kwargs):
        seen["model"] = config.tui.default_model
        seen["project_dir"] = __import__("os").environ.get("NEMO_OO_PROJECT_DIR")

    with patch("nooa_cli.tui.main.main", fake_main):
        result = CliRunner().invoke(command, ["--working-dir", str(workspace)])

    assert result.exit_code == 0, result.output
    assert seen == {
        "model": "workspace-model",
        "project_dir": str(project_config),
    }
    assert "NEMO_OO_PROJECT_DIR" not in __import__("os").environ


def test_get_llm_uses_direct_endpoint_overrides(monkeypatch) -> None:
    calls = []

    def fake_get_llm_client(name: str, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace(_registry_config=None)

    monkeypatch.setattr(unifiedllm, "MODELS", {})
    monkeypatch.setattr(unifiedllm, "get_llm_client", fake_get_llm_client)
    monkeypatch.setenv("MY_API_KEY", "secret-value")

    config = Config()
    config.tui.default_model = "openai/custom/model"
    config.tui.api_base = "https://gateway.example.com/v1"
    config.tui.api_key_env = "MY_API_KEY"

    llm = get_llm(config)

    assert calls == [
        (
            "openai/custom/model",
            {"api_base": "https://gateway.example.com/v1", "api_key": "secret-value"},
        )
    ]
    assert llm._registry_config == {
        "api_base": "https://gateway.example.com/v1",
        "api_key_env": "MY_API_KEY",
    }


def test_get_llm_applies_endpoint_overrides_to_registry_aliases(monkeypatch) -> None:
    calls = []

    def fake_get_llm_client(name: str, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace(_registry_config={"api_base": "https://registry.example/v1"})

    monkeypatch.setattr(
        unifiedllm,
        "MODELS",
        {"alias": {"model_name": "openai/provider/model", "api_base": "https://old.example/v1"}},
    )
    monkeypatch.setattr(unifiedllm, "get_llm_client", fake_get_llm_client)
    monkeypatch.setenv("MY_API_KEY", "secret-value")

    config = Config()
    config.tui.default_model = "alias"
    config.tui.api_base = "https://override.example/v1"
    config.tui.api_key_env = "MY_API_KEY"

    llm = get_llm(config)

    assert calls == [
        ("alias", {"api_base": "https://override.example/v1", "api_key": "secret-value"})
    ]
    assert llm._registry_config == {
        "api_base": "https://override.example/v1",
        "api_key_env": "MY_API_KEY",
    }


def test_get_llm_keeps_native_anthropic_model_off_direct_endpoint(monkeypatch) -> None:
    calls = []

    def fake_get_llm_client(name: str, **kwargs):
        calls.append((name, kwargs))
        raise AssertionError("native Anthropic should not use get_llm_client overrides")

    def fake_completion_client(*, model: str):
        return SimpleNamespace(model=model, _registry_config=None)

    monkeypatch.setattr(unifiedllm, "MODELS", {})
    monkeypatch.setattr(unifiedllm, "get_llm_client", fake_get_llm_client)
    monkeypatch.setattr(unifiedllm, "CompletionClient", fake_completion_client)

    config = Config()
    config.tui.default_model = "claude-sonnet-4-5"
    config.tui.api_base = "https://gateway.example.com/v1"
    config.tui.api_key_env = "MY_API_KEY"

    llm = get_llm(config)

    assert calls == []
    assert llm.model == "claude-sonnet-4-5"
    assert llm._registry_config is None


def test_get_llm_does_not_apply_endpoint_overrides_to_anthropic_alias(monkeypatch) -> None:
    calls = []

    def fake_get_llm_client(name: str, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace(_registry_config={"model_name": "anthropic/claude-sonnet-4-5"})

    monkeypatch.setattr(
        unifiedllm,
        "MODELS",
        {"claude-alias": {"model_name": "anthropic/claude-sonnet-4-5"}},
    )
    monkeypatch.setattr(unifiedllm, "get_llm_client", fake_get_llm_client)

    config = Config()
    config.tui.default_model = "claude-alias"
    config.tui.api_base = "https://gateway.example.com/v1"
    config.tui.api_key_env = "MY_API_KEY"

    llm = get_llm(config)

    assert calls == [("claude-alias", {})]
    assert llm._registry_config == {"model_name": "anthropic/claude-sonnet-4-5"}
