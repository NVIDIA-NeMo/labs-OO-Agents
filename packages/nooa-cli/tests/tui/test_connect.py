# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native dialogs use the same session controller as ACP."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from nooa_cli.tui.commands import ConnectCommand
from nooa_cli.tui.config import TUIConfig

from nooa.unifiedllm import connect


@pytest.fixture
def command(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / ".nooa"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    frontend = SimpleNamespace(
        prompt_choice=AsyncMock(side_effect=["openai", "org/model"]),
        prompt_text=AsyncMock(return_value="work"),
        prompt_sensitive=AsyncMock(return_value="private-key"),
    )
    command = ConnectCommand(frontend, TUIConfig(), SimpleNamespace(cwd=tmp_path))
    command._reload_model_registry = Mock()
    discover = AsyncMock(
        return_value=connect.Discovery("https://api.openai.com/v1", ({"id": "org/model"},))
    )
    monkeypatch.setattr(connect, "discover", discover)
    return command


async def test_picker_previews_then_explicit_save_persists_key(command, tmp_path):
    result = await command.execute([])
    assert result.success
    assert command._control().proposal.entry["transport"] == "direct"
    assert command._control().proposal.entry["api_style"] == "responses"
    assert not (tmp_path / ".nooa/llm_config.yaml").exists()
    assert not (tmp_path / ".nooa/secrets.yaml").exists()
    assert "OPENAI_API_KEY" not in os.environ
    assert "private-key" not in str(result)
    assert (await command.execute([])).success  # Review, no second picker.
    assert command.frontend.prompt_choice.await_count == 2
    assert (await command.execute(["save"])).success
    entry = yaml.safe_load((tmp_path / ".nooa/llm_config.yaml").read_text())["models"]["work"]
    assert entry["model_name"] == "openai/org/model"
    assert "private-key" not in str(entry)
    assert (
        yaml.safe_load((tmp_path / ".nooa/secrets.yaml").read_text())["env"]["OPENAI_API_KEY"]
        == "private-key"
    )
    assert (tmp_path / ".nooa/secrets.yaml").stat().st_mode & 0o777 == 0o600
    assert os.environ["OPENAI_API_KEY"] == "private-key"
    command._reload_model_registry.assert_called_once()
    assert command._control().proposal is None


async def test_cancel_discards_masked_key_without_writing(command, tmp_path):
    command.frontend.prompt_choice.side_effect = ["openai", None]
    assert (await command.execute([])).success
    assert command._control()._api_key is None
    assert not (tmp_path / ".nooa").exists()


async def test_existing_environment_key_is_not_copied_to_workspace(command, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "existing-key")
    assert (await command.execute([])).success
    command.frontend.prompt_sensitive.assert_not_awaited()
    assert (await command.execute(["save"])).success
    assert not (tmp_path / ".nooa/secrets.yaml").exists()


async def test_staged_commands_reuse_draft_and_do_not_open_dialogs(command, tmp_path):
    assert command.validate_args(["model", "org/model", "--as", "work"]) == (True, None)
    assert (await command.execute(["http://localhost:8000/v1"])).success
    assert (await command.execute(["model", "org/model", "--as", "work"])).success
    assert (await command.execute(["save"])).success
    command.frontend.prompt_choice.assert_not_awaited()
    assert (tmp_path / ".nooa/llm_config.yaml").exists()


async def test_overwrite_requires_explicit_replace_and_keeps_pending_key(command, tmp_path):
    assert (await command.execute([])).success
    path = tmp_path / ".nooa/llm_config.yaml"
    path.parent.mkdir()
    path.write_text("models:\n  work:\n    model_name: openai/old\n")
    assert not (await command.execute(["save"])).success
    assert command._control()._api_key == "private-key"
    assert not path.with_name("secrets.yaml").exists()
    assert (await command.execute(["save", "--replace"])).success
    assert yaml.safe_load(path.read_text())["models"]["work"]["model_name"] == "openai/org/model"


async def test_failed_secret_write_reports_partial_save(command, tmp_path, monkeypatch):
    assert (await command.execute([])).success
    monkeypatch.setattr("nooa.secrets.write_secret_env", Mock(side_effect=OSError("private-key")))
    result = await command.execute(["save"])
    assert not result.success
    assert "settings were saved" in str(result)
    assert "private-key" not in str(result)
    assert (tmp_path / ".nooa/llm_config.yaml").exists()


@pytest.mark.parametrize(
    "provider,endpoint,style,key_env",
    [
        ("anthropic", "https://api.anthropic.com/v1", "anthropic", "ANTHROPIC_API_KEY"),
        ("Ollama local", "http://localhost:11434/v1", "chat", ""),
    ],
)
async def test_picker_routes_provider_through_library(
    command, monkeypatch, provider, endpoint, style, key_env
):
    if key_env:
        monkeypatch.setenv(key_env, "existing-key")
    command.frontend.prompt_choice.side_effect = [provider, "org/model"]
    assert (await command.execute([])).success
    connect.discover.assert_awaited_once_with(
        endpoint, api_style=style, api_key="existing-key" if key_env else None
    )
    command.frontend.prompt_sensitive.assert_not_awaited()


async def test_failed_discovery_allows_manual_model(command, monkeypatch):
    monkeypatch.setattr(
        connect, "discover", AsyncMock(side_effect=connect.DiscoveryError("private-key"))
    )
    result = await command.execute([])
    assert not result.success and "private-key" not in str(result)
    assert (await command.execute(["model", "manual-model", "--as", "work"])).success
    assert command._control().proposal.entry["model_name"] == "openai/manual-model"


async def test_cancelled_discovery_does_not_write(command, monkeypatch, tmp_path):
    import asyncio

    monkeypatch.setattr(connect, "discover", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await command.execute([])
    assert not (tmp_path / ".nooa").exists()
    assert (await command.execute(["cancel"])).success
    assert command._control()._api_key is None


async def test_session_swap_discards_the_previous_sessions_draft(command):
    assert (await command.execute([])).success
    old = command._control()
    command.session_manager = SimpleNamespace(session_id="new-session")
    assert command._control() is not old
    assert old._api_key is None
    assert command._control().proposal is None
    assert command._control()._api_key is None
