# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP-only verbosity uses the shared layered Atom settings path."""

import pytest
import yaml
from acp import RequestError, text_block
from acp.schema import AgentMessageChunk, Implementation
from atom_test_agents import ScriptedModels
from nooa_atom.workspace.settings import (
    load_acp_verbosity,
    resolve_behavior_settings,
    settings_path,
)


def selected(response):
    return next(o.current_value for o in response.config_options if o.id == "verbosity")


@pytest.mark.parametrize("route", ["command", "config", "picker"])
async def test_changes_save_workspace_settings_and_restore_in_new_adapter(
    make_adapter, workspace, client, route
):
    path = settings_path(workspace=workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = {
        "atom": {"default_model": "saved-model"},
        "coding": {"theme": "other"},
        "extra": {"keep": True},
    }
    path.write_text(yaml.safe_dump(original))
    adapter = await make_adapter(
        ScriptedModels(), client_info=Implementation(name="pool", version="1.0.16")
    )
    session_id = (await adapter.new_session(str(workspace))).session_id
    if route == "command":
        await adapter.prompt(session_id, [text_block("/verbosity concise")])
    elif route == "config":
        await adapter.set_config_option("verbosity", session_id, "concise")
    else:
        client.ext_answers = [{"action": "accept", "content": {"verbosity": "concise"}}]
        await adapter.prompt(session_id, [text_block("/verbosity")])
    saved = yaml.safe_load(path.read_text())
    assert saved == {**original, "atom": {"default_model": "saved-model", "verbosity": "concise"}}
    assert load_acp_verbosity(workspace) == "concise"
    assert "verbosity" not in resolve_behavior_settings(saved)
    assert adapter.session(session_id).info.reply_count == 0
    second = await make_adapter(ScriptedModels())
    assert selected(await second.new_session(str(workspace))) == "concise"
    await adapter.close_session(session_id)
    assert selected(await second.load_session(str(workspace), session_id)) == "concise"


async def test_user_setting_is_inherited_then_workspace_overrides(
    make_adapter, workspace, tmp_path
):
    user = settings_path("user")
    user.parent.mkdir(parents=True, exist_ok=True)
    user.write_text("atom:\n  verbosity: silent\n  default_model: inherited-model\n")
    adapter = await make_adapter(ScriptedModels())
    response = await adapter.new_session(str(workspace))
    assert selected(response) == "silent"
    await adapter.set_config_option("verbosity", response.session_id, "concise")
    project = yaml.safe_load(settings_path(workspace=workspace).read_text())
    assert project == {"atom": {"verbosity": "concise"}}
    assert yaml.safe_load(user.read_text())["atom"]["verbosity"] == "silent"
    assert selected(await adapter.new_session(str(workspace))) == "concise"
    other = tmp_path / "other-workspace"
    other.mkdir()
    assert selected(await adapter.new_session(str(other))) == "silent"


@pytest.mark.parametrize(
    "value", ["invalid", True, ["concise"], {"level": "concise"}, 1, "CONCISE", None]
)
async def test_invalid_saved_verbosity_defaults_without_breaking_agent_settings(
    make_adapter, workspace, caplog, value
):
    path = settings_path(workspace=workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"atom": {"verbosity": value, "default_model": "kept-model"}}))
    adapter = await make_adapter(ScriptedModels())
    assert selected(await adapter.new_session(str(workspace))) == "verbose"
    if value is not None:  # YAML null removes the key in layered settings.
        assert "Invalid atom.verbosity" in caplog.text
    assert resolve_behavior_settings(yaml.safe_load(path.read_text())) == {
        "default_model": "kept-model"
    }


async def test_environment_override_remains_highest_priority(
    make_adapter, workspace, tmp_path, monkeypatch
):
    override = tmp_path / "override.yaml"
    override.write_text("atom:\n  verbosity: silent\n")
    monkeypatch.setenv("NEMO_OO_SETTINGS", str(override))
    adapter = await make_adapter(ScriptedModels())
    response = await adapter.new_session(str(workspace))
    assert selected(response) == "silent"
    await adapter.set_config_option("verbosity", response.session_id, "concise")
    assert adapter.bridge(response.session_id).verbosity == "concise"
    assert load_acp_verbosity(workspace) == "silent"
    assert selected(await adapter.new_session(str(workspace))) == "silent"


async def test_save_failure_does_not_change_bridge_or_claim_success(
    make_adapter, workspace, client, monkeypatch
):
    from nooa_atom.acp import server

    def fail(*args, **kwargs):
        raise OSError("read-only settings")

    adapter = await make_adapter(ScriptedModels())
    session_id = (await adapter.new_session(str(workspace))).session_id
    monkeypatch.setattr(server, "write_settings_updates", fail)
    with pytest.raises(RequestError):
        await adapter.set_config_option("verbosity", session_id, "concise")
    assert adapter.bridge(session_id).verbosity == "verbose"
    response = await adapter.prompt(session_id, [text_block("/verbosity silent")])
    assert response.stop_reason == "end_turn"
    assert "could not save" in client.texts(AgentMessageChunk, session_id)[-1]
    assert adapter.bridge(session_id).verbosity == "verbose"


@pytest.mark.parametrize("command", ["/verbosity invalid", "/verbosity silent extra", "/verbosity"])
async def test_invalid_or_declined_commands_do_not_save(make_adapter, workspace, client, command):
    adapter = await make_adapter(
        ScriptedModels(), client_info=Implementation(name="pool", version="1.0.16")
    )
    session_id = (await adapter.new_session(str(workspace))).session_id
    client.ext_answers = [{"action": "decline"}]
    await adapter.prompt(session_id, [text_block(command)])
    assert not settings_path(workspace=workspace).exists()
    assert adapter.bridge(session_id).verbosity == "verbose"


async def test_saving_current_level_makes_inherited_value_explicit(make_adapter, workspace):
    user = settings_path("user")
    user.parent.mkdir(parents=True, exist_ok=True)
    user.write_text("atom:\n  verbosity: concise\n")
    adapter = await make_adapter(ScriptedModels())
    response = await adapter.new_session(str(workspace))
    assert selected(response) == "concise"
    await adapter.set_config_option("verbosity", response.session_id, "concise")
    assert yaml.safe_load(settings_path(workspace=workspace).read_text()) == {
        "atom": {"verbosity": "concise"}
    }


async def test_concurrent_local_changes_keep_save_and_apply_ordered(
    make_adapter, workspace, monkeypatch
):
    import asyncio
    import threading

    from nooa_atom.acp import server

    original = server.write_settings_updates
    loop_thread = threading.get_ident()
    observations = []

    def tracked_write(updates, **kwargs):
        # Like existing settings controls, writes execute without yielding to
        # another ACP request before the bridge value is applied.
        assert threading.get_ident() == loop_thread
        observations.append(updates[("atom", "verbosity")])
        return original(updates, **kwargs)

    adapter = await make_adapter(ScriptedModels())
    session_id = (await adapter.new_session(str(workspace))).session_id
    monkeypatch.setattr(server, "write_settings_updates", tracked_write)
    await asyncio.gather(
        adapter.set_config_option("verbosity", session_id, "concise"),
        adapter.set_config_option("verbosity", session_id, "silent"),
    )
    assert observations == ["concise", "silent"]
    assert adapter.bridge(session_id).verbosity == load_acp_verbosity(workspace) == "silent"
