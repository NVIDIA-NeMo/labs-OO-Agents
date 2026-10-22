# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP terminal auth: ``nooa atom --connect`` runs the ``nooa connect`` wizard.

The wizard itself is mocked; nothing here touches the network.
"""

import json
import time

import click
import click.testing
import pytest
import yaml
from acp import PROTOCOL_VERSION, RequestError, text_block
from acp.schema import (
    AgentMessageChunk,
    AuthCapabilities,
    ClientCapabilities,
    ConfigOptionUpdate,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
)
from atom_test_agents import ScriptedModels
from nooa_atom.acp import terminal_auth
from nooa_atom.acp.protocol import initialize_response

TERMINAL = ClientCapabilities(auth=AuthCapabilities(terminal=True))
FORMS = ClientCapabilities(elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities()))
FORMS_AND_TERMINAL = ClientCapabilities(
    auth=AuthCapabilities(terminal=True),
    elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities()),
)
METHOD = {
    "id": "nooa-connect",
    "name": "Connect a model",
    "description": "Set up a model provider in a terminal",
    "type": "terminal",
    "args": ["--connect"],
}


@pytest.fixture(autouse=True)
def user_dir(tmp_path, monkeypatch):
    user = tmp_path / "user"
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.delenv("NOOA_MODEL", raising=False)
    monkeypatch.delenv("NEMO_OO_LLM_CONFIG", raising=False)
    # The saved models name this key variable; it stays unset, and a model
    # whose key is missing still counts as configured.
    monkeypatch.delenv("K", raising=False)
    return user


def _save_user_model(user_dir, alias="local", model="openai/m", api_base="https://api.test/v1"):
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "llm_config.yaml").write_text(
        yaml.safe_dump(
            {"models": {alias: {"model_name": model, "api_base": api_base, "api_key_env": "K"}}}
        )
    )


def _auth_methods(capabilities):
    response = initialize_response(PROTOCOL_VERSION, client_capabilities=capabilities)
    return response.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True)[
        "authMethods"
    ]


# ---- advertising ---------------------------------------------------------------


def test_the_method_is_advertised_only_to_clients_that_run_terminal_auth():
    assert _auth_methods(TERMINAL) == [METHOD]
    assert _auth_methods(None) == []
    assert _auth_methods(ClientCapabilities()) == []
    assert _auth_methods(ClientCapabilities(auth=AuthCapabilities(terminal=False))) == []
    assert _auth_methods(FORMS) == []
    # Without the capability the answer is exactly the one before terminal auth.
    assert initialize_response(PROTOCOL_VERSION, client_capabilities=FORMS) == initialize_response(
        PROTOCOL_VERSION
    )


def test_the_wire_field_is_client_capabilities_auth_terminal():
    from acp.schema import InitializeRequest

    request = InitializeRequest.model_validate(
        {"protocolVersion": 1, "clientCapabilities": {"auth": {"terminal": True}}}
    )
    assert _auth_methods(request.client_capabilities) == [METHOD]


async def test_the_adapter_answers_initialize_with_the_method(make_adapter, client):
    adapter = await make_adapter(ScriptedModels({None: []}))
    plain = await adapter.initialize(PROTOCOL_VERSION, client_capabilities=FORMS)
    assert plain.auth_methods == []
    offered = await adapter.initialize(PROTOCOL_VERSION, client_capabilities=TERMINAL)
    assert [m.model_dump(mode="json", exclude_none=True) for m in offered.auth_methods] == [METHOD]


# ---- nooa atom --connect --------------------------------------------------------


@pytest.fixture
def no_server(monkeypatch):
    """--connect must not take over stdio or serve ACP."""

    def fail(*args, **kwargs):
        raise AssertionError("--connect started the ACP server")

    monkeypatch.setattr("nooa_atom.acp.cli.run", fail)
    monkeypatch.setattr("nooa_atom.acp.cli.reserve_stdio_for_acp", fail)


@pytest.fixture
def wizard(monkeypatch, user_dir):
    """Replace the CLI wizard: ``wizard.outcome`` is the alias to save, None, or an exception."""

    class Wizard:
        outcome: object = "local"
        calls: list[dict] = []

    def run_wizard(**options):
        Wizard.calls.append(options)
        if isinstance(Wizard.outcome, BaseException):
            raise Wizard.outcome
        if Wizard.outcome:
            _save_user_model(user_dir, Wizard.outcome, api_base="https://user:pw@api.test/v1?k=s")
        return Wizard.outcome

    Wizard.calls = []
    monkeypatch.setattr("nooa_cli.commands._connect_wizard.run_wizard", run_wizard)
    return Wizard


def _connect(*args):
    from nooa_atom.acp.cli import command

    return click.testing.CliRunner().invoke(command, ["--connect", *args])


def test_connect_exits_0_after_a_save_and_leaves_a_result_without_secrets(
    wizard, no_server, user_dir
):
    result = _connect()
    assert result.exit_code == 0, result.output
    saved = json.loads((user_dir / terminal_auth.RESULT_FILE).read_text())
    assert saved["alias"] == "local" and saved["model"] == "openai/m"
    # No user info, query or fragment from the URL; never a key.
    assert saved["server"] == "https://api.test/v1"
    assert set(saved) == {"alias", "model", "server", "saved_at"}
    assert (user_dir / terminal_auth.RESULT_FILE).stat().st_mode & 0o077 == 0


@pytest.mark.parametrize(
    "outcome", [None, click.ClickException("checks failed"), click.Abort(), KeyboardInterrupt()]
)
def test_connect_exits_non_zero_when_nothing_was_saved(wizard, no_server, user_dir, outcome):
    # A result from an earlier run must not survive a cancelled or failed one.
    terminal_auth.write_result("old", "openai/old", "https://old.test")
    wizard.outcome = outcome
    result = _connect()
    assert result.exit_code != 0
    assert not (user_dir / terminal_auth.RESULT_FILE).exists()


def test_connect_runs_the_wizard_with_the_defaults_of_nooa_connect(wizard, no_server):
    from nooa_cli.commands.connect import command as nooa_connect

    assert _connect().exit_code == 0
    assert click.testing.CliRunner().invoke(nooa_connect, []).exit_code == 0
    via_acp, via_cli = wizard.calls
    assert via_acp == via_cli


# ---- first run: no model ----------------------------------------------------------


async def test_session_new_without_a_model_asks_terminal_clients_to_connect(
    make_adapter, workspace, user_dir
):
    adapter = await make_adapter(ScriptedModels({None: []}), capabilities=TERMINAL)
    with pytest.raises(RequestError) as failed:
        await adapter.new_session(str(workspace))
    assert failed.value.code == -32000
    # The wizard saved a model; the retried session/new (any process) succeeds.
    _save_user_model(user_dir)
    retry = await make_adapter(ScriptedModels({None: []}), capabilities=TERMINAL)
    assert (await retry.new_session(str(workspace))).session_id


async def test_session_new_without_a_model_is_unchanged_for_other_clients(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels({None: []}), capabilities=FORMS)
    assert (await adapter.new_session(str(workspace))).session_id


async def test_session_new_with_a_server_model_is_not_auth(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels({None: []}), capabilities=TERMINAL, model="m")
    assert (await adapter.new_session(str(workspace))).session_id


async def test_a_prompt_without_a_model_asks_terminal_clients_to_connect(
    make_adapter, workspace, user_dir
):
    _save_user_model(user_dir)
    adapter = await make_adapter(ScriptedModels({None: []}), capabilities=TERMINAL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    (user_dir / "llm_config.yaml").unlink()
    with pytest.raises(RequestError) as failed:
        await adapter.prompt(session_id, [text_block("hello")])
    assert failed.value.code == -32000
    assert adapter.session(session_id).info.reply_count == 0


# ---- /connect -----------------------------------------------------------------------


async def _connect_prompt(adapter, session_id):
    return await adapter.prompt(session_id, [text_block("/connect")])


async def test_connect_fails_with_auth_required_then_reports_the_save(
    atom_adapter, workspace, client
):
    adapter = await atom_adapter([], capabilities=TERMINAL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    with pytest.raises(RequestError) as failed:
        await _connect_prompt(adapter, session_id)
    assert failed.value.code == -32000
    # The client ran `nooa atom --connect` (it saved) and reconnected: a new
    # agent process, with no memory of the first /connect.
    await adapter.close()
    restarted = await atom_adapter([], capabilities=TERMINAL)
    session_id = (await restarted.new_session(str(workspace))).session_id
    terminal_auth.write_result("local", "openai/m", "https://key:secret@api.test/v1?x=1")
    before = len(client.updates(session_id, ConfigOptionUpdate))
    response = await _connect_prompt(restarted, session_id)
    assert response.stop_reason == "end_turn"
    text = "".join(client.texts(AgentMessageChunk, session_id))
    assert "Saved model alias local: openai/m at https://api.test/v1" in text
    assert "secret" not in text
    assert len(client.updates(session_id, ConfigOptionUpdate)) == before + 1
    assert restarted.session(session_id).info.reply_count == 0
    # One save answers one /connect: the next one asks again.
    with pytest.raises(RequestError) as again:
        await _connect_prompt(restarted, session_id)
    assert again.value.code == -32000


@pytest.mark.parametrize(
    "content",
    [
        None,  # no result: the wizard was cancelled or never ran
        "stale",
        "future",
        "not json",
        json.dumps({"alias": "", "saved_at": 0}),
        json.dumps(["local"]),
    ],
)
async def test_connect_without_a_fresh_result_asks_again(
    atom_adapter, workspace, user_dir, content
):
    adapter = await atom_adapter([], capabilities=TERMINAL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    path = user_dir / terminal_auth.RESULT_FILE
    if content == "stale":
        terminal_auth.write_result(
            "local", "m", "", now=time.time() - terminal_auth.RESULT_MAX_AGE - 1
        )
    elif content == "future":
        terminal_auth.write_result("local", "m", "", now=time.time() + 3600)
    elif content is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    with pytest.raises(RequestError) as failed:
        await _connect_prompt(adapter, session_id)
    assert failed.value.code == -32000
    assert not path.exists()


async def test_session_new_drops_a_left_over_result(atom_adapter, workspace, user_dir):
    # First start: the wizard saved, session/new succeeded. A later /connect
    # must run the wizard, not report that old save.
    terminal_auth.write_result("local", "m", "")
    adapter = await atom_adapter([], capabilities=TERMINAL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    assert not (user_dir / terminal_auth.RESULT_FILE).exists()
    with pytest.raises(RequestError):
        await _connect_prompt(adapter, session_id)


async def test_terminal_auth_wins_over_forms_for_connect(atom_adapter, workspace, client):
    adapter = await atom_adapter([], capabilities=FORMS_AND_TERMINAL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    with pytest.raises(RequestError):
        await _connect_prompt(adapter, session_id)
    assert not [entry for entry in client.log if entry[0] == "elicitation"]


@pytest.mark.parametrize("capabilities", [None, FORMS])
async def test_clients_without_terminal_auth_keep_todays_connect(
    atom_adapter, workspace, client, user_dir, capabilities
):
    from nooa_atom.acp.server import _CONNECT_TEXT

    # Even a fresh result does not change their /connect.
    terminal_auth.write_result("local", "m", "")
    adapter = await atom_adapter([], capabilities=capabilities)
    session_id = (await adapter.new_session(str(workspace))).session_id
    response = await _connect_prompt(adapter, session_id)
    assert response.stop_reason == "end_turn"
    assert _CONNECT_TEXT in "".join(client.texts(AgentMessageChunk, session_id))
    assert not [entry for entry in client.log if entry[0] == "elicitation"]
    assert (user_dir / terminal_auth.RESULT_FILE).exists()
