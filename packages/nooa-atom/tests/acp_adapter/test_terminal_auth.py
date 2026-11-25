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
from atom_test_agents import ModelFactory, ScriptedModels, reply
from nooa_atom.acp import terminal_auth
from nooa_atom.acp.protocol import initialize_response
from nooa_atom.acp.terminal_auth import ANY, session_key, workspace_key

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


def _saved(user_dir, alias):
    from nooa_cli.commands._connect_wizard import SavedModel

    return SavedModel(alias, user_dir / "llm_config.yaml")


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
            return _saved(user_dir, Wizard.outcome)
        return None

    Wizard.calls = []
    monkeypatch.setattr("nooa_cli.commands._connect_wizard.run_wizard", run_wizard)
    return Wizard


def _connect(*args):
    from nooa_atom.acp.cli import command

    return click.testing.CliRunner().invoke(command, ["--connect", *args])


def _result_file(user_dir, key):
    return user_dir / terminal_auth.HANDOFF_DIR / f"result-{key}.json"


def _results(user_dir):
    return sorted(path.name for path in (user_dir / terminal_auth.HANDOFF_DIR).glob("result-*"))


def test_connect_exits_0_after_a_save_and_leaves_a_result_without_secrets(
    wizard, no_server, user_dir
):
    result = _connect()
    assert result.exit_code == 0, result.output
    path = _result_file(user_dir, ANY)  # run by hand: no session asked
    saved = json.loads(path.read_text())
    assert saved["alias"] == "local" and saved["model"] == "openai/m"
    # Scheme, host and port only: no user info, path, query or fragment; never a key.
    assert saved["server"] == "https://api.test"
    assert set(saved) == {"alias", "model", "server", "fingerprint", "saved_at", "applied"}
    assert path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize(
    "outcome", [None, click.ClickException("checks failed"), click.Abort(), KeyboardInterrupt()]
)
def test_connect_exits_non_zero_when_nothing_was_saved(wizard, no_server, user_dir, outcome):
    # An earlier result for the same request must not survive a cancelled or
    # failed run; another session's unclaimed save must.
    terminal_auth.write_result(ANY, "old", "openai/old", "https://old.test")
    terminal_auth.write_result(session_key("other"), "theirs", "openai/t", "")
    wizard.outcome = outcome
    result = _connect()
    assert result.exit_code != 0
    assert _results(user_dir) == [f"result-{session_key('other')}.json"]


def test_connect_runs_the_wizard_with_the_defaults_of_nooa_connect(wizard, no_server):
    from nooa_cli.commands.connect import command as nooa_connect

    assert _connect().exit_code == 0
    assert click.testing.CliRunner().invoke(nooa_connect, []).exit_code == 0
    via_acp, via_cli = wizard.calls
    assert via_acp == via_cli


def test_connect_binds_the_save_to_the_newest_request(wizard, no_server, user_dir):
    terminal_auth.record_request(session_id="older", workspace="/w", now=time.time() - 60)
    terminal_auth.record_request(session_id="newer", workspace="/w")
    terminal_auth.record_request(session_id="stale", workspace="/w", now=time.time() - 3600)
    assert _connect().exit_code == 0
    assert _results(user_dir) == [f"result-{session_key('newer')}.json"]
    # The claimed and the stale requests are gone; the other one waits for its own run.
    requests = sorted((user_dir / terminal_auth.HANDOFF_DIR).glob("request-*"))
    assert [path.name for path in requests] == [f"request-{session_key('older')}.json"]


def test_claim_request_skips_one_another_process_already_claimed(no_server, user_dir, monkeypatch):
    # Two terminals list the same fresh request before either reserves it; only
    # one may win. Simulate the loser by moving the file away first, then let
    # the real ``os.replace`` raise ``FileNotFoundError`` as it would for them.
    import os as os_module

    terminal_auth.record_request(session_id="only", workspace="/w")
    path = next((user_dir / terminal_auth.HANDOFF_DIR).glob("request-*.json"))
    real_replace = os_module.replace
    raced = path.with_name(".raced-away")

    def racing_replace(src, dst):
        real_replace(src, raced)
        raise FileNotFoundError

    monkeypatch.setattr(terminal_auth.os, "replace", racing_replace)
    assert terminal_auth.claim_request() == ANY
    # The file is gone from the handoff dir (the "other process" has it); no crash.
    assert not list((user_dir / terminal_auth.HANDOFF_DIR).glob("request-*.json"))


def test_claim_request_reserves_the_key_so_a_second_claim_gets_none(no_server, user_dir):
    terminal_auth.record_request(session_id="only", workspace="/w")
    assert terminal_auth.claim_request() == session_key("only")
    assert terminal_auth.claim_request() == ANY


def test_connect_exits_0_when_only_the_note_cannot_be_written(
    wizard, no_server, user_dir, monkeypatch
):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(terminal_auth, "write_result", fail)
    result = _connect()
    assert result.exit_code == 0
    assert "Saved model alias local, but could not tell the client (disk full)" in result.output


@pytest.mark.parametrize(
    ("url", "label"),
    [
        ("https://user:pw@api.test:8443/v1/secret-token?k=s#f", "https://api.test:8443"),
        ("http://[::1]:8000/v1", "http://[::1]:8000"),
        ("not a url", ""),
        (None, ""),
    ],
)
def test_the_server_label_keeps_scheme_host_and_port_only(url, label):
    assert terminal_auth._server_label(url) == label


def test_a_decision_entry_reports_its_endpoint(tmp_path):
    path = tmp_path / "llm_config.yaml"
    entry = {"model_name": "m", "client_type": "decision", "endpoint": "https://d.test/x/decide"}
    path.write_text(yaml.safe_dump({"models": {"d": entry}}))
    saved = terminal_auth._saved_entry("d", path)
    assert saved.server == "https://d.test/x/decide"
    assert saved.fingerprint == terminal_auth.entry_fingerprint(entry)


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
    assert (
        user_dir / terminal_auth.HANDOFF_DIR / f"request-{session_key(session_id)}.json"
    ).exists()


# ---- /connect -----------------------------------------------------------------------


async def _connect_prompt(adapter, session_id):
    return await adapter.prompt(session_id, [text_block("/connect")])


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
    key = session_key(session_id)
    path = _result_file(user_dir, key)
    if content == "stale":
        terminal_auth.write_result(
            key, "local", "m", "", now=time.time() - terminal_auth.RESULT_MAX_AGE - 1
        )
    elif content == "future":
        terminal_auth.write_result(key, "local", "m", "", now=time.time() + 3600)
    elif content is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    with pytest.raises(RequestError) as failed:
        await _connect_prompt(adapter, session_id)
    assert failed.value.code == -32000
    assert not path.exists()


async def test_session_new_takes_a_save_made_without_a_request(atom_adapter, workspace, user_dir):
    # `nooa atom --connect` run by hand, then a new session: it uses that
    # save once; a later /connect runs the wizard rather than report it.
    terminal_auth.write_result(ANY, "local", "m", "")
    adapter = await atom_adapter([], capabilities=TERMINAL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    assert _results(user_dir) == []
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
    terminal_auth.write_result(ANY, "local", "m", "")
    adapter = await atom_adapter([], capabilities=capabilities)
    session_id = (await adapter.new_session(str(workspace))).session_id
    response = await _connect_prompt(adapter, session_id)
    assert response.stop_reason == "end_turn"
    assert _CONNECT_TEXT in "".join(client.texts(AgentMessageChunk, session_id))
    assert not [entry for entry in client.log if entry[0] == "elicitation"]
    assert _result_file(user_dir, ANY).exists()


async def test_the_connect_command_description_matches_the_client(make_adapter, workspace, client):
    from acp.schema import AvailableCommandsUpdate

    descriptions = {}
    for name, capabilities in (("terminal", TERMINAL), ("plain", FORMS)):
        adapter = await make_adapter(ScriptedModels(), capabilities=capabilities, model="m")
        sid = (await adapter.new_session(str(workspace))).session_id
        await client.wait_for(lambda sid=sid: client.updates(sid, AvailableCommandsUpdate))
        [commands] = client.updates(sid, AvailableCommandsUpdate)
        [command] = [c for c in commands.available_commands if c.name == "connect"]
        descriptions[name] = command.description
    assert "in a terminal" in descriptions["terminal"]
    assert "shows how to run nooa connect" in descriptions["plain"]


# ---- after a save: the session uses the saved alias ---------------------------------


@pytest.fixture
def saving_wizard(monkeypatch, user_dir):
    """The wizard's own save (``connect.write`` to the user registry), without its screens."""
    from nooa import paths
    from nooa.unifiedllm import connect

    class Wizard:
        alias = "second"
        model = None

    def run_wizard(**options):
        path = paths.get_user_dir("llm_config.yaml")
        connect.write(
            {
                "model_name": Wizard.model or f"openai/{Wizard.alias}",
                "api_base": "https://api.test/v1",
                "api_key_env": "K",
            },
            path,
            alias=Wizard.alias,
        )
        return _saved(user_dir, Wizard.alias)

    monkeypatch.setattr("nooa_cli.commands._connect_wizard.run_wizard", run_wizard)
    return Wizard


class WorkspaceDefault(ModelFactory):
    """Like the server's factory: no alias means the workspace's default model."""

    def __call__(self, alias, workspace):
        from nooa_atom.workspace.models import default_model

        return super().__call__(alias or default_model(workspace), workspace)


def _model_option(config_options):
    (option,) = [option for option in config_options or [] if option.id == "model"]
    return option.current_value, [choice.value for choice in option.options]


async def _adapter(make_adapter, factory):
    return await make_adapter(
        ScriptedModels({None: []}), capabilities=TERMINAL, llm_factory=factory
    )


async def _connected_session(make_adapter, workspace, user_dir, factory):
    """A session on ``first``; its /connect asks for auth and the wizard saves ``second``."""
    _save_user_model(user_dir, "first")
    adapter = await _adapter(make_adapter, factory)
    new = await adapter.new_session(str(workspace))
    assert _model_option(new.config_options) == ("first", ["first"])
    with pytest.raises(RequestError) as failed:
        await _connect_prompt(adapter, new.session_id)
    assert failed.value.code == -32000
    assert _connect().exit_code == 0  # the client runs `nooa atom --connect`
    return adapter, new.session_id


async def test_a_retried_connect_switches_the_session_to_the_saved_alias(
    make_adapter, workspace, client, user_dir, saving_wizard, no_server
):
    factory = WorkspaceDefault()
    adapter, session_id = await _connected_session(make_adapter, workspace, user_dir, factory)
    response = await _connect_prompt(adapter, session_id)
    assert response.stop_reason == "end_turn"
    assert adapter.session(session_id).info.model == "second"
    (update,) = client.updates(session_id, ConfigOptionUpdate)
    assert _model_option(update.config_options) == ("second", ["first", "second"])
    text = "".join(client.texts(AgentMessageChunk, session_id))
    assert "Saved model alias second: openai/second" in text
    assert "This session uses it from now on." in text


async def test_the_pool_flow_reloads_the_session_then_reports_the_save(
    make_adapter, workspace, client, user_dir, saving_wizard, no_server
):
    # Pool runs the wizard, starts a new agent process, closes the old one,
    # loads the same session by id, and the user types /connect again.
    factory = WorkspaceDefault()
    adapter, session_id = await _connected_session(make_adapter, workspace, user_dir, factory)
    restarted = await _adapter(make_adapter, factory)
    await adapter.close()
    loaded = await restarted.load_session(str(workspace), session_id)
    assert _model_option(loaded.config_options) == ("second", ["first", "second"])
    assert restarted.session(session_id).info.model == "second"
    response = await _connect_prompt(restarted, session_id)
    assert response.stop_reason == "end_turn"
    text = "".join(client.texts(AgentMessageChunk, session_id))
    assert "Saved model alias second: openai/second" in text
    assert "This session uses it." in text
    # One save answers one /connect: the next one asks again.
    with pytest.raises(RequestError) as again:
        await _connect_prompt(restarted, session_id)
    assert again.value.code == -32000


@pytest.mark.parametrize("then", ["prompt", "authenticate", "load"])
async def test_the_saved_alias_is_used_without_a_second_connect(
    make_adapter, workspace, client, user_dir, saving_wizard, no_server, then
):
    factory = WorkspaceDefault({"second": [[reply("From the saved model.")]]})
    adapter, session_id = await _connected_session(make_adapter, workspace, user_dir, factory)
    if then == "prompt":
        response = await adapter.prompt(session_id, [text_block("hello")])
        assert response.stop_reason == "end_turn"
        texts = client.texts(AgentMessageChunk, session_id)
        assert texts[-1] == "From the saved model.\n\n"
        assert any("This session uses it from now on." in text for text in texts)
    elif then == "authenticate":
        await adapter.authenticate(terminal_auth.CONNECT_METHOD_ID)
    else:
        # The client reconnected and loads the session again.
        await adapter.close_session(session_id)
        loaded = await adapter.load_session(str(workspace), session_id)
        assert _model_option(loaded.config_options) == ("second", ["first", "second"])
    assert adapter.session(session_id).info.model == "second"
    if then in {"prompt", "authenticate"}:
        update = client.updates(session_id, ConfigOptionUpdate)[-1]
        assert _model_option(update.config_options) == ("second", ["first", "second"])
        # The save was taken: a later /connect runs the wizard again.
        with pytest.raises(RequestError):
            await _connect_prompt(adapter, session_id)


async def test_a_second_cold_restore_does_not_reapply_an_already_applied_alias(
    make_adapter, workspace, client, user_dir, saving_wizard, no_server
):
    factory = WorkspaceDefault({"second": [[reply("From the saved model.")]]})
    adapter, session_id = await _connected_session(make_adapter, workspace, user_dir, factory)
    # First cold restore: applies "second" and marks the result applied.
    await adapter.close_session(session_id)
    loaded = await adapter.load_session(str(workspace), session_id)
    assert _model_option(loaded.config_options) == ("second", ["first", "second"])
    assert adapter.session(session_id).info.model == "second"
    # The user then picks a different model from the menu.
    await adapter.session(session_id).set_model("first")
    assert adapter.session(session_id).info.model == "first"
    # A second cold restore (new process, same 15-minute window) must not
    # re-apply the already-applied save and override that later choice.
    await adapter.close_session(session_id)
    reloaded = await adapter.load_session(str(workspace), session_id)
    assert _model_option(reloaded.config_options) == ("first", ["first", "second"])
    assert adapter.session(session_id).info.model == "first"


async def test_another_window_does_not_take_the_save(
    make_adapter, workspace, user_dir, saving_wizard, no_server
):
    factory = WorkspaceDefault()
    adapter, session_id = await _connected_session(make_adapter, workspace, user_dir, factory)
    other = await make_adapter(
        ScriptedModels({None: [reply("Hi.")]}), capabilities=TERMINAL, llm_factory=factory
    )
    other_id = (await other.new_session(str(workspace))).session_id
    assert (await other.prompt(other_id, [text_block("hello")])).stop_reason == "end_turn"
    assert other.session(other_id).info.model == "first"
    # The window that asked still gets its save.
    response = await _connect_prompt(adapter, session_id)
    assert response.stop_reason == "end_turn"
    assert adapter.session(session_id).info.model == "second"


async def test_a_new_session_after_a_failed_session_new_uses_the_saved_alias(
    make_adapter, workspace, user_dir, saving_wizard, no_server
):
    adapter = await _adapter(make_adapter, WorkspaceDefault())
    with pytest.raises(RequestError):
        await adapter.new_session(str(workspace))
    assert (
        user_dir / terminal_auth.HANDOFF_DIR / f"request-{workspace_key(workspace)}.json"
    ).exists()
    assert _connect().exit_code == 0
    new = await adapter.new_session(str(workspace))
    assert _model_option(new.config_options) == ("second", ["second"])
    assert adapter.session(new.session_id).info.model == "second"
    assert _results(user_dir) == []


async def test_a_failed_session_new_keeps_the_save_for_the_retry(
    make_adapter, workspace, user_dir, saving_wizard, no_server, monkeypatch
):
    from nooa_atom.session.registry import SessionRegistry

    adapter = await _adapter(make_adapter, WorkspaceDefault())
    with pytest.raises(RequestError):
        await adapter.new_session(str(workspace))
    assert _connect().exit_code == 0
    create = SessionRegistry.create

    async def failing(self, *args, **kwargs):
        raise RuntimeError("could not start")

    monkeypatch.setattr(SessionRegistry, "create", failing)
    with pytest.raises(RuntimeError):
        await adapter.new_session(str(workspace))
    monkeypatch.setattr(SessionRegistry, "create", create)
    new = await adapter.new_session(str(workspace))
    assert adapter.session(new.session_id).info.model == "second"


async def test_resaving_the_current_alias_rebuilds_its_client(
    make_adapter, workspace, client, user_dir, saving_wizard, no_server
):
    factory = WorkspaceDefault()
    saving_wizard.alias = "first"
    saving_wizard.model = "openai/first-v2"
    adapter, session_id = await _connected_session(make_adapter, workspace, user_dir, factory)
    built = len(factory.calls)
    await _connect_prompt(adapter, session_id)
    assert [alias for alias, _ in factory.calls[built:]] == ["first"]
    assert "This session uses it from now on." in "".join(
        client.texts(AgentMessageChunk, session_id)
    )


async def test_a_workspace_entry_that_shadows_the_save_is_reported(
    make_adapter, workspace, client, user_dir, saving_wizard, no_server
):
    factory = WorkspaceDefault()
    adapter, session_id = await _connected_session(make_adapter, workspace, user_dir, factory)
    own = workspace / ".nooa" / "llm_config.yaml"
    own.parent.mkdir(exist_ok=True)
    own.write_text(yaml.safe_dump({"models": {"second": {"model_name": "openai/elsewhere"}}}))
    await _connect_prompt(adapter, session_id)
    assert adapter.session(session_id).info.model == "first"
    text = "".join(client.texts(AgentMessageChunk, session_id))
    assert "configures second differently" in text


async def test_an_alias_the_workspace_cannot_use_is_reported_not_switched_to(
    make_adapter, workspace, client, user_dir
):
    _save_user_model(user_dir, "first")
    adapter = await _adapter(make_adapter, WorkspaceDefault())
    session_id = (await adapter.new_session(str(workspace))).session_id
    terminal_auth.write_result(session_key(session_id), "elsewhere", "openai/x", "")
    await _connect_prompt(adapter, session_id)
    assert adapter.session(session_id).info.model == "first"
    assert "not in the model configuration" in "".join(client.texts(AgentMessageChunk, session_id))


async def test_an_unreadable_handoff_does_not_block_prompts(
    make_adapter, workspace, client, user_dir
):
    _save_user_model(user_dir, "first")
    adapter = await make_adapter(
        ScriptedModels({None: [reply("Still here.")]}),
        capabilities=TERMINAL,
        llm_factory=WorkspaceDefault(),
    )
    session_id = (await adapter.new_session(str(workspace))).session_id
    terminal_auth.write_result(session_key(session_id), "first", "m", "")
    handoff = user_dir / terminal_auth.HANDOFF_DIR
    handoff.chmod(0o500)
    try:
        response = await adapter.prompt(session_id, [text_block("hello")])
        assert response.stop_reason == "end_turn"
        assert client.texts(AgentMessageChunk, session_id)[-1] == "Still here.\n\n"
        # An explicit /connect says why it cannot finish.
        await _connect_prompt(adapter, session_id)
        assert "Could not read the result of `nooa connect`" in "".join(
            client.texts(AgentMessageChunk, session_id)
        )
    finally:
        handoff.chmod(0o700)


async def test_authenticate_rejects_an_unknown_method(make_adapter):
    adapter = await make_adapter(ScriptedModels({None: []}), capabilities=TERMINAL)
    with pytest.raises(RequestError):
        await adapter.authenticate("other")


async def test_a_session_that_ran_connect_can_be_loaded_after_the_restart(
    atom_adapter, workspace, client
):
    # Pool runs the terminal Connect, starts a new agent process, closes the
    # old one, then loads the same session by id. The old process must not
    # discard it as an empty session when it closes.
    adapter = await atom_adapter([], capabilities=TERMINAL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    with pytest.raises(RequestError) as failed:
        await _connect_prompt(adapter, session_id)
    assert failed.value.code == -32000
    restarted = await atom_adapter([], capabilities=TERMINAL)
    await adapter.close()
    await restarted.load_session(str(workspace), session_id)
    assert restarted.session(session_id).id == session_id


async def test_an_untouched_session_is_still_discarded_at_close(atom_adapter, workspace):
    adapter = await atom_adapter([], capabilities=TERMINAL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    await adapter.close()
    restarted = await atom_adapter([], capabilities=TERMINAL)
    with pytest.raises(RequestError):
        await restarted.load_session(str(workspace), session_id)


def test_a_client_without_auth_capabilities_is_not_a_terminal_client():
    from acp.schema import ClientCapabilities

    # A client that leaves ``auth`` out gets the schema's plain-dict default.
    assert terminal_auth.supports_terminal_auth(ClientCapabilities()) is False
    assert terminal_auth.supports_terminal_auth(None) is False
