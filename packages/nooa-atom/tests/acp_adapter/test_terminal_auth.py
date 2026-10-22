# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP terminal auth: ``nooa atom --connect`` runs the ``nooa connect`` wizard.

The wizard itself is mocked; nothing here touches the network.
"""

import json

import click
import click.testing
import pytest
import yaml
from acp import PROTOCOL_VERSION
from acp.schema import (
    AuthCapabilities,
    ClientCapabilities,
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
