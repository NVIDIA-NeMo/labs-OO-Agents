# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model sources: installed packages that add models and a terminal sign-in."""

import importlib.metadata
import json
import logging
import types

import click.testing
import pytest
import yaml
from nooa_atom.acp import terminal_auth
from nooa_atom.workspace import model_sources
from nooa_atom.workspace.models import workspace_models


class Source:
    """A model source for tests: ``outcome`` is what ``sign_in`` returns or raises."""

    def __init__(self, id="hub", models=None, outcome="hub-a"):
        self.id = id
        self.name = f"{id.title()} models"
        self.description = f"Sign in to {id}"
        self._models = (
            {"hub-a": {"model_name": "openai/a", "api_base": "https://hub.test/v1"}}
            if models is None
            else models
        )
        self.outcome = outcome
        self.calls = 0

    def models(self, workspace):
        if isinstance(self._models, BaseException):
            raise self._models
        return self._models

    def sign_in(self):
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class EntryPoint:
    def __init__(self, name, load):
        self.name = name
        self._load = load

    def load(self):
        return self._load()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    user = tmp_path / "user"
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.delenv("NEMO_OO_LLM_CONFIG", raising=False)
    model_sources.reset_model_sources()
    yield user
    model_sources.reset_model_sources()


def _entry_points(monkeypatch, *entries):
    def entry_points(*, group):
        return list(entries) if group == model_sources.ENTRY_POINT_GROUP else []

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)


def _install(monkeypatch, *sources):
    _entry_points(monkeypatch, *(EntryPoint(s.id, lambda s=s: s) for s in sources))


# ---- loading -------------------------------------------------------------------


def test_sources_load_once_and_bad_ones_are_skipped(monkeypatch, caplog):
    good = Source("hub")

    def broken():
        raise ImportError("no module named hub_missing")

    no_name = types.SimpleNamespace(id="x", description="", models=dict, sign_in=lambda: None)
    loads = []
    _entry_points(
        monkeypatch,
        EntryPoint("good", lambda: loads.append(1) or good),
        EntryPoint("broken", broken),
        EntryPoint("no-name", lambda: no_name),
        EntryPoint("bad-id", lambda: Source("Bad_Id")),
        EntryPoint("again", lambda: Source("hub")),
        EntryPoint("factory", lambda: lambda: Source("other")),
        EntryPoint("factory-fails", lambda: lambda: 1 / 0),
    )
    with caplog.at_level(logging.WARNING, logger=model_sources.__name__):
        sources = model_sources.model_sources()
    assert [source.id for source in sources] == ["hub", "other"]
    assert sources[0] is good
    text = caplog.text
    for name in ("broken", "no-name", "bad-id", "again", "factory-fails"):
        assert f"Skipped model source {name}" in text
    assert "no 'name'" in text and "another source has the id 'hub'" in text
    model_sources.model_sources()
    assert loads == [1]
    assert model_sources.find_source("other").id == "other"
    assert model_sources.find_source("missing") is None


def test_a_source_class_is_instantiated(monkeypatch):
    class Hub:
        id = "hub"
        name = "Hub"
        description = "d"

        def models(self, workspace):
            return {}

        def sign_in(self):
            return None

    _entry_points(monkeypatch, EntryPoint("hub", lambda: Hub))
    (source,) = model_sources.model_sources()
    assert isinstance(source, Hub)


def test_no_sources_installed(monkeypatch):
    _entry_points(monkeypatch)
    assert model_sources.model_sources() == []
    assert terminal_auth.auth_method_ids() == [terminal_auth.CONNECT_METHOD_ID]


# ---- workspace_models ----------------------------------------------------------


def _write(path, models):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"models": models}))


def test_source_models_are_the_lowest_layer(monkeypatch, tmp_path, isolated, caplog):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _install(
        monkeypatch,
        Source(
            "hub",
            models={
                "hub-a": {"model_name": "openai/a"},
                "hub-b": {"model_name": "openai/b"},
                "hub-c": {"model_name": "openai/c"},
                "bad": "not a mapping",
            },
        ),
        Source("down", models=RuntimeError("offline")),
    )
    assert list(workspace_models(workspace)) == ["hub-a", "hub-b", "hub-c"]
    _write(isolated / "llm_config.yaml", {"local": {"model_name": "openai/local"}})
    _write(
        workspace / ".nooa" / "llm_config.yaml",
        {"hub-b": {"model_name": "openai/mine"}, "hub-c": None},
    )
    with caplog.at_level(logging.WARNING, logger=model_sources.__name__):
        models = workspace_models(workspace)
    # File aliases first (the default model stays a file's), then the rest of the sources.
    assert list(models) == ["local", "hub-b", "hub-a"]
    assert models["hub-b"] == {"model_name": "openai/mine"}
    assert "Model source down could not list its models" in caplog.text


def test_source_models_are_read_on_every_call(monkeypatch, tmp_path):
    source = Source("hub", models={})
    _install(monkeypatch, source)
    assert workspace_models(tmp_path) == {}
    source._models = {"hub-a": {"model_name": "openai/a"}}
    assert list(workspace_models(tmp_path)) == ["hub-a"]


# ---- auth methods --------------------------------------------------------------


def test_terminal_clients_are_offered_connect_and_each_source(monkeypatch):
    from acp import PROTOCOL_VERSION
    from acp.schema import AuthCapabilities, ClientCapabilities
    from nooa_atom.acp.protocol import initialize_response

    _install(monkeypatch, Source("hub"), Source("lab"))

    def offered(capabilities):
        response = initialize_response(PROTOCOL_VERSION, client_capabilities=capabilities)
        return [m.model_dump(mode="json", exclude_none=True) for m in response.auth_methods]

    assert offered(ClientCapabilities(auth=AuthCapabilities(terminal=True))) == [
        {
            "id": "nooa-connect",
            "name": "Other models (nooa connect)",
            "description": "Set up a model provider in a terminal",
            "type": "terminal",
            "args": ["--connect"],
        },
        {
            "id": "nooa-source-hub",
            "name": "Hub models",
            "description": "Sign in to hub",
            "type": "terminal",
            "args": ["--sign-in", "hub"],
        },
        {
            "id": "nooa-source-lab",
            "name": "Lab models",
            "description": "Sign in to lab",
            "type": "terminal",
            "args": ["--sign-in", "lab"],
        },
    ]
    assert offered(ClientCapabilities()) == []
    assert offered(None) == []
    assert terminal_auth.is_auth_method("nooa-source-hub")
    assert terminal_auth.is_auth_method("nooa-connect")
    assert not terminal_auth.is_auth_method("nooa-source-other")
    assert not terminal_auth.is_auth_method(None)


# ---- nooa atom --sign-in -------------------------------------------------------


@pytest.fixture
def no_server(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("--sign-in started the ACP server")

    monkeypatch.setattr("nooa_atom.acp.cli.run", fail)
    monkeypatch.setattr("nooa_atom.acp.cli.reserve_stdio_for_acp", fail)


def _sign_in(source_id):
    from nooa_atom.acp.cli import command

    return click.testing.CliRunner().invoke(command, ["--sign-in", source_id])


def test_sign_in_saves_the_result(monkeypatch, isolated, no_server):
    source = Source("hub")
    _install(monkeypatch, source)
    result = _sign_in("hub")
    assert result.exit_code == 0, result.output
    assert "Return to the client" in result.output
    saved = json.loads((isolated / terminal_auth.RESULT_FILE).read_text())
    assert saved["alias"] == "hub-a" and saved["model"] == "openai/a"
    assert saved["server"] == "https://hub.test/v1"
    assert source.calls == 1


THREE = {
    "hub-a": {"model_name": "openai/a", "api_base": "https://hub.test/v1"},
    "hub-b": {"model_name": "openai/b", "api_base": "https://hub.test/v1"},
    "hub-c": {"model_name": "openai/c", "api_base": "https://hub.test/v1"},
}


@pytest.mark.parametrize(
    ("answers", "alias"),
    [("hub-c\n", "hub-c"), ("\n", "hub-a"), ("nope\nhub-b\n", "hub-b")],
)
def test_sign_in_asks_which_model_to_use(monkeypatch, isolated, no_server, answers, alias):
    _install(monkeypatch, Source("hub", models=THREE))
    from nooa_atom.acp.cli import command

    result = click.testing.CliRunner().invoke(command, ["--sign-in", "hub"], input=answers)
    assert result.exit_code == 0, result.output
    assert "3 Hub models models. Type part of a name" in result.output
    saved = json.loads((isolated / terminal_auth.RESULT_FILE).read_text())
    assert saved["alias"] == alias
    assert saved["model"] == THREE[alias]["model_name"]


class PickingSource(Source):
    """Lists nothing; ``choices`` is the whole catalogue."""

    def __init__(self):
        super().__init__("hub", models=THREE)

    def choices(self, workspace):
        return THREE

    def models(self, workspace):
        return {}


def test_sign_in_picks_from_choices_and_writes_only_the_pick(monkeypatch, isolated, no_server):
    from nooa_atom.acp.cli import command

    from nooa import paths

    _install(monkeypatch, PickingSource())
    result = click.testing.CliRunner().invoke(command, ["--sign-in", "hub"], input="hub-b\n")
    assert result.exit_code == 0, result.output
    saved = json.loads((isolated / terminal_auth.RESULT_FILE).read_text())
    assert saved["alias"] == "hub-b" and saved["model"] == "openai/b"
    written = yaml.safe_load(paths.get_user_dir("llm_config.yaml").read_text())["models"]
    assert list(written) == ["hub-b"]
    assert written["hub-b"]["model_name"] == "openai/b"


@pytest.mark.parametrize(
    ("source_id", "outcome", "code"),
    [("hub", None, 1), ("hub", "", 1), ("hub", RuntimeError("bad key"), 1), ("nope", "x", 2)],
)
def test_sign_in_without_a_model_saves_nothing(
    monkeypatch, isolated, no_server, source_id, outcome, code
):
    _install(monkeypatch, Source("hub", outcome=outcome))
    terminal_auth.write_result("old", "openai/old", "")
    result = _sign_in(source_id)
    assert result.exit_code == code
    assert not (isolated / terminal_auth.RESULT_FILE).exists()
    if isinstance(outcome, BaseException):
        assert "bad key" in result.output


def test_sign_in_lets_ctrl_c_through(monkeypatch, isolated):
    _install(monkeypatch, Source("hub", outcome=KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        terminal_auth.run_terminal_sign_in("hub")
    assert not (isolated / terminal_auth.RESULT_FILE).exists()


def test_sign_in_with_models_that_fail_still_saves_the_alias(monkeypatch, isolated):
    source = Source("hub")
    _install(monkeypatch, source)
    model_sources.model_sources()
    source._models = RuntimeError("offline")
    assert terminal_auth.run_terminal_sign_in("hub") == 0
    saved = json.loads((isolated / terminal_auth.RESULT_FILE).read_text())
    assert saved["alias"] == "hub-a" and saved["model"] == ""
