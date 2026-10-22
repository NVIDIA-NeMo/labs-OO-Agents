# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""/connect: the host-side wizard, with a scripted asker and no network."""

from types import SimpleNamespace

import pytest
import yaml
from nooa_atom.agent import connect_command
from nooa_atom.agent.connect_command import CANCELLED, CONNECT_FALLBACK, run_connect
from nooa_atom.agent.slash_commands import BUILTIN_COMMANDS, RESERVED_COMMAND_NAMES

from nooa.elicit import ElicitationUnavailable, Elicited, bind_asker
from nooa.unifiedllm import connect as real

MODELS = [{"id": f"vendor/model-{n}"} for n in range(40)] + [
    {"id": "acme/llama-big", "context_length": 131072}
]


class FakeConnect:
    """The real library with its network calls replaced."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.keys: list[str | None] = []

    def __getattr__(self, name):
        return getattr(real, name)

    async def discover(self, endpoint, *, api_style="chat", api_key=None):
        self.calls.append("discover")
        self.keys.append(api_key)
        return real.Discovery(api_base=endpoint, models=tuple(MODELS))

    async def catalogue(self):
        self.calls.append("catalogue")
        return [
            {"id": "acme/llama-big", "context_length": 131072, "top_provider": {}, "reasoning": {}}
        ]

    async def check_interfaces(self, alias, model, api_base, api_key_env, **kwargs):
        self.calls.append("check_interfaces")
        self.keys.append(kwargs.get("api_key"))
        entry = real.plan(alias, model, "chat", api_base, api_key_env).entry
        entry["provenance"]["probes"]["routing"] = {"outcome": "accepted"}
        yield real.InterfaceResult(
            results={"chat": real.ConnectResult(alias, entry)}, tokens_charged_to_budget=10
        )

    async def run_steps(self, proposal, *, approved, api_key=None):
        self.calls.append(f"run_steps:{approved}")
        self.keys.append(api_key)
        yield real.ConnectResult(proposal.alias, proposal.entry)

    def verdict(self, entry):
        return real.Verdict(("routing", "tools"), (), (), ())


@pytest.fixture
def fake(monkeypatch):
    library = FakeConnect()
    monkeypatch.setattr(connect_command, "_connect", lambda: library)
    return library


def scripted(*answers):
    """An asker that answers each form in turn; records (message, field names)."""
    forms = []
    queue = list(answers)

    async def asker(request):
        forms.append((request.message, list(request.fields), request))
        answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, str):
            return Elicited(action=answer)
        return Elicited(action="accept", values=answer)

    asker.forms = forms
    return asker


HAPPY = (
    {"provider": "nvidia"},
    {"endpoint": "https://integrate.api.nvidia.com/v1"},
    {"api_key_env": "NVIDIA_API_KEY"},
    {"filter": "llama"},
    {"model": "acme/llama-big"},
    {"context_window": 100000, "max_tokens": 4096},
    {"alias": "big"},
)
ENV = {"NVIDIA_API_KEY": "sk-test-secret"}


async def _run(asker, path, environ=ENV):
    with bind_asker(asker):
        return await run_connect("", path=path, environ=environ)


async def test_the_happy_path_writes_the_config(fake, tmp_path):
    path = tmp_path / "llm_config.yaml"
    asker = scripted(*HAPPY)
    text = await _run(asker, path)
    assert "Saved big" in text and str(path) in text
    saved = yaml.safe_load(path.read_text())["models"]["big"]
    assert saved["api_base"] == "https://integrate.api.nvidia.com/v1"
    assert saved["api_key_env"] == "NVIDIA_API_KEY"
    assert saved["context_window"] == 100000
    assert saved["max_tokens"] == 4096
    assert "sk-test-secret" not in path.read_text() and "sk-test-secret" not in text
    # The order of the CLI wizard, and the defaults it uses.
    fields = [names for _message, names, _request in asker.forms]
    assert fields == [
        ["provider"],
        ["endpoint"],
        ["api_key_env"],
        ["filter"],
        ["model"],
        ["context_window", "max_tokens"],
        ["alias"],
    ]
    requests = [request for *_rest, request in asker.forms]
    assert "custom" in requests[0].fields["provider"].values
    assert requests[1].fields["endpoint"].default == "https://integrate.api.nvidia.com/v1"
    assert requests[2].fields["api_key_env"].default == "NVIDIA_API_KEY"
    assert requests[4].fields["model"].values == ("acme/llama-big",)
    assert requests[5].fields["context_window"].default == 131072
    assert requests[6].fields["alias"].default == "llama-big"
    # No form ever asks for a key, and the key came from the environment.
    assert not any("key" in name and name != "api_key_env" for names in fields for name in names)
    assert set(fake.keys) == {"sk-test-secret"}
    assert fake.calls == ["discover", "check_interfaces", "catalogue", "run_steps:all"]


async def test_replacing_an_alias_asks_first(fake, tmp_path):
    path = tmp_path / "llm_config.yaml"
    path.write_text(yaml.safe_dump({"models": {"big": {"model_name": "old"}}}))
    text = await _run(scripted(*HAPPY, {"replace": False}), path)
    assert text.endswith(CANCELLED)
    assert yaml.safe_load(path.read_text())["models"]["big"] == {"model_name": "old"}
    text = await _run(scripted(*HAPPY, {"replace": True}), path)
    assert "Saved big" in text
    assert yaml.safe_load(path.read_text())["models"]["big"]["model_name"] != "old"


async def test_a_missing_key_variable_stops_with_instructions(fake, tmp_path):
    path = tmp_path / "llm_config.yaml"
    text = await _run(scripted(*HAPPY[:3]), path, environ={})
    assert "NVIDIA_API_KEY is not set" in text
    assert "export NVIDIA_API_KEY=" in text and "Nothing written." in text
    assert not path.exists()
    assert fake.calls == []


@pytest.mark.parametrize("step", range(len(HAPPY)))
@pytest.mark.parametrize("action", ["decline", "cancel"])
async def test_declining_any_step_writes_nothing(fake, tmp_path, step, action):
    path = tmp_path / "llm_config.yaml"
    text = await _run(scripted(*HAPPY[:step], action), path)
    assert text.endswith(CANCELLED)
    assert not path.exists()


async def test_no_forms_gives_the_fallback_text(fake, tmp_path):
    path = tmp_path / "llm_config.yaml"
    assert await run_connect("", path=path, environ=ENV) == CONNECT_FALLBACK
    text = await _run(scripted(ElicitationUnavailable("no forms")), path)
    assert text == CONNECT_FALLBACK
    assert not path.exists()


async def test_a_bad_url_stops(fake, tmp_path):
    path = tmp_path / "llm_config.yaml"
    text = await _run(scripted({"provider": "custom"}, {"endpoint": "ftp://x"}), path)
    assert "cannot be used" in text and "Nothing written." in text
    assert not path.exists()


async def test_a_rejected_key_stops(fake, tmp_path, monkeypatch):
    async def rejected(*_args, **_kwargs):
        raise real.DiscoveryError("unauthorized", status_code=401)

    monkeypatch.setattr(fake, "discover", rejected, raising=False)
    path = tmp_path / "llm_config.yaml"
    text = await _run(scripted(*HAPPY[:3]), path)
    assert "rejected the key" in text and not path.exists()


def test_connect_is_a_host_side_builtin():
    command = BUILTIN_COMMANDS["connect"]
    assert command.output_to_agent is False and command._method is not None
    assert not command.body
    # Still reserved: no skill can claim /connect.
    assert "connect" in RESERVED_COMMAND_NAMES


async def test_the_registry_runs_connect_without_the_agent(fake, tmp_path, monkeypatch):
    from nooa_atom.agent.slash_commands import SlashCommandRegistry

    monkeypatch.setattr(connect_command, "default_registry_path", lambda: tmp_path / "c.yaml")
    agent = SimpleNamespace(cwd=tmp_path)
    registry = SlashCommandRegistry(agent, bind_registry=False)
    result = await registry.invoke("connect", "")
    assert result.output_to_agent is False
    assert result.text == CONNECT_FALLBACK
