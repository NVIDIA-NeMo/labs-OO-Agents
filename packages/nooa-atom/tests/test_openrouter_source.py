# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The built-in OpenRouter model source (no network: HTTP is patched)."""

import asyncio
import importlib.metadata
import json
import os
import time
from pathlib import Path

import click.testing
import httpx
import pytest
import yaml
from nooa_atom.workspace import model_sources
from nooa_atom.workspace import openrouter_source as orsrc
from nooa_atom.workspace.models import workspace_models

from nooa.unifiedllm import connect

FIXTURE = Path(__file__).parent / "fixtures" / "openrouter_models.json"
RAW = json.loads(FIXTURE.read_text())["data"]
RECORDS = [connect.openrouter_facts(item) for item in RAW]
IDS = sorted(item["id"] for item in RAW)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    user = tmp_path / "user"
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.setenv("NEMO_OO_SECRETS", str(tmp_path / "no-secrets.yaml"))
    monkeypatch.delenv("NEMO_OO_LLM_CONFIG", raising=False)
    monkeypatch.delenv(orsrc.KEY_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    orsrc.reset()
    model_sources.reset_model_sources()
    yield user
    orsrc.reset()
    model_sources.reset_model_sources()


class Fetch:
    """A patched catalogue fetch: returns ``RECORDS`` or raises ``error``."""

    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def __call__(self, timeout=orsrc.FETCH_TIMEOUT):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return [dict(record) for record in RECORDS]


def _fetch(monkeypatch, error=None):
    fetch = Fetch(error)
    monkeypatch.setattr(orsrc, "_fetch", fetch)
    return fetch


def _cache(age=0.0):
    path = orsrc.write_cache(RECORDS)
    data = json.loads(path.read_text())
    data["fetched_at"] = time.time() - age
    path.write_text(json.dumps(data))
    return path


def _aliases():
    return [orsrc.ALIAS_PREFIX + model for model in IDS]


# ---- models() ------------------------------------------------------------------


def test_no_key_lists_nothing_and_fetches_nothing(monkeypatch, tmp_path):
    fetch = _fetch(monkeypatch)
    _cache()
    assert orsrc.source.models(tmp_path) == {}
    assert fetch.calls == 0


def test_a_key_saved_in_secrets_counts_and_is_loaded(monkeypatch, tmp_path):
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text(yaml.safe_dump({"env": {orsrc.KEY_ENV: "sk-or-saved"}}))
    monkeypatch.setenv("NEMO_OO_SECRETS", str(secrets))
    _fetch(monkeypatch)
    _cache()
    assert list(orsrc.source.choices(tmp_path)) == _aliases()
    # The model client reads the key from the environment.
    assert os.environ[orsrc.KEY_ENV] == "sk-or-saved"


def test_fresh_cache_needs_no_network_and_is_memoised(monkeypatch, tmp_path):
    monkeypatch.setenv(orsrc.KEY_ENV, "sk-or-test")
    fetch = _fetch(monkeypatch, error=AssertionError("no network"))
    _cache(age=60)
    first = orsrc.source.choices(tmp_path)
    assert list(first) == _aliases()
    built = []
    monkeypatch.setattr(orsrc, "entries", lambda records: built.append(records) or {})
    assert orsrc.source.choices(tmp_path) == first
    assert built == [] and fetch.calls == 0


def test_missing_cache_is_fetched_once(monkeypatch, tmp_path):
    monkeypatch.setenv(orsrc.KEY_ENV, "sk-or-test")
    fetch = _fetch(monkeypatch)
    assert list(orsrc.source.choices(tmp_path)) == _aliases()
    assert orsrc.cache_path().is_file()
    assert list(orsrc.source.choices(tmp_path)) == _aliases()
    assert fetch.calls == 1


def test_stale_cache_is_refreshed(monkeypatch, tmp_path):
    monkeypatch.setenv(orsrc.KEY_ENV, "sk-or-test")
    fetch = _fetch(monkeypatch)
    _cache(age=orsrc.MAX_AGE + 60)
    assert list(orsrc.source.choices(tmp_path)) == _aliases()
    assert fetch.calls == 1
    fetched_at = json.loads(orsrc.cache_path().read_text())["fetched_at"]
    assert time.time() - fetched_at < 60


def test_stale_cache_and_failed_fetch_keep_the_stale_models(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv(orsrc.KEY_ENV, "sk-or-test")
    fetch = _fetch(monkeypatch, error=httpx.ConnectError("offline"))
    _cache(age=orsrc.MAX_AGE + 60)
    assert list(orsrc.source.choices(tmp_path)) == _aliases()
    assert list(orsrc.source.choices(tmp_path)) == _aliases()
    # One attempt, one warning: a failure is not retried on every call.
    assert fetch.calls == 1
    assert [r.levelname for r in caplog.records if "OpenRouter" in r.getMessage()] == ["WARNING"]


def test_missing_cache_and_failed_fetch_list_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv(orsrc.KEY_ENV, "sk-or-test")
    _fetch(monkeypatch, error=TimeoutError())
    assert orsrc.source.choices(tmp_path) == {}


def test_an_unreadable_cache_is_fetched_again(monkeypatch, tmp_path):
    monkeypatch.setenv(orsrc.KEY_ENV, "sk-or-test")
    fetch = _fetch(monkeypatch)
    orsrc.cache_path().parent.mkdir(parents=True)
    orsrc.cache_path().write_text("{not json")
    assert list(orsrc.source.choices(tmp_path)) == _aliases()
    assert fetch.calls == 1


def test_the_fetch_asks_for_text_models_with_a_short_timeout(monkeypatch):
    seen = {}

    async def catalogue(*, timeout, outputs):
        seen.update(timeout=timeout, outputs=outputs)
        return RECORDS

    monkeypatch.setattr(connect, "catalogue", catalogue)

    async def inside_a_running_loop():
        return orsrc._fetch()

    assert asyncio.run(inside_a_running_loop()) == RECORDS
    assert seen == {"timeout": orsrc.FETCH_TIMEOUT, "outputs": "text"}


# ---- parity with nooa connect ----------------------------------------------------


def _connect_saved_entries(monkeypatch, tmp_path):
    """What Connect's OpenRouter path saves for each fixture model.

    As the wizard does for ``--provider openrouter``: read the endpoint's
    listing (OpenRouter's catalogue), plan with the listing record and its
    catalogue facts, enable the suggested media without testing, then
    ``write`` the entry and read the file back.
    """
    original = httpx.AsyncClient.__init__

    def init(self, **kwargs):
        kwargs["transport"] = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": RAW})
        )
        original(self, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    preset = connect.PROVIDERS["openrouter"]
    discovery = asyncio.run(connect.discover(preset.api_base, api_style=preset.api_style))
    assert discovery.models_url == connect.CATALOGUE_URL
    registry = tmp_path / "llm_config.yaml"
    for model in discovery.models:
        arguments = {
            "catalogue": connect.openrouter_facts(model["public_model"]),
            "endpoint_model": model,
            "endpoint_models_url": discovery.models_url,
        }
        args = (model["id"], model["id"], preset.api_style, preset.api_base, preset.api_key_env)
        suggested = connect.plan(*args, **arguments).entry["provenance"]["media_suggestions"]
        proposal = connect.plan(*args, **arguments, media_inputs=suggested["suggested"])
        connect.write(proposal.entry, registry, alias=orsrc.ALIAS_PREFIX + model["id"])
    return yaml.safe_load(registry.read_text())["models"]


def test_entries_equal_what_nooa_connect_saves(monkeypatch, tmp_path):
    saved = _connect_saved_entries(monkeypatch, tmp_path)
    ours = orsrc.entries(RECORDS)
    assert sorted(saved) == list(ours) == _aliases()
    for alias in ours:
        assert ours[alias] == saved[alias], alias


def test_entries_are_configured_from_the_catalogue():
    ours = orsrc.entries(RECORDS)
    claude = ours["openrouter/anthropic/claude-sonnet-4.5"]
    assert claude["model_name"] == "openai/anthropic/claude-sonnet-4.5"
    assert claude["api_base"] == "https://openrouter.ai/api/v1"
    assert claude["api_key_env"] == "OPENROUTER_API_KEY"
    assert claude["api_style"] == "chat"
    assert claude["media_inputs"] == ["image"]
    assert claude["context_window"] == 1_000_000
    # Connect's default reply cap: at most 32768 and the model's limits.
    assert claude["max_tokens"] == 32768
    gpt = ours["openrouter/openai/gpt-5"]
    assert list(gpt["reasoning_levels"]) == ["high", "medium", "low", "minimal"]
    assert gpt["reasoning_levels"]["low"] == {"reasoning_effort": "low"}
    assert gpt["reasoning_default"] == "medium"
    llama = ours["openrouter/meta-llama/llama-3.3-70b-instruct"]
    assert llama["media_inputs"] == []
    assert llama["context_window"] == 128000  # The top provider's lower limit.
    assert "reasoning_levels" not in llama


def test_a_record_connect_rejects_is_skipped():
    assert list(orsrc.entries([{"id": ""}, *RECORDS])) == _aliases()


# ---- sign_in() ------------------------------------------------------------------


class Key:
    """A patched ``/key`` check that records the keys it saw."""

    def __init__(self, status=200):
        self.status = status
        self.keys = []

    def __call__(self, url, *, headers, timeout, follow_redirects):
        assert url == orsrc.KEY_URL and timeout <= 10
        self.keys.append(headers["Authorization"].removeprefix("Bearer "))
        return httpx.Response(self.status, json={"data": {}})


def _sign_in(monkeypatch, answers, status=200):
    key = Key(status)
    monkeypatch.setattr(httpx, "get", key)
    saved = []
    from nooa import secrets

    original = secrets.write_secret_env
    monkeypatch.setattr(
        secrets,
        "write_secret_env",
        lambda path, name, value: saved.append((Path(path), name)) or original(path, name, value),
    )
    result = {}

    @click.command()
    def command():
        result["alias"] = orsrc.source.sign_in()

    run = click.testing.CliRunner().invoke(command, input=answers)
    return result.get("alias"), run, key, saved


def test_sign_in_saves_the_key_writes_the_cache_and_picks_a_default(monkeypatch, isolated):
    fetch = _fetch(monkeypatch)
    alias, run, key, saved = _sign_in(monkeypatch, "sk-or-new\n")
    assert run.exit_code == 0, run.output
    assert alias == "openrouter/anthropic/claude-sonnet-4.5"
    assert key.keys == ["sk-or-new"] and fetch.calls == 1
    assert saved == [(isolated / "secrets.yaml", "OPENROUTER_API_KEY")]
    stored = yaml.safe_load((isolated / "secrets.yaml").read_text())
    assert stored == {"env": {"OPENROUTER_API_KEY": "sk-or-new"}}
    assert "sk-or-new" not in run.output
    assert "3 OpenRouter models are now available" in run.output
    assert orsrc.cache_path().is_file()
    # The models are listed at once, without another fetch.
    assert list(orsrc.source.choices(Path.cwd())) == _aliases()
    assert fetch.calls == 1


def test_sign_in_keeps_an_existing_key(monkeypatch):
    monkeypatch.setenv(orsrc.KEY_ENV, "sk-or-env")
    _fetch(monkeypatch)
    alias, run, key, saved = _sign_in(monkeypatch, "y\n")
    assert run.exit_code == 0, run.output
    assert alias == "openrouter/anthropic/claude-sonnet-4.5"
    assert key.keys == ["sk-or-env"] and saved == []
    assert "sk-or-env" not in run.output


def test_sign_in_with_a_bad_key_saves_nothing(monkeypatch, isolated):
    fetch = _fetch(monkeypatch)
    alias, run, key, saved = _sign_in(monkeypatch, "sk-or-bad\n", status=401)
    assert alias is None
    assert "OpenRouter did not accept this key. Nothing was saved." in run.output
    assert saved == [] and fetch.calls == 0
    assert not (isolated / "secrets.yaml").exists()
    assert "sk-or-bad" not in run.output


def test_sign_in_without_a_key_sets_nothing_up(monkeypatch):
    _fetch(monkeypatch)
    alias, run, key, saved = _sign_in(monkeypatch, "\n")
    assert alias is None and key.keys == [] and saved == []


def test_sign_in_lets_ctrl_c_through(monkeypatch):
    _fetch(monkeypatch)
    monkeypatch.setattr(click, "prompt", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        orsrc.source.sign_in()


def test_default_alias_falls_back_to_the_first_model():
    assert orsrc.default_alias({"openrouter/x/a": {}, "openrouter/x/b": {}}) == "openrouter/x/a"


# ---- registration ----------------------------------------------------------------


def test_the_entry_point_is_registered():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    import tomllib

    group = tomllib.loads(pyproject.read_text())["project"]["entry-points"][
        model_sources.ENTRY_POINT_GROUP
    ]
    assert group == {"openrouter": "nooa_atom.workspace.openrouter_source:source"}
    installed = {
        entry.name: entry.value
        for entry in importlib.metadata.entry_points(group=model_sources.ENTRY_POINT_GROUP)
    }
    if installed:  # Only when nooa-atom's metadata is installed in this environment.
        assert installed.get("openrouter") == group["openrouter"]


def test_the_source_is_valid_and_listed_after_the_files(monkeypatch, tmp_path):
    class EntryPoint:
        name = "openrouter"

        def load(self):
            return orsrc.source

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *, group: [EntryPoint()] if group == model_sources.ENTRY_POINT_GROUP else [],
    )
    assert model_sources._check(orsrc.source) is None
    assert model_sources.find_source("openrouter") is orsrc.source
    monkeypatch.setenv(orsrc.KEY_ENV, "sk-or-test")
    _fetch(monkeypatch)
    config = tmp_path / ".nooa" / "llm_config.yaml"
    config.parent.mkdir()
    config.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "mine": {"model_name": "openai/mine", "api_base": "https://mine.test/v1"},
                    "openrouter/openai/gpt-5": None,
                }
            }
        )
    )
    monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(config))
    assert list(workspace_models(tmp_path)) == ["mine"]  # the source lists nothing


def test_models_lists_nothing_and_choices_list_the_catalogue(monkeypatch, tmp_path):
    monkeypatch.setenv(orsrc.KEY_ENV, "sk-or-test")
    _fetch(monkeypatch)
    assert orsrc.source.models(tmp_path) == {}
    assert list(orsrc.source.choices(tmp_path)) == _aliases()
