# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The same onboarding plan can be approved by a CLI or a TUI."""

import json
from copy import deepcopy

import httpx
import pytest
import yaml

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, mock_post, response_body


def make_plan(**kwargs):
    return connect.plan(
        "local", "gateway/model", "chat", "https://api.test/v1", "CONNECT_TEST_KEY", **kwargs
    )


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"reasoning_effort": "high"}, True),
        ({"reasoning": {"effort": "high"}}, True),
        ({"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}, True),
        ({"thinking": {"type": "enabled", "budget_tokens": 1024}}, True),
        ({"chat_template_kwargs": {"enable_thinking": True}}, True),
        ({"reasoning_effort": "none"}, False),
        ({"reasoning": {"effort": "off"}}, False),
        ({"thinking": {"type": "disabled"}}, False),
        ({"chat_template_kwargs": {"enable_thinking": False}}, False),
        ({"temperature": 0.5}, False),
    ],
)
def test_unobserved_reasoning_checks_use_settings_not_label_names(params, expected):
    entry = make_plan(reasoning_levels={"custom": params}).entry
    record = {"outcome": "accepted", "reasoning_observed": False}
    entry["provenance"]["probes"]["level:custom"] = record
    original = deepcopy(entry)
    assert connect.unobserved_reasoning_levels(entry) == (["custom"] if expected else [])
    assert entry == original
    record["reasoning_observed"] = True
    assert connect.unobserved_reasoning_levels(entry) == []
    record["reasoning_observed"] = False
    for outcome in ("rejected", "not_probed"):
        record["outcome"] = outcome
        assert connect.unobserved_reasoning_levels(entry) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
async def test_discover_normalizes_root_and_uses_temporary_key(monkeypatch, style):
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path == "/models":
            return httpx.Response(404)
        assert request.url.path == "/v1/models"
        key = "x-api-key" if style == "anthropic" else "authorization"
        assert request.headers[key].endswith("temporary-secret")
        return httpx.Response(200, json={"data": [{"id": "model-b"}, {"id": "model-a"}]})

    mock_http(monkeypatch, handle)
    found = await connect.discover("https://api.test", api_style=style, api_key="temporary-secret")
    assert found.api_base == "https://api.test/v1"
    assert [model["id"] for model in found.models] == ["model-a", "model-b"]
    assert "temporary-secret" not in repr(found)


@pytest.mark.asyncio
async def test_discovery_auth_failure_is_structured_and_does_not_retry(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(401, text="echo-secret")

    mock_http(monkeypatch, handle)
    with pytest.raises(connect.DiscoveryError) as error:
        await connect.discover("https://api.test", api_key="echo-secret")
    assert error.value.status_code == 401
    assert "echo-secret" not in str(error.value)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_discover_anthropic_pagination_and_optional_auth(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        assert "authorization" not in request.headers
        assert "x-api-key" not in request.headers
        assert request.headers["anthropic-version"] == "2023-06-01"
        if "after_id" not in request.url.params:
            return httpx.Response(
                200, json={"data": [{"id": "a"}], "has_more": True, "last_id": "a"}
            )
        assert request.url.params["after_id"] == "a"
        return httpx.Response(200, json={"data": [{"id": "b"}], "has_more": False})

    mock_http(monkeypatch, handle)
    found = await connect.discover("http://localhost:8000/v1/models", api_style="anthropic")
    assert found.api_base == "http://localhost:8000/v1"
    assert len(found.models) == len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key_env,key", [("CONNECT_TEST_KEY", "temporary-key"), ("", "temporary-key")]
)
async def test_run_transient_or_no_key_never_changes_environment(monkeypatch, key_env, key):
    monkeypatch.delenv("CONNECT_TEST_KEY", raising=False)
    proposal = connect.plan("local", "model", "chat", "http://localhost:8000/v1", key_env)
    requests = []

    async def post(self, url, **kwargs):
        requests.append(kwargs)
        return httpx.Response(200, json={"choices": [{"message": {"content": "323"}}]})

    mock_post(monkeypatch, post)
    result = await connect.run(proposal, approved="minimal", api_key=key)
    assert len(requests) == 1
    assert requests[0]["headers"].get("Authorization") == (f"Bearer {key}" if key else None)
    assert "CONNECT_TEST_KEY" not in __import__("os").environ
    assert "temporary-key" not in repr(proposal) + repr(result)


@pytest.mark.asyncio
async def test_cancel_after_discovery_leaves_credentials_and_files_untouched(tmp_path, monkeypatch):
    original = httpx.AsyncClient
    monkeypatch.delenv("CONNECT_TEST_KEY", raising=False)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": [{"id": "model"}]})
            ),
            **kw,
        ),
    )
    found = await connect.discover("https://api.test/v1", api_key="temporary-key")
    assert found.models
    # The frontend cancels here: discovery must not persist any key or model.
    assert "CONNECT_TEST_KEY" not in __import__("os").environ
    assert not list(tmp_path.iterdir())


def test_plan_is_data_without_credentials_or_network(monkeypatch):
    monkeypatch.delenv("CONNECT_TEST_KEY", raising=False)
    proposal = make_plan(reasoning_levels={"high": {"reasoning_effort": "high"}})
    assert proposal.entry["model_name"] == "openai/gateway/model"
    assert proposal.entry["client_type"] == "completion"
    assert proposal.entry["max_tokens"] == 32768
    assert "context_window" not in proposal.entry
    assert proposal.price_estimate is None
    assert [p.name for p in proposal.probes] == ["routing", "tools", "level:high"]
    assert [p.body["max_tokens"] for p in proposal.probes] == [32768, 32768, 32768]
    assert proposal.probes[-1].body["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_no_approval_needs_no_credentials_and_does_no_io(monkeypatch):
    monkeypatch.delenv("CONNECT_TEST_KEY", raising=False)
    proposal = make_plan()
    before = deepcopy(proposal.entry)
    result = await connect.run(proposal, approved="none")
    assert proposal.entry == before
    assert all(p["outcome"] == "not_probed" for p in result.entry["provenance"]["probes"].values())


@pytest.mark.parametrize(
    "style,suffix,token_key",
    [
        ("chat", "/chat/completions", "max_tokens"),
        ("responses", "/responses", "max_output_tokens"),
        ("anthropic", "/messages", "max_tokens"),
    ],
)
@pytest.mark.asyncio
async def test_minimal_approval_posts_exact_plan_once(monkeypatch, style, suffix, token_key):
    monkeypatch.setenv("CONNECT_TEST_KEY", "secret-test-key")
    proposal = connect.plan("local", "wire/model", style, "https://api.test/v1", "CONNECT_TEST_KEY")
    requests = []

    async def post(self, url, **kwargs):
        requests.append((url, kwargs))
        return httpx.Response(
            200,
            json=response_body(style),
        )

    mock_post(monkeypatch, post)
    result = await connect.run(proposal, approved="minimal")
    assert len(requests) == 1
    url, kwargs = requests[0]
    assert url.endswith(suffix)
    expected = deepcopy(proposal.probes[0].body)
    if style == "responses":
        expected["truncation"] = "disabled"
    elif style == "anthropic":
        expected["messages"][0]["content"] = [
            {"type": "text", "text": expected["messages"][0]["content"]}
        ]
    assert kwargs["json"] == expected
    assert kwargs["json"][token_key] == 32768
    assert kwargs["json"]["model"] == "wire/model"
    assert result.entry["provenance"]["probes"]["routing"]["outcome"] == "accepted"
    assert "secret-test-key" not in json.dumps(result.entry)


@pytest.mark.asyncio
async def test_failures_do_not_retry_or_mark_levels_unsupported(monkeypatch):
    monkeypatch.setenv("CONNECT_TEST_KEY", "secret-test-key")
    calls = []

    async def post(self, url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(401, json={"error": {"message": "secret-test-key rejected"}})

    mock_post(monkeypatch, post)
    result = await connect.run(
        make_plan(reasoning_levels={"high": {"reasoning_effort": "high"}}), approved="all"
    )
    assert len(calls) == 1
    assert "secret-test-key" not in json.dumps(result.entry)
    assert result.entry["reasoning_levels"] == {"high": {"reasoning_effort": "high"}}
    assert result.entry["provenance"]["probes"]["level:high"]["outcome"] == "not_probed"


@pytest.mark.asyncio
async def test_budget_stops_before_second_call_and_reconnect_skips_accepted(monkeypatch):
    monkeypatch.setenv("CONNECT_TEST_KEY", "secret-test-key")
    bodies = []

    async def post(self, url, **kwargs):
        bodies.append(kwargs["json"])
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "323"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    mock_post(monkeypatch, post)
    first = await connect.run(make_plan(budget_tokens=33280), approved="all")
    assert len(bodies) == 1
    assert first.entry["provenance"]["probes"]["tools"]["outcome"] == "not_probed"
    assert first.entry["provenance"]["probes"]["tools"]["reason"] == "budget exhausted"
    await connect.run(make_plan(existing_entry=first.entry), approved="minimal")
    assert len(bodies) == 1


def test_write_preserves_other_aliases(tmp_path):
    path = tmp_path / "connected.yaml"
    proposal = make_plan()
    connect.write(proposal.entry, path, alias="first")
    connect.write(proposal.entry, path, alias="second")
    assert set(yaml.safe_load(path.read_text())["models"]) == {"first", "second"}


def test_write_replaces_hand_written_alias_preserving_neighbors(tmp_path):
    path = tmp_path / "llm_config.yaml"
    path.write_text(
        "# human notes\nmodels:\n  local:\n    model_name: openai/old\n"
        "  other: {model_name: openai/other} # keep this\n"
        "settings: true # keep this too\n"
    )
    connect.write({"model_name": "openai/new"}, path, alias="local")
    text = path.read_text()
    assert text.startswith("# human notes\n")
    assert "  other: {model_name: openai/other} # keep this\n" in text
    assert "settings: true # keep this too\n" in text
    assert yaml.safe_load(text)["models"]["local"] == connect.configure_entry(
        {"model_name": "openai/new"}
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"model": "other"},
        {"messages": []},
        {"api_key": "secret"},
        {"extra_body": {}},
        {"client": "other"},
    ],
)
def test_level_cannot_change_route_or_credentials(patch):
    with pytest.raises(ValueError):
        make_plan(reasoning_levels={"high": patch})


def test_catalogue_metadata_is_not_claimed_as_probe_evidence():
    proposal = make_plan(
        catalogue={
            "id": "provider/model",
            "context_length": 128000,
            "top_provider": {"max_completion_tokens": 8192},
        }
    )
    assert proposal.entry["context_window"] == 128000
    assert proposal.entry["provenance"]["catalogue_limits"]["max_completion_tokens"] == 8192
    assert proposal.entry["max_tokens"] == 8192
    assert "context_window" in proposal.entry["provenance"]["not_probed"]


def test_connect_uses_existing_registry_discovery(tmp_path, monkeypatch):
    from nooa import layered_config, llm_config

    user = tmp_path / "user"
    project = tmp_path / "project"
    user.mkdir()
    project.mkdir()
    manual = user / "llm_config.yaml"
    manual.write_text("models: {local: {model_name: openai/manual}}\n")
    connect.write(make_plan().entry, manual, alias="local")
    monkeypatch.setattr(llm_config, "bundled_config_paths", lambda: [])
    monkeypatch.setattr(layered_config, "get_user_dir", lambda name: user / name)
    monkeypatch.setattr(layered_config, "get_project_dir", lambda name: project / name)
    monkeypatch.delenv("NEMO_OO_LLM_CONFIG", raising=False)
    assert llm_config.llm_config_chain() == [manual]
    assert yaml.safe_load(manual.read_text())["models"]["local"] == make_plan().entry


@pytest.mark.parametrize(
    "style,patch,response",
    [
        (
            "chat",
            {"reasoning_effort": "high"},
            {"choices": [{"message": {"content": "323", "reasoning_content": "reason"}}]},
        ),
        ("chat", {"thinking": {"type": "enabled"}}, {"choices": [{"message": {"content": "323"}}]}),
        (
            "chat",
            {"chat_template_kwargs": {"enable_thinking": True}},
            {"choices": [{"message": {"content": "323"}}]},
        ),
        (
            "responses",
            {"reasoning": {"effort": "high"}},
            response_body("responses"),
        ),
        (
            "anthropic",
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
            response_body("anthropic"),
        ),
    ],
)
@pytest.mark.asyncio
async def test_real_httpx_serialization_keeps_level_blocks(monkeypatch, style, patch, response):
    if style == "chat" and "thinking" in patch:
        # This case specifically exercises LiteLLM's local rejection; direct
        # compatible transports forward unknown fields to the server.
        monkeypatch.setenv("NOOA_LLM_TRANSPORT", "litellm")
    monkeypatch.setenv("CONNECT_TEST_KEY", "private-test-key")
    sent = []

    async def handle(request):
        assert (
            request.headers.get("x-api-key") == "private-test-key"
            if style == "anthropic"
            else request.headers["authorization"] == "Bearer private-test-key"
        )
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response)

    mock_http(monkeypatch, handle)
    proposal = connect.plan(
        "local",
        "claude-sonnet-4-6" if style == "anthropic" else "gpt-5.1",
        style,
        "https://api.test/v1",
        "CONNECT_TEST_KEY",
        reasoning_levels={"high": patch},
    )
    result = await connect.run(proposal, approved="all")
    if style == "chat" and "thinking" in patch:
        # Explicitly preserving an incompatible field must not produce a green
        # probe after silently dropping it. The legacy SDK rejects this shape.
        assert len(sent) == 2
        assert result.entry["provenance"]["probes"]["level:high"]["outcome"] == "not_probed"
        return
    assert len(sent) == 3, result.entry["provenance"]["probes"]
    assert all(
        body.get("max_output_tokens", body.get("max_tokens", body.get("max_completion_tokens")))
        == proposal.entry["max_tokens"]
        for i, body in enumerate(sent)
    )
    assert all(sent[-1][key] == value for key, value in patch.items())
    assert result.entry["provenance"]["probes"]["level:high"]["client"] == "unifiedllm"
    assert "private-test-key" not in yaml.safe_dump(result.entry)


@pytest.mark.asyncio
async def test_budget_template_cannot_raise_approved_output_cap(monkeypatch):
    monkeypatch.setenv("CONNECT_TEST_KEY", "key")
    bodies = []

    async def post(self, url, **kwargs):
        bodies.append(kwargs["json"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "323"}}]})

    mock_post(monkeypatch, post)
    proposal = make_plan(
        reasoning_levels={
            "high": {"thinking": {"type": "enabled", "budget_tokens": 4096}, "max_tokens": 5120}
        }
    )
    result = await connect.run(proposal, approved="all")
    assert len(bodies) == 2
    assert result.entry["provenance"]["probes"]["level:high"]["outcome"] == "not_probed"


@pytest.mark.asyncio
async def test_alternate_cap_cannot_bypass_budget(monkeypatch):
    monkeypatch.setenv("CONNECT_TEST_KEY", "key")
    bodies = []

    async def post(self, url, **kwargs):
        bodies.append(kwargs["json"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "323"}}]})

    mock_post(monkeypatch, post)
    result = await connect.run(
        make_plan(
            reasoning_levels={"high": {"max_completion_tokens": 100000}},
            budget_tokens=131072,
        ),
        approved="all",
    )
    assert len(bodies) == 2
    assert result.entry["provenance"]["probes"]["level:high"]["outcome"] == "not_probed"


def test_fuzzy_match_finds_a_gateway_routed_model_by_its_catalogue_name():
    catalogue = [
        {"id": "anthropic/claude-opus-5"},
        {"id": "anthropic/claude-opus-4.5"},
        {"id": "openai/gpt-5.6"},
        {"id": "google/gemini-3-pro"},
    ]
    assert connect.match_models("aws/anthropic/bedrock-claude-opus-5", catalogue) == []
    matches = connect.fuzzy_match_models("aws/anthropic/bedrock-claude-opus-5", catalogue)
    ids = [item["id"] for item in matches]
    assert "anthropic/claude-opus-5" in ids
    assert len(matches) <= 3
    assert "openai/gpt-5.6" not in ids
    assert "google/gemini-3-pro" not in ids


def test_fuzzy_match_returns_nothing_below_the_cutoff():
    catalogue = [{"id": "openai/gpt-5.6"}, {"id": "google/gemini-3-pro"}]
    assert connect.fuzzy_match_models("totally-unrelated-vendor/made-up-model", catalogue) == []


def test_entry_loads_through_main_registry(tmp_path, monkeypatch):
    from nooa.unifiedllm import registry

    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "_loaded", False)
    monkeypatch.setenv("CONNECT_TEST_KEY", "key")
    entry = make_plan(catalogue={"id": "provider/model", "context_length": 12345}).entry
    path = tmp_path / "connected.yaml"
    connect.write(entry, path, alias="local")
    registry.reload_registry(path)
    with registry.get_llm_client("local") as client:
        assert client.model == "openai/gateway/model"
        assert client.context_window == 12345


def test_library_source_has_no_ui_or_provider_imports():
    import ast
    from pathlib import Path

    imported = []
    for source in Path(connect.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
    assert all(
        name.split(".")[0]
        not in {
            "nooa_cli",
            "click",
            "prompt_toolkit",
            "rich",
            "litellm",
            "openai",
            "anthropic",
            "textual",
        }
        for name in imported
    )


def test_library_plan_and_save_without_cli_dependencies(tmp_path):
    import subprocess
    import sys

    code = """
import importlib.abc
import sys
from pathlib import Path

# Isolate Connect's dependency boundary from the existing core runtime imports.
import nooa.unifiedllm

class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'nooa_cli', 'click', 'prompt_toolkit', 'rich', 'textual'}:
            raise AssertionError('Connect imported a frontend dependency: ' + fullname)

sys.meta_path.insert(0, NoFrontend())
from nooa.unifiedllm import connect
from nooa.unifiedllm.connect import _diagnostics, _session

proposal = connect.plan('local', 'example/model', 'chat', 'https://api.test/v1', '')
assert proposal.entry['max_tokens'] > 0
connect.write(proposal.entry, Path(sys.argv[1]), alias='local')
"""
    path = tmp_path / "models.yaml"
    result = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert yaml.safe_load(path.read_text())["models"]["local"]["max_tokens"] > 0


def test_catalogue_efforts_propose_complete_blocks_and_default():
    proposal = make_plan(
        catalogue={
            "id": "provider/model",
            "reasoning": {"supported_efforts": ["low", "high"], "default_effort": "high"},
        }
    )
    assert proposal.entry["reasoning_levels"] == {
        "low": {"reasoning_effort": "low"},
        "high": {"reasoning_effort": "high"},
    }
    assert proposal.entry["reasoning_default"] == "high"
    assert proposal.entry["provenance"]["reasoning_levels"]["source"] == "catalogue"


def test_missing_default_is_not_replaced_when_user_changes_levels():
    proposal = make_plan(
        catalogue={
            "id": "provider/model",
            "reasoning": {"supported_efforts": ["low", "high"], "default_effort": "high"},
        },
        reasoning_levels={"low": {"reasoning_effort": "low"}},
    )
    assert "reasoning_default" not in proposal.entry


@pytest.mark.parametrize(
    "levels", [[], 5, {True: {"thinking": True}}, {"high": {}}, {" ": {"thinking": True}}]
)
def test_bad_declarations_fail_before_probe(levels):
    with pytest.raises(ValueError):
        make_plan(reasoning_levels=levels)


def test_comment_only_registry_and_quoted_alias_round_trip(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("# Keep this note\n")
    connect.write({"model_name": "openai/m"}, path, alias="off")
    connect.write({"model_name": "openai/n"}, path, alias="off")
    assert path.read_text().startswith("# Keep this note\n")
    assert yaml.safe_load(path.read_text())["models"] == {
        "off": connect.configure_entry({"model_name": "openai/n"})
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", [{"n": 20}, {"stream": True}])
async def test_probe_cannot_multiply_generations_or_stream(monkeypatch, settings):
    calls = []

    async def post(self, url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(200, json={"choices": [{"message": {"content": "323"}}]})

    mock_post(monkeypatch, post)
    result = await connect.run(
        make_plan(reasoning_levels={"high": settings}), approved="all", api_key="test-key"
    )
    assert len(calls) == 2
    assert result.entry["provenance"]["probes"]["level:high"]["outcome"] == "not_probed"


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
def test_written_metadata_does_not_become_output_allocation(tmp_path, monkeypatch, style):
    from nooa.unifiedllm import registry

    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "_loaded", False)
    entry = connect.plan(
        "local",
        "wire/model",
        style,
        "https://api.test/v1",
        "",
        catalogue={"id": "provider/model", "top_provider": {"max_completion_tokens": 128000}},
        reasoning_levels={"low": {"reasoning_effort": "low"}},
    ).entry
    path = tmp_path / "models.yaml"
    connect.write(entry, path, alias="local")
    registry.reload_registry(path)
    with registry.get_llm_client("local", api_key="test-key") as client:
        assert "max_output_tokens" not in client.config
        assert client.config.get("max_tokens") != 128000
        assert client.reasoning_levels == ("low",)


@pytest.mark.asyncio
async def test_reconnect_keeps_observed_tools_without_spending_again(monkeypatch):
    calls = []

    async def post(self, url, **kwargs):
        calls.append(kwargs)
        data = response_body("chat")
        data["choices"][0]["finish_reason"] = "tool_calls"
        data["choices"][0]["message"]["tool_calls"] = [
            {
                "id": "probe-call",
                "type": "function",
                "function": {"name": "probe_tool", "arguments": "{}"},
            }
        ]
        return httpx.Response(200, json=data)

    mock_post(monkeypatch, post)
    first = await connect.run(make_plan(), approved="all", api_key="test-key")
    second = await connect.run(
        make_plan(existing_entry=first.entry), approved="all", api_key="test-key"
    )
    assert len(calls) == 2
    assert first.entry["provenance"]["probes"]["tools"]["tool_observed"] is True
    assert second.entry["provenance"]["probes"]["tools"]["tool_observed"] is True


@pytest.mark.asyncio
async def test_modified_plan_cannot_remove_the_required_output_cap(monkeypatch):
    async def post(*args, **kwargs):
        raise AssertionError("Uncapped request sent")

    mock_post(monkeypatch, post)
    proposal = make_plan()
    proposal.probes[0].body.pop("max_tokens")
    result = await connect.run(proposal, approved="minimal", api_key="test-key")
    assert result.entry["provenance"]["probes"]["routing"]["outcome"] == "not_probed"


@pytest.mark.asyncio
async def test_progress_events_arrive_before_and_after_each_request(monkeypatch):
    events, calls = [], []

    async def post(self, url, **kwargs):
        assert events[-1].outcome["outcome"] == "running"
        calls.append(kwargs)
        return httpx.Response(200, json={"choices": [{"message": {"content": "323"}}]})

    mock_post(monkeypatch, post)
    async for event in connect.run_steps(make_plan(), approved="all", api_key="test-key"):
        events.append(event)
    assert [(e.name, e.outcome["outcome"]) for e in events[:-1]] == [
        ("routing", "running"),
        ("routing", "accepted"),
        ("tools", "running"),
        ("tools", "accepted"),
    ]
    assert isinstance(events[-1], connect.ConnectResult)
    assert len(calls) == 2


def test_responses_style_does_not_enable_unprobed_explicit_caching():
    entry = connect.plan("local", "model", "responses", "https://api.test/v1", "").entry
    assert "cache_breakpoint" not in entry
    assert "cache_breakpoint" in entry["provenance"]["not_probed"]


@pytest.mark.asyncio
async def test_catalogue_missing_data_is_a_contract_error(monkeypatch):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={})), **kw
        ),
    )
    with pytest.raises(ValueError, match="model"):
        await connect.catalogue()


@pytest.mark.asyncio
async def test_closing_progress_iterator_before_send_closes_client(monkeypatch):
    from contextlib import aclosing

    closed = []
    real_client = httpx.AsyncClient

    class Client(real_client):
        async def __aexit__(self, *args):
            closed.append(True)
            return await super().__aexit__(*args)

        async def post(self, *args, **kwargs):
            raise AssertionError("Cancelled before dispatch")

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    async with aclosing(connect.run_steps(make_plan(), approved="minimal", api_key="key")) as steps:
        event = await anext(steps)
        assert event.outcome["outcome"] == "running"
    assert closed == []  # No runtime client exists until dispatch starts.
