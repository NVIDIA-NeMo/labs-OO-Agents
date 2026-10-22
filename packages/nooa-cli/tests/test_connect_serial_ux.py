# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small offline whole-flow regressions for the serialized human UX policies."""

import asyncio
import json
from copy import deepcopy

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands.connect import command
from pydantic import BaseModel

from nooa import Agent, llm_config, strategy
from nooa.strategies.predict import PredictStrategy
from nooa.unifiedllm import connect, registry
from nooa.unifiedllm.connect._reasoning import native_levels, reasoning_metadata
from tests.unifiedllm.connect.connect_http import mock_http, response_body


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("UX_KEY", "PRIVATE-KEY")
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "_loaded", False)

    async def catalogue():
        return []

    monkeypatch.setattr(connect, "catalogue", catalogue)


def native_model(manual=False, structured=True, cap=8192, context=32768):
    return {
        "id": "claude-opus-4-7",
        "max_input_tokens": context,
        "max_tokens": cap,
        "capabilities": {
            "structured_outputs": {"supported": structured},
            "thinking": {
                "supported": True,
                "types": {
                    "enabled": {"supported": manual},
                    "adaptive": {"supported": not manual},
                    "disabled": {"supported": True},
                },
            },
            "effort": {"supported": not manual, "high": {"supported": not manual}},
        },
    }


def args(path, style="anthropic", reply=4096):
    return [
        "claude-opus-4-7",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        style,
        "--api-key-env",
        "UX_KEY",
        "--media-inputs",
        "",
        "--reply-tokens",
        str(reply),
        "--probe",
        "minimal",
        "--output",
        str(path),
    ]


class Answer(BaseModel):
    value: int


class OrdinaryPredict(Agent):
    @strategy(PredictStrategy())
    async def answer(self) -> Answer:
        """Return the integer 323."""
        ...


def test_native_metadata_default_ui_predict_save_reload(monkeypatch, tmp_path):
    calls = []

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [native_model()]})
        return httpx.Response(200, json=response_body("anthropic"))

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(command, args(path), input="high\ntest\ny\n")
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["structured_output_mode"] == "anthropic"
    assert entry["reasoning_level"] == "high" and entry["max_tokens"] == 4096
    assert "PRIVATE-KEY" not in result.output
    registry.reload_registry(path)
    client = registry.get_llm_client("local")
    try:
        assert asyncio.run(OrdinaryPredict(llm=client).answer()).value == 323
    finally:
        asyncio.run(client.aclose())
    posts = [json.loads(r.content) for r in calls if r.method == "POST"]
    assert len(posts) == 2
    for body in posts:
        assert body["output_config"]["format"]["schema"]["properties"]["value"]["type"] == "integer"
        assert body["output_config"]["effort"] == "high"
        assert body["thinking"] == {"type": "adaptive"}
        assert body["max_tokens"] == 4096
        assert not {"tools", "tool_choice", "response_format"} & body.keys()


@pytest.mark.parametrize("mode", [None, "auto", "anthropic"])
def test_explicit_profile_not_recommended_over(monkeypatch, mode):
    model = native_model()
    kwargs = {"endpoint_model": model, "endpoint_models_url": "https://api.test/v1/models"}
    proposal = connect.plan(
        "local",
        model["id"],
        "anthropic",
        "https://api.test/v1",
        "",
        structured_output_mode=mode,
        **kwargs,
    )
    assert proposal.entry["structured_output_mode"] == mode
    assert not proposal.entry["provenance"]["structured_output_profile"].get("recommended")


@pytest.mark.parametrize(
    "style, source, supported",
    [
        ("chat", "https://api.test/v1/models", True),
        ("anthropic", "https://foreign.test/models", True),
        ("anthropic", "https://api.test/v1/models", False),
        ("responses", "https://api.test/v1/models", True),
    ],
)
def test_unknown_or_foreign_metadata_never_invents_native(style, source, supported):
    model = native_model(structured=supported)
    entry = connect.plan(
        "local",
        model["id"],
        style,
        "https://api.test/v1",
        "",
        endpoint_model=model,
        endpoint_models_url=source,
    ).entry
    assert "structured_output_mode" not in entry


def test_bounded_structured_capability_and_provenance():
    model = native_model()
    model["capabilities"]["structured_outputs"]["secret"] = "DO-NOT-COPY"
    clean = reasoning_metadata(model)
    assert clean["capabilities"]["structured_outputs"] == {"supported": True}
    entry = connect.plan(
        "local",
        model["id"],
        "anthropic",
        "https://api.test/v1",
        "",
        endpoint_model=model,
        endpoint_models_url="https://api.test/v1/models",
    ).entry
    profile = entry["provenance"]["structured_output_profile"]
    assert profile["source"] == "https://api.test/v1/models" and profile["recommended"]
    assert connect.configure_entry(entry)["provenance"]["structured_output_profile"] == profile


@pytest.mark.parametrize(
    "manual, selection, expected",
    [(True, "on", "on"), (True, "off", "off"), (False, "high", "high")],
)
def test_manual_reasoning_ui_save_reload(monkeypatch, tmp_path, manual, selection, expected):
    calls = []

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(
                200, json={"data": [native_model(manual=manual, structured=False)]}
            )
        return httpx.Response(200, json=response_body("anthropic"))

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        args(path),
        input=selection + ("\n1024" if selection == "on" else "") + "\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert all(r.method == "GET" for r in calls)
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["reasoning_level"] == expected
    if selection == "on":
        assert entry["reasoning_levels"]["on"] == {
            "thinking": {"type": "enabled", "budget_tokens": 1024}
        }
    if not manual:
        assert "Reasoning token allowance" not in result.output
    registry.reload_registry(path)
    client = registry.get_llm_client("local")
    try:
        asyncio.run(client.acall([{"role": "user", "content": "hi"}]))
    finally:
        asyncio.run(client.aclose())
    body = json.loads(calls[-1].content)
    assert body["thinking"]["type"] == (
        "enabled" if selection == "on" else "disabled" if selection == "off" else "adaptive"
    )
    assert body["max_tokens"] == entry["max_tokens"] == 4096
    if selection == "on":
        assert body["thinking"]["budget_tokens"] == 1024


@pytest.mark.parametrize("action", ["cancel", "adjust"])
def test_manual_on_small_explicit_cap_explains_adjust_or_cancel(monkeypatch, tmp_path, action):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        return httpx.Response(200, json={"data": [native_model(manual=True, structured=False)]})

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    answers = "on\n1024\n" + action + ("\ncustom\n2048\nsave\ny\n" if action == "adjust" else "\n")
    result = CliRunner().invoke(command, args(path, reply=1024), input=answers)
    assert result.exit_code == 0, result.output
    assert "No cap was changed" in result.output
    if action == "cancel":
        assert not path.exists()
    else:
        entry = yaml.safe_load(path.read_text())["models"]["local"]
        assert entry["max_tokens"] == 2048
        assert "max_tokens" not in entry["reasoning_levels"]["on"]


@pytest.mark.parametrize("selection", ["default", "off"])
def test_inactive_manual_on_does_not_raise_small_cap(monkeypatch, tmp_path, selection):
    def handle(request):
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={"data": [native_model(manual=True, structured=False, cap=1024, context=1536)]},
        )

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(command, args(path, reply=512), input=selection + "\nsave\ny\n")
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["max_tokens"] == 512 and "on" not in entry.get("reasoning_levels", {})


def test_native_enabled_false_has_no_manual_ui():
    model = native_model(manual=False)
    levels = native_levels(
        model,
        model_id=model["id"],
        api_style="anthropic",
        endpoint="https://api.test/v1",
        source="https://api.test/v1/models",
    )
    assert set(levels) == {"high", "off"}


@pytest.mark.parametrize("status", [400, 401, 500])
def test_yes_approved_failure_nonzero_and_no_key_or_registry(monkeypatch, tmp_path, status):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "PRIVATE-KEY private failure"}})

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(command, args(path, style="chat") + ["--no-catalogue", "--yes"])
    assert result.exit_code != 0 and not path.exists()
    assert "PRIVATE-KEY" not in result.output and "private failure" not in result.output
    assert len(calls) == 1


def test_unknown_chat_failure_plain_format_recovery_cumulative_budget(monkeypatch, tmp_path):
    posts = []

    def handle(request):
        body = json.loads(request.content)
        posts.append(body)
        if len(posts) == 1:
            assert "response_format" in body
            return httpx.Response(400, json={"error": {"message": "carrier rejected"}})
        assert "output_config" in body and "response_format" not in body
        return httpx.Response(200, json=response_body("chat"))

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        args(path, style="chat") + ["--no-catalogue", "--budget-tokens", "9216"],
        input="32768\n32768\ntest\nstructured\nanthropic\ntest\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(posts) == 2
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["structured_output_mode"] == "anthropic"
    assert "Change answer format" in result.output


def test_retry_exhaustion_sends_zero_extra_posts(monkeypatch, tmp_path):
    posts = []

    def handle(request):
        posts.append(request)
        return httpx.Response(400, json={"error": {"message": "failed"}})

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        args(path, style="chat") + ["--no-catalogue", "--budget-tokens", "4608"],
        input="32768\n32768\ntest\nreview\ntest\ncancel\n",
    )
    assert result.exit_code != 0, result.output
    assert len(posts) == 1 and not path.exists()
    assert "budget used up" in result.output


@pytest.mark.parametrize("save_untested", [False, True])
def test_discovery_failure_changed_unknown_route_requires_new_manual_review(
    monkeypatch, tmp_path, save_untested
):
    events = []
    original = prompts.labelled_choice

    def choose(label, options, **kw):
        answer = original(label, options, **kw)
        events.append((label, answer))
        return answer

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    calls = []

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            model = (
                {"id": "A", "max_input_tokens": 99999, "max_output_tokens": 8192}
                if request.url.host == "a.test"
                else {"id": "B"}
            )
            return httpx.Response(200, json={"data": [model]})
        if request.url.host == "a.test":
            return httpx.Response(400, json={"error": {"message": "failure"}})
        reviews = [
            answer for label, answer in events if label == "Would you like to test these settings?"
        ]
        assert reviews == ["test", "test"] or reviews == ["test", "test", "test"]
        assert "PRIVATE-KEY" not in str(request.headers)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=response_body("chat"))
        return httpx.Response(400, json={"error": {"message": "unsupported"}})

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    flags = [
        "A",
        "--as",
        "local",
        "--endpoint",
        "https://a.test/v1",
        "--api-key-env",
        "UX_KEY",
        "--media-inputs",
        "",
        "--reply-tokens",
        "4096",
        "--probe",
        "minimal",
        "--output",
        str(path),
    ]
    answers = "discover\ntest\nserver\nhttps://b.test/v1\nB\n\n32768\n16384\n" + (
        "save\ny\nchat\nsave\ny\n" if save_untested else "test\ntest\ny\n"
    )
    result = CliRunner().invoke(command, flags, input=answers)
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["model_name"] == "openai/B" and entry["context_window"] == 16384
    assert entry["api_key_env"] == "" and entry["max_tokens"] == 4096
    assert result.output.count("Context window (tokens)") == 1
    assert "Previous source, context and test proof no longer apply" in " ".join(
        result.output.split()
    )
    assert result.output.count("Would you like to test these settings?") == 3
    bposts = [r for r in calls if r.method == "POST" and r.url.host == "b.test"]
    assert not bposts if save_untested else bool(bposts)
    assert all(json.loads(r.content)["model"] == "B" for r in bposts)


class OrdinaryCodeAct(Agent):
    async def answer(self) -> Answer:
        """Return the integer 323."""
        ...


def test_default_agent_uses_saved_native_profile_without_forcing_a_typed_carrier(
    monkeypatch, tmp_path
):
    calls = []
    runtime = False

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [native_model()]})
        body = json.loads(request.content)
        if not runtime:
            return httpx.Response(200, json=response_body("anthropic"))
        tool = next(t for t in body["tools"] if t["name"] in {"execute_python", "python_cell"})
        data = response_body("anthropic")
        data["stop_reason"] = "tool_use"
        data["content"] = [
            {
                "type": "tool_use",
                "id": "cell",
                "name": tool["name"],
                "input": {"code": "return_result(Answer(value=323))"},
            }
        ]
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(command, args(path), input="high\ntest\ny\n")
    assert result.exit_code == 0, result.output
    registry.reload_registry(path)
    client = registry.get_llm_client("local")
    runtime = True
    agent = OrdinaryCodeAct(llm=client)
    try:
        assert asyncio.run(agent.answer()).value == 323
    finally:
        asyncio.run(agent.aclose())
        asyncio.run(client.aclose())
    body = json.loads(calls[-1].content)
    assert body["tools"] and "format" not in body.get("output_config", {})
    assert body["output_config"]["effort"] == "high" and body["max_tokens"] == 4096


@pytest.mark.parametrize("mode", ["none", "auto", "anthropic"])
def test_plain_native_recommendation_can_be_overridden_explicitly(monkeypatch, tmp_path, mode):
    def handle(request):
        assert request.method == "GET"
        return httpx.Response(200, json={"data": [native_model()]})

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, args(path), input="default\nstructured\n" + mode + "\nsave\ny\n"
    )
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["structured_output_mode"] == (None if mode == "none" else mode)
    assert "Answer format" in result.output


def test_manual_on_baseline_uses_validated_budget_and_current_fingerprint(monkeypatch, tmp_path):
    calls, entries = [], []
    original = connect.run_steps

    async def capture(proposal, **kwargs):
        async for event in original(proposal, **kwargs):
            if isinstance(event, connect.ConnectResult):
                entries.append(deepcopy(event.entry))
            yield event

    monkeypatch.setattr(connect, "run_steps", capture)

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [native_model(manual=True)]})
        return httpx.Response(200, json=response_body("anthropic"))

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(command, args(path), input="on\n2048\ntest\ny\n")
    assert result.exit_code == 0, result.output
    entry = entries[0]
    record = entry["provenance"]["probes"]["routing"]
    assert connect.routing_allows_save(entry, entry)
    snapshot = connect.effective_config_snapshot(entry)
    assert snapshot["level"] == "on" and snapshot["params"]["thinking"]["budget_tokens"] == 2048
    assert record["verification_config"] == snapshot
    saved = yaml.safe_load(path.read_text())["models"]["local"]
    assert saved["reasoning_levels"]["on"] == {
        "thinking": {"type": "enabled", "budget_tokens": 2048}
    }
    assert saved["max_tokens"] == 4096 and saved["reasoning_level"] == "on"
    assert len([r for r in calls if r.method == "POST"]) == 1


@pytest.mark.parametrize("allowance", ["1023", "invalid"])
def test_manual_on_invalid_allowance_never_posts_or_writes(monkeypatch, tmp_path, allowance):
    def handle(request):
        assert request.method == "GET"
        return httpx.Response(200, json={"data": [native_model(manual=True)]})

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(command, args(path), input="on\n" + allowance + "\n")
    assert result.exit_code != 0 and not path.exists()
    assert "at least 1,024" in result.output


def test_identity_unchanged_native_base_does_not_reprepare_metadata(monkeypatch):
    state = wizard.WizardState(
        model="A",
        api_style="anthropic",
        endpoint="https://api.test/v1",
        api_key_env="UX_KEY",
        tokens_spent=1234,
    )
    marker = object()
    state.server_details = marker
    monkeypatch.setattr(
        wizard, "prepare_metadata", lambda s: pytest.fail("same effective identity")
    )
    assert wizard.review_replacement_route(state, ("https://api.test", "A"))
    assert state.server_details is marker and state.tokens_spent == 1234


def test_route_replacement_preserves_explicit_ui_format_but_not_old_facts(monkeypatch):
    state = wizard.WizardState(
        model="B",
        api_style="chat",
        endpoint="https://api.test/v1",
        budget_tokens=10000,
        output_tokens=2048,
        reasoning_output_tokens=4096,
        approval="minimal",
        probe="minimal",
        guided=True,
        api_key_env="",
        media_inputs=[],
        reply_tokens=4096,
        candidate={"id": "A", "context_length": 999999},
    )
    state.structured_output_mode = "anthropic"  # choice made after constructing the wizard
    state.structured_output_mode_set = True
    state.server_details = object()
    state.metadata_prepared = state.metadata_displayed = True
    state.tokens_spent = 4608

    def prep(s):
        assert s.server_details is None and s.candidate is None
        assert not s.metadata_displayed and not s.metadata_prepared
        s.manual_limits = {"context_window": 32768, "max_input_tokens": 16384}
        s.context_window = 16384
        return True

    monkeypatch.setattr(wizard, "prepare_metadata", prep)
    monkeypatch.setattr(prompts, "labelled_choice", lambda label, opts, **kw: "test")
    assert wizard.review_replacement_route(state, ("https://api.test/v1", "A"))
    assert state.proposal.entry["structured_output_mode"] == "anthropic"
    assert state.proposal.entry["context_window"] == 16384
    assert state.proposal.budget_tokens == 5392 and state.tokens_spent == 4608
    assert not state.proposal.entry["provenance"]["probes"]


@pytest.mark.parametrize("action", ["cancel", "adjust"])
def test_manual_on_reselection_keeps_owned_budget_and_never_adds_level_cap(
    monkeypatch, tmp_path, action
):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        return httpx.Response(200, json={"data": [native_model(manual=True, structured=False)]})

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    answers = "on\n2048\nchange\n32768\n8192\ncustom\n1024\non\n\n" + action
    answers += "\ncustom\n4096\nsave\ny\n" if action == "adjust" else "\n"
    result = CliRunner().invoke(command, args(path), input=answers)
    assert result.exit_code == 0, result.output
    assert "No cap was changed" in result.output
    if action == "cancel":
        assert not path.exists()
    else:
        entry = yaml.safe_load(path.read_text())["models"]["local"]
        assert entry["max_tokens"] == 4096
        assert entry["reasoning_levels"]["on"] == {
            "thinking": {"type": "enabled", "budget_tokens": 2048}
        }


def test_keep_explicit_on_from_native_metadata_still_validates_wizard_budget(monkeypatch, tmp_path):
    def handle(request):
        assert request.method == "GET"
        return httpx.Response(200, json={"data": [native_model(manual=True, structured=False)]})

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, args(path, reply=1024) + ["--reasoning-level", "on"], input="keep\n1024\ncancel\n"
    )
    assert result.exit_code == 0, result.output
    assert "Reasoning token allowance" in result.output and "No cap was changed" in result.output
    assert not path.exists()


def test_discovered_native_carrier_requires_current_review_without_reasoning_metadata(
    monkeypatch, tmp_path
):
    calls, reviews = [], []
    original = prompts.labelled_choice

    def choose(label, options, **kw):
        answer = original(label, options, **kw)
        if label == "Would you like to test these settings?":
            reviews.append(answer)
        return answer

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    model = native_model()
    model["capabilities"] = {"structured_outputs": {"supported": True}}

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [model]})
        if request.url.path == "/v1/messages":
            body = json.loads(request.content)
            if "format" in body.get("output_config", {}):
                assert reviews == [
                    "test",
                    "test",
                ]  # native carrier was reviewed before configured POST
            return httpx.Response(200, json=response_body("anthropic"))
        return httpx.Response(400, json={"error": {"message": "unsupported"}})

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    flags = args(path)
    index = flags.index("--api-style")
    del flags[index : index + 2]
    result = CliRunner().invoke(command, flags, input="discover\ntest\ntest\ny\n")
    assert result.exit_code == 0, result.output
    assert len(reviews) == 2
    assert result.output.count("Model details\n") == 1
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["structured_output_mode"] == "anthropic"
