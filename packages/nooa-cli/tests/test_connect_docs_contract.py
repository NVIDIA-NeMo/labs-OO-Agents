# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable documentation contracts: actual Click/library/SDKs, offline HTTP.

Shell examples are parsed, never executed by a shell. Redirects are emulated in
an isolated workspace. Only human wizard prompts are answered deterministically;
agent stages must not prompt. Shared SDK HTTP fixtures retain the real clients.
"""

import json
import re
import shlex
import socket
import urllib.error
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands import _connect_stages as stages
from nooa_cli.commands.connect import command

from nooa import agent as agent_module
from nooa import llm_config, tracing
from nooa.skill import TextSkill
from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body

ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT / "skills/nooa-connect/SKILL.md"
GUIDE = ROOT / "docs/model-connect.md"
REPORT_KEYS = {
    "version",
    "stage",
    "ok",
    "data",
    "error",
    "checks",
    "warnings",
    "run_context",
    "diagnostic_prompt",
}
BASE = [
    "your-model",
    "--as",
    "work",
    "--endpoint",
    "https://gateway.example/v1",
    "--api-style",
    "chat",
    "--api-key-env",
    "MODEL_KEY",
    "--max-tokens",
    "1024",
    "--budget-tokens",
    "65536",
]


def shell_commands(path):
    """Enumerate every shell fence, including multi-command blocks and exports."""
    commands = []
    for block in re.findall(r"```(?:sh|bash)\n(.*?)```", path.read_text(), re.S):
        for line in block.replace("\\\n", " ").splitlines():
            tokens = shlex.split(line, comments=True)
            if tokens:
                commands.append(tokens)
    return commands


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for variable, path in {
        "HOME": tmp_path / "home",
        "XDG_CONFIG_HOME": tmp_path / "config",
        "NEMO_OO_USER_DIR": tmp_path / "user",
        "NEMO_OO_PROJECT_DIR": tmp_path / ".nooa",
        "NEMO_OO_LLM_CONFIG": tmp_path / "override.yaml",
        "NEMO_OO_SECRETS": tmp_path / "secrets.yaml",
    }.items():
        monkeypatch.setenv(variable, str(path))
    (tmp_path / "override.yaml").write_text("models: {}\n")
    (tmp_path / "secrets.yaml").write_text("env: {}\n")
    for variable in [
        "MODEL_KEY",
        "MY_MODEL_KEY",
        "DECISION_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
    ]:
        monkeypatch.setenv(variable, "dummy-docs-key")
    monkeypatch.setattr(llm_config, "bundled_config_paths", lambda: [])
    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("NOOA_LLM_TRANSPORT", raising=False)
    monkeypatch.setattr(agent_module, "_auto_tracing_attempted", False)
    monkeypatch.setattr(tracing, "_enabled", False)
    monkeypatch.setattr(tracing, "_probe_failed", False)
    # Trace health I/O is independent of SDK HTTP. No viewer and no file fallback.
    monkeypatch.setattr(tracing, "resolve_otlp_endpoint", Mock(return_value=None))
    monkeypatch.setattr(
        tracing.exporters_mod, "journal", Mock(side_effect=AssertionError("exporter"))
    )
    monkeypatch.setattr(
        tracing.urllib.request, "urlopen", Mock(side_effect=AssertionError("urllib I/O"))
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Documentation contracts must never open a network socket")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    sent = []

    def handle(request):
        sent.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "your-model"}]})
        body = json.loads(request.content)
        if "questions" in body:
            if request.url.path.endswith("systemone") or "alpha/decisions" in request.url.path:
                data = {"model": body["model"], "answers": {"supported": {"noul": 0.7}}}
            else:
                data = {
                    "model": body["model"],
                    "answers": [{"type": "predicate", "name": "supported", "probability": 0.7}],
                }
            return httpx.Response(200, json=data)
        style = (
            "responses"
            if request.url.path.endswith("/responses")
            else "anthropic"
            if request.url.path.endswith("/messages")
            else "chat"
        )
        text = '{"value":323}'
        if "Can you access and interpret" in json.dumps(body):
            text = '{"value":true}'
        if connect.REASONING_CHECK_PROMPT in json.dumps(body):
            text = "B G D A C E F H"
        data = response_body(style, text)
        if style == "chat":
            choice = data["choices"][0]
            choice["message"]["reasoning_content"] = "private mock reasoning"
            data["usage"]["prompt_tokens_details"] = {"cached_tokens": 15}
            if any(tool["function"]["name"] == "probe_tool" for tool in body.get("tools", [])):
                choice["finish_reason"] = "tool_calls"
                choice["message"]["tool_calls"] = [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "probe_tool", "arguments": '{"value":"ok"}'},
                    }
                ]
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    return sent


def invoke(args):
    result = CliRunner().invoke(command, args, input="")
    return result


def report(result, code=0):
    assert result.exit_code == code, result.output
    data = json.loads(result.stdout)
    assert set(data) == REPORT_KEYS
    assert data["version"] == 1
    assert data["ok"] is (code == 0)
    assert "dummy-docs-key" not in result.output
    assert "private mock reasoning" not in result.output
    return data


def approved_human_answers(monkeypatch):
    """Bounded human input only; preserve wizard execution and its real checks."""
    asked = []

    def answer(text, **kwargs):
        asked.append(text)
        assert len(asked) < 40, "Unexpected wizard loop"
        if text == "Choose a provider":
            return "custom"
        if text == "Model server URL":
            return "https://gateway.example/v1"
        if text.startswith("Key environment variable"):
            return "MODEL_KEY"
        if text.startswith("Search advertised") or text.startswith("Exact model"):
            return "your-model"
        if text == "API format":
            return "chat"
        if text == "Save this model as":
            return "wizard-work"
        assert "default" in kwargs, (text, kwargs)
        return kwargs["default"]

    monkeypatch.setattr(prompts, "prompt", answer)
    monkeypatch.setattr(prompts, "confirm", lambda text, **kw: True)
    monkeypatch.setattr(prompts, "choose_reply_limit", lambda suggested, *args, **kw: suggested)
    monkeypatch.setattr(prompts, "choose_media_mode", lambda **kw: "manual")
    monkeypatch.setattr(prompts, "choose_media_inputs", lambda current=(): list(current))
    monkeypatch.setattr(prompts, "review_media_inputs", lambda current=(): list(current))
    return asked


@pytest.mark.parametrize("path,expected", [(SKILL, 9), (GUIDE, 10)], ids=["skill", "guide"])
def test_every_documented_shell_command(path, expected, offline, monkeypatch, tmp_path):
    commands = shell_commands(path)
    assert len(commands) == expected, "Review new/removed shell examples explicitly"
    executed = []
    for index, tokens in enumerate(commands):
        # Each standalone human example owns a fresh isolated registry.
        if tokens[0] == "export":
            assert tokens == ["export", "NEMO_OO_LLM_CONFIG=$PWD/llm_config.yaml"]
            monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(tmp_path / "llm_config.yaml"))
            assert (tmp_path / "llm_config.yaml").is_file()
            executed.append(index)
            continue
        assert tokens[:4] == ["uv", "run", "nooa", "connect"]
        args = tokens[4:]
        redirect = None
        if ">" in args:
            offset = args.index(">")
            assert len(args[offset:]) == 2
            redirect = Path(args[offset + 1])
            args = args[:offset]
        before = len(offline)
        if "--stage" not in args and "--help" not in args:
            monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / f"wizard-{index}"))
            with monkeypatch.context() as human:
                approved_human_answers(human)
                result = invoke(args)
            assert result.exit_code == 0, result.output
            registry = tmp_path / f"wizard-{index}/llm_config.yaml"
            assert registry.exists(), result.output
            assert len(offline) > before, "Wizard must run its actual documented checks"
            entries = yaml.safe_load(registry.read_text())["models"]
            assert entries
        else:
            # Agent examples must not read stdin or prompt.
            with monkeypatch.context() as headless:
                headless.setattr(
                    prompts, "prompt", Mock(side_effect=AssertionError("stage prompt"))
                )
                headless.setattr(
                    prompts, "confirm", Mock(side_effect=AssertionError("stage confirmation"))
                )
                result = invoke(args)
            if "--help" in args:
                assert result.exit_code == 0
                assert all(
                    f"--{option}" in result.stdout for option in ["stage", "api-style", "input"]
                )
            else:
                data = report(result)
                stage = args[args.index("--stage") + 1]
                assert data["stage"] == stage
                if stage in {"plan", "save"}:
                    assert len(offline) == before
                else:
                    assert len(offline) > before
                if stage == "routing":
                    assert data["checks"]["routing"]["outcome"] == "accepted"
                    if data["data"]["entry"].get("client_type") != "decision":
                        assert data["checks"]["routing"]["typed_result_validated"] is True
                        assert data["checks"]["routing"]["answer_correct"] is True
                if stage == "save":
                    assert Path(data["data"]["path"]).exists()
        if redirect:
            redirect.write_text(result.stdout)
            json.loads(redirect.read_text())
        executed.append(index)
    assert executed == list(range(len(commands)))


@pytest.mark.parametrize(
    "stage,count",
    [
        ("discover", 1),
        ("catalogue", 1),
        ("interfaces", 3),
        ("plan", 0),
        ("routing", 1),
        ("tools", 1),
        ("reasoning", 2),
        ("media", 3),
        ("session", 3),
        ("all", 9),
        ("save", 0),
    ],
)
def test_every_stage_independent(stage, count, offline, tmp_path):
    assert stage in stages.STAGES
    levels = tmp_path / "levels.yaml"
    levels.write_text("high: {reasoning_effort: high}\nlow: {reasoning_effort: low}\n")
    args = [*BASE, "--stage", stage]
    if stage in {"reasoning", "all"}:
        args += ["--levels-file", str(levels)]
    if stage in {"media", "all"}:
        args += ["--test-media-inputs", "image,pdf"]
    if stage == "save":
        plan = invoke([*BASE, "--stage", "plan"])
        report(plan)
        source = tmp_path / "proposal.json"
        source.write_text(plan.stdout)
        args = ["--stage", "save", "--input", str(source), "--output", "models.yaml"]
    data = report(invoke(args))
    assert len(offline) == count
    if stage in {"routing", "tools", "reasoning", "media", "session", "all"}:
        assert data["checks"]
    if stage == "media":
        assert data["data"]["entry"]["media_inputs"] == ["image", "pdf"]


def test_documented_stage_table_covers_enum():
    for path in [SKILL, GUIDE]:
        table = set(re.findall(r"^\| `([a-z]+)` \|", path.read_text(), re.M))
        assert table == set(stages.STAGES)


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
def test_explicit_chat_family_interfaces_still_compares_three(style, offline):
    args = BASE.copy()
    args[args.index("chat")] = style
    data = report(invoke([*args, "--stage", "interfaces", "--output-tokens", "200"]))
    assert data["data"]["accepted"] == ["chat", "responses", "anthropic"]
    assert [request.url.path for request in offline] == [
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/messages",
    ]
    assert "entry" not in data["data"]


@pytest.mark.parametrize("style", ["systemone", "openai"])
@pytest.mark.parametrize("stage", ["plan", "interfaces", "routing", "all", "save"])
def test_native_decision_contract(style, stage, offline, tmp_path):
    base = [
        "decision-model",
        "--as",
        "decisions",
        "--endpoint",
        "https://decision.example/v1/" + ("systemone" if style == "systemone" else "decisions"),
        "--api-style",
        style,
        "--api-key-env",
        "DECISION_KEY",
        "--budget-tokens",
        "4096",
    ]
    args = [*base, "--stage", stage]
    if stage == "save":
        source = tmp_path / "decision-plan.json"
        source.write_text(invoke([*base, "--stage", "plan"]).stdout)
        args = ["--stage", "save", "--input", str(source), "--output", "decision.yaml"]
    data = report(invoke(args))
    assert len(offline) == (0 if stage in {"plan", "save"} else 1)
    if offline:
        body = json.loads(offline[0].content)
        if style == "openai":
            assert body["questions"][0]["type"] == "predicate"
        else:
            assert "supported" in body["questions"]
        assert "max_tokens" not in body and "max_output_tokens" not in body
    if stage in {"plan", "routing", "all", "save"}:
        entry = data["data"]["entry"]
        assert entry["client_type"] == "decision" and entry["model_name"] == "decision-model"
        assert "max_tokens" not in entry


@pytest.mark.parametrize("failed", [False, True], ids=["untested-plan", "failed-result"])
def test_permissive_save_and_authorized_whole_alias_replacement(
    failed, offline, tmp_path, monkeypatch
):
    if failed:

        async def reject(client, request, **kwargs):
            offline.append(request)
            return httpx.Response(
                401, request=request, json={"error": {"message": "denied dummy-docs-key"}}
            )

        monkeypatch.setattr(httpx.AsyncClient, "send", reject)
        plan = report(invoke([*BASE, "--stage", "routing"]), 1)
    else:
        plan = report(invoke([*BASE, "--stage", "plan"]))
    before_save = len(offline)
    target = tmp_path / "models.yaml"
    target.write_text(
        "# approved registry\nmodels:\n  other: {model_name: openai/keep, custom: retained}\n  work: {model_name: openai/old, custom: needs-review}\n"
    )
    original = target.read_text()
    source = tmp_path / "proposal.json"
    source.write_text(json.dumps(plan))
    args = ["--stage", "save", "--input", str(source), "--output", str(target)]
    report(invoke(args), 2)
    assert target.read_text() == original
    result = invoke([*args, "--yes"])
    report(result)
    assert "replacing alias" in result.stderr
    assert ("checks did not pass" in result.stderr) is failed
    saved = yaml.safe_load(target.read_text())["models"]
    assert saved["other"] == {"model_name": "openai/keep", "custom": "retained"}
    assert "# approved registry" in target.read_text()
    assert "custom" not in saved["work"], "Fresh plans replace, not merge selected aliases"
    # Carrying forward an approved field requires an explicit reviewed input.
    plan["data"]["entry"]["custom"] = "reviewed"
    source.write_text(json.dumps(plan))
    report(invoke([*args, "--yes"]))
    assert yaml.safe_load(target.read_text())["models"]["work"]["custom"] == "reviewed"
    assert len(offline) == before_save


@pytest.mark.parametrize(
    "extra",
    [
        ["--input", "prior.json"],
        ["--output", "registry.yaml"],
        ["--working-dir", "."],
        ["--provider", "openai"],
        ["--catalogue-model", "model"],
        ["--edit-model", "work"],
        ["--levels", "high"],
        ["--no-probe"],
        ["--probe", "minimal"],
        ["--show-config"],
        ["--reasoning-template", "effort"],
        ["--reasoning-level", "high"],
        ["--catalogue-file", "prior.json"],
        ["--models-url", "https://listing.example"],
        ["--test-media-inputs", "image", "--media-inputs", "image"],
    ],
)
def test_documented_option_incompatibilities(extra, offline, tmp_path):
    (tmp_path / "prior.json").write_text("{}")
    result = invoke([*BASE, "--stage", "routing", *extra])
    assert result.exit_code == 2, result.output
    assert not offline


def test_checks_build_fresh_plans_and_do_not_accumulate_evidence(offline, tmp_path):
    first = report(invoke([*BASE, "--stage", "tools"]))
    assert set(first["checks"]) == {"tools"}
    second = report(invoke([*BASE, "--stage", "routing", "--reply-tokens", "512"]))
    assert set(second["checks"]) == {"routing"}
    assert second["data"]["entry"]["max_tokens"] == 512
    assert json.loads(offline[-1].content)["max_tokens"] == 512
    assert {path.name for path in tmp_path.glob("*.yaml")} == {"override.yaml", "secrets.yaml"}


def test_no_catalogue_flag_does_not_disable_catalogue_stage(offline):
    report(invoke([*BASE, "--stage", "catalogue", "--no-catalogue"]))
    assert len(offline) == 1 and offline[0].method == "GET"


def test_connect_skill_loads_as_text_skill_and_links_resolve():
    skill = TextSkill(path=SKILL.parent)
    assert skill.id == "nooa-connect"
    assert "permission-first" in skill.description
    assert "--stage" in type(skill).__doc__
    assert skill.read_file("SKILL.md") == SKILL.read_text()
    paths = [
        SKILL,
        GUIDE,
        ROOT / "AGENTS.md",
        ROOT / "docs/media.md",
        ROOT / "skills/README.md",
        *[
            ROOT / "skills" / name / "SKILL.md"
            for name in [
                "nooa-agent-authoring",
                "nooa-media",
                "nooa-model-configuration",
                "nooa-tools-and-skills",
            ]
        ],
    ]
    for path in paths:
        for destination in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            if "://" in destination or destination.startswith("#"):
                continue
            local, _, anchor = destination.partition("#")
            linked = (path.parent / local).resolve()
            assert linked.exists(), (path.relative_to(ROOT), destination)
            if anchor and linked.suffix == ".md":
                headings = re.findall(r"^#+ (.+)$", linked.read_text(), re.M)
                slugs = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}
                assert anchor in slugs, (linked, anchor)


@pytest.mark.parametrize("reachable", [False, True])
def test_default_trace_selection_mocked_resolver_and_exporter(
    offline, monkeypatch, capsys, reachable
):
    endpoint = "http://localhost:5001/v1/traces"
    resolver = Mock(return_value=endpoint if reachable else None)
    exporter = Mock(return_value=object())
    monkeypatch.setattr(tracing, "resolve_otlp_endpoint", resolver)
    monkeypatch.setattr(tracing.exporters_mod, "journal", exporter)
    selected = tracing._default_exporters()
    resolver.assert_called_once_with(endpoint)
    if reachable:
        assert selected == [exporter.return_value]
        exporter.assert_called_once_with(endpoint=endpoint)
    else:
        assert selected is None
        exporter.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert not offline


def test_default_trace_health_probe_mocked_urllib(offline, monkeypatch):
    opened = Mock(side_effect=urllib.error.URLError("offline viewer"))
    monkeypatch.setattr(tracing.urllib.request, "urlopen", opened)
    assert tracing._probe("http://localhost:5001/v1/traces", 0.01) == "down"
    assert opened.call_args.args[0].full_url == "http://localhost:5001/api/eval/health"
    assert opened.call_args.kwargs == {"timeout": 0.01}


def test_automatic_agent_trace_path_keeps_stage_stdout_quiet_json(offline):
    result = invoke([*BASE, "--stage", "routing"])
    report(result)
    assert agent_module._auto_tracing_attempted is True
    tracing.resolve_otlp_endpoint.assert_called_once_with("http://localhost:5001/v1/traces")
    tracing.urllib.request.urlopen.assert_not_called()
    tracing.exporters_mod.journal.assert_not_called()
    assert result.stderr == ""
    assert tracing._enabled is False


def test_reachable_viewer_automatic_agent_path_is_quiet_with_in_memory_exporter(
    offline, monkeypatch, tmp_path
):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from nooa.runtime.hooks import get_hooks, set_hooks
    from nooa.tracing._session import get_session, set_session

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    prior_hooks, prior_session = get_hooks(), get_session()
    monkeypatch.setattr(tracing, "_provider", None)
    monkeypatch.setattr(tracing, "_hooks", None)
    monkeypatch.setattr(tracing.trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(tracing, "_instrument_litellm", lambda provider: None)
    monkeypatch.setattr(
        tracing, "resolve_otlp_endpoint", Mock(return_value="http://localhost:5001/v1/traces")
    )
    monkeypatch.setattr(tracing.exporters_mod, "journal", Mock(return_value=exporter))
    try:
        result = invoke([*BASE, "--stage", "routing"])
        report(result)
        assert result.stderr == ""
        assert tracing._enabled is True
        assert agent_module._auto_tracing_attempted is True
        tracing.exporters_mod.journal.assert_called_once()
        tracing.urllib.request.urlopen.assert_not_called()
        provider.force_flush()
        assert not (tmp_path / ".nooa/traces").exists()
    finally:
        provider.shutdown()
        set_hooks(prior_hooks)
        set_session(prior_session)


@pytest.mark.parametrize("style", ["systemone", "openai"])
@pytest.mark.parametrize(
    "extra",
    [
        ["--stage", "discover"],
        ["--stage", "tools"],
        ["--stage", "reasoning"],
        ["--stage", "media"],
        ["--stage", "session"],
        ["--stage", "routing", "--max-tokens", "1024"],
        ["--stage", "routing", "--discovery-file", "discovery.json"],
    ],
)
def test_native_decision_rejects_chat_options(style, extra, offline, tmp_path):
    (tmp_path / "discovery.json").write_text("{}")
    result = invoke(
        [
            "decision-model",
            "--as",
            "decisions",
            "--endpoint",
            "https://decision.example/v1/decisions",
            "--api-style",
            style,
            *extra,
        ]
    )
    # CLI prevalidation is exit 2; library-only guards are handled failures (1).
    expected = 2 if extra[1] in {"discover", "reasoning"} or "--discovery-file" in extra else 1
    report(result, expected)
    assert not offline


def test_plan_offline_without_credential_and_save_raw_data_to_project(
    offline, monkeypatch, tmp_path
):
    monkeypatch.delenv("MODEL_KEY")
    planned = report(invoke([*BASE, "--stage", "plan", "--test-media-inputs", "image,pdf"]))
    assert planned["data"]["session_checks"] is False
    source = tmp_path / "raw.json"
    source.write_text(json.dumps(planned["data"]))
    args = ["--stage", "save", "--input", str(source), "--working-dir", str(tmp_path)]
    blocked = invoke([*args, "--output", "other.yaml"])
    assert blocked.exit_code == 2
    assert "mutually exclusive" in blocked.stderr
    saved = report(invoke(args))
    assert Path(saved["data"]["path"]) == tmp_path / ".nooa/llm_config.yaml"
    assert not offline


def test_interface_reasoning_compatibility_cap_is_on_wire(offline):
    report(
        invoke(
            [
                *BASE,
                "--stage",
                "interfaces",
                "--output-tokens",
                "200",
                "--reasoning-template",
                "effort",
                "--reasoning-level",
                "high",
                "--reasoning-output-tokens",
                "64",
            ]
        )
    )
    assert len(offline) == 6
    ordinary, reasoning = offline[::2], offline[1::2]

    def cap(request):
        body = json.loads(request.content)
        return body.get("max_tokens", body.get("max_output_tokens"))

    assert [cap(request) for request in ordinary] == [200] * 3
    assert [cap(request) for request in reasoning] == [64] * 3
