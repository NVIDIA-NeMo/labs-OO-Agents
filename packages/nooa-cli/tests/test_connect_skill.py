# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical agent Connect skill examples run headlessly with mocked transport."""

import ast
import json
import re
import shlex
from pathlib import Path

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands.connect import command

from nooa import TextSkill
from tests.unifiedllm.connect.connect_http import mock_http, response_body
from tests.unifiedllm.connect.test_connect_decision_smoke import fixture_response

ROOT = Path(__file__).parents[3]
SKILL = ROOT / "skills/nooa-connect/SKILL.md"


def snippets():
    # One logical shell command per item, including grouped metadata-first steps.
    commands = []
    for block in re.findall(r"```sh\n(.*?)```", SKILL.read_text(), re.S):
        pending = ""
        for line in block.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            pending += line + "\n"
            if not line.endswith("\\"):
                commands.append(pending)
                pending = ""
        assert not pending, "Unfinished shell continuation"
    return commands


def test_skill_metadata_links_and_permission_contract():
    skill = TextSkill(path=SKILL.parent)
    assert skill.id == "nooa-connect"
    text = SKILL.read_text()
    assert "Get permission before configuring the model" in text
    assert "Agents use **`--stage NAME`**" in text
    assert "--input` is for **save only**" in text
    assert "each invocation\nstarts a new budget" in text
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if "://" not in target:
            assert (SKILL.parent / target.split("#")[0]).is_file(), target
    for python in re.findall(r"```python\n(.*?)```", text, re.S):
        ast.parse(python)
    assert "defaults to 2,048 output tokens" in text
    assert "explicitly selects `--output-tokens 200`" in text
    for name in (
        "nooa-agent-authoring",
        "nooa-model-configuration",
        "nooa-tools-and-skills",
        "nooa-media",
    ):
        other = (ROOT / "skills" / name / "SKILL.md").read_text()
        assert "../nooa-connect/SKILL.md" in other
        assert not re.search(r"uv run nooa connect\b", other)
    for path, target in (
        (ROOT / "AGENTS.md", "skills/nooa-connect/SKILL.md"),
        (ROOT / "docs/media.md", "../skills/nooa-connect/SKILL.md"),
    ):
        assert target in path.read_text()


def test_documented_headless_workflow_with_mocked_native_requests(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("MODEL_KEY", "DECISION_KEY"):
        monkeypatch.setenv(name, "skill-example-secret")
    seen = []

    def handle(request):
        seen.append(request)
        if request.method == "GET":
            if request.url.host == "openrouter.ai":
                assert request.url.path == "/api/v1/models"
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "openai/example-model",
                                "context_length": 16384,
                                "top_provider": {"max_completion_tokens": 4096},
                                "architecture": {"input_modalities": ["image", "file"]},
                            }
                        ]
                    },
                )
            assert request.url.path == "/v1/models"
            return httpx.Response(200, json={"data": [{"id": "your-model"}]})
        if request.url.path.endswith(("/decisions", "/systemone")):
            return fixture_response(request)
        style = (
            "anthropic"
            if request.url.path.endswith("/messages")
            else "responses"
            if request.url.path.endswith("/responses")
            else "chat"
        )
        return httpx.Response(200, json=response_body(style))

    mock_http(monkeypatch, handle)
    reports = {}
    for snippet in snippets():
        tokens = shlex.split(snippet.replace("\\\n", " "))
        assert tokens[:4] == ["uv", "run", "nooa", "connect"]
        if ">" in tokens:
            redirect = tokens.index(">")
            output = Path(tokens[redirect + 1])
            args = tokens[4:redirect]
        else:
            output = None
            args = tokens[4:]
        assert "--prompt-key" not in args and "--yes" not in args
        assert "--stage" in args or args == ["--help"]
        before = len(seen)
        result = CliRunner().invoke(command, args, input="")
        assert result.exit_code == 0, result.output
        assert "skill-example-secret" not in result.output
        if output:
            output.write_text(result.stdout)
            assert result.stdout.lstrip().startswith("{"), (args, result.stdout)
            report = json.loads(result.stdout)
            reports[report["stage"]] = report
            assert report["ok"] is True
            if report["stage"] in {"plan", "save"}:
                assert len(seen) == before
        if "--stage" in args and args[args.index("--stage") + 1] in {"plan", "save"}:
            assert len(seen) == before  # Includes the metadata-only save with no redirect.
        assert result.stderr == "" or "Warning" in result.stderr

    entry = yaml.safe_load(Path(".nooa/llm_config.yaml").read_text())["models"]["work"]
    assert entry["max_tokens"] == 1024
    assert entry["context_window"] == 16384
    assert entry["api_key_env"] == "MODEL_KEY"
    assert "provenance" not in entry
    image = reports["media"]
    assert image["data"]["entry"]["media_inputs"] == ["image"]
    assert image["checks"]["media:image"]["attachment_observed"] is True
    assert image["checks"]["media:image"]["answer_correct"] is True
    assert len(seen) == 12  # 1 routing, 2 metadata GETs, 4 interface, 5 native decision/image.
    metadata_entry = yaml.safe_load(Path("llm_config.yaml").read_text())["models"]["work"]
    assert metadata_entry["model_name"] == "openai/example-model"
    assert metadata_entry["media_inputs"] == []

    # Same documented save refuses replacement until explicitly authorized.
    refused = CliRunner().invoke(
        command,
        [
            "--stage",
            "save",
            "--input",
            "model-result.json",
            "--output",
            ".nooa/llm_config.yaml",
        ],
    )
    assert refused.exit_code == 2
    assert json.loads(refused.stdout)["error"]["type"] == "UsageError"


def test_headless_plan_needs_no_live_credentials_or_network(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODEL_KEY", raising=False)
    mock_http(monkeypatch, lambda request: pytest.fail("Plan must not make HTTP calls"))
    snippet = next(s for s in snippets() if "--stage plan" in s)
    args = shlex.split(snippet.replace("\\\n", " "))[4:]
    args = args[: args.index(">")]
    result = CliRunner().invoke(command, args, input="")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["data"]["entry"]["api_key_env"] == "MODEL_KEY"


def test_probe_constructor_announcements_do_not_pollute_output(capsys):
    import sys

    from nooa.unifiedllm.connect._baseline import _probe_agent

    def announcing_constructor(**kwargs):
        print("OTel tracing enabled: example")
        print("setup diagnostic", file=sys.stderr)
        return kwargs

    assert _probe_agent(announcing_constructor, value=7) == {"value": 7}
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_diagnostics_point_to_canonical_headless_skill():
    from nooa.unifiedllm import connect
    from nooa.unifiedllm.connect._diagnostics import installation_context

    context = installation_context()
    assert context["reference_paths"][0] == str(SKILL.resolve())
    handoff = connect.diagnostic_prompt("routing", {}, {})
    assert "First read the nooa-connect skill" in handoff
    assert "does not authorize additional paid calls, model configuration or writes" in handoff
