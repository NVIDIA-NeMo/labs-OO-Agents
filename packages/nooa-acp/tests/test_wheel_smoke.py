# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failures that must not be mistaken for a clean installed-wheel acceptance."""

from __future__ import annotations

import http.client
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest


@pytest.fixture
def wheel_smoke():
    path = Path(__file__).resolve().parents[3] / "scripts" / "smoke_installed_wheels.py"
    spec = importlib.util.spec_from_file_location("wheel_smoke_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wheel_layout(tmp_path, monkeypatch, wheel_smoke):
    prefix = tmp_path / "wheel-venv"
    site = prefix / "Lib" / "site-packages"
    package = site / "nooa" / "__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("", encoding="utf-8")
    metadata = site / "nooa-1.2.3.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.3\nName: nooa\nVersion: 1.2.3\n", encoding="utf-8"
    )
    (metadata / "WHEEL").write_text("Wheel-Version: 1.0\n", encoding="utf-8")
    (metadata / "RECORD").write_text(
        "\n".join(
            f"{name},,"
            for name in (
                "nooa/__init__.py",
                f"{metadata.name}/METADATA",
                f"{metadata.name}/WHEEL",
                f"{metadata.name}/RECORD",
            )
        ),
        encoding="utf-8",
    )
    distribution = importlib.metadata.PathDistribution(metadata)
    module_spec = importlib.util.spec_from_file_location("nooa", package)
    monkeypatch.setattr(wheel_smoke.importlib.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(wheel_smoke.importlib.util, "find_spec", lambda name: module_spec)
    return prefix, metadata, module_spec


def test_wheel_provenance_accepts_an_installed_local_archive(wheel_smoke, wheel_layout):
    prefix, metadata, module_spec = wheel_layout
    (metadata / "direct_url.json").write_text(
        json.dumps({"url": "file:///build/nooa-1.2.3-py3-none-any.whl", "archive_info": {}}),
        encoding="utf-8",
    )
    evidence = wheel_smoke._distribution_evidence("nooa", "nooa", prefix)
    assert evidence["version"] == "1.2.3"
    assert Path(evidence["module_path"]) == Path(module_spec.origin).resolve()
    assert Path(evidence["metadata_path"]) == metadata / "METADATA"


@pytest.mark.parametrize("directory_info", [{}, {"editable": True}], ids=["source", "editable"])
def test_wheel_provenance_rejects_directory_installs(wheel_smoke, wheel_layout, directory_info):
    prefix, metadata, _ = wheel_layout
    (metadata / "direct_url.json").write_text(
        json.dumps({"url": "file:///checkout", "dir_info": directory_info}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="source-directory/editable"):
        wheel_smoke._distribution_evidence("nooa", "nooa", prefix)


def test_wheel_metadata_cannot_mask_a_source_import(wheel_smoke, wheel_layout, tmp_path):
    prefix, _, module_spec = wheel_layout
    source = tmp_path / "checkout" / "nooa" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    module_spec.origin = str(source)
    with pytest.raises(RuntimeError, match="nooa import is not an installed file"):
        wheel_smoke._distribution_evidence("nooa", "nooa", prefix)


def test_record_cannot_mask_a_missing_wheel_file(wheel_smoke, wheel_layout):
    prefix, metadata, _ = wheel_layout
    (metadata / "WHEEL").unlink()
    with pytest.raises(RuntimeError, match="nooa (WHEEL|is missing wheel)"):
        wheel_smoke._distribution_evidence("nooa", "nooa", prefix)


@pytest.mark.parametrize("explicit_cache", [False, True], ids=["isolated", "prepared-cache"])
def test_clean_environment_excludes_parent_sources_and_credentials(
    wheel_smoke, tmp_path, monkeypatch, explicit_cache
):
    support = wheel_smoke._companion("_acp_smoke_support.py")
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path / "Windows"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "checkout" / "src"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "wrong-python"))
    monkeypatch.setenv("OPENAI_API_KEY", "parent-credential-must-not-be-forwarded")
    monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(tmp_path / "parent-config.yaml"))
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "ambient-cache"))
    monkeypatch.setenv("CUSTOM_TIKTOKEN_CACHE_DIR", str(tmp_path / "ambient-cache"))
    module_cache = str(tmp_path / "machine-cache" / "ModuleAnalysisCache")
    monkeypatch.setenv("PSModuleAnalysisCachePath", module_cache)
    cache = tmp_path / "prepared-cache" if explicit_cache else None
    environment = support.child_environment(
        tmp_path / "中文 repo's [literal]",
        tmp_path / "config",
        "http://127.0.0.1:12345",
        tmp_path / "shell" / "powershell.exe",
        tokenizer_cache=cache,
    )
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "OPENAI_API_KEY" not in environment
    assert environment["PSModuleAnalysisCachePath"] == module_cache
    assert environment["NEMO_OO_LLM_CONFIG"] == str(tmp_path / "config" / "llm_config.yaml")
    assert environment["NOOA_NATIVE_TEST_KEY"] == support._TEST_KEY
    assert environment["PYTHONUTF8"] == "0"
    assert environment["PYTHONNOUSERSITE"] == "1"
    for name in ("TIKTOKEN_CACHE_DIR", "CUSTOM_TIKTOKEN_CACHE_DIR"):
        if explicit_cache:
            assert environment[name] == str(cache)
        else:
            assert name not in environment


@pytest.mark.parametrize("unrelated_resource", [False, True], ids=["absent", "wrong-resource"])
def test_missing_tokenizer_data_requires_explicit_preparation(
    wheel_smoke, tmp_path, unrelated_resource
):
    cache = tmp_path / "tokenizer-cache"
    if unrelated_resource:
        cache.mkdir()
        (cache / "unrelated-tokenizer").write_text("unrelated resource", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Prepared cl100k_base tokenizer data is missing"):
        wheel_smoke._prepared_tokenizer_cache(cache)


@pytest.mark.parametrize("failure", [False, True], ids=["response", "callback-error"])
def test_scripted_callback_records_requests_and_reports_failures(wheel_smoke, failure):
    support = wheel_smoke._companion("_acp_smoke_support.py")
    request = {"model": support._MODEL, "messages": [{"role": "user", "content": "验收"}]}

    def respond(received):
        assert received == request
        if failure:
            raise RuntimeError("synthetic callback error")
        return "return_result('verified')"

    with support.ScriptedModelServer(responder=respond) as server:
        url = urlsplit(server.url)
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(request),
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {support._TEST_KEY}",
                },
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            assert response.status == (400 if failure else 200)
        finally:
            connection.close()
        assert server.requests == [request]
        assert server.remaining == 0
        if failure:
            assert server.errors == ["Scripted model responder failed: synthetic callback error"]
        else:
            assert not server.errors
            function = payload["choices"][0]["message"]["tool_calls"][0]["function"]
            assert function["name"] == "execute_python"
            assert json.loads(function["arguments"])["code"] == "return_result('verified')"


@pytest.fixture
def command_cards(wheel_smoke):
    from acp import start_tool_call, text_block, tool_content, update_tool_call

    command = f"Write-Output '{wheel_smoke._STDOUT_MARKER}'; python verify.py"
    updates = [start_tool_call("python-cell", "Running Python", kind="other")]
    for index, exit_code in enumerate((1, 0)):
        command_id = f"shell-{index}"
        command_content = f"```powershell\n{command}\n```"
        updates.append(
            start_tool_call(
                command_id,
                "Running command",
                kind="other",
                status="in_progress",
                content=[tool_content(text_block(command_content))],
                raw_input={"command": command},
            )
        )
        diagnostic = "FAILED (failures=2)" if exit_code else "OK"
        rendered = (
            f"{command_content}\n\nStdout:\n\n"
            f"```text\n{wheel_smoke._STDOUT_MARKER}\n```\n\n"
            f"Stderr:\n\n```text\nRan 4 tests in 0.001s\n{diagnostic}\n```\n\n"
            f"Exit code: {exit_code}"
        )
        updates.append(
            update_tool_call(
                command_id,
                title=f"Command failed (exit {exit_code})" if exit_code else "Ran command (exit 0)",
                status="failed" if exit_code else "completed",
                content=[tool_content(text_block(rendered))],
                raw_output={"exit_code": exit_code, "timed_out": False, "output_truncated": False},
            )
        )
    return updates


def test_command_cards_verify_standard_content_and_ignore_unrelated_tools(
    wheel_smoke, command_cards
):
    evidence = wheel_smoke._command_cards(command_cards)
    assert [card["exit_code"] for card in evidence] == [1, 0]
    assert [card["status"] for card in evidence] == ["failed", "completed"]
    assert all(card["stdout_visible"] and card["stderr_visible"] for card in evidence)


@pytest.mark.parametrize(
    ("regression", "error"),
    [
        ("execute-kind", "generic kind"),
        ("start-content", "standard text content"),
        ("lost-command", "retain the full fenced command"),
        ("lost-labels", "label its stdout and stderr"),
        ("lost-stdout", "fenced stdout"),
        ("lost-stderr", "fenced stderr test diagnostics"),
        ("raw-only-exit", "exit code in standard content"),
        ("wrong-status", "status and title"),
    ],
)
def test_command_cards_reject_ui_regressions_despite_correct_raw_exit_codes(
    wheel_smoke, command_cards, regression, error
):
    start, finish = command_cards[1:3]
    if regression == "execute-kind":
        start.kind = "execute"
    elif regression == "start-content":
        start.content = None
    elif regression == "wrong-status":
        finish.status = "completed"
    else:
        text = finish.content[0].content
        if regression == "lost-command":
            text.text = text.text.replace(start.raw_input["command"], "truncated command")
        elif regression == "lost-labels":
            text.text = text.text.replace("Stdout:", "")
        elif regression == "lost-stdout":
            # The marker still occurs in the command; it must also be actual output.
            text.text = text.text.replace(f"```text\n{wheel_smoke._STDOUT_MARKER}\n```", "")
        elif regression == "lost-stderr":
            text.text = text.text.replace("FAILED (failures=2)", "test result missing")
        elif regression == "raw-only-exit":
            text.text = text.text.replace("Exit code: 1", "")
    with pytest.raises(RuntimeError, match=error):
        wheel_smoke._command_cards(command_cards)
