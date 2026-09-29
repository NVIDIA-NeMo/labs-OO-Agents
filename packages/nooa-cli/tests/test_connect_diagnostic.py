# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic handoffs explain reproduction and sources without exposing credentials."""

import shlex
from pathlib import Path

from nooa_cli.commands._connect_registry import diagnostic_context

from nooa.unifiedllm import connect


def test_target_and_effective_source_are_distinct(tmp_path, monkeypatch):
    from nooa import llm_config

    target = tmp_path / "target file.yaml"
    override = tmp_path / "override.yaml"
    override.write_text("models: {saved: {model_name: openai/model}}\n")
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [override])
    monkeypatch.setenv("CHECK_KEY", "private-value")
    context = diagnostic_context(
        target=target,
        alias="saved",
        model="model; not a command",
        endpoint="https://api.test/v1",
        api_key_env="CHECK_KEY",
        budget=131072,
        remaining=128936,
        stage="interfaces",
    )
    assert context["target_file"] == str(target)
    assert context["effective_alias_source"] == str(override)
    assert context["registry_files"] == [str(override)]
    assert context["credential_available"] is True
    assert shlex.split(context["rerun_command"])[4] == "model; not a command"
    assert "private-value" not in repr(context)
    prompt = connect.diagnostic_prompt(
        "interfaces", {}, {}, run_context={**context, "api_key": "secret"}
    )
    assert '"api_key"' not in prompt
    assert "git clone" not in prompt
    assert context["source_root"].startswith("/")
    assert all(path.startswith("/") for path in context["reference_paths"])
    from nooa.skill import _parse_skill_md

    skill_path = Path(context["reference_paths"][0])
    name, _, _ = _parse_skill_md(skill_path.parent)
    assert name == "nooa-model-configuration"
    assert str(skill_path) in prompt
    assert "nooa-agent-authoring" not in prompt
    assert context["target_in_registry_chain"] is False
    assert shlex.split(context["rerun_command"])[-1] == "2136"
    assert "does not authorize additional paid calls" in prompt


def test_context_does_not_mask_broken_yaml_or_leak_pasted_key(tmp_path, monkeypatch):
    from nooa import llm_config

    broken = tmp_path / "broken.yaml"
    broken.write_text("models: [invalid: : private-secret\n")
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [broken])
    context = diagnostic_context(
        target=tmp_path / "private-secret.yaml",
        alias="saved",
        api_key="private-secret",
        model="model",
        endpoint="https://u:private-secret@api.test/v1",
        remaining=0,
        stage="interfaces",
    )
    assert "private-secret" not in repr(context)
    assert "rerun_command" not in context
    assert context["credential_source"] == "pasted"


def test_pasted_rerun_uses_masked_prompt_and_proxy_presence(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "https://user:secret@proxy.test")
    context = diagnostic_context(
        model="model",
        endpoint="https://api.test/v1",
        api_key="pasted-secret",
        remaining=9000,
        stage="interfaces",
    )
    assert "--prompt-key" in context["rerun_command"]
    assert "--api-key-env" not in context["rerun_command"]
    assert context["proxy_variables_set"]["HTTPS_PROXY"] is True
    assert "user:secret" not in repr(context)


def test_wheel_references_pin_version_not_main(monkeypatch, tmp_path):
    from nooa import _version
    from nooa.unifiedllm.connect._diagnostics import installation_context

    monkeypatch.setattr(_version, "__file__", str(tmp_path / "site-packages/nooa/_version.py"))
    monkeypatch.setattr(_version, "__version__", "1.2.3")
    result = installation_context()
    assert result["source_root"] is None
    assert "checkout --detach v1.2.3" in result["reference_command"]
    assert "skills/nooa-model-configuration/SKILL.md" in result["reference_guidance"]
    assert "nooa-agent-authoring" not in result["reference_guidance"]
    monkeypatch.setattr(_version, "__version__", "0.0.0+unknown")
    result = installation_context()
    assert "reference_command" not in result
    assert "revision is unknown" in result["reference_guidance"]


def test_wrapper_timeout_is_not_claimed_as_server_receipt():
    from nooa.unifiedllm.connect._diagnostics import timeout_details

    details = timeout_details(TimeoutError("private"), deadline_expired=True)
    assert details["outcome"] == "not_confirmed"
    assert details["timeout_kind"] == "probe_deadline"
    assert "unknown" in details["reason"]
    assert "private" not in repr(details)
