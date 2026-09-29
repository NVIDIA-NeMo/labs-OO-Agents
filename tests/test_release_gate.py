# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the release-gate alias resolution (no provider calls)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from nooa.unifiedllm.registry import reload_registry
from tests.integration import _release_gate as gate


def _write(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body))
    return path


@pytest.fixture
def layers(tmp_path, monkeypatch):
    """Isolated registry: one bundled YAML plus an empty user and project layer."""
    user = tmp_path / "user"
    project = tmp_path / "project"
    user.mkdir()
    project.mkdir()
    bundled = _write(
        tmp_path / "bundled.yaml",
        """
        models:
          release-gate-openai:
            model_name: openai/route-from-wheel
            api_base: https://gateway.example/v1
            api_key_env: GATE_TEST_KEY
        """,
    )
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project))
    monkeypatch.delenv("NEMO_OO_LLM_CONFIG", raising=False)
    monkeypatch.setattr("nooa.llm_config.bundled_config_paths", lambda: [bundled])
    monkeypatch.chdir(tmp_path)
    reload_registry()
    yield {"user": user, "bundled": bundled, "tmp": tmp_path}
    monkeypatch.undo()
    reload_registry()


def test_alias_from_bundled_package_resolves(layers):
    assert gate.gate_host("openai") == "gateway.example"


def test_missing_alias_skips(layers):
    with pytest.raises(pytest.skip.Exception, match="release-gate-kimi"):
        gate.gate_host("kimi")


@pytest.mark.parametrize("layer", ["user", "env"])
def test_alias_shadowed_by_a_higher_layer_fails(layers, monkeypatch, layer):
    body = """
        models:
          release-gate-openai:
            model_name: openai/some-other-route
            api_base: https://elsewhere.example/v1
        """
    if layer == "user":
        _write(layers["user"] / "llm_config.yaml", body)
    else:
        override = _write(layers["tmp"] / "override.yaml", body)
        monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(override))
    reload_registry()
    with pytest.raises(pytest.fail.Exception, match="outside the bundled-config package"):
        gate.gate_host("openai")
    with pytest.raises(pytest.fail.Exception, match="outside the bundled-config package"):
        gate.gate_client("openai")


def test_alias_without_api_base_fails_clearly(layers):
    _write(
        layers["bundled"],
        """
        models:
          release-gate-openai:
            model_name: openai/route-from-wheel
        """,
    )
    reload_registry()
    with pytest.raises(pytest.fail.Exception, match="no api_base"):
        gate.gate_host("openai")


def test_openai_cache_case_pins_the_responses_api(layers, monkeypatch):
    """The case renders Responses input, so the alias must not choose the API style."""
    from nooa.unifiedllm import ResponsesClient
    from tests.integration.test_cache_resume_live import _client

    monkeypatch.setenv("GATE_TEST_KEY", "unused")
    assert isinstance(_client("openai"), ResponsesClient)


def test_problems_report_missing_alias_and_credential(layers, monkeypatch):
    monkeypatch.delenv("GATE_TEST_KEY", raising=False)
    problems = gate.gate_problems(("openai", "kimi"))
    assert any("release-gate-openai" in p and "GATE_TEST_KEY" in p for p in problems)
    assert any("release-gate-kimi" in p and "not installed" in p for p in problems)
    monkeypatch.setenv("GATE_TEST_KEY", "set")
    assert gate.gate_problems(("openai",)) == []


def test_problems_report_shadowed_alias_and_missing_api_base(layers, monkeypatch):
    monkeypatch.setenv("GATE_TEST_KEY", "set")
    _write(
        layers["user"] / "llm_config.yaml",
        """
        models:
          release-gate-openai:
            model_name: openai/some-other-route
        """,
    )
    _write(
        layers["bundled"],
        """
        models:
          release-gate-openai:
            model_name: openai/route-from-wheel
          release-gate-kimi:
            model_name: openai/kimi
        """,
    )
    reload_registry()
    problems = gate.gate_problems(("openai", "kimi"))
    assert any("outside the bundled-config package" in p for p in problems)
    assert any("release-gate-kimi" in p and "api_base" in p for p in problems)
