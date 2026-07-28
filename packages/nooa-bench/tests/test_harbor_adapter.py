# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the example Harbor adapter command construction."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest


class _FakeBaseInstalledAgent:
    def __init__(self, logs_dir: Path, *args: Any, **kwargs: Any) -> None:
        self.logs_dir = logs_dir
        self.model_name = "gpt-5.6-sol"
        self.root_commands: list[str] = []
        self.agent_commands: list[tuple[str, dict[str, str]]] = []

    async def exec_as_root(self, environment: Any, command: str) -> None:
        self.root_commands.append(command)

    async def exec_as_agent(
        self, environment: Any, command: str, env: dict[str, str], cwd: str
    ) -> None:
        self.agent_commands.append((command, env))


def _install_harbor_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        "harbor": types.ModuleType("harbor"),
        "harbor.agents": types.ModuleType("harbor.agents"),
        "harbor.agents.installed": types.ModuleType("harbor.agents.installed"),
        "harbor.agents.installed.base": types.ModuleType("harbor.agents.installed.base"),
        "harbor.environments": types.ModuleType("harbor.environments"),
        "harbor.environments.base": types.ModuleType("harbor.environments.base"),
        "harbor.models": types.ModuleType("harbor.models"),
        "harbor.models.agent": types.ModuleType("harbor.models.agent"),
        "harbor.models.agent.context": types.ModuleType("harbor.models.agent.context"),
    }
    modules["harbor.agents.installed.base"].BaseInstalledAgent = _FakeBaseInstalledAgent
    modules["harbor.environments.base"].BaseEnvironment = object
    modules["harbor.models.agent.context"].AgentContext = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _load_adapter(monkeypatch: pytest.MonkeyPatch) -> Any:
    _install_harbor_fakes(monkeypatch)
    module_path = Path("examples/benchmarks/harbor_adapter.py").resolve()
    spec = importlib.util.spec_from_file_location("harbor_adapter_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_bench_install_does_not_download_copilot_runtime(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    agent = adapter.NooaBenchAgent(Path.cwd(), git_url="https://example.test/repo.git")

    await agent.install(object())

    command = agent.root_commands[-1]
    assert "download-runtime" not in command
    assert "COPILOT_CLI_EXTRACT_DIR" not in command


@pytest.mark.asyncio
async def test_copilot_install_downloads_runtime(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    agent = adapter.NooaBenchAgent(
        Path.cwd(), git_url="https://example.test/repo.git", agent_type="copilot"
    )

    await agent.install(object())

    command = agent.root_commands[-1]
    assert "python3 -m copilot download-runtime" in command
    assert "COPILOT_CLI_EXTRACT_DIR=/opt/nooa-copilot-runtime" in command
    assert "chmod -R a+rX /opt/nooa-copilot-runtime" in command
    assert "a+rwx" not in command


@pytest.mark.asyncio
async def test_bench_run_does_not_forward_copilot_credentials(monkeypatch):
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "secret-token")
    adapter = _load_adapter(monkeypatch)
    agent = adapter.NooaBenchAgent(Path.cwd(), git_url="https://example.test/repo.git")

    await agent.run("fix it", object(), object())

    command, env = agent.agent_commands[-1]
    assert "COPILOT_GITHUB_TOKEN" not in env
    assert "nooa-copilot-home" not in command


@pytest.mark.asyncio
async def test_copilot_run_uses_private_ephemeral_home(monkeypatch):
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "secret-token")
    adapter = _load_adapter(monkeypatch)
    agent = adapter.NooaBenchAgent(
        Path.cwd(), git_url="https://example.test/repo.git", agent_type="copilot"
    )

    await agent.run("fix it", object(), object())

    command, env = agent.agent_commands[-1]
    assert env["COPILOT_GITHUB_TOKEN"] == "secret-token"
    assert env["COPILOT_HOME"] == "/tmp/nooa-copilot-home"
    assert "nooa-copilot-home" not in command
    assert "/logs/artifacts/copilot-home" not in command
