# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core workflows exercised without a POSIX shell or a live model provider."""

import json
import os

import pytest

from nooa import Agent
from nooa.layered_config import load_layered_yaml
from nooa.secrets import write_secret_env
from nooa.storage import SQLiteStorageManager
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall, connect


class _PlatformAgent(Agent, llm=FakeLLMClient()):
    value: str = ""

    def remember(self, text: str) -> str:
        self.value = text
        return text

    async def remember_text(self, text: str) -> str:
        """Remember the supplied text using self.remember and return it."""
        ...


async def test_codeact_tool_call_and_sqlite_snapshot_round_trip(tmp_path):
    text = "中文 text with emoji 🙂"
    llm = FakeLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="remember",
                        name="execute_python",
                        arguments=json.dumps({"code": "return_result(self.remember(text))"}),
                    )
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    agent = _PlatformAgent(llm=llm)
    assert await agent.remember_text(text) == text
    assert llm.call_count == 1

    path = tmp_path / "中文 session.db"
    with SQLiteStorageManager(path) as storage:
        storage.save_snapshot(agent)
    restored = _PlatformAgent()
    with SQLiteStorageManager(path) as storage:
        assert storage.restore_latest_snapshot(restored)
    assert restored.value == text


def test_utf8_registry_write_and_layered_read_preserve_crlf(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    path = tmp_path / "中文 configuration.yaml"
    path.write_bytes("# 保留注释 🙂\r\nmodels: {}\r\n".encode())
    connect.write({"model_name": "openai/test-model"}, path, alias="中文模型")

    content = path.read_bytes().decode("utf-8")
    assert content.startswith("# 保留注释 🙂\r\n")
    assert "\n" not in content.replace("\r\n", "")
    monkeypatch.setenv("NOOA_TEST_PLATFORM_CONFIG", str(path))
    loaded = load_layered_yaml("config.yaml", "NOOA_TEST_PLATFORM_CONFIG")
    assert loaded["models"]["中文模型"]["model_name"] == "openai/test-model"

    from nooa_cli.commands._connect_registry import entries
    from nooa_cli.commands.config import _count_aliases

    from nooa import llm_config
    from nooa.unifiedllm.registry import _load_models_from_yaml

    assert _load_models_from_yaml(path)["中文模型"]["model_name"] == "openai/test-model"
    assert _count_aliases(path) == 1
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    assert entries(path)["中文模型"][0]["model_name"] == "openai/test-model"


def test_secret_update_preserves_utf8_metadata(tmp_path):
    import yaml

    path = tmp_path / "secrets.yaml"
    path.write_text("env: {}\nmetadata: 中文 🙂\n", encoding="utf-8")
    write_secret_env(path, "NOOA_TEST_ONLY", "test-only-value")
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "env": {"NOOA_TEST_ONLY": "test-only-value"},
        "metadata": "中文 🙂",
    }


def test_debug_handler_is_optional_when_sigusr2_is_unavailable(monkeypatch):
    from nooa.runtime import debug_handler

    monkeypatch.setattr(debug_handler, "_handler_installed", False)
    monkeypatch.delattr(debug_handler.signal, "SIGUSR2", raising=False)

    def unexpected_signal(*args):
        pytest.fail("A platform without SIGUSR2 must not register its handler")

    monkeypatch.setattr(debug_handler.signal, "signal", unexpected_signal)
    debug_handler.install_debug_handler()
    assert not debug_handler._handler_installed


@pytest.mark.skipif(os.name != "nt", reason="Windows has no Unix sandbox mechanisms")
def test_windows_sandbox_capabilities_can_be_queried(tmp_path, monkeypatch):
    from nooa.runtime.sandbox.guards import probe_capabilities

    # Pin the root: otherwise the answer depends on whether this machine has
    # built one at the default location.
    monkeypatch.setenv("NOOA_SANDBOX_RUNTIME", str(tmp_path))
    capabilities = probe_capabilities()
    # The AppContainer mechanisms need a working runtime root. Without one each
    # guardrail reports None, so the executor fails closed instead of running
    # cells unconfined, and the reason names the root that was checked.
    assert capabilities.filesystem is None
    assert capabilities.network is None
    assert capabilities.memory is None
    assert capabilities.cpu is None
    assert capabilities.landlock_abi == 0
    assert str(tmp_path.resolve()) in capabilities.detail["reason"]


@pytest.mark.skipif(os.name != "nt", reason="the runtime root is a Windows concept")
def test_sandbox_validates_the_configured_runtime_root(tmp_path, monkeypatch):
    """Construction checks the root the config names, not the environment's.

    The launcher runs the worker from the configured root, so probing any other
    lets construction and launch disagree about whether the sandbox is usable.
    """
    from nooa.runtime.sandbox import executor
    from nooa.runtime.sandbox.config import SandboxConfig
    from nooa.runtime.sandbox.errors import SandboxUnavailable

    monkeypatch.setattr(executor, "_CAPS_CACHE", {})
    monkeypatch.setenv("NOOA_SANDBOX_RUNTIME", str(tmp_path / "from-env"))
    configured = tmp_path / "from-config"
    configured.mkdir()
    config = SandboxConfig(
        start_method="spawn", backend="appcontainer", runtime_root=str(configured)
    )
    with pytest.raises(SandboxUnavailable) as excinfo:
        executor.SandboxedExecutor(_PlatformAgent(), config, cell_timeout=1)
    assert str(configured.resolve()) in str(excinfo.value)


@pytest.mark.parametrize("require", [False, True])
def test_sandbox_requires_a_supported_process_start_method(monkeypatch, require):
    from nooa.runtime.sandbox import executor
    from nooa.runtime.sandbox.config import SandboxConfig
    from nooa.runtime.sandbox.errors import SandboxUnavailable

    monkeypatch.setattr(executor.mp, "get_all_start_methods", lambda: ["spawn"])
    with pytest.raises(SandboxUnavailable, match="fork"):
        executor.SandboxedExecutor(None, SandboxConfig(require=require), cell_timeout=1)
