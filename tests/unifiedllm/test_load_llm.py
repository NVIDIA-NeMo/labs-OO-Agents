# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``load_llm``: a registry entry in, today's client out."""

import textwrap

import pytest

from nooa.unifiedllm import CompletionClient, ResponsesClient, load_llm, reload_registry


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    """A clean registry with no user, project or bundled entries."""
    for name in ("user", "project"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setattr("nooa.llm_config.bundled_config_paths", lambda: [])
    monkeypatch.delenv("NEMO_OO_LLM_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    reload_registry()
    yield
    reload_registry()


def _registry(tmp_path, body):
    path = tmp_path / "llm_config.yaml"
    path.write_text(textwrap.dedent(body))
    reload_registry(path)


@pytest.mark.parametrize("adapter", [None, "litellm"])
def test_loads_todays_client(tmp_path, adapter):
    line = f"adapter: {adapter}" if adapter else ""
    _registry(
        tmp_path,
        f"""
        models:
          chat:
            model_name: openai/chat-model
            api_base: https://gw.example.com/v1
            api_key: test
            media_inputs: []
            {line}
          resp:
            model_name: openai/resp-model
            client_type: responses
            api_key: test
            media_inputs: []
            {line}
        """,
    )
    chat = load_llm("chat", temperature=0.1)
    resp = load_llm("resp")
    assert type(chat) is CompletionClient and chat.model == "openai/chat-model"
    assert chat.config["temperature"] == 0.1
    assert type(resp) is ResponsesClient and resp.model == "openai/resp-model"
    # The adapter field selects the adapter; it is never sent to the provider.
    assert "adapter" not in chat.config and "adapter" not in resp.config
    chat.close()
    resp.close()


def test_unknown_adapter_is_rejected(tmp_path):
    _registry(
        tmp_path,
        """
        models:
          odd:
            model_name: openai/x
            adapter: anthropic-messages
        """,
    )
    with pytest.raises(ValueError, match="unsupported adapter 'anthropic-messages'.*litellm"):
        load_llm("odd")


def test_plain_model_name_passes_through():
    client = load_llm("openai/gpt-4o-mini", api_key="test")
    assert type(client) is CompletionClient and client.model == "openai/gpt-4o-mini"
    client.close()
