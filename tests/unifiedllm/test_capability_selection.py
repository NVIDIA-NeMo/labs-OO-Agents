# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capability-aware UnifiedLLM client selection."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from litellm.exceptions import BadRequestError

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies import CodeActStrategy
from nooa.unifiedllm import (
    CompletionClient,
    LLMRequirements,
    LLMResponse,
    ResponsesClient,
    RetryConfig,
    ToolCall,
    get_llm_client,
    reload_registry,
    resolve_llm_client_for_requirements,
    with_retry,
)

CODEACT_REQUIREMENTS = LLMRequirements(
    function_tools=True,
    structured_result=True,
    multi_turn_tools=True,
    reasoning="preserve_model_default",
)


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    user = tmp_path / "user"
    project = tmp_path / "project"
    user.mkdir()
    project.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project))
    monkeypatch.setattr("nooa.llm_config.bundled_config_paths", lambda: [])
    monkeypatch.delenv("NEMO_OO_LLM_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    reload_registry()
    yield
    reload_registry()


def _write_config(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body))
    return path


def test_tool_requirements_route_raw_openai_gpt5_to_responses():
    llm = get_llm_client("gpt-5.6", llm_requirements=CODEACT_REQUIREMENTS)

    assert isinstance(llm, ResponsesClient)
    assert llm.model == "gpt-5.6"
    assert llm._client_type == "responses"


def test_tool_requirements_route_provider_qualified_openai_gpt5_to_responses():
    llm = get_llm_client("openai/gpt-5.6", llm_requirements=CODEACT_REQUIREMENTS)

    assert isinstance(llm, ResponsesClient)
    assert llm.model == "openai/gpt-5.6"
    assert llm._client_type == "responses"


def test_registry_aliases_resolve_same_client_type_for_same_openai_gpt5(tmp_path):
    path = _write_config(
        tmp_path / "models.yaml",
        """
        models:
          raw-alias:
            model_name: gpt-5.6
          qualified-alias:
            model_name: openai/gpt-5.6
        """,
    )
    reload_registry(path)

    raw = get_llm_client("raw-alias", llm_requirements=CODEACT_REQUIREMENTS)
    qualified = get_llm_client("qualified-alias", llm_requirements=CODEACT_REQUIREMENTS)

    assert isinstance(raw, ResponsesClient)
    assert isinstance(qualified, ResponsesClient)
    assert raw._client_type == qualified._client_type == "responses"


def test_explicit_client_type_overrides_requirement_selection():
    llm = get_llm_client(
        "gpt-5.6",
        client_type="completion",
        llm_requirements=CODEACT_REQUIREMENTS,
    )

    assert isinstance(llm, CompletionClient)
    assert llm._client_type == "completion"
    assert llm._client_type_source == "explicit"


def test_registry_client_type_overrides_requirement_selection(tmp_path):
    path = _write_config(
        tmp_path / "models.yaml",
        """
        models:
          explicit-completion:
            model_name: gpt-5.6
            client_type: completion
        """,
    )
    reload_registry(path)

    llm = get_llm_client("explicit-completion", llm_requirements=CODEACT_REQUIREMENTS)

    assert isinstance(llm, CompletionClient)
    assert llm._client_type == "completion"
    assert llm._client_type_source == "registry"


def test_unknown_raw_model_with_tool_requirements_fails_actionably():
    with pytest.raises(ValueError, match="client_type"):
        get_llm_client("some-unknown-model-xyz", llm_requirements=CODEACT_REQUIREMENTS)


def test_known_non_gpt5_models_keep_completion_transport():
    llm = get_llm_client("gpt-4o-mini", llm_requirements=CODEACT_REQUIREMENTS)

    assert isinstance(llm, CompletionClient)
    assert llm._client_type == "completion"


def test_direct_completion_client_is_not_replanned():
    direct = CompletionClient("some-unknown-model-xyz")

    assert resolve_llm_client_for_requirements(direct, CODEACT_REQUIREMENTS) is direct


def test_requirement_upgrade_cache_closes_with_source_client():
    source = get_llm_client("gpt-5.6")
    routed = resolve_llm_client_for_requirements(source, CODEACT_REQUIREMENTS)

    assert isinstance(routed, ResponsesClient)
    assert routed is resolve_llm_client_for_requirements(source, CODEACT_REQUIREMENTS)

    source.close()

    assert routed._http is not None
    assert routed._http._sync_closed is True


@pytest.mark.asyncio
async def test_codeact_uses_requirements_to_upgrade_auto_gpt5_before_llm_call():
    auto_llm = get_llm_client(
        "gpt-5.6",
        api_base="https://example.test/v1",
        api_key="test-key",
    )
    seen: dict[str, object] = {}

    async def fake_responses_acall(self, messages, tools=None, **kwargs):
        seen["client"] = self
        seen["messages"] = messages
        seen["tools"] = tools
        seen["kwargs"] = kwargs
        return LLMResponse(
            raw_response=None,
            content="",
            tool_calls=[
                ToolCall(
                    id="call_execute",
                    name="execute_python",
                    arguments=json.dumps({"code": "return_result(7)"}),
                )
            ],
            finish_reason="tool_calls",
            assistant_message={"role": "assistant", "content": ""},
        )

    class ToolAgent(Agent, llm=auto_llm):
        @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=3, max_retries=1)))
        async def value(self) -> int:
            """Return a number."""
            ...

    with patch.object(ResponsesClient, "acall", fake_responses_acall):
        result = await ToolAgent().value()

    assert result == 7
    routed_client = seen["client"]
    assert isinstance(routed_client, ResponsesClient)
    assert routed_client.config["api_base"] == "https://example.test/v1"
    assert routed_client.config["api_key"] == "test-key"
    tools = seen["tools"]
    assert isinstance(tools, list)
    tool_names = {tool.name for tool in tools}
    assert tool_names == {"execute_python", "return_result"}


@pytest.mark.asyncio
async def test_deterministic_bad_request_is_not_retried():
    calls = 0

    async def deterministic_bad_request():
        nonlocal calls
        calls += 1
        raise BadRequestError(
            message="Function tools with reasoning_effort are not supported here",
            model="gpt-5.6",
            llm_provider="openai",
        )

    with pytest.raises(BadRequestError):
        await with_retry(
            deterministic_bad_request,
            config=RetryConfig(max_retries=3, base_delay=0.01, jitter_factor=0.0),
        )

    assert calls == 1
