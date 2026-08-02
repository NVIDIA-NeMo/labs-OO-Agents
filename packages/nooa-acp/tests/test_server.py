# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ACP server surface."""

import asyncio
import threading
from unittest.mock import AsyncMock, patch

import pytest
from acp import PROTOCOL_VERSION, RequestError, resource_link_block, text_block
from acp.schema import AgentMessageChunk, EnvVariable, McpServerStdio
from click.testing import CliRunner
from nooa_acp.cli import command
from nooa_acp.server import CodingACPAdapter
from nooa_cli.commands import discover_commands

from nooa.errors import GenerationError
from nooa.interactive import RespondReason, RespondResult
from nooa.unifiedllm import FakeLLMClient


def _completed_llm() -> FakeLLMClient:
    return FakeLLMClient.with_tool_call(
        "execute_python",
        {
            "code": (
                "self.message('ACP response')\n"
                "return_result(RespondReason.DONE, explanation='request complete')"
            )
        },
    )


class _RecordingClient:
    def __init__(self) -> None:
        self.updates: list[object] = []

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append(update)


def test_acp_command_is_discovered_as_cli_plugin():
    assert dict(discover_commands())["acp"] is command


def test_acp_command_defaults_to_public_nvidia_model():
    runner = CliRunner()
    with (
        patch("nooa.secrets.load_secrets_into_env"),
        patch("nooa.unifiedllm.get_llm_client") as get_llm_client,
        patch("nooa_acp.server.serve") as serve,
        patch("nooa_acp.cli.asyncio.run"),
    ):
        result = runner.invoke(command, env={"NVIDIA_API_KEY": "nvapi-test"})

    assert result.exit_code == 0
    llm_factory = serve.call_args.args[0]
    llm_factory()
    get_llm_client.assert_called_once_with(
        "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        client_type=None,
        api_key="nvapi-test",
    )


class _MCPTools:
    async def lookup(self, query: str) -> str:
        """Look up a value in the test MCP server."""
        return query


async def test_adapter_completes_one_session_prompt(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    initialized = await adapter.initialize(PROTOCOL_VERSION)
    session = await adapter.new_session(str(tmp_path))
    response = await adapter.prompt(session.session_id, [text_block("do the work")])

    assert initialized.protocol_version == PROTOCOL_VERSION
    assert initialized.agent_info is not None
    assert initialized.agent_info.name == "nooa-acp"
    assert response.stop_reason == "end_turn"
    assert any(
        isinstance(update, AgentMessageChunk) and update.content.text == "ACP response"
        for update in client.updates
    )
    await adapter.close()


async def test_adapter_includes_resource_links_in_prompt(tmp_path):
    llm = _completed_llm()
    client = _RecordingClient()
    adapter = CodingACPAdapter(lambda: llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    session = await adapter.new_session(str(tmp_path))
    await adapter.prompt(
        session.session_id,
        [resource_link_block("README", "file:///workspace/README.md")],
    )

    assert "Resource README: file:///workspace/README.md" in str(llm.last_messages)
    await adapter.close()


async def test_adapter_preserves_prompt_whitespace(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    session = await adapter.new_session(str(tmp_path))
    assert adapter._dispatcher is not None
    result = RespondResult(kind=RespondReason.DONE, explanation="done")
    submit = AsyncMock(return_value=result)
    with patch.object(adapter._dispatcher, "submit", submit):
        await adapter.prompt(session.session_id, [text_block("  indented\n")])

    submit.assert_awaited_once_with("  indented\n")
    await adapter.close()


async def test_adapter_connects_baseline_stdio_mcp_servers(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    server = McpServerStdio(
        name="lookup",
        command="lookup-server",
        args=["--stdio"],
        env=[EnvVariable(name="TOKEN", value="test")],
    )

    with patch(
        "nooa_acp.server.MCPManager.create_stdio_server",
        new=AsyncMock(return_value=_MCPTools()),
    ) as create:
        await adapter.new_session(str(tmp_path), mcp_servers=[server])

    create.assert_awaited_once_with(
        "lookup",
        command="lookup-server",
        args=["--stdio"],
        env={"TOKEN": "test"},
    )
    await adapter.close()


async def test_adapter_rejects_concurrent_session_creation(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    server = McpServerStdio(name="lookup", command="lookup-server", args=[], env=[])
    started = threading.Event()
    release = threading.Event()

    async def create_mcp(*args, **kwargs):
        started.set()
        await asyncio.to_thread(release.wait, 2)
        return _MCPTools()

    with patch("nooa_acp.server.MCPManager.create_stdio_server", side_effect=create_mcp):
        first_session = asyncio.create_task(
            adapter.new_session(str(tmp_path), mcp_servers=[server])
        )
        await asyncio.to_thread(started.wait, 1)
        with pytest.raises(RequestError):
            await adapter.new_session(str(tmp_path))
        release.set()
        await first_session

    await adapter.close()


async def test_adapter_rejects_duplicate_mcp_names_before_startup(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    servers = [
        McpServerStdio(name="lookup", command="first", args=[], env=[]),
        McpServerStdio(name="lookup", command="second", args=[], env=[]),
    ]

    with patch(
        "nooa_acp.server.MCPManager.create_stdio_server",
        new=AsyncMock(return_value=_MCPTools()),
    ) as create:
        with pytest.raises(RequestError):
            await adapter.new_session(str(tmp_path), mcp_servers=servers)

    create.assert_not_awaited()
    await adapter.close()


@pytest.mark.parametrize(
    ("message", "stop_reason"),
    [
        (
            "Empty response: the model used all available output tokens on reasoning; "
            "increase `max_tokens`.",
            "max_tokens",
        ),
        (
            "Generation failed after 10 iterations (max_iterations=10).",
            "max_turn_requests",
        ),
    ],
)
async def test_adapter_maps_generation_limits_to_stop_reasons(
    tmp_path, message: str, stop_reason: str
):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    session = await adapter.new_session(str(tmp_path))
    assert adapter._dispatcher is not None

    with patch.object(adapter._dispatcher, "submit", side_effect=GenerationError(message)):
        response = await adapter.prompt(session.session_id, [text_block("do the work")])

    assert response.stop_reason == stop_reason
    await adapter.close()


async def test_adapter_propagates_unrelated_generation_errors(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    session = await adapter.new_session(str(tmp_path))
    assert adapter._dispatcher is not None

    with (
        patch.object(
            adapter._dispatcher,
            "submit",
            side_effect=GenerationError("LLM API error after retries"),
        ),
        pytest.raises(GenerationError, match="LLM API error"),
    ):
        await adapter.prompt(session.session_id, [text_block("do the work")])

    await adapter.close()
