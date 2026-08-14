# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ACP server surface."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from acp import PROTOCOL_VERSION, RequestError, resource_link_block, text_block
from acp.schema import AgentMessageChunk, EnvVariable, McpServerStdio, UserMessageChunk
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


async def _session(adapter: CodingACPAdapter, session_id: str):
    return (await adapter._sessions.get(session_id)).value


def test_acp_command_is_discovered_as_cli_plugin():
    assert dict(discover_commands())["acp"] is command


def test_acp_command_passes_the_nvidia_key_for_nvidia_models():
    runner = CliRunner()
    with (
        patch("nooa.secrets.load_secrets_into_env"),
        patch("nooa.unifiedllm.get_llm_client") as get_llm_client,
        patch("nooa_acp.server.serve") as serve,
        patch("nooa_acp.cli.asyncio.run"),
    ):
        result = runner.invoke(
            command,
            ["--model", "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"],
            env={"NVIDIA_API_KEY": "nvapi-test"},
        )

    assert result.exit_code == 0
    llm_factory = serve.call_args.args[0]
    llm_factory()
    get_llm_client.assert_called_once_with(
        "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        client_type=None,
        api_key="nvapi-test",
    )


def test_acp_command_leaves_the_key_alone_for_other_providers():
    runner = CliRunner()
    with (
        patch("nooa.secrets.load_secrets_into_env"),
        patch("nooa.unifiedllm.get_llm_client") as get_llm_client,
        patch("nooa_acp.server.serve") as serve,
        patch("nooa_acp.cli.asyncio.run"),
    ):
        result = runner.invoke(
            command,
            ["--model", "openai/gpt-4o-mini"],
            env={"NVIDIA_API_KEY": "nvapi-test"},
        )

    assert result.exit_code == 0
    serve.call_args.args[0]()
    get_llm_client.assert_called_once_with("openai/gpt-4o-mini", client_type=None)


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
    assert initialized.agent_capabilities.load_session is True
    capabilities = initialized.agent_capabilities.session_capabilities
    assert capabilities.list is not None
    assert capabilities.close is not None
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
    runtime = await _session(adapter, session.session_id)
    result = RespondResult(kind=RespondReason.DONE, explanation="done")
    submit = AsyncMock(return_value=result)
    with patch.object(runtime.dispatcher, "submit", submit):
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
        session = await adapter.new_session(str(tmp_path), mcp_servers=[server])

    create.assert_awaited_once_with(
        "lookup",
        command="lookup-server",
        args=["--stdio"],
        env={"TOKEN": "test"},
    )
    runtime = await _session(adapter, session.session_id)
    assert "mcp.lookup" in runtime.agent.skills.loaded()
    assert "mcp.lookup" in runtime.agent.skills.activated()
    await adapter.close()


async def test_adapter_allows_independent_session_creation(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    server = McpServerStdio(name="lookup", command="lookup-server", args=[], env=[])
    started = asyncio.Event()
    release = asyncio.Event()

    async def create_mcp(*args, **kwargs):
        started.set()
        await release.wait()
        return _MCPTools()

    with patch("nooa_acp.server.MCPManager.create_stdio_server", side_effect=create_mcp):
        first_session = asyncio.create_task(
            adapter.new_session(str(tmp_path), mcp_servers=[server])
        )
        await asyncio.wait_for(started.wait(), 1)
        second = await asyncio.wait_for(adapter.new_session(str(tmp_path)), 1)
        release.set()
        first = await first_session

    assert first.session_id != second.session_id
    await adapter.close()


async def test_adapter_rejects_two_prompts_for_same_session(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    created = await adapter.new_session(str(tmp_path))
    session = await _session(adapter, created.session_id)
    started = asyncio.Event()
    release = asyncio.Event()

    async def submit(_text: str):
        started.set()
        await release.wait()
        return RespondResult(kind=RespondReason.DONE, explanation="done")

    with patch.object(session.dispatcher, "submit", side_effect=submit):
        first = asyncio.create_task(adapter.prompt(created.session_id, [text_block("first")]))
        await asyncio.wait_for(started.wait(), 1)
        with pytest.raises(RequestError):
            await adapter.prompt(created.session_id, [text_block("second")])
        release.set()
        assert (await first).stop_reason == "end_turn"

    await adapter.close()


async def test_adapter_lists_closes_loads_and_replays_durable_session(tmp_path):
    first_client = _RecordingClient()
    first_adapter = CodingACPAdapter(_completed_llm)
    first_adapter.on_connect(first_client)  # type: ignore[arg-type]
    created = await first_adapter.new_session(str(tmp_path))
    await first_adapter.prompt(created.session_id, [text_block("remember this")])

    listed = await first_adapter.list_sessions(str(tmp_path))
    assert [session.session_id for session in listed.sessions] == [created.session_id]
    assert listed.sessions[0].cwd == str(tmp_path)
    await first_adapter.close_session(created.session_id)
    with pytest.raises(RequestError):
        await first_adapter.prompt(created.session_id, [text_block("closed")])
    await first_adapter.close()

    replay_client = _RecordingClient()
    replay_adapter = CodingACPAdapter(_completed_llm)
    replay_adapter.on_connect(replay_client)  # type: ignore[arg-type]
    await replay_adapter.load_session(str(tmp_path), created.session_id)

    replayed_text = [
        update.content.text
        for update in replay_client.updates
        if isinstance(update, AgentMessageChunk)
    ]
    assert replayed_text == ["ACP response"]
    replayed_user_text = [
        update.content.text
        for update in replay_client.updates
        if isinstance(update, UserMessageChunk)
    ]
    assert replayed_user_text == ["remember this"]
    await replay_adapter.close()


async def test_adapter_reports_unknown_session_on_load(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    with pytest.raises(RequestError):
        await adapter.load_session(str(tmp_path), "missing")

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
    runtime = await _session(adapter, session.session_id)

    with patch.object(runtime.dispatcher, "submit", side_effect=GenerationError(message)):
        response = await adapter.prompt(session.session_id, [text_block("do the work")])

    assert response.stop_reason == stop_reason
    await adapter.close()


async def test_adapter_propagates_unrelated_generation_errors(tmp_path):
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]
    session = await adapter.new_session(str(tmp_path))
    runtime = await _session(adapter, session.session_id)

    with (
        patch.object(
            runtime.dispatcher,
            "submit",
            side_effect=GenerationError("LLM API error after retries"),
        ),
        pytest.raises(GenerationError, match="LLM API error"),
    ):
        await adapter.prompt(session.session_id, [text_block("do the work")])

    await adapter.close()


async def test_prompt_on_a_closing_session_is_a_clean_protocol_error(tmp_path):
    """A prompt racing a close must not surface as a raw internal error.

    turn() raises SessionRuntimeClosedError as well as SessionBusyError; only
    the latter was translated, so the client got -32603 with no actionable
    reason instead of a typed protocol error.
    """
    client = _RecordingClient()
    adapter = CodingACPAdapter(_completed_llm)
    adapter.on_connect(client)  # type: ignore[arg-type]

    await adapter.initialize(PROTOCOL_VERSION)
    session = await adapter.new_session(str(tmp_path))
    runtime = await adapter._sessions.get(session.session_id)
    await runtime.close()

    with pytest.raises(RequestError):
        await adapter.prompt(session.session_id, [text_block("do the work")])
