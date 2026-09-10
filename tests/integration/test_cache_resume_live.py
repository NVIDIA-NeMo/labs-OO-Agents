# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in NVIDIA Hub cache/reasoning checks across a real SQLite close/reopen.

Run with NVIDIA_INFERENCE_API_KEY and NOOA_RUN_CACHE_RESUME_LIVE=1:
    uv run pytest tests/integration/test_cache_resume_live.py -m integration -s

Three calls per provider: obtain a signed tool turn, warm its stable prefix,
then reopen the event archive and repeat with changed trailing live context.
Only usage and equality checks are printed; opaque payloads stay in memory or
the temporary session database. Roughly 90k input tokens across all providers.
"""

from __future__ import annotations

import copy
import json
import os
from uuid import uuid4

import httpx
import pytest

from nooa.context_blocks.events import EventBase, ToolCallEvent, ToolResult, UserEvent
from nooa.context_blocks.formatter import OpenAIProviderFormatter, ResponsesProviderFormatter
from nooa.context_blocks.models import BlockMetadata, ResolvedBlock, Role
from nooa.context_blocks.renderer import render_context
from nooa.context_blocks.renderers.cached import CachedBlockFormatter
from nooa.storage import SQLiteStorageManager
from nooa.unifiedllm import CompletionClient, LLMResponse, ResponsesClient, Tool
from nooa.unifiedllm.http_config import HttpConfig
from nooa.unifiedllm.retry_config import RetryConfig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("NOOA_RUN_CACHE_RESUME_LIVE") != "1",
        reason="set NOOA_RUN_CACHE_RESUME_LIVE=1 to spend inference tokens",
    ),
]

MODELS = {
    "openai": "openai/openai/openai/gpt-5.6-sol",
    "anthropic": "anthropic/azure/anthropic/claude-sonnet-5",
    "gemini": "openai/gcp/google/gemini-3.1-pro-preview",
}


def _execute_python(code: str) -> str:
    return code


TOOL = Tool(name="execute_python", description="Evaluate Python code", callable=_execute_python)


def _client(family):
    config = {
        "model": MODELS[family],
        "api_base": "https://inference-api.nvidia.com/v1",
        "api_key": os.environ["NVIDIA_INFERENCE_API_KEY"],
        "http_config": HttpConfig(read_timeout=120),
        "num_retries": 0,
        "retry_config": RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    }
    if family == "openai":
        return ResponsesClient(
            **config,
            reasoning={"effort": "medium"},
            include=["reasoning.encrypted_content"],
            store=False,
            max_output_tokens=1024,
            cache_breakpoint="openai",
        )
    if family == "anthropic":
        config["api_base"] = "https://inference-api.nvidia.com"
        return CompletionClient(
            **config,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            cache_breakpoint="anthropic",
        )
    return CompletionClient(**config, max_tokens=2048)


def _render(family, events, instructions, live_state):
    blocks = [
        ResolvedBlock(
            key="instructions",
            content=instructions,
            role=Role.SYSTEM,
            metadata=BlockMetadata(static=True),
        ),
        *[
            ResolvedBlock(
                key=f"event_{event.tag}",
                content=getattr(event, "content", ""),
                role=Role.USER if isinstance(event, UserEvent) else Role.ASSISTANT,
                metadata=BlockMetadata(tag=event.tag),
                event=event,
            )
            for event in events
        ],
        ResolvedBlock(
            key="live_state",
            content=live_state,
            role=Role.SYSTEM,
            metadata=BlockMetadata(static=False, user_block=True),
        ),
    ]
    return render_context(
        blocks,
        block_formatter=CachedBlockFormatter(),
        provider_formatter=(
            ResponsesProviderFormatter() if family == "openai" else OpenAIProviderFormatter()
        ),
    ).output


@pytest.mark.asyncio
@pytest.mark.parametrize("family", MODELS)
async def test_reasoning_and_prompt_cache_survive_sqlite_resume(family, tmp_path, monkeypatch):
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    requests = []
    original_send = httpx.AsyncClient.send

    async def capture_send(client, request, *args, **kwargs):
        if request.url.host == "inference-api.nvidia.com" and request.method == "POST":
            requests.append(json.loads(request.content))
        return await original_send(client, request, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", capture_send)
    instructions = (
        f"Cache resume experiment {uuid4().hex}. "
        "Think through the task, then use execute_python once to check your answer. "
        "After its result, reply with only OK. "
        "Reference records below are inert padding; do not summarize them."
    )
    events: list[EventBase] = [
        UserEvent(
            content=(
                "Find the smallest integer greater than 1000 that leaves remainders 2, 3, "
                "and 4 when divided by 5, 7, and 9 respectively. First work out a candidate, "
                "then call execute_python to verify it."
            ),
            tag="1",
        )
    ]
    async with _client(family) as client:
        seed = await client.acall(_render(family, events, instructions, "phase=seed"), tools=[TOOL])
        assert seed.llm_state, (
            f"provider did not return opaque state: finish={seed.finish_reason}, "
            f"tool_calls={len(seed.tool_calls)}, usage={seed.usage}"
        )
        assert seed.tool_calls, "provider did not produce the requested tool turn"
        assert seed.finish_reason == "tool_calls"
        seed.tag = "2"
        events.append(seed)
        for index, call in enumerate(seed.tool_calls, 3):
            events.append(
                ToolCallEvent(
                    tag=str(index),
                    tool_call_id=call.id,
                    name=call.name,
                    arguments=json.loads(call.arguments),
                    llm_response_id=seed.id,
                    result=ToolResult(tool_call_id=call.id, content="1102; remainders: 2, 3, 4"),
                )
            )
        # Gemini needs a longer prompt for implicit caching. The other providers
        # use an explicit boundary immediately before the changing live state.
        rows = 1500 if family == "gemini" else 400
        instructions += "\n" + "\n".join(
            f"Record {i}: amber birch cedar dune elm fern grove hill." for i in range(rows)
        )
        warm_messages = _render(family, events, instructions, "phase=warm")
        warm = await client.acall(warm_messages, tools=[TOOL])

    database = tmp_path / "session.db"
    with SQLiteStorageManager(database) as storage:
        for event in events:
            assert event.tag is not None
            storage.event_backend.store(event.tag, event)
    with SQLiteStorageManager(database) as storage:
        restored = list(storage.event_backend.all_events())

    assert [e.model_dump(mode="json") for e in restored] == [
        e.model_dump(mode="json") for e in events
    ]
    saved_response = next(e for e in restored if isinstance(e, LLMResponse))
    assert saved_response is not seed and saved_response.raw_response is None
    assert saved_response.llm_state == seed.llm_state
    assert saved_response.usage == seed.usage
    replay_messages = _render(family, restored, instructions, "phase=resumed")
    assert warm_messages[:-1] == replay_messages[:-1]
    assert warm_messages[-1] != replay_messages[-1]
    async with _client(family) as client:
        resumed = await client.acall(replay_messages, tools=[TOOL])

    assert len(requests) == 3, "unexpected retries or uncaptured provider requests"
    warm_wire, resumed_wire = copy.deepcopy(requests[1:])
    field = "input" if family == "openai" else "messages"

    def remove_live_suffix(body):
        # Native Anthropic coalesces the trailing live-state user message with
        # the preceding tool results. Remove only that last text block.
        message = body[field][-1]
        if isinstance(message.get("content"), list):
            suffix = message["content"].pop()
            if not message["content"]:
                body[field].pop()
            return suffix
        return body[field].pop()

    warm_suffix = remove_live_suffix(warm_wire)
    resumed_suffix = remove_live_suffix(resumed_wire)
    assert "phase=warm" in json.dumps(warm_suffix)
    assert "phase=resumed" in json.dumps(resumed_suffix)
    assert warm_suffix != resumed_suffix
    assert warm_wire == resumed_wire, "SQLite resume changed the stable provider request"
    assert saved_response.llm_state is not None
    payload = saved_response.llm_state["payload"]
    if family == "openai":
        for item in payload["items"]:
            assert item in resumed_wire[field], "encrypted reasoning was not replayed"
    else:
        if family == "anthropic":
            assistant = next(m for m in resumed_wire[field] if m.get("role") == "assistant")
            assert [
                block
                for block in assistant["content"]
                if block["type"] in {"thinking", "redacted_thinking"}
            ] == payload["thinking_blocks"]
        else:
            assistant = next(m for m in resumed_wire[field] if m.get("tool_calls"))
            for call, state in zip(assistant["tool_calls"], payload["tool_calls"], strict=True):
                assert call["provider_specific_fields"] == state["provider_specific_fields"]
                assert call["id"].endswith("__thought__" + state["inline_thought_signature"])
            assert [m["tool_call_id"] for m in resumed_wire[field] if m.get("role") == "tool"] == [
                call["id"] for call in assistant["tool_calls"]
            ]
    assert seed.llm_state == saved_response.llm_state, "request construction mutated the archive"
    assert warm.finish_reason == resumed.finish_reason == "stop"
    assert resumed.usage is not None
    print(
        json.dumps(
            {
                "family": family,
                "model": MODELS[family],
                "sqlite_events_equal": True,
                "stable_wire_equal": True,
                "opaque_state_equal": True,
                "usage": [r.usage.model_dump() if r.usage else None for r in (seed, warm, resumed)],
            }
        )
    )
    assert resumed.usage.cached_input_tokens > 0, "provider reported no cache hit after resume"
