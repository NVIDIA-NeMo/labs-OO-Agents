# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in NVIDIA Hub cache/reasoning checks across a real SQLite close/reopen.

Run with NVIDIA_INFERENCE_API_KEY and NOOA_RUN_CACHE_RESUME_LIVE=1:
    uv run pytest tests/integration/test_cache_resume_live.py -m integration -s

Three calls per provider: obtain a signed tool turn, warm its stable prefix,
then reopen the event archive and repeat with changed trailing live context.
Only usage and equality checks are printed; opaque payloads stay in memory or
the temporary session database. Roughly 90k input tokens across all providers.

Additional opt-in cases cover readable reasoning from Nemotron/Qwen/DeepSeek
and cross-provider replay. Saved-turn switch cases require NOOA_LIVE_ARCHIVE_ROOT
pointing to a completed three-provider run's pytest-N directory, so they reuse
its seeds instead of spending inference tokens regenerating them.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from nooa._immutable_json import json_containers
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


def _secrets(response):
    """Extract only native replay strings, without printing their contents."""
    keys = {
        "encrypted_content",
        "signature",
        "data",
        "thought_signature",
        "inline_thought_signature",
    }

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in keys and isinstance(child, str):
                    yield child
                else:
                    yield from visit(child)
        elif isinstance(value, list):
            for child in value:
                yield from visit(child)

    return list(visit([json_containers(part.native) for part in response.parts if part.native]))


def _report_usage(family, phase, response):
    print(
        json.dumps(
            {
                "family": family,
                "phase": phase,
                "finish_reason": response.finish_reason,
                "native_strings": len(_secrets(response)),
                "usage": response.usage.model_dump() if response.usage else None,
            }
        ),
        flush=True,
    )


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
        _report_usage(family, "seed", seed)
        assert _secrets(seed), (
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
        _report_usage(family, "warm", warm)

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
    assert saved_response.parts == seed.parts
    assert saved_response.replay_scope == seed.replay_scope
    assert saved_response.usage == seed.usage
    replay_messages = _render(family, restored, instructions, "phase=resumed")
    # Archive loading intentionally omits transient SDK responses and parsed
    # objects. Compare the public messages here; parts/native are checked above.
    assert [dict(message) for message in warm_messages[:-1]] == [
        dict(message) for message in replay_messages[:-1]
    ]
    assert warm_messages[-1] != replay_messages[-1]
    async with _client(family) as client:
        resumed = await client.acall(replay_messages, tools=[TOOL])
        _report_usage(family, "resumed", resumed)

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
    wire_json = json.dumps(resumed_wire)
    assert all(secret in wire_json for secret in _secrets(saved_response)), (
        "native replay string missing on wire"
    )
    assert seed.parts == saved_response.parts, "request construction mutated the archive"
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


@pytest.mark.asyncio
@pytest.mark.parametrize("source,target", [(s, t) for s in MODELS for t in MODELS if s != t])
async def test_saved_turn_switches_provider_without_private_state(source, target, monkeypatch):
    """Reuse the three saved live sessions; never send foreign state even during a test."""
    archive_root = os.getenv("NOOA_LIVE_ARCHIVE_ROOT")
    if not archive_root:
        pytest.skip("set NOOA_LIVE_ARCHIVE_ROOT to the completed resume run's pytest directory")
    database = (
        Path(archive_root)
        / f"test_reasoning_and_prompt_cach{list(MODELS).index(source)}"
        / "session.db"
    )
    await _check_provider_switch(source, target, database, monkeypatch)


async def _check_provider_switch(source, target, database, monkeypatch, *, require_private=True):
    with SQLiteStorageManager(database) as storage:
        events = list(storage.event_backend.all_events())
    response = next(event for event in events if isinstance(event, LLMResponse))
    native_strings = _secrets(response)
    if require_private:
        assert native_strings, "source archive has no opaque state"
    readable = [part.text for part in response.parts if part.kind == "reasoning" and part.text]
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    original_send = httpx.AsyncClient.send
    requests = []

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)

    async def guarded_send(client, request, *args, **kwargs):
        if request.url.host == "inference-api.nvidia.com" and request.method == "POST":
            body = json.loads(request.content)
            encoded = json.dumps(body)
            # Enforce before the network call, not after a potential disclosure.
            assert not any(secret in encoded for secret in native_strings), (
                "foreign opaque state reached dispatch"
            )
            assert all(any(text in value for value in strings(body)) for text in readable), (
                "portable reasoning missing on wire"
            )
            assert "nooa_turn" not in encoded
            requests.append(body)
        return await original_send(client, request, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", guarded_send)
    messages = _render(
        target,
        events,
        "The verification tool has completed. Reply only OK.",
        "phase=model-switched",
    )
    async with _client(target) as client:
        result = await client.acall(messages, tools=[TOOL])
    assert len(requests) == 1
    _report_usage(f"{source}->{target}", "switched_after_resume", result)
    assert result.finish_reason == "stop"
    print(
        json.dumps(
            {
                "source": source,
                "target": target,
                "source_native_strings": len(native_strings),
                "opaque_state_stripped": True,
                "readable_reasoning_parts_preserved": len(readable),
            }
        ),
        flush=True,
    )


@pytest.mark.asyncio
async def test_readable_reasoning_survives_sqlite_and_provider_switch(tmp_path, monkeypatch):
    """Demand nonempty real reasoning text, so the transfer check is not vacuous."""
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    events = [
        UserEvent(
            tag="1",
            content="Find the smallest integer greater than 1000 that leaves remainders 2, 3, and 4 when divided by 5, 7, and 9 respectively. Explain briefly why no smaller qualifying integer works.",
        )
    ]
    async with _client("openai") as client:
        seed = await client.acall(
            _render(
                "openai",
                events,
                "Solve the user's arithmetic question carefully.",
                "phase=readable-seed",
            ),
            reasoning={"effort": "medium", "summary": "auto"},
        )
    _report_usage("openai", "readable_seed", seed)
    assert seed.reasoning, "provider returned no readable summary; text transfer not verified"
    assert _secrets(seed), "provider returned no opaque state alongside summary"
    seed.tag = "2"
    events.append(seed)
    database = tmp_path / "test_reasoning_and_prompt_cach0" / "session.db"
    database.parent.mkdir()
    with SQLiteStorageManager(database) as storage:
        for event in events:
            storage.event_backend.store(event.tag, event)
    with SQLiteStorageManager(database) as storage:
        restored = next(e for e in storage.event_backend.all_events() if isinstance(e, LLMResponse))
    assert restored.reasoning == seed.reasoning
    assert restored.parts == seed.parts
    for target in ("anthropic", "gemini"):
        with monkeypatch.context() as isolated:
            await _check_provider_switch("openai", target, database, isolated)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,model",
    [
        ("nemotron", "openai/nvidia/nvidia/nemotron-3-ultra"),
        ("qwen", "openai/nvidia/qwen/qwen3-5-397b-a17b"),
        ("deepseek", "openai/nvidia/deepseek-ai/deepseek-v4-pro"),
    ],
)
async def test_plain_reasoning_from_hub_models_survives_resume(
    source, model, tmp_path, monkeypatch
):
    """Verify actual reasoning_content, not a synthetic thought or a provider summary."""
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    events = [
        UserEvent(
            tag="1",
            content="Is 17 times 19 smaller than 18 squared? Work it out briefly and give the difference.",
        )
    ]
    async with CompletionClient(
        model=model,
        api_base="https://inference-api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_INFERENCE_API_KEY"],
        max_tokens=1536,
        http_config=HttpConfig(read_timeout=120),
        num_retries=0,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    ) as client:
        seed = await client.acall(
            _render(source, events, "Solve the arithmetic problem concisely.", "phase=plain-seed")
        )
    _report_usage(source, "plain_seed", seed)
    raw_text = getattr(seed.raw_response.choices[0].message, "reasoning_content", None)
    assert raw_text, "route did not return reasoning_content"
    assert seed.reasoning == raw_text, "normalization lost plain reasoning"
    assert not _secrets(seed), "expected portable text, not opaque reasoning"
    assert seed.finish_reason == "stop"
    seed.tag = "2"
    events.append(seed)
    database = tmp_path / "plain-session.db"
    with SQLiteStorageManager(database) as storage:
        for event in events:
            storage.event_backend.store(event.tag, event)
    with SQLiteStorageManager(database) as storage:
        restored = next(e for e in storage.event_backend.all_events() if isinstance(e, LLMResponse))
    assert restored.raw_response is None
    assert restored.reasoning == raw_text
    print(
        json.dumps(
            {
                "source": source,
                "model": model,
                "reasoning_characters": len(raw_text),
                "sqlite_reasoning_equal": True,
            }
        ),
        flush=True,
    )
    for target in MODELS:
        with monkeypatch.context() as isolated:
            await _check_provider_switch(source, target, database, isolated, require_private=False)
