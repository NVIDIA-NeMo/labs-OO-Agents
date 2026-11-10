# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden wire bodies for CompletionClient and ResponsesClient.

Each case patches the LiteLLM entry points, records the exact keyword
arguments the client passes on, and compares them with a JSON file under
``snapshots/wire``. The files were recorded on the code before the adapter
split, so any change in what reaches LiteLLM fails here.

To re-record after an intended wire change, run with
``NOOA_UPDATE_WIRE_SNAPSHOTS=1`` and review the JSON diff.
"""

import copy
import json
import os
from pathlib import Path
from typing import Any

import httpx
import litellm
import pytest
from pydantic import BaseModel, Field

from nooa.llm_types import CacheBoundary, LLMResponse
from nooa.unifiedllm import CompletionClient, ResponsesClient, Tool
from nooa.unifiedllm.retry_config import RetryConfig

SNAPSHOT_DIR = Path(__file__).parent / "snapshots" / "wire"
UPDATE = os.environ.get("NOOA_UPDATE_WIRE_SNAPSHOTS") == "1"
NO_RETRY = RetryConfig(max_retries=0, rate_limit_extra_retries=0)
IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


def lookup(query: str, limit: int = 3) -> str:
    """Look something up."""
    raise AssertionError("never executed")


TOOL = Tool(name="lookup", description="Look something up.", callable=lookup)


class Answer(BaseModel):
    value: int
    label: str


class Loose(BaseModel):
    counts: dict[str, int]


class Bounded(BaseModel):
    score: int = Field(ge=0, le=10)
    name: str = Field(max_length=5)


# ---------------------------------------------------------------------------
# Canned provider replies
# ---------------------------------------------------------------------------


def chat_reply(content=None, *, tool_calls=None, reasoning=None, thinking=None, model="m"):
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if thinking is not None:
        message["thinking_blocks"] = thinking
    return litellm.ModelResponse(
        id="chat_1",
        created=0,
        model=model,
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    )


def responses_reply(output):
    return litellm.ResponsesAPIResponse(
        id="resp_1",
        object="response",
        created_at=0,
        status="completed",
        model="gpt-5.6",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
    )


def responses_text(text):
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "lookup", "arguments": '{"query": "x"}'},
}


# ---------------------------------------------------------------------------
# Stable serialization
# ---------------------------------------------------------------------------


def normalize(value: Any) -> Any:
    if isinstance(value, type) and issubclass(value, BaseModel):
        return {"__pydantic_model__": value.__name__, "schema": value.model_json_schema()}
    if isinstance(value, BaseModel) and not isinstance(value, (LLMResponse, CacheBoundary)):
        return {"__pydantic__": type(value).__name__, "data": normalize(value.model_dump())}
    if isinstance(value, (LLMResponse, CacheBoundary)):
        return {"__unexpected__": type(value).__name__}
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__name__}>"


def normalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in kwargs.items():
        if key == "client":
            out[key] = {
                "__client__": type(value).__name__,
                "max_retries": getattr(value, "max_retries", None),
            }
        else:
            out[key] = normalize(value)
    return out


def summarize(response: LLMResponse) -> dict[str, Any]:
    return {
        "content": response.content,
        "reasoning": response.reasoning,
        "tool_calls": [[c.id, c.name, c.arguments] for c in response.tool_calls],
        "parsed": None if response.parsed is None else response.parsed.model_dump(),
        "finish_reason": response.finish_reason,
        "usage": None if response.usage is None else response.usage.model_dump(),
        "replay_scope": response.replay_scope,
        "parts": [type(p).__name__ for p in response.parts],
    }


# ---------------------------------------------------------------------------
# Wire capture
# ---------------------------------------------------------------------------


@pytest.fixture
def wire(monkeypatch):
    """Patch every LiteLLM call the clients make; fail on any real network use."""
    captured: list[dict[str, Any]] = []
    replies: list[Any] = []

    def record(api, kwargs):
        captured.append({"api": api, "kwargs": json.loads(json.dumps(normalize_kwargs(kwargs)))})
        return copy.deepcopy(replies.pop(0))

    async def acompletion(**kwargs):
        return record("acompletion", kwargs)

    async def aresponses(**kwargs):
        return record("aresponses", kwargs)

    def no_network(*args, **kwargs):
        raise AssertionError("wire snapshot test reached the network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    monkeypatch.setattr(litellm, "completion", lambda **kw: record("completion", kw))
    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(litellm, "responses", lambda **kw: record("responses", kw))
    monkeypatch.setattr(litellm, "aresponses", aresponses)
    return captured, replies


async def run_case(case, is_async):
    """Run one case; return the LLMResponse of every call in order."""
    results = []
    client = case["client"]()
    try:
        history: list[Any] = []
        for step in case["steps"]:
            messages = step["messages"](history)
            kwargs = step.get("kwargs", {})
            args = (messages, step.get("tools"), step.get("output_model"))
            result = (
                await client.acall(*args, **kwargs) if is_async else client.call(*args, **kwargs)
            )
            history.append(result)
            results.append(result)
    finally:
        await client.aclose()
    return results


def chat(model="openai/gpt-4o-mini", **kwargs):
    kwargs.setdefault("api_key", "sk-test")
    kwargs.setdefault("api_base", "https://gateway.test/v1")
    return lambda: CompletionClient(model, retry_config=NO_RETRY, **kwargs)


def responses(model="openai/gpt-5.6", **kwargs):
    kwargs.setdefault("api_key", "sk-test")
    return lambda: ResponsesClient(model, retry_config=NO_RETRY, **kwargs)


SYSTEM = {"role": "system", "content": "You are terse."}
USER = {"role": "user", "content": "Hello"}


def const(*messages):
    return lambda history: [copy.deepcopy(m) for m in messages]


def tool_history(history):
    """User, the previous reply passed back unchanged, then its tool result."""
    return [
        SYSTEM,
        USER,
        history[0],
        {"role": "tool", "tool_call_id": history[0].tool_calls[0].id, "content": "found"},
    ]


CASES: dict[str, dict[str, Any]] = {
    "chat_plain": {
        "client": chat(temperature=0.2),
        "steps": [{"messages": const(SYSTEM, USER), "kwargs": {"prompt_cache_key": "conv-1"}}],
        "replies": [chat_reply("Hi")],
    },
    "chat_tools": {
        "client": chat(),
        "steps": [
            {"messages": const(SYSTEM, USER), "tools": [TOOL]},
            {"messages": const(SYSTEM, USER), "tools": [TOOL], "kwargs": {"tool_choice": "auto"}},
            {
                "messages": const(SYSTEM, USER),
                "tools": [TOOL],
                "kwargs": {"tool_choice": "required"},
            },
            {"messages": const(SYSTEM, USER), "kwargs": {"tool_choice": "required"}},
        ],
        "replies": [
            chat_reply(None, tool_calls=[TOOL_CALL]),
            chat_reply("plain"),
            chat_reply(None, tool_calls=[TOOL_CALL]),
            chat_reply("no tools"),
        ],
    },
    "chat_output_strict": {
        "client": chat(),
        "steps": [{"messages": const(USER), "output_model": Answer}],
        "replies": [chat_reply('{"value": 3, "label": "x"}')],
    },
    "chat_output_loose": {
        "client": chat(),
        "steps": [{"messages": const(USER), "output_model": Loose}],
        "replies": [chat_reply('```json\n{"counts": {"a": 1}}\n```')],
    },
    "chat_output_native_anthropic": {
        "client": chat(
            "anthropic/claude-sonnet-4-5",
            api_base="https://anthropic.test",
            structured_output_mode="anthropic",
            max_tokens=512,
        ),
        "steps": [{"messages": const(USER), "output_model": Answer}],
        "replies": [chat_reply('{"value": 1, "label": "a"}')],
    },
    "chat_output_native_gateway": {
        "client": chat(
            "openai/azure/anthropic/claude-opus-5-5",
            structured_output_mode="anthropic",
            max_tokens=512,
            thinking={"type": "adaptive"},
        ),
        "steps": [{"messages": const(USER), "output_model": Answer}],
        "replies": [chat_reply('{"value": 1, "label": "a"}')],
    },
    "chat_image": {
        "client": chat(media_inputs=["image"]),
        "steps": [
            {
                "messages": const(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            {"type": "image_url", "image_url": {"url": IMAGE}},
                        ],
                    }
                )
            }
        ],
        "replies": [chat_reply("A pixel.")],
    },
    "chat_image_text_only": {
        "client": chat(),
        "steps": [
            {
                "messages": const(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            {"type": "image_url", "image_url": {"url": IMAGE}},
                        ],
                    }
                )
            }
        ],
        "replies": [chat_reply("No image.")],
    },
    "chat_cache_boundary_anthropic": {
        "client": chat("anthropic/claude-sonnet-4-5", api_base="https://anthropic.test"),
        "steps": [
            {
                "messages": const(
                    SYSTEM,
                    {"role": "user", "content": "stable context"},
                    CacheBoundary(),
                    {"role": "user", "content": "fresh question"},
                ),
                "tools": [TOOL],
            }
        ],
        "replies": [chat_reply("ok")],
    },
    "chat_cache_boundary_openai": {
        "client": chat(),
        "steps": [
            {
                "messages": const(
                    SYSTEM,
                    {"role": "user", "content": "stable context"},
                    CacheBoundary(),
                    {"role": "user", "content": "fresh question"},
                )
            }
        ],
        "replies": [chat_reply("ok")],
    },
    "chat_replay_anthropic_thinking": {
        "client": chat("anthropic/claude-sonnet-4-5", api_base="https://anthropic.test"),
        "steps": [
            {"messages": const(SYSTEM, USER), "tools": [TOOL]},
            {"messages": tool_history, "tools": [TOOL]},
        ],
        "replies": [
            chat_reply(
                None,
                tool_calls=[TOOL_CALL],
                reasoning="thinking it over",
                thinking=[
                    {"type": "thinking", "thinking": "thinking it over", "signature": "sig-1"}
                ],
            ),
            chat_reply("done"),
        ],
    },
    "chat_replay_reasoning_content": {
        "client": chat("openai/gateway-deepseek"),
        "steps": [
            {"messages": const(SYSTEM, USER), "tools": [TOOL]},
            {"messages": tool_history, "tools": [TOOL]},
        ],
        "replies": [
            chat_reply(None, tool_calls=[TOOL_CALL], reasoning="plain reasoning"),
            chat_reply("done"),
        ],
    },
    "chat_bedrock": {
        "client": chat(
            "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            api_base=None,
            aws_region_name="us-east-1",
        ),
        "steps": [
            {"messages": const(SYSTEM, USER), "tools": [TOOL]},
            {"messages": tool_history, "output_model": Bounded},
        ],
        "replies": [
            chat_reply(None, tool_calls=[TOOL_CALL]),
            chat_reply('{"score": 4, "name": "abc"}'),
        ],
    },
    "chat_reasoning_level": {
        "client": chat(
            reasoning_levels={
                "low": {"reasoning_effort": "low"},
                "high": {"reasoning_effort": "high", "max_tokens": 4096},
            },
            reasoning_level="high",
        ),
        "steps": [
            {"messages": const(USER)},
            {"messages": const(USER), "kwargs": {"reasoning_level": "low"}},
        ],
        "replies": [chat_reply("a"), chat_reply("b")],
    },
    "responses_plain": {
        "client": responses(temperature=0.2),
        "steps": [{"messages": const(SYSTEM, USER), "kwargs": {"prompt_cache_key": "conv-1"}}],
        "replies": [responses_reply([responses_text("Hi")])],
    },
    "responses_tools": {
        "client": responses(),
        "steps": [
            {"messages": const(SYSTEM, USER), "tools": [TOOL]},
            {
                "messages": const(SYSTEM, USER),
                "tools": [TOOL],
                "kwargs": {"tool_choice": "required"},
            },
        ],
        "replies": [
            responses_reply(
                [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": '{"query": "x"}',
                        "status": "completed",
                    }
                ]
            ),
            responses_reply([responses_text("plain")]),
        ],
    },
    "responses_output_strict": {
        "client": responses(),
        "steps": [{"messages": const(USER), "output_model": Answer}],
        "replies": [responses_reply([responses_text('{"value": 3, "label": "x"}')])],
    },
    "responses_output_loose": {
        "client": responses(),
        "steps": [{"messages": const(USER), "output_model": Loose}],
        "replies": [responses_reply([responses_text('{"counts": {"a": 1}}')])],
    },
    "responses_image": {
        "client": responses(media_inputs=["image"]),
        "steps": [
            {
                "messages": const(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            {"type": "image_url", "image_url": {"url": IMAGE, "detail": "low"}},
                        ],
                    }
                )
            }
        ],
        "replies": [responses_reply([responses_text("A pixel.")])],
    },
    "responses_cache_boundary": {
        "client": responses(),
        "steps": [
            {
                "messages": const(
                    SYSTEM,
                    {"role": "user", "content": "stable context " * 20},
                    CacheBoundary(),
                    {"role": "user", "content": "fresh question"},
                )
            }
        ],
        "replies": [responses_reply([responses_text("ok")])],
    },
    "responses_replay_encrypted_reasoning": {
        "client": responses(),
        "steps": [
            {"messages": const(SYSTEM, USER), "tools": [TOOL]},
            {"messages": tool_history, "tools": [TOOL]},
        ],
        "replies": [
            responses_reply(
                [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "enc-1",
                        "summary": [{"type": "summary_text", "text": "checked"}],
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": '{"query": "x"}',
                        "status": "completed",
                    },
                ]
            ),
            responses_reply([responses_text("done")]),
        ],
    },
    "responses_reasoning_level": {
        "client": responses(
            reasoning_levels={
                "low": {"reasoning": {"effort": "low"}},
                "high": {"reasoning": {"effort": "high"}, "max_tokens": 4096},
            },
            reasoning_level="high",
        ),
        "steps": [
            {"messages": const(USER)},
            {"messages": const(USER), "kwargs": {"reasoning_level": "low"}},
        ],
        "replies": [responses_reply([responses_text("a")]), responses_reply([responses_text("b")])],
    },
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.asyncio
async def test_wire_snapshot(wire, name):
    captured, replies = wire
    case = CASES[name]
    observed: dict[str, Any] = {}
    for mode in ("sync", "async"):
        captured.clear()
        replies[:] = list(case["replies"])
        results = await run_case(case, mode == "async")
        assert not replies, "every canned reply must be consumed"
        observed[mode] = {
            "calls": copy.deepcopy(captured),
            "responses": [summarize(r) for r in results],
        }
    text = json.dumps(observed, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path = SNAPSHOT_DIR / f"{name}.json"
    if UPDATE or not path.exists():
        if not UPDATE:
            pytest.fail(f"missing wire snapshot {path}; record with NOOA_UPDATE_WIRE_SNAPSHOTS=1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert text == path.read_text(), f"wire body changed for {name}"
