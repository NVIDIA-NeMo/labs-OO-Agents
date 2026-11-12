# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden wire bodies for CompletionClient and ResponsesClient.

Each case patches the LiteLLM entry points, records the exact keyword
arguments the client passes on, and compares them with a JSON file under
``snapshots/wire``. The files were recorded on the code before the adapter
split, so any change in what reaches LiteLLM fails here. Each file stores
common calls/responses once plus explicit per-mode API and client types;
only the expected snapshot is expanded before the exact comparison.

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


def _without_owned_dispatch_carriers(kwargs):
    """Compare the wire body, validating only the SDK's two private carriers."""
    carriers = {"litellm_logging_obj", "litellm_call_id"}
    present = carriers.intersection(kwargs)
    if present:
        from uuid import UUID

        from litellm.litellm_core_utils.litellm_logging import Logging

        assert present == carriers
        assert isinstance(kwargs["litellm_logging_obj"], Logging)
        assert UUID(kwargs["litellm_call_id"])
        assert (
            kwargs["litellm_logging_obj"].model_call_details.get("_nooa_logical_call_owned") is True
        )
    return {key: value for key, value in kwargs.items() if key not in carriers}


def normalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in _without_owned_dispatch_carriers(kwargs).items():
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


def snapshot_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def pack_snapshot(observed: dict[str, Any]) -> dict[str, Any]:
    """Deduplicate only API/client type; reject any other mode differences."""
    assert set(observed) == {"sync", "async"}
    common = {}
    modes = {}
    for mode, payload in copy.deepcopy(observed).items():
        assert payload["calls"], "a snapshot must capture at least one call"
        metadata = []
        for call in payload["calls"]:
            metadata.append(
                {"api": call.pop("api"), "client_type": call["kwargs"]["client"].pop("__client__")}
            )
        assert all(item == metadata[0] for item in metadata), "API/client type varies within mode"
        common[mode] = payload
        modes[mode] = metadata[0]
    assert snapshot_text(common["sync"]) == snapshot_text(common["async"]), (
        "sync/async common wire payload differs"
    )
    packed = {"common": common["sync"], "modes": modes}
    assert snapshot_text(unpack_snapshot(packed)) == snapshot_text(observed)
    return packed


def unpack_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct independent full expectations, preserving all client metadata."""
    assert set(snapshot) == {"common", "modes"}
    assert set(snapshot["modes"]) == {"sync", "async"}
    expected = {}
    for mode, metadata in snapshot["modes"].items():
        assert set(metadata) == {"api", "client_type"}
        payload = copy.deepcopy(snapshot["common"])
        for call in payload["calls"]:
            assert "api" not in call
            assert "__client__" not in call["kwargs"]["client"]
            call["api"] = metadata["api"]
            call["kwargs"]["client"]["__client__"] = metadata["client_type"]
        expected[mode] = payload
    return expected


def check_snapshot(observed: dict[str, Any], path: Path) -> None:
    if UPDATE:
        # Pack before writing: unsupported differences must never overwrite a golden.
        text = snapshot_text(pack_snapshot(observed))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    if not path.exists():
        pytest.fail(f"missing wire snapshot {path}; record with NOOA_UPDATE_WIRE_SNAPSHOTS=1")
    expected = unpack_snapshot(json.loads(path.read_text()))
    assert snapshot_text(observed) == snapshot_text(expected), f"wire body changed for {path.stem}"


@pytest.mark.parametrize("name", sorted(CASES))
def test_snapshot_contract_round_trip(name):
    packed = json.loads((SNAPSHOT_DIR / f"{name}.json").read_text())
    original = copy.deepcopy(packed)
    expanded = unpack_snapshot(packed)
    assert pack_snapshot(expanded) == packed
    assert expanded["sync"]["responses"] == packed["common"]["responses"]
    for mode, payload in expanded.items():
        for call, common in zip(payload["calls"], packed["common"]["calls"], strict=True):
            assert call["api"] == packed["modes"][mode]["api"]
            assert call["kwargs"]["client"] == {
                **common["kwargs"]["client"],
                "__client__": packed["modes"][mode]["client_type"],
            }
    expanded["sync"]["calls"][0]["kwargs"]["client"]["max_retries"] = 99
    assert packed == original
    assert expanded["async"]["calls"] == unpack_snapshot(original)["async"]["calls"]


@pytest.mark.parametrize(
    "field",
    ["kwargs", "client_retries", "client_extra", "response", "call_count", "retry_type"],
)
def test_snapshot_contract_rejects_common_difference(field, tmp_path, monkeypatch):
    packed = json.loads((SNAPSHOT_DIR / "chat_plain.json").read_text())
    observed = unpack_snapshot(packed)
    payload = observed["async"]
    kwargs = payload["calls"][0]["kwargs"]
    if field == "kwargs":
        kwargs["temperature"] = 0.9
    elif field == "client_retries":
        kwargs["client"]["max_retries"] = 7
    elif field == "client_extra":
        kwargs["client"]["extra_metadata"] = "must not disappear"
    elif field == "response":
        payload["responses"][0]["content"] = "different"
    elif field == "call_count":
        payload["calls"].append(copy.deepcopy(payload["calls"][0]))
    else:
        kwargs["max_retries"] = False  # JSON false is not JSON 0.
    unchanged = copy.deepcopy(observed)
    with pytest.raises(AssertionError, match="common wire payload differs"):
        pack_snapshot(observed)
    path = tmp_path / "snapshot.json"
    path.write_text(snapshot_text(packed))
    monkeypatch.setattr(f"{__name__}.UPDATE", True)
    with pytest.raises(AssertionError, match="common wire payload differs"):
        check_snapshot(observed, path)
    assert path.read_text() == snapshot_text(packed)
    assert observed == unchanged


def test_snapshot_contract_explicit_update(tmp_path, monkeypatch):
    packed = json.loads((SNAPSHOT_DIR / "chat_plain.json").read_text())
    observed = unpack_snapshot(packed)
    for payload in observed.values():
        payload["calls"][0]["kwargs"]["client"]["extra_metadata"] = "retained"
    path = tmp_path / "new.json"
    monkeypatch.setattr(f"{__name__}.UPDATE", False)
    with pytest.raises(pytest.fail.Exception, match="missing wire snapshot"):
        check_snapshot(observed, path)
    assert not path.exists()
    monkeypatch.setattr(f"{__name__}.UPDATE", True)
    check_snapshot(observed, path)
    assert unpack_snapshot(json.loads(path.read_text())) == observed
    monkeypatch.setattr(f"{__name__}.UPDATE", False)
    check_snapshot(observed, path)


@pytest.mark.parametrize("field", ["api", "client_type"])
def test_snapshot_contract_preserves_mode_difference(field, tmp_path, monkeypatch):
    packed = json.loads((SNAPSHOT_DIR / "chat_tools.json").read_text())
    observed = unpack_snapshot(packed)
    for call in observed["async"]["calls"]:
        if field == "api":
            call["api"] = "completion"
        else:
            call["kwargs"]["client"]["__client__"] = "OpenAI"
    assert unpack_snapshot(pack_snapshot(observed)) == observed
    path = tmp_path / "snapshot.json"
    path.write_text(snapshot_text(packed))
    monkeypatch.setattr(f"{__name__}.UPDATE", False)
    with pytest.raises(AssertionError, match="wire body changed"):
        check_snapshot(observed, path)
    observed["async"]["calls"][0]["api"] = "different"
    with pytest.raises(AssertionError, match="varies within mode"):
        pack_snapshot(observed)


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
    check_snapshot(observed, SNAPSHOT_DIR / f"{name}.json")


@pytest.mark.parametrize(
    "bad",
    [
        {"litellm_logging_obj": object()},
        {"litellm_call_id": "not-a-uuid"},
        {"litellm_logging_obj": object(), "litellm_call_id": "not-a-uuid"},
    ],
)
def test_wire_carrier_normalization_rejects_invalid_carriers(bad):
    with pytest.raises(AssertionError):
        _without_owned_dispatch_carriers(bad)


def test_wire_carrier_normalization_keeps_all_other_fields():
    original = {"metadata": {"user": "caller", "nested": [1, 2]}, "native_param": 9}
    result = _without_owned_dispatch_carriers(original)
    assert result == original
    assert result["metadata"] is original["metadata"]
