# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental public shell: neutral preflight and shared logical execution."""

import ast
import asyncio
import importlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from nooa.llm_types import AssistantReasoning, AssistantText, CacheBoundary, LLMResponse
from nooa.unifiedllm import LLM, LLMFailure, LLMRequest, Tool
from nooa.unifiedllm.adapters import Capabilities
from nooa.unifiedllm.errors import EmptyContentError, _ReplyParseError
from nooa.unifiedllm.logical import logical_async, logical_sync
from nooa.unifiedllm.retry_config import RetryConfig

MODE = pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
REPLY = LLMResponse(parts=(AssistantText(text="ok"),), finish_reason="stop")
TOOL = Tool("lookup", "Look up", lambda: None)
DEFAULT_FACTS = Capabilities()
HISTORY = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]


class StrictAdapter:
    """No provider hooks, admission wrapper or configuration side channel."""

    def __init__(self, facts=DEFAULT_FACTS, replies=()):
        self.facts = facts
        self.reads = 0
        self.seen = []
        self.replies = iter(replies)

    @property
    def capabilities(self):
        self.reads += 1
        return self.facts

    def send_sync(self, request):
        self.seen.append(request)
        reply = next(self.replies, REPLY)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def send(self, request):
        await asyncio.sleep(0)
        return self.send_sync(request)


async def run(llm, request, is_async):
    return await llm.acall(request) if is_async else llm.call(request)


@MODE
@pytest.mark.parametrize(
    "fields, facts, kind",
    [
        ({"tools": [TOOL]}, Capabilities(tools=False), "bad_request"),
        ({"tool_choice": "missing"}, Capabilities(), "bad_request"),
        ({"tool_choice": "required"}, Capabilities(), "bad_request"),
        ({"tools": [object()]}, Capabilities(), "bad_request"),
        ({"messages": [object()]}, Capabilities(), "bad_request"),
        ({"output_type": int}, Capabilities(), "bad_request"),
    ],
)
async def test_preflight_has_zero_sends(is_async, fields, facts, kind):
    adapter = StrictAdapter(facts)
    with pytest.raises(LLMFailure) as caught:
        await run(LLM(adapter), LLMRequest(**fields), is_async)
    assert caught.value.kind == kind
    assert adapter.seen == []


@MODE
async def test_wrong_call_type_is_typed(is_async):
    adapter = StrictAdapter()
    with pytest.raises(LLMFailure, match="LLMRequest"):
        await run(LLM(adapter), [], is_async)
    assert adapter.seen == []


@MODE
async def test_unknown_facts_and_four_field_custom_request(is_async):
    adapter = StrictAdapter(Capabilities(tools=None))
    llm = LLM(adapter)
    for choice in ("auto", "none"):
        await run(llm, LLMRequest(tools=[TOOL], tool_choice=choice), is_async)
    assert [req.tool_choice for req in adapter.seen] == ["auto", "none"]
    assert llm.capabilities.tools is None
    assert llm.capabilities.reasoning_levels is None
    assert llm.capabilities.context_window is None
    assert adapter.reads == 1


@MODE
async def test_switching_routes_preserves_history_and_native_identities(is_async):
    boundary = CacheBoundary()
    history = [*HISTORY, REPLY, boundary]
    request = LLMRequest(messages=history)
    image = LLM(StrictAdapter(Capabilities(media_kinds=frozenset({"image"}), context_window=1000)))
    text = LLM(StrictAdapter(Capabilities()))
    again = LLM(StrictAdapter(Capabilities(media_kinds=frozenset({"image"}))))
    for llm in (image, text, again):
        await run(llm, request, is_async)
        sent = llm.adapter.seen[-1].messages
        assert sent is not history
        assert sent[1] is REPLY and sent[2] is boundary
    assert image.adapter.seen[0].messages[0] == HISTORY[0]
    assert "removed" in text.adapter.seen[0].messages[0]["content"][0]["text"]
    assert again.adapter.seen[0].messages[0] == HISTORY[0]
    assert request.messages is history and history[0] is HISTORY[0]
    assert HISTORY[0]["content"][0]["type"] == "image_url"
    assert text.capabilities.context_window is None
    assert again.capabilities.context_window is None


@MODE
@pytest.mark.parametrize(
    "message",
    [
        HISTORY[0],
        {"type": "input_image", "image_url": "x"},
        {"type": "input_audio", "input_audio": {"data": "x", "format": "wav"}},
        {"type": "video_url", "video_url": {"url": "x"}},
        {"type": "file", "file": {"mime_type": "application/pdf", "file_data": "x"}},
        {"type": "file", "file": {"mime_type": "application/msword", "file_data": "x"}},
        {"role": "assistant", "audio": {"data": "x"}, "content": "hello"},
        {"type": "function_call_output", "output": ({"type": "input_image", "image_url": "x"},)},
    ],
)
async def test_strict_media_mode_has_zero_sends(is_async, message):
    adapter = StrictAdapter(Capabilities())
    with pytest.raises(LLMFailure) as caught:
        await run(
            LLM(adapter, on_unsupported_media="error"), LLMRequest(messages=[message]), is_async
        )
    assert caught.value.kind == "media_rejected"
    assert adapter.seen == []


def test_facts_snapshot_and_invalid_configuration():
    media, levels = {"image"}, ["low"]
    adapter = StrictAdapter(Capabilities(media_kinds=media, reasoning_levels=levels))
    llm = LLM(adapter)
    media.clear()
    levels.append("high")
    adapter.facts = Capabilities()
    assert llm.capabilities.media_kinds == frozenset({"image"})
    assert llm.capabilities.reasoning_levels == ("low",)
    with pytest.raises(FrozenInstanceError):
        llm.capabilities.tools = False
    assert adapter.reads == 1
    for facts in (
        None,
        Capabilities(media_kinds={"unknown"}),
        Capabilities(context_window=0),
        Capabilities(max_output_tokens=-1),
        Capabilities(tools="yes"),
        Capabilities(reasoning_levels="low"),
        Capabilities(reasoning_levels=("low", "low")),
    ):
        with pytest.raises(LLMFailure) as caught:
            LLM(StrictAdapter(facts))
        assert caught.value.kind == "config"
    with pytest.raises(LLMFailure, match="on_unsupported_media"):
        LLM(adapter, on_unsupported_media="ignore")


@MODE
@pytest.mark.parametrize("ceiling", [10, 20])
async def test_independent_route_ceilings_are_facts_not_request_fields(is_async, ceiling):
    adapter = StrictAdapter(Capabilities(context_window=10, max_output_tokens=ceiling))
    llm = LLM(adapter)
    assert await run(llm, LLMRequest(), is_async) is REPLY
    assert llm.capabilities.max_output_tokens == ceiling
    assert len(adapter.seen) == 1


@MODE
async def test_retry_owner_and_typed_terminal_parse(is_async):
    adapter = StrictAdapter(replies=[LLMFailure("transient", "status 503"), REPLY])
    llm = LLM(adapter, retry_config=RetryConfig(max_retries=1, base_delay=0, jitter_factor=0))
    assert await run(llm, LLMRequest(), is_async) is REPLY
    assert len(adapter.seen) == 2
    error = ValueError("status 503 in malformed reply")
    adapter = StrictAdapter(replies=[_ReplyParseError(error)])
    with pytest.raises(LLMFailure) as caught:
        await run(LLM(adapter, retry_config=llm._retry_config), LLMRequest(), is_async)
    assert caught.value.kind == "bad_request"
    assert caught.value.__cause__ is error
    assert len(adapter.seen) == 1


@MODE
@pytest.mark.parametrize("enabled, configured", [(False, False), (False, True), (True, True)])
async def test_public_reasoning_only_retry_opt_in(is_async, enabled, configured):
    empty = LLMResponse(parts=(AssistantReasoning(text="thinking"),), finish_reason="stop")
    adapter = StrictAdapter(replies=[empty, REPLY])
    config = (
        RetryConfig(max_retries=1, retry_on_empty_content=enabled, base_delay=0, jitter_factor=0)
        if configured
        else None
    )
    result = await run(LLM(adapter, retry_config=config), LLMRequest(), is_async)
    assert result is (REPLY if enabled else empty)
    assert len(adapter.seen) == (2 if enabled else 1)


@MODE
async def test_public_reasoning_only_retry_exhaustion(is_async):
    empty = LLMResponse(parts=(AssistantReasoning(text="thinking"),), finish_reason="stop")
    adapter = StrictAdapter(replies=[empty, empty])
    config = RetryConfig(max_retries=1, retry_on_empty_content=True, base_delay=0, jitter_factor=0)
    with pytest.raises(LLMFailure) as caught:
        await run(LLM(adapter, retry_config=config), LLMRequest(), is_async)
    assert caught.value.kind == "output_limit"
    assert isinstance(caught.value.__cause__, EmptyContentError)
    assert caught.value.__cause__.reasoning == "thinking"
    assert len(adapter.seen) == 2


@pytest.mark.asyncio
async def test_concurrent_public_requests_do_not_leak_media_state():
    adapter = StrictAdapter()
    llm = LLM(adapter)
    requests = [
        LLMRequest(messages=[*HISTORY, {"role": "user", "content": str(i)}]) for i in range(20)
    ]
    await asyncio.gather(*(llm.acall(request) for request in requests))
    assert len(adapter.seen) == 20
    assert {req.messages[-1]["content"] for req in adapter.seen} == {str(i) for i in range(20)}
    assert all("removed" in req.messages[0]["content"][0]["text"] for req in adapter.seen)
    assert all(req.messages[0]["content"][0]["type"] == "image_url" for req in requests)


@MODE
@pytest.mark.parametrize(
    "terminal", [RuntimeError("failed"), KeyboardInterrupt(), asyncio.CancelledError()]
)
async def test_shared_logical_rolls_back_on_all_terminal_failures(is_async, terminal):
    staged, events = [], []

    def send(_):
        events.append("send")
        raise ValueError("rejected") if len(events) == 1 else terminal

    async def asend(value):
        return send(value)

    def recover(exc):
        assert str(exc) == "rejected"
        staged.append("exclusion")
        return True

    kwargs = {"recovery_attempts": 1, "recover": recover, "rollback": staged.clear}
    with pytest.raises(type(terminal)):
        if is_async:
            await logical_async(lambda: None, asend, **kwargs)
        else:
            logical_sync(lambda: None, send, **kwargs)
    assert len(events) == 2 and staged == []


@MODE
async def test_shared_logical_success_commits_staging(is_async):
    staged = []

    def send(_):
        if not staged:
            raise ValueError("rejected")
        return REPLY

    async def asend(value):
        return send(value)

    def recover(exc):
        staged.append("exclusion")
        return True

    kwargs = {"recovery_attempts": 1, "recover": recover, "rollback": staged.clear}
    result = (
        await logical_async(lambda: None, asend, **kwargs)
        if is_async
        else logical_sync(lambda: None, send, **kwargs)
    )
    assert result is REPLY and staged == ["exclusion"]


def test_neutral_import_edges_and_public_export():
    assert importlib.import_module("nooa.unifiedllm.llm").LLM is LLM
    for name in ("llm", "logical"):
        module = importlib.import_module("nooa.unifiedllm." + name)
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ("unifiedllm", "registry", "nooa.unifiedllm.unifiedllm")
                assert not (node.module or "").startswith(
                    ("litellm", "openai", "anthropic", "nooa.runtime")
                )
            elif isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith(
                        (
                            "litellm",
                            "openai",
                            "anthropic",
                            "nooa.runtime",
                            "nooa.unifiedllm.unifiedllm",
                        )
                    )
                    for alias in node.names
                )
