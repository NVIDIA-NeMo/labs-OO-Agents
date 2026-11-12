# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in numeric sideband: wire projection, paid failures and scope isolation."""

import asyncio
import copy
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace

import litellm
import pytest
from pydantic import BaseModel

from nooa.llm_types import LLMResponse, LLMUsage
from nooa.unifiedllm import CompletionClient, LLMRequest, ResponsesClient, Tool
from nooa.unifiedllm import unifiedllm as core
from nooa.unifiedllm._observations import _collect_observations, _current_collector
from nooa.unifiedllm.adapters import Capabilities
from nooa.unifiedllm.adapters._numeric_usage import _numeric_usage
from nooa.unifiedllm.errors import _ReplyParseError
from nooa.unifiedllm.llm import LLM
from nooa.unifiedllm.retry_config import RetryConfig

MESSAGES = [
    {"role": "system", "content": "private instructions"},
    {"role": "user", "content": "Hi"},
]
USAGE = {
    "input_tokens": 7,
    "output_tokens": 2,
    "cached_input_tokens": 3,
    "cache_write_input_tokens": 1,
    "reasoning_tokens": 1,
    "total_tokens": 9,
    "cost_usd": 0.03,
}


def lookup(query: str) -> str:
    """Look up a query."""
    raise AssertionError("not executed")


TOOL = Tool(name="lookup", description="Look up.", callable=lookup)


class Answer(BaseModel):
    count: int


def make_client(family):
    cls = CompletionClient if family == "chat" else ResponsesClient
    return cls("openai/gpt-4o-mini", api_key="private-key", api_base="https://private.test/v1")


def reply(family, *, content="ok", usage=USAGE):
    if family == "chat":
        result = litellm.ModelResponse(
            choices=[{"index": 0, "message": {"role": "assistant", "content": content}}]
        )
    else:
        result = litellm.ResponsesAPIResponse(
            id="private-id",
            object="response",
            created_at=0,
            status="completed",
            model="private-model",
            output=[
                {
                    "type": "message",
                    "id": "private-msg",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": content, "annotations": []}],
                }
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
    result.usage = copy.deepcopy(usage)
    return result


def stub_provider(monkeypatch, family, function):
    name = "completion" if family == "chat" else "responses"
    monkeypatch.setattr(litellm, name, function)

    async def afunction(**kwargs):
        return function(**kwargs)

    monkeypatch.setattr(litellm, "a" + name, afunction)


async def send(client, request, is_async):
    return await client._adapter.send(request) if is_async else client._adapter.send_sync(request)


@pytest.mark.parametrize("family", ["chat", "responses"])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("outcome", ["success", "malformed", "empty", "missing", "zero"])
@pytest.mark.asyncio
async def test_delivered_usage_before_parse_and_single_calibration(
    monkeypatch, family, is_async, outcome
):
    updates = []
    monkeypatch.setattr(
        core, "_token_calibration", SimpleNamespace(update=lambda *args: updates.append(args))
    )
    monkeypatch.setattr(litellm, "token_counter", lambda **kwargs: 23)
    usage = (
        None if outcome == "missing" else (dict.fromkeys(USAGE, 0) if outcome == "zero" else USAGE)
    )
    raw = reply(
        family,
        content="" if outcome == "empty" else "not JSON" if outcome == "malformed" else "ok",
        usage=usage,
    )
    stub_provider(monkeypatch, family, lambda **kwargs: raw)
    client = make_client(family)
    request = LLMRequest(
        messages=MESSAGES, tools=[TOOL], output_type=Answer if outcome == "malformed" else None
    )
    parse = client._adapter.parse
    with _collect_observations() as collector:

        def checked_parse(*args, **kwargs):
            (record,) = collector.records
            assert record.response_received and not record.finished
            assert record.raw_input_estimate == 23
            return parse(*args, **kwargs)

        monkeypatch.setattr(client._adapter, "parse", checked_parse)
        if outcome == "malformed":
            with pytest.raises(_ReplyParseError):
                await send(client, request, is_async)
        else:
            result = await send(client, request, is_async)
            assert result.content == ("" if outcome == "empty" else "ok")
        (record,) = collector.records
        assert record.finished and record.transmission_started and record.response_received
        assert len(updates) == (0 if outcome in {"missing", "zero"} else 1)
        assert client._adapter._route_calibration_record is None
        if outcome == "missing":
            assert record.usage is None
        else:
            assert asdict(record.usage) == {**usage, "cost_source": "provider"}
        snapshot = collector.records
        raw.usage = {"input_tokens": 1000}
        assert collector.records == snapshot
        with pytest.raises(FrozenInstanceError):
            record.finished = False
        text = repr(asdict(record))
        for secret in ("private", "openai", "lookup", "not JSON", "https"):
            assert secret not in text
    assert _current_collector.get() is None
    await client.aclose()


@pytest.mark.parametrize("family", ["chat", "responses"])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_projection_tools_instructions_and_no_wire_change(monkeypatch, family, is_async):
    estimates, wires = [], []

    def counter(**kwargs):
        estimates.append(copy.deepcopy(kwargs))
        return 19 if "tools" in kwargs else 11

    monkeypatch.setattr(litellm, "token_counter", counter)

    def provider(**kwargs):
        wires.append(copy.deepcopy({k: v for k, v in kwargs.items() if k != "client"}))
        if _current_collector.get() is not None:
            (observed,) = _current_collector.get().records
            assert observed.raw_input_estimate == 19 and not observed.response_received
        return reply(family)

    stub_provider(monkeypatch, family, provider)
    client = make_client(family)
    request = LLMRequest(messages=MESSAGES, tools=[TOOL])
    await send(client, request, is_async)
    baseline = copy.deepcopy(estimates)
    estimates.clear()
    with _collect_observations() as collector:
        await send(client, request, is_async)
        assert collector.records[0].raw_input_estimate == 19
    assert wires[0] == wires[1]
    assert wires[0]["max_retries"] == wires[0]["num_retries"] == 0
    # Two raw estimates with active collection, but exactly one calibration;
    # inactive collection adds no estimate beyond the existing success hook.
    assert estimates == baseline + baseline
    assert baseline[-1]["tools"] == wires[0]["tools"]
    expected = core._token_counter_messages(wires[0]["messages" if family == "chat" else "input"])
    if family == "responses":
        expected = [{"role": "system", "content": wires[0]["instructions"]}, *expected]
        assert wires[0]["instructions"] == "private instructions"
    assert baseline[-1]["messages"] == expected
    await client.aclose()


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_dummy_tool_projection(monkeypatch, is_async):
    seen = []
    monkeypatch.setattr(
        litellm, "token_counter", lambda **kw: seen.append(kw) or (40 if kw.get("tools") else 10)
    )
    monkeypatch.setattr(
        "nooa.unifiedllm.adapters.litellm_chat._needs_dummy_tool", lambda model: True
    )
    wires = []
    stub_provider(monkeypatch, "chat", lambda **kw: wires.append(kw) or reply("chat"))
    client = make_client("chat")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]
    with _collect_observations() as collector:
        await send(client, LLMRequest(messages=messages), is_async)
        assert collector.records[0].raw_input_estimate == 40
    assert seen[1]["tools"] == wires[0]["tools"] == [core._DUMMY_TOOL_SCHEMA]
    await client.aclose()


@pytest.mark.parametrize(
    "usage,hidden,cost,source",
    [
        (None, {}, None, "unknown"),
        (None, {"response_cost": 0}, 0, "response_metadata"),
        ({"input_tokens": 0, "output_tokens": 0}, {}, None, "unknown"),
        (LLMUsage(input_tokens=0, output_tokens=0), {}, None, "unknown"),
        (LLMUsage(cost_usd=0), {}, 0, "provider"),
        ({"cost": 0}, {}, 0, "provider"),
        ({"cost": 0.2}, {"response_cost": 0}, 0, "response_metadata"),
        ({"cost": 0.2}, {"response_cost": float("nan")}, 0.2, "provider"),
        ({"cost": 0.2}, {"response_cost": True}, 0.2, "provider"),
        ({"cost": -1}, {}, None, "unknown"),
    ],
)
def test_cost_provenance_and_absent_token_fields(usage, hidden, cost, source):
    result = _numeric_usage(SimpleNamespace(usage=usage, _hidden_params=hidden))
    if result is None:
        assert usage is None and cost is None
    else:
        assert result.cost_usd == cost and result.cost_source == source
        if isinstance(usage, dict) and "input_tokens" in usage:
            assert result.input_tokens == result.output_tokens == result.total_tokens == 0
        if usage is None or (isinstance(usage, dict) and "input_tokens" not in usage):
            assert result.input_tokens is None and result.output_tokens is None


@pytest.mark.parametrize("family", ["chat", "responses"])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_no_scope_no_sideband_and_original_provider_error(monkeypatch, family, is_async):
    client = make_client(family)
    name = "nooa.unifiedllm.adapters.litellm_" + family + "._numeric_usage"
    monkeypatch.setattr(name, lambda raw: pytest.fail("inactive numeric projection"))
    estimates = []
    monkeypatch.setattr(litellm, "token_counter", lambda **kw: estimates.append(kw) or 17)
    stub_provider(monkeypatch, family, lambda **kw: reply(family, usage=None))
    await send(client, LLMRequest(messages=MESSAGES), is_async)
    assert estimates == []
    error = RuntimeError("original provider error")

    def failing(**kw):
        raise error

    stub_provider(monkeypatch, family, failing)
    with _collect_observations() as collector:
        with pytest.raises(RuntimeError) as caught:
            await send(client, LLMRequest(messages=MESSAGES), is_async)
        assert caught.value is error
        (record,) = collector.records
        assert record.finished and record.transmission_started and not record.response_received
        assert record.usage is None
    assert _current_collector.get() is None
    await client.aclose()


@pytest.mark.parametrize("family", ["chat", "responses"])
@pytest.mark.asyncio
async def test_nested_and_concurrent_scopes(monkeypatch, family):
    client = make_client(family)
    monkeypatch.setattr(litellm, "token_counter", lambda **kw: 12)
    name = "acompletion" if family == "chat" else "aresponses"
    nested_collectors = []

    async def provider(**kw):
        messages = kw["messages" if family == "chat" else "input"]
        if "outer" in repr(messages):
            parent = _current_collector.get()
            with _collect_observations() as child:
                nested_collectors.append(child)
                await client._adapter.send(
                    LLMRequest(messages=[{"role": "user", "content": "nested"}])
                )
            assert _current_collector.get() is parent
        await asyncio.sleep(0)
        return reply(family)

    monkeypatch.setattr(litellm, name, provider)

    async def caller(content):
        with _collect_observations() as collector:
            await client._adapter.send(LLMRequest(messages=[{"role": "user", "content": content}]))
            assert len(collector.records) == 1
        assert _current_collector.get() is None
        return collector

    first, second = await asyncio.gather(caller("outer"), caller("other"))
    assert first is not second and len(nested_collectors) == 1
    assert all(
        c.records[0].response_received and c.records[0].finished
        for c in (first, second, *nested_collectors)
    )
    await client.aclose()


@pytest.mark.parametrize("family", ["chat", "responses"])
@pytest.mark.asyncio
async def test_cancellation_reset_and_late_shielded_result_stays_unknown(monkeypatch, family):
    client = make_client(family)
    monkeypatch.setattr(litellm, "token_counter", lambda **kw: 12)
    entered, release = asyncio.Event(), asyncio.Event()
    provider_tasks = []
    name = "acompletion" if family == "chat" else "aresponses"

    async def provider(**kw):
        provider_tasks.append(asyncio.current_task())
        entered.set()
        await release.wait()
        return reply(family)

    monkeypatch.setattr(litellm, name, provider)
    collectors = []

    async def caller():
        try:
            with _collect_observations() as collector:
                collectors.append(collector)
                await client._adapter.send(LLMRequest(messages=MESSAGES))
        finally:
            assert _current_collector.get() is None

    task = asyncio.create_task(caller())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    frozen = collectors[0].records
    assert frozen[0].finished and frozen[0].usage is None and not frozen[0].response_received
    assert not provider_tasks[0].done()  # actual existing shield behavior
    with _collect_observations() as unrelated:
        release.set()
        await provider_tasks[0]
        assert unrelated.records == ()
    assert collectors[0].records == frozen  # no invented late settlement
    await client.aclose()


def test_bounded_collector_and_scope_restoration():
    from nooa.unifiedllm._observations import _observe_attempt

    with _collect_observations(limit=1) as parent:
        with _observe_attempt() as writer:
            writer.projected(0)
            snapshot = parent.records
            with pytest.raises(ValueError), _collect_observations():
                raise ValueError("scope failure")
            assert _current_collector.get() is parent
        assert not snapshot[0].finished and parent.records[0].finished
        with _observe_attempt() as dropped:
            assert dropped is None
        assert len(parent.records) == parent.dropped_attempts == 1
    assert _current_collector.get() is None
    for limit in (0, 257, True):
        with pytest.raises(ValueError), _collect_observations(limit=limit):
            pass


@pytest.mark.asyncio
async def test_custom_adapter_protocol_unaffected_no_fabricated_fallback():
    class Custom:
        capabilities = Capabilities()

        def send_sync(self, request):
            return LLMResponse(usage=LLMUsage(input_tokens=5))

        async def send(self, request):
            return self.send_sync(request)

    llm = LLM(Custom())
    with _collect_observations() as collector:
        assert llm.call(LLMRequest(messages=MESSAGES)).usage.input_tokens == 5
        assert (await llm.acall(LLMRequest(messages=MESSAGES))).usage.input_tokens == 5
        assert collector.records == ()


@pytest.mark.parametrize("family", ["chat", "responses"])
@pytest.mark.asyncio
async def test_legacy_success_hooks_single_update(monkeypatch, family):
    updates = []
    monkeypatch.setattr(
        core, "_token_calibration", SimpleNamespace(update=lambda *args: updates.append(args))
    )
    monkeypatch.setattr(litellm, "token_counter", lambda **kw: 12)
    stub_provider(monkeypatch, family, lambda **kw: reply(family))
    client = make_client(family)
    with _collect_observations() as collector:
        client.call(MESSAGES)
        await client.acall(MESSAGES)
        assert len(collector.records) == len(updates) == 2
        assert client._adapter._route_calibration_record is None
    await client.aclose()


@pytest.mark.parametrize("family", ["chat", "responses"])
@pytest.mark.asyncio
async def test_empty_paid_attempt_retry_keeps_separate_records(monkeypatch, family):
    calls = []

    def provider(**kw):
        raw = reply(family, content="" if not calls else "ok")
        if not calls:
            if family == "chat":
                raw.choices[0].message.reasoning_content = "reasoning"
            else:
                raw.output = [
                    {
                        "type": "reasoning",
                        "id": "r1",
                        "summary": [{"type": "summary_text", "text": "reasoning"}],
                    }
                ]
        calls.append(kw)
        return raw

    stub_provider(monkeypatch, family, provider)
    monkeypatch.setattr(litellm, "token_counter", lambda **kw: 12)
    client = make_client(family)
    llm = LLM(
        client._adapter,
        retry_config=RetryConfig(
            max_retries=1, base_delay=0, max_delay=0, jitter_factor=0, retry_on_empty_content=True
        ),
    )
    with _collect_observations() as collector:
        result = await llm.acall(LLMRequest(messages=MESSAGES))
        assert result.content == "ok" and result.usage.input_tokens == 7
        assert len(collector.records) == len(calls) == 2
        assert all(r.response_received and r.usage.input_tokens == 7 for r in collector.records)
    await client.aclose()


@pytest.mark.parametrize("family", ["chat", "responses"])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_preparation_failure_resets_scope_without_transmission(monkeypatch, family, is_async):
    client = make_client(family)
    error = ValueError("preflight failure")

    def prepare(*args, **kwargs):
        raise error

    monkeypatch.setattr(client._adapter, "prepare", prepare)
    with pytest.raises(ValueError) as caught, _collect_observations() as collector:
        await send(client, LLMRequest(messages=MESSAGES), is_async)
    assert caught.value is error and _current_collector.get() is None
    (record,) = collector.records
    assert record.finished and not record.transmission_started
    assert record.raw_input_estimate is None and record.usage is None
    await client.aclose()


def test_attempt_writer_captures_owner_even_with_another_scope_active():
    from nooa.unifiedllm._observations import _NumericUsage, _observe_attempt

    with _collect_observations() as parent, _observe_attempt() as writer:
        with _collect_observations() as child:
            writer.projected(0)
            writer.transmitting()
            writer.delivered(_NumericUsage(input_tokens=0))
            assert child.records == ()
            assert parent.records[0].usage.input_tokens == 0
    assert parent.records[0].finished


def test_raw_estimator_retains_legacy_fallback_and_failure_is_unknown(monkeypatch):
    def counter(**kwargs):
        if "messages" in kwargs:
            raise ValueError("unsupported projection")
        return len(kwargs["text"])

    monkeypatch.setattr(litellm, "token_counter", counter)
    projected = [
        {
            "role": "tool",
            "type": "function_call_output",
            "output": [{"type": "input_text", "text": "tool result"}],
        }
    ]
    assert core._estimate_raw_input_tokens("model", projected, instructions="sys") == len(
        "sys"
    ) + len("tool result")

    def failing(**kwargs):
        raise ValueError("tokenizer failure")

    monkeypatch.setattr(litellm, "token_counter", failing)
    assert core._estimate_raw_input_tokens("model", projected) is None


def test_malformed_usage_projection_is_best_effort():
    class Broken:
        def model_dump(self, **kwargs):
            raise ValueError("malformed usage")

    assert _numeric_usage(SimpleNamespace(usage=Broken())) is None
