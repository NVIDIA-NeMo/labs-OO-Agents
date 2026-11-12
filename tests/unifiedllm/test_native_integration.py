# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real SDK integration with shared calibration, broker and logical journal."""

import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel

from nooa.unifiedllm import AdmissionBroker, AdmissionCallCapError, registry
from nooa.unifiedllm._accounting import _accounting_receiver
from nooa.unifiedllm._observations import _collect_observations
from nooa.unifiedllm.admission import _admission_controller_scope
from nooa.unifiedllm.errors import LLMFailure
from nooa.unifiedllm.request import LLMRequest
from tests.unifiedllm.test_native_load_llm import borrowed, dispose, load, paid_payload, zero_delay
from tests.unifiedllm.test_native_projection import FAMILIES


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(registry, "MODELS", {})


class LooseAnswer(BaseModel):
    values: list


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_actual_loose_schema_and_received_once_calibration(family, mode, monkeypatch):
    seen, updates = [], []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=paid_payload(family), headers={"x-response-cost": "0"})

    sdk, http = borrowed(family, mode, handler)
    llm = load(family, sdk, retry_config=False, context_window=1)
    record = llm.adapter._route_calibration_record
    original = record.update

    def update(estimated, actual):
        updates.append((estimated, actual))
        original(estimated, actual)

    monkeypatch.setattr(record, "update", update)
    receipts = []
    with _accounting_receiver(receipts.append), _collect_observations() as observations:
        with pytest.raises(LLMFailure):
            request = LLMRequest(
                messages=[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "q"},
                ],
                output_type=LooseAnswer,
            )
            llm.call(request) if mode == "sync" else await llm.acall(request)
    body = seen[0]
    schema = (
        body["response_format"]["json_schema"]
        if family == "openai-chat"
        else body["text"]["format"]
        if family == "openai-responses"
        else body["output_config"]["format"]
    )
    if family != "anthropic-messages":
        assert schema["strict"] is False
    assert schema["schema"]["properties"]["values"]["type"] == "array"
    assert (
        body[
            {
                "openai-chat": "max_completion_tokens",
                "openai-responses": "max_output_tokens",
                "anthropic-messages": "max_tokens",
            }[family]
        ]
        == 100
    )
    assert len(updates) == len(seen) == len(receipts) == 1
    assert updates[0][1] == 3
    assert observations.records[0].response_received and observations.records[0].finished
    assert receipts[0].subtotal("cost_usd").complete and receipts[0].subtotal("cost_usd").known == 0
    await llm.aclose()
    assert not http.is_closed
    await dispose(sdk, mode)


@pytest.mark.parametrize("family", FAMILIES)
async def test_sdk_paid_empty_and_preack_cancel_share_broker_call_cap(family, monkeypatch):
    seen, receipts = [], []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json=paid_payload(family, empty=True, input_tokens=7, output_tokens=2),
            headers={"x-response-cost": ".03"},
        )

    sdk, http = borrowed(family, "async", handler)
    llm = load(family, sdk, retry_config=zero_delay(empty=True))
    with AdmissionBroker(max_in_flight=1, max_calls=1) as broker:
        controller = broker.controller(queue_timeout=2)
        original = controller._readline
        ready = asyncio.Event()
        release = asyncio.Event()

        async def gate(reader, started):
            line = await original(reader, started)
            if json.loads(line).get("status") == "ready":
                ready.set()
                await release.wait()
            return line

        monkeypatch.setattr(controller, "_readline", gate)
        with _admission_controller_scope(controller), _accounting_receiver(receipts.append):
            task = asyncio.create_task(
                llm.acall(LLMRequest(messages=[{"role": "user", "content": "q"}]))
            )
            await asyncio.wait_for(ready.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        for _ in range(200):
            state = broker.snapshot()
            if state.active == state.queued == 0:
                break
            await asyncio.sleep(0.005)
        assert state.admitted_calls == 0 and not seen
        assert len(receipts) == 1 and not receipts[0].attempts[0].provider_entered
        monkeypatch.setattr(controller, "_readline", original)
        with _admission_controller_scope(controller), _accounting_receiver(receipts.append):
            with pytest.raises(LLMFailure) as caught:
                await llm.acall(LLMRequest(messages=[{"role": "user", "content": "q"}]))
        assert isinstance(caught.value.__cause__, AdmissionCallCapError)
        for _ in range(200):
            state = broker.snapshot()
            if state.active == state.queued == 0:
                break
            await asyncio.sleep(0.005)
        assert len(seen) == state.admitted_calls == 1 and state.active == state.queued == 0
        assert len(receipts) == 2
        paid, refused = receipts[1].attempts
        assert paid.response_received and paid.provider_entered
        assert not refused.provider_entered and not refused.response_received
        assert receipts[1].subtotal("input_tokens").known == 7
        assert receipts[1].subtotal("output_tokens").known == 2
        assert receipts[1].subtotal("cost_usd").known == pytest.approx(0.03)
        assert receipts[1].subtotal("cost_usd").complete
        controller.close()
    await llm.aclose()
    assert not http.is_closed
    await sdk.close()


@pytest.mark.parametrize("family", FAMILIES)
async def test_sdk_logical_journal_viewer_replay_privacy_and_callback_isolation(
    family, monkeypatch
):
    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, get_value

    from nooa.tracing import _journal_api
    from nooa.tracing._context_sideband import set_journal_payload
    from nooa.tracing._litellm_journal import FileMessageJournalCallback
    from nooa.unifiedllm import _logical_trace
    from nooa.viewer.otlp_store import _resolve_message
    from tests.tracing.test_journal_llm_call import FakeWriter

    class DirectObserver(CustomLogger):
        def __init__(self):
            self.done = asyncio.Event()
            self.events = []

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.events.append(kwargs)
            self.done.set()

    writer = FakeWriter()
    direct = DirectObserver()
    # Observer is the completion barrier AFTER the real journal sink.
    callbacks = [FileMessageJournalCallback(writer), direct]
    # Isolate SDK-owned setup caches BETWEEN parametrized tests, before native
    # entry. They remain untouched throughout native and subsequent direct C.
    for name in (
        "input_callback",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, "callbacks", callbacks)
    monkeypatch.setattr(_logical_trace, "_journal_renderer", _journal_api._render_logical_journal)
    set_journal_payload(None)
    seen = []
    data = paid_payload(family)
    if family == "openai-chat":
        data["choices"][0]["message"].update(
            reasoning_content="thought", private_wire="private-native"
        )
    elif family == "openai-responses":
        data["output"].insert(
            0,
            {
                "id": "r",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thought"}],
                "encrypted_content": "private-native",
            },
        )
    else:
        data["content"].insert(
            0, {"type": "thinking", "thinking": "thought", "signature": "private-native"}
        )

    def handler(request):
        assert get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True
        seen.append(request)
        return httpx.Response(200, json=data, headers={"x-response-cost": ".03"})

    sdk, http = borrowed(family, "async", handler)
    llm = load(family, sdk, retry_config=False)
    first = await llm.acall(LLMRequest(messages=[{"role": "user", "content": "question"}]))
    await llm.acall(LLMRequest(messages=[first, {"role": "user", "content": "follow"}]))
    assert len(writer.calls) == len(seen) == 2
    call = writer.calls[1][1]
    messages = [_resolve_message(item, writer.blocks) for item in call["input_skeleton"]]
    assert messages[0]["content"] == "answer"
    assert messages[-1]["content"] == "follow"
    assert call["model"] == "model" and call["tokens"]["prompt"] == 3
    rendered = json.dumps([writer.calls, writer.blocks])
    assert all(
        secret not in rendered for secret in ("private-native", "fixture-key", "fixture.invalid")
    )
    assert litellm.callbacks is callbacks and callbacks[-1] is direct
    assert not get_value(_SUPPRESS_INSTRUMENTATION_KEY)
    assert not direct.events

    # The SAME installed callback chain still receives a real unrelated LiteLLM
    # SDK/HTTP call after native calls; no manual callback invocation/setup.
    import openai

    direct_requests = []

    def direct_reply(request):
        assert not get_value(_SUPPRESS_INSTRUMENTATION_KEY)
        direct_requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "direct",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "direct answer"},
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(direct_reply)) as direct_http:
        direct_sdk = openai.AsyncOpenAI(
            api_key="fixture-key",
            base_url="https://fixture.invalid/v1",
            http_client=direct_http,
            max_retries=0,
        )
        result = await litellm.acompletion(
            model="openai/gpt-4o-mini",
            api_key="fixture-key",
            api_base="https://fixture.invalid/v1",
            client=direct_sdk,
            messages=[{"role": "user", "content": "unrelated C"}],
            num_retries=0,
        )
        await asyncio.wait_for(direct.done.wait(), 5)
        assert result.choices[0].message.content == "direct answer"
    assert len(direct_requests) == len(direct.events) == 1
    assert len(writer.calls) == 3
    assert litellm.callbacks is callbacks and callbacks[-1] is direct
    await llm.aclose()
    assert not http.is_closed
    await sdk.close()
