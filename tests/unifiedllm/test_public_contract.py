# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from nooa.unifiedllm import (
    CompletionClient,
    LLMChunk,
    LLMClient,
    LLMContextLengthError,
    LLMGatewayTimeoutError,
    LLMToolCallChunk,
    LLMTransportError,
    LLMUsage,
    ResponsesClient,
)
from nooa.unifiedllm._anyllm import _normalize_error


def _chat_chunk(text="", finish=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=[]), finish_reason=finish
            )
        ],
        usage=None,
    )


def test_public_protocol_and_sync_stream_closes(monkeypatch):
    client = CompletionClient("openai/test")
    closed = []

    class Stream:
        def __iter__(self):
            yield _chat_chunk("hello")
            yield _chat_chunk(finish="stop")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(client._transport, "completion", lambda **kwargs: Stream())
    assert isinstance(client, LLMClient)
    assert [chunk.content for chunk in client.stream([{"role": "user", "content": "hi"}])] == [
        "hello",
        "",
    ]
    assert closed == [True]
    client.close()


def test_stream_tool_calls_are_provider_neutral(monkeypatch):
    client = CompletionClient("openai/test")
    provider_call = SimpleNamespace(
        index=0, id="call-1", function=SimpleNamespace(name="lookup", arguments="{}")
    )
    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=[provider_call]),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
    )
    monkeypatch.setattr(client._transport, "completion", lambda **kwargs: iter([chunk]))

    normalized = list(client.stream([{"role": "user", "content": "hi"}]))[0]

    assert normalized.tool_calls == (LLMToolCallChunk(0, "call-1", "lookup", "{}"),)
    client.close()


@pytest.mark.asyncio
async def test_async_stream_cancellation_closes(monkeypatch):
    client = CompletionClient("openai/test")
    closed = []

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            return _chat_chunk("hello")

        async def aclose(self):
            closed.append(True)

    async def make(**kwargs):
        return Stream()

    monkeypatch.setattr(client._transport, "acompletion", make)
    stream = client.astream([{"role": "user", "content": "hi"}])
    assert isinstance(await anext(stream), LLMChunk)
    await stream.aclose()
    assert closed == [True]
    await client.aclose()


@pytest.mark.asyncio
async def test_responses_stream_aggregation_in_acall(monkeypatch):
    client = ResponsesClient("openai/test", capabilities={"responses": True})

    async def events():
        yield {"type": "response.output_text.delta", "delta": "hel"}
        yield {"type": "response.output_text.delta", "delta": "lo"}

    async def make(**kwargs):
        return events()

    monkeypatch.setattr(client._transport, "aresponses", make)
    response = await client.acall([{"role": "user", "content": "hi"}], stream=True)
    assert response.content == "hello"
    await client.aclose()


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            type(
                "RawBadRequest",
                (Exception,),
                {"status_code": 400, "body": {"error": {"code": "context_length_exceeded"}}},
            )("long"),
            LLMContextLengthError,
        ),
        (type("APIConnectionError", (Exception,), {})("down"), LLMTransportError),
        (type("APITimeoutError", (Exception,), {})("slow"), LLMGatewayTimeoutError),
    ],
)
def test_raw_sdk_error_normalization(exc, expected):
    assert isinstance(_normalize_error(exc, "openai"), expected)


def test_stream_lifecycle_finishes_once_with_normalized_response(monkeypatch):
    import nooa.runtime.llm_lifecycle as lifecycle

    client = CompletionClient("openai/test")
    terminal = []

    class Stream:
        def __iter__(self):
            yield _chat_chunk("hel")
            yield _chat_chunk("lo", "stop")

        def close(self):
            pass

    monkeypatch.setattr(client._transport, "completion", lambda **kwargs: Stream())
    monkeypatch.setattr(lifecycle, "begin_llm_call", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        lifecycle,
        "end_llm_call",
        lambda call, response=None, exception=None: terminal.append((response, exception)),
    )

    assert [chunk.content for chunk in client.stream([{"role": "user", "content": "hi"}])] == [
        "hel",
        "lo",
    ]
    assert len(terminal) == 1
    response, error = terminal[0]
    assert error is None
    assert response.content == "hello"
    assert response.assistant_message == {"role": "assistant", "content": "hello"}
    client.close()


@pytest.mark.asyncio
async def test_async_stream_close_emits_one_cancelled_terminal_hook(monkeypatch):
    import nooa.runtime.llm_lifecycle as lifecycle

    client = CompletionClient("openai/test")
    terminal = []

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            return _chat_chunk("hello")

        async def aclose(self):
            pass

    async def make(**kwargs):
        return Stream()

    monkeypatch.setattr(client._transport, "acompletion", make)
    monkeypatch.setattr(lifecycle, "begin_llm_call", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        lifecycle,
        "end_llm_call",
        lambda call, response=None, exception=None: terminal.append((response, exception)),
    )

    stream = client.astream([{"role": "user", "content": "hi"}])
    await anext(stream)
    await stream.aclose()
    assert len(terminal) == 1
    assert terminal[0][0] is None
    assert isinstance(terminal[0][1], GeneratorExit)
    await client.aclose()


def test_response_contract_has_no_provider_escape_hatch(monkeypatch):
    client = CompletionClient("openai/test")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="hello", tool_calls=[], reasoning_content=None),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    monkeypatch.setattr(client._transport, "completion", lambda **kwargs: response)

    normalized = client.call([{"role": "user", "content": "hi"}])

    assert not hasattr(normalized, "raw_response")
    assert all("any_llm" not in type(value).__module__ for value in vars(normalized).values())
    client.close()


def test_normalization_failure_emits_one_error_terminal(monkeypatch):
    import nooa.runtime.llm_lifecycle as lifecycle

    client = CompletionClient("openai/test")
    terminal = []
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="not json", tool_calls=[], reasoning_content=None),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    monkeypatch.setattr(client._transport, "completion", lambda **kwargs: response)
    monkeypatch.setattr(lifecycle, "begin_llm_call", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        lifecycle,
        "end_llm_call",
        lambda call, response=None, exception=None: terminal.append((response, exception)),
    )

    from pydantic import BaseModel

    class Output(BaseModel):
        value: int

    with pytest.raises(ValueError):
        client.call([{"role": "user", "content": "hi"}], output_model=Output)

    assert len(terminal) == 1
    assert terminal[0][0] is None
    assert terminal[0][1] is not None
    client.close()


def test_reasoning_client_terminalizes_after_think_tag_normalization(monkeypatch):
    import nooa.runtime.llm_lifecycle as lifecycle
    from nooa.unifiedllm import ReasoningCompletionClient

    client = ReasoningCompletionClient("openai/test")
    terminal = []
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="<think>work</think>answer", tool_calls=[], reasoning_content=None
                ),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    monkeypatch.setattr(client._transport, "completion", lambda **kwargs: response)
    monkeypatch.setattr(lifecycle, "begin_llm_call", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        lifecycle,
        "end_llm_call",
        lambda call, response=None, exception=None: terminal.append((response, exception)),
    )

    normalized = client.call([{"role": "user", "content": "hi"}])

    assert normalized.content == "answer"
    assert normalized.reasoning == "work"
    assert terminal == [(normalized, None)]
    client.close()


def test_response_stores_canonical_usage_with_read_only_compatibility_view():
    from nooa.unifiedllm import LLMResponse

    usage = LLMUsage(input_tokens=2, output_tokens=3, cached_input_tokens=1)
    response = LLMResponse(
        "ok",
        [],
        "stop",
        {"role": "assistant", "content": "ok"},
        usage={"prompt_tokens": 2, "completion_tokens": 3, "cached_input_tokens": 1},
    )

    assert response.reported_usage == usage
    assert response.usage == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "cached_input_tokens": 1,
    }
    assert "reported_usage" not in vars(response)


def test_responses_public_sync_stream_normalizes_and_closes(monkeypatch):
    client = ResponsesClient("openai/test", capabilities={"responses": True})
    closed = []

    class Stream:
        def __iter__(self):
            yield {"type": "response.output_text.delta", "delta": "hello"}
            yield {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                },
            }

        def close(self):
            closed.append(True)

    monkeypatch.setattr(client._transport, "responses", lambda **kwargs: Stream())
    chunks = list(client.stream([{"role": "user", "content": "hi"}]))

    assert [chunk.content for chunk in chunks] == ["hello", ""]
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage == LLMUsage(input_tokens=2, output_tokens=1)
    assert closed == [True]
    client.close()


@pytest.mark.asyncio
async def test_responses_public_async_stream_close_terminalizes_once(monkeypatch):
    import nooa.runtime.llm_lifecycle as lifecycle

    client = ResponsesClient("openai/test", capabilities={"responses": True})
    closed = []
    terminal = []

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            return {"type": "response.output_text.delta", "delta": "hello"}

        async def aclose(self):
            closed.append(True)

    async def make(**kwargs):
        return Stream()

    monkeypatch.setattr(client._transport, "aresponses", make)
    monkeypatch.setattr(lifecycle, "begin_llm_call", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        lifecycle,
        "end_llm_call",
        lambda call, response=None, exception=None: terminal.append((response, exception)),
    )
    stream = client.astream([{"role": "user", "content": "hi"}])
    assert (await anext(stream)).content == "hello"
    await stream.aclose()

    assert closed == [True]
    assert len(terminal) == 1
    assert isinstance(terminal[0][1], GeneratorExit)
    await client.aclose()


def test_responses_call_stream_true_aggregates_normalized_chunks(monkeypatch):
    client = ResponsesClient("openai/test", capabilities={"responses": True})

    class Stream:
        def __iter__(self):
            yield {"type": "response.output_text.delta", "delta": "hel"}
            yield {"type": "response.output_text.delta", "delta": "lo"}
            yield {"type": "response.completed", "response": {"status": "completed"}}

        def close(self):
            pass

    monkeypatch.setattr(client._transport, "responses", lambda **kwargs: Stream())
    response = client.call([{"role": "user", "content": "hi"}], stream=True)

    assert response.content == "hello"
    assert response.finish_reason == "stop"
    client.close()
