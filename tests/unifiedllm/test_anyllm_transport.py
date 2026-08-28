from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nooa.unifiedllm import CompletionClient, LLMResponse, ResponsesClient


def completion_response(content="hello", *, finish="stop", usage=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=usage
    )


def test_completion_normalizes_provider_response_without_leak():
    client = CompletionClient("gpt-test", api_key="test")
    client._transport.completion = Mock(
        return_value=completion_response(
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
        )
    )
    response = client.call([{"role": "user", "content": "hi"}])
    assert isinstance(response, LLMResponse)
    assert response.content == "hello"
    assert response.usage == {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    assert response.to_wire()["message"]["content"] == "hello"


@pytest.mark.asyncio
async def test_completion_async_normalizes_tool_call():
    function = SimpleNamespace(name="lookup", arguments='{"id": 1}')
    provider_call = SimpleNamespace(id="call-1", function=function)
    client = CompletionClient("gpt-test", api_key="test")
    client._transport.acompletion = AsyncMock(
        return_value=completion_response("", finish="tool_calls", tool_calls=[provider_call])
    )
    response = await client.acall([{"role": "user", "content": "hi"}])
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "lookup"


def test_responses_normalizes_output_and_finish_reason():
    client = ResponsesClient(
        "gpt-test",
        api_key="test",
        cache_control_injection_points=[],
        capabilities={"responses": True},
    )
    raw = SimpleNamespace(
        output=[],
        output_text="done",
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
        usage=None,
    )
    client._transport.responses = Mock(return_value=raw)
    response = client.call([{"role": "user", "content": "hi"}])
    assert response.content == "done"
    assert response.finish_reason == "length"
    assert response.to_wire()["message"]["content"] == "done"


def test_sync_stream_aggregation_does_not_mutate_chunks():
    from nooa.unifiedllm.unifiedllm import _collect_sync

    first_delta = SimpleNamespace(content="hel", reasoning_content="why ", tool_calls=None)
    second_delta = SimpleNamespace(content="lo", reasoning_content="now", tool_calls=None)
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=first_delta, finish_reason=None)], usage=None
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=second_delta, finish_reason="stop")],
            usage={"total_tokens": 2},
        ),
    ]
    result = _collect_sync(iter(chunks))
    assert result.choices[0].message.content == "hello"
    assert result.choices[0].message.reasoning_content == "why now"
    assert result.usage == {"total_tokens": 2}
    assert not hasattr(chunks[-1].choices[0], "message")


@pytest.mark.asyncio
async def test_async_stream_aggregates_split_tool_calls():
    from nooa.unifiedllm.unifiedllm import _collect_async

    async def stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_",
                                function=SimpleNamespace(name="look", arguments='{"id":'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0, id="1", function=SimpleNamespace(name="up", arguments="1}")
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        )

    result = await _collect_async(stream())
    call = result.choices[0].message.tool_calls[0]
    assert (call.id, call.function.name, call.function.arguments) == (
        "call_1",
        "lookup",
        '{"id":1}',
    )


def test_provider_routing_supports_explicit_and_legacy_bare_names():
    from nooa.unifiedllm._anyllm import AnyLLMTransport

    assert AnyLLMTransport("anthropic:claude-3-5-sonnet", {}).provider_name == "anthropic"
    assert AnyLLMTransport("claude-sonnet-4-5-20250514", {}).provider_name == "anthropic"
    assert AnyLLMTransport("gemini-2.5-pro", {}).provider_name == "gemini"
    assert AnyLLMTransport("gpt-5-mini", {}).provider_name == "openai"


def test_provider_options_are_merged_into_request_not_constructor(monkeypatch):
    from nooa.unifiedllm._anyllm import AnyLLM, AnyLLMTransport

    provider = Mock()
    provider.completion.return_value = completion_response()
    create = Mock(return_value=provider)
    monkeypatch.setattr(AnyLLM, "create", create)
    transport = AnyLLMTransport(
        "openai:gpt-test", {"api_key": "test", "provider_options": {"extra_body": {"x": 1}}}
    )

    transport.completion(messages=[])

    assert "extra_body" not in create.call_args.kwargs
    assert provider.completion.call_args.kwargs["extra_body"] == {"x": 1}


def test_responses_requires_declared_capability():
    with pytest.raises(ValueError, match="capabilities.responses=true"):
        ResponsesClient("openai:gpt-test")


def test_openai_compatible_responses_uses_openai_provider(monkeypatch):
    from nooa.unifiedllm._anyllm import AnyLLM, AnyLLMTransport

    create = Mock(return_value=Mock())
    compatible = Mock()
    monkeypatch.setattr(AnyLLM, "create", create)
    monkeypatch.setattr(AnyLLM, "create_openai_compatible", compatible)
    transport = AnyLLMTransport(
        "gpt-test",
        {
            "provider": "openai-compatible",
            "endpoint": "https://example.test/v1",
            "_responses_api": True,
        },
    )

    _ = transport.provider

    create.assert_called_once()
    assert create.call_args.args == ("openai",)
    compatible.assert_not_called()


def test_provider_options_reject_normalized_request_collisions():
    with pytest.raises(ValueError, match="max_tokens"):
        CompletionClient("openai:gpt-test", provider_options={"max_tokens": 7})
    with pytest.raises(ValueError, match="response_format"):
        CompletionClient("openai:gpt-test", provider_options={"response_format": {}})
