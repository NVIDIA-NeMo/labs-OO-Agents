# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A reply that arrived but fails to parse is never re-sent by the transport retry.

The adapters parse inside the retry. A parse error can quote the reply, so
text such as "rate limit" or "status 503" must not make it look transient.
"""

import json

import litellm
import pytest
from pydantic import BaseModel, ValidationError

from nooa.unifiedllm import CompletionClient, ResponsesClient
from nooa.unifiedllm.retry_config import RetryConfig

MESSAGES = [{"role": "user", "content": "Answer."}]
RETRY = RetryConfig(max_retries=2, base_delay=0, rate_limit_base_delay=0, jitter_factor=0)


class Answer(BaseModel):
    value: int


def chat_reply(text):
    return litellm.ModelResponse(
        choices=[{"index": 0, "message": {"role": "assistant", "content": text}}],
        usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    )


def responses_reply(text):
    return litellm.ResponsesAPIResponse(
        id="resp_1",
        object="response",
        created_at=0,
        status="completed",
        model="gpt-5.6",
        output=[
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


class RateLimited(Exception):
    """A transport failure the retry recognises by its status code."""

    status_code = 429


def install(monkeypatch, api, replies):
    """Serve ``replies`` in order (an exception is raised); return the call log."""
    seen = []
    reply = chat_reply if api == "chat" else responses_reply
    sync_name = "completion" if api == "chat" else "responses"

    def call(**kwargs):
        item = replies[min(len(seen), len(replies) - 1)]
        seen.append(kwargs)
        if isinstance(item, Exception):
            raise item
        return reply(item)

    async def acall(**kwargs):
        return call(**kwargs)

    monkeypatch.setattr(litellm, sync_name, call)
    monkeypatch.setattr(litellm, "a" + sync_name, acall)
    return seen


def client(api):
    if api == "chat":
        return CompletionClient("openai/gpt-4o-mini", api_key="sk-test", retry_config=RETRY)
    return ResponsesClient("openai/gpt-5.6", api_key="sk-test", retry_config=RETRY)


async def run(llm, is_async):
    if is_async:
        return await llm.acall(MESSAGES, output_model=Answer)
    return llm.call(MESSAGES, output_model=Answer)


API = pytest.mark.parametrize("api", ["chat", "responses"])
MODE = pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])


@API
@MODE
@pytest.mark.parametrize(
    "text",
    [
        "Sorry, I hit a rate limit and cannot answer {",
        "upstream said status 503, service unavailable",
        '{"value": "rate limit"}',
    ],
    ids=["json-rate-limit", "json-status-503", "schema-rate-limit"],
)
@pytest.mark.asyncio
async def test_malformed_reply_is_sent_once_and_keeps_its_error(monkeypatch, api, is_async, text):
    seen = install(monkeypatch, api, [text])
    llm = client(api)
    with pytest.raises(Exception) as caught:
        await run(llm, is_async)
    assert len(seen) == 1
    # The caller sees the parse error itself, as when parsing ran after the retry.
    expected = ValidationError if text.startswith("{") else json.JSONDecodeError
    assert type(caught.value) is expected


@API
@MODE
@pytest.mark.asyncio
async def test_transport_rate_limit_is_still_retried(monkeypatch, api, is_async):
    seen = install(monkeypatch, api, [RateLimited("rate limited"), '{"value": 7}'])
    response = await run(client(api), is_async)
    assert len(seen) == 2
    assert response.parsed == Answer(value=7)
