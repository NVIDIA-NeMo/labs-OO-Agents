# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed recovery facts dominate diagnostic causes and legacy prose."""

import pytest

from nooa import Agent
from nooa.events import PythonOutput, ResultStatus, Summary
from nooa.runtime.actor import (
    ActorRuntime,
    _is_context_window_error,
    _parse_context_window_tokens,
    _parse_prompt_tokens,
)
from nooa.runtime.event_query import EventQuery
from nooa.runtime.media_recovery import image_limit_recovery
from nooa.unifiedllm import FailureKind as K
from nooa.unifiedllm import FakeLLMClient, LLMFailure
from nooa.unifiedllm.media_inputs import explicitly_rejected, image_limit


class Opaque(Exception):
    def __str__(self):
        raise AssertionError("typed cause must not be inspected")


@pytest.mark.parametrize("kind", list(K))
def test_typed_recovery_does_not_consult_cause(kind):
    error = LLMFailure(
        kind,
        "context length exceeded; too many images: 9 (max 1); invalid image",
        cause=Opaque(),
        prompt_tokens=0,
        context_window=8192,
        media_kind="image",
        media_count=9,
        media_count_limit=0,
    )
    assert _is_context_window_error(error) == (kind == K.CONTEXT_WINDOW)
    assert _parse_prompt_tokens(error) == (0 if kind == K.CONTEXT_WINDOW else None)
    assert _parse_context_window_tokens(error) == (8192 if kind == K.CONTEXT_WINDOW else None)
    assert image_limit(error) == (0 if kind == K.MEDIA_LIMIT else None)
    assert explicitly_rejected(error, "image") == (kind == K.MEDIA_REJECTED)
    assert not explicitly_rejected(error, "audio")


def test_missing_facts_are_not_extracted_from_typed_message_or_cause():
    cause = RuntimeError("context length exceeded 4096; too many images: 9 (max 1)")
    cause.__cause__ = cause
    window = LLMFailure(K.CONTEXT_WINDOW, str(cause), cause=cause)
    media = LLMFailure(K.MEDIA_LIMIT, str(cause), media_kind="image", cause=cause)
    assert _is_context_window_error(window)
    assert _parse_prompt_tokens(window) is None
    assert _parse_context_window_tokens(window) is None
    assert image_limit(media) is None


def test_raw_duck_fallback_stays_available():
    class ContextWindowExceededError(Exception):
        pass

    class RequestError(Exception):
        status_code = 400

    raw = ContextWindowExceededError(
        "maximum context length is 4096 tokens; prompt contains at least 2500 input tokens"
    )
    assert _is_context_window_error(raw)
    assert _parse_prompt_tokens(raw) == 2500
    assert _parse_context_window_tokens(raw) == 4096
    assert image_limit(RequestError("Too many images: 7 (max 5)")) == 5
    assert explicitly_rejected(RequestError("Invalid image"), "image")


def test_zero_limit_collapses_only_included_events_and_keeps_originals():
    class A(Agent, llm=FakeLLMClient()):
        pass

    agent = A()
    em = agent.event_manager

    def image(n):
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{n}"}}

    tags = [
        em.add(
            PythonOutput(
                tool_call_id=str(n),
                execution_count=n,
                execution_status=ResultStatus.COMPLETE,
                stdout=f"printed-{n}",
                images=[image(n)],
            )
        )
        for n in (1, 2)
    ]
    recovery = image_limit_recovery(agent.runtime)
    em.set_event_query(EventQuery(limit=1))
    error = LLMFailure(
        K.MEDIA_LIMIT, "zero images", media_kind="image", media_count_limit=0, cause=Opaque()
    )
    try:
        assert recovery.recover(error) is True
    finally:
        em.set_event_query(None)
    assert recovery.limit == 0
    active = list(em.values())
    assert sum(isinstance(event, PythonOutput) and bool(event.images) for event in active) == 1
    assert sum(isinstance(event, Summary) for event in active) == 1
    recovery.before_request()
    assert all(not isinstance(event, PythonOutput) or not event.images for event in em.values())
    for tag in tags:
        original = em[tag]
        assert original.images
        assert original.stdout.startswith("printed-")
    assert recovery.recover(error) is None


async def test_typed_context_and_image_recovery_each_run_once():
    window = LLMFailure(
        K.CONTEXT_WINDOW, "window", prompt_tokens=100, context_window=8192, cause=Opaque()
    )
    media = LLMFailure(
        K.MEDIA_LIMIT, "image", media_kind="image", media_count_limit=0, cause=Opaque()
    )

    class Recovery:
        def recover(self, error):
            return True if image_limit(error) == 0 else None

    class Runtime:
        _request_recoveries = [Recovery()]

    errors = [window, media]
    rebuilt = []

    async def call():
        if errors:
            raise errors.pop(0)
        return "ok"

    async def rebuild():
        rebuilt.append("rebuild")

    def recover_context(error):
        assert error is window
        rebuilt.append("window")
        return True

    assert await ActorRuntime._call_with_recovery(Runtime(), call, rebuild, recover_context) == "ok"
    assert rebuilt == ["window", "rebuild", "rebuild"]
    errors[:] = [window, window]
    with pytest.raises(LLMFailure) as caught:
        await ActorRuntime._call_with_recovery(Runtime(), call, rebuild, recover_context)
    assert caught.value is window
    errors[:] = [media, media]
    with pytest.raises(LLMFailure) as caught:
        await ActorRuntime._call_with_recovery(Runtime(), call, rebuild, recover_context)
    assert caught.value is media
    terminal = LLMFailure(K.AUTH, "context window; too many images", cause=window)
    errors[:] = [terminal]
    with pytest.raises(LLMFailure) as caught:
        await ActorRuntime._call_with_recovery(Runtime(), call, rebuild, recover_context)
    assert caught.value is terminal


@pytest.mark.parametrize(
    "facts, expected", [({"prompt_tokens": 0, "context_window": 8192}, 7936), ({}, 32000)]
)
async def test_generate_uses_typed_provider_window_and_zero_prompt(monkeypatch, facts, expected):
    from nooa.runtime.actor import _current_llm_var, _current_method_var

    class Client(FakeLLMClient):
        @property
        def context_window(self):
            return 200000

    llm = Client()

    class A(Agent, llm=llm):
        async def respond(self, prompt: str) -> str:
            """Respond to {prompt}."""
            ...

    agent = A()
    error = LLMFailure(K.CONTEXT_WINDOW, "misleading prompt 199999 tokens", cause=Opaque(), **facts)
    seen = []
    original = llm.acall

    async def acall(messages, **kwargs):
        seen.append(kwargs.get("max_tokens"))
        if len(seen) == 1:
            raise error
        return await original(messages, **kwargs)

    monkeypatch.setattr(llm, "acall", acall)
    lt = _current_llm_var.set(llm)
    mt = _current_method_var.set(type(agent).respond)
    try:
        await agent.runtime.generate(tools=[], max_tokens=64000)
    finally:
        _current_llm_var.reset(lt)
        _current_method_var.reset(mt)
    assert seen == [64000, expected]
