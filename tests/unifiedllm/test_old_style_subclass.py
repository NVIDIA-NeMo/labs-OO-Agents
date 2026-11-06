# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UnifiedLLM subclasses written against the older contract (override call/acall) still work."""

from __future__ import annotations

import pytest

from nooa.unifiedllm import AssistantText, LLMResponse, UnifiedLLM


class OldStyleLLM(UnifiedLLM):
    """Implements only call/acall, as subclasses did before _call_once existed."""

    def call(self, messages, tools=None, output_model=None, **kwargs):
        return LLMResponse(parts=(AssistantText(text="sync"),))

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        return LLMResponse(parts=(AssistantText(text="async"),))


class NewStyleLLM(UnifiedLLM):
    def _call_once(self, messages, tools=None, output_model=None, **kwargs):
        return LLMResponse(parts=(AssistantText(text="once"),))

    async def _acall_once(self, messages, tools=None, output_model=None, **kwargs):
        return LLMResponse(parts=(AssistantText(text="async once"),))


class EmptyLLM(UnifiedLLM):
    pass


def _text(response: LLMResponse) -> str:
    return "".join(p.text for p in response.parts if isinstance(p, AssistantText))


async def test_an_old_style_subclass_can_be_built_and_called():
    llm = OldStyleLLM("old-model")
    assert _text(llm.call([{"role": "user", "content": "hi"}])) == "sync"
    assert _text(await llm.acall([{"role": "user", "content": "hi"}])) == "async"


async def test_a_new_style_subclass_goes_through_call_once():
    llm = NewStyleLLM("new-model")
    assert _text(llm.call([{"role": "user", "content": "hi"}])) == "once"
    assert _text(await llm.acall([{"role": "user", "content": "hi"}])) == "async once"


async def test_a_subclass_with_neither_fails_clearly_when_called():
    llm = EmptyLLM("empty-model")  # building it is fine
    with pytest.raises(NotImplementedError, match=r"must implement _call_once\(\)"):
        llm.call([{"role": "user", "content": "hi"}])
    with pytest.raises(NotImplementedError, match=r"must implement _acall_once\(\)"):
        await llm.acall([{"role": "user", "content": "hi"}])
