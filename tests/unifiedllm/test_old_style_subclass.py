# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UnifiedLLM subclasses implement the abstract _call_once/_acall_once."""

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


def _text(response: LLMResponse) -> str:
    return "".join(p.text for p in response.parts if isinstance(p, AssistantText))


def test_a_subclass_that_only_overrides_call_cannot_be_built():
    with pytest.raises(TypeError, match="abstract"):
        OldStyleLLM("old-model")


async def test_a_subclass_goes_through_call_once():
    llm = NewStyleLLM("new-model")
    assert _text(llm.call([{"role": "user", "content": "hi"}])) == "once"
    assert _text(await llm.acall([{"role": "user", "content": "hi"}])) == "async once"
