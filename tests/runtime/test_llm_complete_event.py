# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLMComplete event lifecycle.

Pins that ``runtime.generate()`` emits exactly one ``LLMComplete`` event
per LLM round-trip, with the correct payload assembled from
``LLMResponse.usage`` / ``.tool_calls`` / ``.reasoning`` and the
surrounding generation_id context.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from nooa import strategy
from nooa.events import LLMComplete
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _resp(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    *,
    usage: dict | None = None,
    reasoning: str | None = None,
) -> LLMResponse:
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
        usage=usage,
        reasoning=reasoning,
    )


class TestLLMCompleteEvent:
    """LLMComplete fires once per runtime.generate() with the right payload."""

    @pytest.mark.asyncio
    async def test_fires_exactly_once_per_generate(self) -> None:
        """Single LLM round-trip emits exactly one LLMComplete event."""
        recorded: list[LLMComplete] = []

        @strategy(
            PredictStrategy(),
            llm=FakeLLMClient(
                scripted_responses=[
                    _resp(
                        content='{"answer":"hi"}',
                        usage={"prompt_tokens": 42, "completion_tokens": 7},
                    )
                ]
            ),
        )
        async def predict_fn(prompt: str) -> dict:
            """{prompt}"""
            ...

        # Hook the standalone agent's event manager: we wrap the standalone
        # wrapper so we can install the handler on the fresh child agent
        # before generation runs.
        from nooa.standalone import _atif_exporter_var

        class _Capture:
            def _attach_child(self, em, child_agent_name: str = "") -> None:
                em.on("LLMComplete", lambda e: recorded.append(e))

            def _detach_child(self, em) -> None:
                pass

        token = _atif_exporter_var.set(_Capture())
        try:
            await predict_fn("say hi")
        finally:
            _atif_exporter_var.reset(token)

        assert len(recorded) == 1, f"expected exactly 1 LLMComplete, got {len(recorded)}"

    @pytest.mark.asyncio
    async def test_payload_carries_model_tokens_cost_generation_id(self) -> None:
        """LLMComplete payload reflects LLMResponse.usage and llm_client.model."""
        recorded: list[LLMComplete] = []

        @strategy(
            PredictStrategy(),
            llm=FakeLLMClient(
                scripted_responses=[
                    _resp(
                        content='{"v":1}',
                        usage={
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "cached_tokens": 30,
                            "cost": 0.0042,
                        },
                        reasoning="I think...",
                    )
                ]
            ),
        )
        async def predict_fn(prompt: str) -> dict:
            """{prompt}"""
            ...

        from nooa.standalone import _atif_exporter_var

        class _Capture:
            def _attach_child(self, em, child_agent_name: str = "") -> None:
                em.on("LLMComplete", lambda e: recorded.append(e))

            def _detach_child(self, em) -> None:
                pass

        token = _atif_exporter_var.set(_Capture())
        try:
            await predict_fn("hi")
        finally:
            _atif_exporter_var.reset(token)

        assert len(recorded) == 1
        ev = recorded[0]
        assert ev.model_name == "fake-model"
        assert ev.prompt_tokens == 100
        assert ev.completion_tokens == 20
        assert ev.cached_tokens == 30
        assert ev.cost_usd == pytest.approx(0.0042)
        assert ev.reasoning_content == "I think..."
        assert ev.tool_calls == []
        # generation_id is non-empty (set by the strategy that wrapped this turn).
        assert ev.generation_id != ""

    @pytest.mark.asyncio
    async def test_carries_structured_tool_calls(self) -> None:
        """tool_calls list mirrors LLMResponse.tool_calls (canonical ids)."""
        recorded: list[LLMComplete] = []
        tcs = [
            ToolCall(
                id="call_alpha", name="execute_python", arguments=json.dumps({"code": "print(1)"})
            ),
            ToolCall(id="call_beta", name="return_result", arguments=json.dumps({"result": 42})),
        ]

        @strategy(
            PredictStrategy(),
            llm=FakeLLMClient(
                scripted_responses=[
                    _resp(
                        content='{"ok":true}',
                        tool_calls=tcs,
                        usage={"prompt_tokens": 1, "completion_tokens": 1},
                    )
                ]
            ),
        )
        async def predict_fn() -> dict:
            """Do something."""
            ...

        from nooa.standalone import _atif_exporter_var

        class _Capture:
            def _attach_child(self, em, child_agent_name: str = "") -> None:
                em.on("LLMComplete", lambda e: recorded.append(e))

            def _detach_child(self, em) -> None:
                pass

        token = _atif_exporter_var.set(_Capture())
        try:
            try:
                await predict_fn()
            except Exception:
                # PredictStrategy may surface a validation error on tool_calls;
                # we only care that LLMComplete fired before that.
                pass
        finally:
            _atif_exporter_var.reset(token)

        assert len(recorded) >= 1
        ev = recorded[0]
        assert [tc["tool_call_id"] for tc in ev.tool_calls] == ["call_alpha", "call_beta"]
        assert ev.tool_calls[0]["function_name"] == "execute_python"
        assert json.loads(ev.tool_calls[0]["arguments"]) == {"code": "print(1)"}

    @pytest.mark.asyncio
    async def test_zero_usage_renders_zero_tokens(self) -> None:
        """When the provider returns no usage block, LLMComplete carries zeros."""
        recorded: list[LLMComplete] = []

        @strategy(
            PredictStrategy(),
            llm=FakeLLMClient(scripted_responses=[_resp(content='{"v":1}', usage=None)]),
        )
        async def predict_fn() -> dict:
            """Do something."""
            ...

        from nooa.standalone import _atif_exporter_var

        class _Capture:
            def _attach_child(self, em, child_agent_name: str = "") -> None:
                em.on("LLMComplete", lambda e: recorded.append(e))

            def _detach_child(self, em) -> None:
                pass

        token = _atif_exporter_var.set(_Capture())
        try:
            await predict_fn()
        finally:
            _atif_exporter_var.reset(token)

        assert len(recorded) == 1
        ev = recorded[0]
        assert ev.prompt_tokens == 0
        assert ev.completion_tokens == 0
        assert ev.cached_tokens == 0
        assert ev.cost_usd == 0.0


class TestAtifExporterContextVar:
    """_atif_exporter_var cascade for standalone agents."""

    def test_default_value_is_none(self) -> None:
        """Outside an install_atif scope, the var resolves to None."""
        from nooa.standalone import _atif_exporter_var

        assert _atif_exporter_var.get() is None

    @pytest.mark.asyncio
    async def test_set_value_visible_to_standalone_wrapper(self) -> None:
        """Setting the var BEFORE a standalone call exposes it during wrapper exec."""
        from nooa.standalone import _atif_exporter_var

        seen_event_managers: list[Any] = []

        class _Capture:
            def _attach_child(self, em, child_agent_name: str = "") -> None:
                seen_event_managers.append(em)
                em.on("LLMComplete", lambda e: None)

            def _detach_child(self, em) -> None:
                pass

        @strategy(
            PredictStrategy(),
            llm=FakeLLMClient(scripted_responses=[_resp(content='{"v":1}')]),
        )
        async def fn() -> dict:
            """Do something."""
            ...

        token = _atif_exporter_var.set(_Capture())
        try:
            await fn()
        finally:
            _atif_exporter_var.reset(token)

        assert len(seen_event_managers) == 1, (
            "Standalone wrapper should call exporter._attach_child(em) "
            "exactly once per call when _atif_exporter_var is set."
        )

    @pytest.mark.asyncio
    async def test_reset_restores_to_none(self) -> None:
        """After reset/exit, the var is back to None and wrapper does not call attach."""
        from nooa.standalone import _atif_exporter_var

        attach_calls: list[Any] = []

        class _Capture:
            def _attach_child(self, em, child_agent_name: str = "") -> None:
                attach_calls.append(em)

            def _detach_child(self, em) -> None:
                pass

        @strategy(
            PredictStrategy(),
            llm=FakeLLMClient(
                scripted_responses=[
                    _resp(content='{"v":1}'),
                    _resp(content='{"v":2}'),
                ]
            ),
        )
        async def fn() -> dict:
            """Do something."""
            ...

        token = _atif_exporter_var.set(_Capture())
        await fn()  # inside scope → attach fires
        _atif_exporter_var.reset(token)
        await fn()  # outside scope → attach should NOT fire

        assert len(attach_calls) == 1, (
            f"attach should fire once inside the scope, not after reset; got {len(attach_calls)}"
        )
        assert _atif_exporter_var.get() is None
