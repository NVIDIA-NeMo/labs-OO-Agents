# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failing test: recovery path doesn't archive when prompt_tokens can't be parsed.

Replicates the bug found in e2e testing: when the API error says
"your request has N input tokens" (not "prompt contains N tokens"),
_parse_prompt_tokens returns None -> _compute_reduced_max_tokens returns None
-> the error is re-raised WITHOUT calling _archive_on_context_error.

Two bugs:
1. _parse_prompt_tokens regex only matches "prompt ... N tokens", not "request has N input tokens"
2. _archive_on_context_error is called AFTER `if _reduced is None: raise`, so it never fires
   when the prompt token count can't be parsed.
"""

import pytest

from nooa import Agent
from nooa.events import Message
from nooa.runtime.actor import (
    _context_window_for_error,
    _current_llm_var,
    _current_method_var,
    _parse_context_window_tokens,
    _parse_prompt_tokens,
)
from nooa.unifiedllm import FakeLLMClient


class _ContextWindowExceededError(Exception):
    """Simulates litellm.BadRequestError wrapping ContextWindowExceededError."""

    pass


class _FakeLLM(FakeLLMClient):
    _cw = 262_144

    @property
    def context_window(self):
        return self._cw


def _mk_llm(context_window=262_144):
    class _LLM(_FakeLLM):
        _cw = context_window

    llm = _LLM()
    llm.model = "gpt-4o"
    return llm


class TestPromptTokensRegex:
    """_parse_prompt_tokens must handle all common API error formats."""

    def test_openai_format(self):
        """OpenAI: 'prompt contains at least 67073 tokens'."""
        exc = Exception(
            "This model's maximum context length is 131072 tokens. "
            "However, you requested 64000 output tokens and your prompt "
            "contains at least 67073 input tokens."
        )
        assert _parse_prompt_tokens(exc) == 67073

    def test_nvidia_gateway_format(self):
        """NVIDIA gateway: 'your request has 1203158 input tokens'."""
        exc = Exception(
            "This model's maximum context length is 262144 tokens. "
            "However, your request has 1203158 input tokens. "
            "Please reduce the length of the input messages."
        )
        result = _parse_prompt_tokens(exc)
        assert result == 1203158, f"Should extract 1203158 from NVIDIA gateway format, got {result}"

    def test_litellm_wrapped_nvidia_format(self):
        """litellm wraps: 'ContextWindowExceededError: ... request has N input tokens'."""
        exc = Exception(
            "litellm.BadRequestError: OpenAIException - "
            "litellm.ContextWindowExceededError: "
            "This model's maximum context length is 262144 tokens. "
            "However, your request has 1203158 input tokens."
        )
        result = _parse_prompt_tokens(exc)
        assert result == 1203158, f"Should handle litellm-wrapped NVIDIA format, got {result}"


class TestContextWindowTokensRegex:
    """_parse_context_window_tokens must handle provider context-limit text."""

    def test_context_window_is_parsed_from_openai_nvidia_format(self):
        """Provider error text can be the only source for model context size."""
        exc = Exception(
            "This model's maximum context length is 262144 tokens. "
            "However, you requested 32000 output tokens and your prompt contains "
            "at least 230145 input tokens."
        )

        assert _parse_context_window_tokens(exc) == 262144

    def test_context_window_is_parsed_from_anthropic_gt_format(self):
        """Anthropic/Azure style 'input > maximum' text exposes the window too."""
        exc = Exception("prompt is too long: 1,017,198 tokens > 1,000,000 maximum")

        assert _parse_context_window_tokens(exc) == 1000000

    def test_provider_reported_window_wins_over_stale_client_value(self):
        """The provider error is authoritative for the endpoint that rejected us."""

        class StaleWindowLLM:
            context_window = 131072

        exc = Exception("This model's maximum context length is 262144 tokens.")

        assert _context_window_for_error(StaleWindowLLM(), exc) == 262144


class TestArchivalFiresOnContextError:
    """_archive_on_context_error must fire even when prompt tokens can't be parsed."""

    @pytest.mark.asyncio
    async def test_archival_fires_when_reduced_is_none(self):
        """When _compute_reduced_max_tokens returns None (can't parse prompt tokens
        and no max_tokens provided), archival should STILL fire before re-raising.

        generate() archives events then re-raises (since it can't compute a
        reduced max_tokens). The caller (e.g. CodeAct) retries with fresh
        messages built from the now-smaller event store.
        """
        from unittest.mock import patch

        llm = _mk_llm(262_144)

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        for i in range(20):
            agent.event_manager.add(Message(content=f"message {i} " * 50))

        n_events_before = len(list(agent.event_manager.keys()))

        error = _ContextWindowExceededError(
            "This model's maximum context length is 262144 tokens. "
            "However, your request has 500000 input tokens."
        )

        summary_events = []
        agent.event_manager.on("Summary", lambda ev: summary_events.append(ev))

        method = type(agent).respond
        llm_token = _current_llm_var.set(llm)
        method_token = _current_method_var.set(method)
        try:
            with patch.object(llm, "acall", side_effect=error):
                with patch(
                    "nooa.runtime.actor._is_context_window_error",
                    side_effect=lambda exc: isinstance(exc, _ContextWindowExceededError),
                ):
                    with pytest.raises(_ContextWindowExceededError):
                        await agent.runtime.generate(tools=[], max_tokens=None)
        finally:
            _current_llm_var.reset(llm_token)
            _current_method_var.reset(method_token)

        # Archival should have fired BEFORE the re-raise
        n_events_after = len(list(agent.event_manager.keys()))
        assert n_events_after < n_events_before, (
            f"Archival should reduce events: {n_events_after} >= {n_events_before}. "
            f"Summary events: {len(summary_events)}"
        )
        assert len(summary_events) >= 1, "Archival should emit Summary events"

    @pytest.mark.asyncio
    async def test_archival_uses_provider_window_when_llm_window_is_missing(self):
        """Direct CompletionClient/custom models may not expose context_window.

        The provider's ContextWindowExceededError still includes the true window,
        so recovery should archive events instead of no-oping and retrying the
        same overlarge prompt until CodeAct exhausts its retries.
        """
        from unittest.mock import patch

        class NoWindowLLM(_FakeLLM):
            @property
            def context_window(self):
                return None

        llm = NoWindowLLM()
        llm.model = "openai/nvidia/qwen/qwen3.5-35b-a3b"

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        for i in range(30):
            agent.event_manager.add(Message(content=f"message {i} " * 200))

        n_events_before = len(list(agent.event_manager.keys()))
        error = _ContextWindowExceededError(
            "This model's maximum context length is 262144 tokens. "
            "However, you requested 32000 output tokens and your prompt contains "
            "at least 230145 input tokens."
        )

        summary_events = []
        agent.event_manager.on("Summary", lambda ev: summary_events.append(ev))

        method = type(agent).respond
        llm_token = _current_llm_var.set(llm)
        method_token = _current_method_var.set(method)
        try:
            with patch.object(llm, "acall", side_effect=error):
                with patch(
                    "nooa.runtime.actor._is_context_window_error",
                    side_effect=lambda exc: isinstance(exc, _ContextWindowExceededError),
                ):
                    with pytest.raises(_ContextWindowExceededError):
                        await agent.runtime.generate(tools=[], max_tokens=32000)
        finally:
            _current_llm_var.reset(llm_token)
            _current_method_var.reset(method_token)

        n_events_after = len(list(agent.event_manager.keys()))
        assert n_events_after < n_events_before
        assert len(summary_events) >= 1

    @pytest.mark.asyncio
    async def test_archival_fires_with_unparseable_token_count_by_shedding_minimum(self):
        """When the error message has no recognizable token count, archive anyway.

        The provider rejected the request, so the prompt is too large even if
        our local estimate is under target. The safety net must shed at least
        one average event rather than retrying the identical overflowing prompt.
        """
        from unittest.mock import patch

        llm = _mk_llm(4_096)  # tiny window so small events overflow

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        for i in range(20):
            agent.event_manager.add(Message(content=f"message {i} " * 50))

        n_events_before = len(list(agent.event_manager.keys()))

        # Error with NO parseable token count — some unknown provider format
        error = _ContextWindowExceededError("context length exceeded: too many tokens in the input")

        summary_events = []
        agent.event_manager.on("Summary", lambda ev: summary_events.append(ev))

        method = type(agent).respond
        llm_token = _current_llm_var.set(llm)
        method_token = _current_method_var.set(method)
        try:
            with patch.object(llm, "acall", side_effect=error):
                with patch(
                    "nooa.runtime.actor._is_context_window_error",
                    side_effect=lambda exc: isinstance(exc, _ContextWindowExceededError),
                ):
                    with pytest.raises(_ContextWindowExceededError):
                        await agent.runtime.generate(tools=[], max_tokens=None)
        finally:
            _current_llm_var.reset(llm_token)
            _current_method_var.reset(method_token)

        # _parse_prompt_tokens should return None for this error
        assert _parse_prompt_tokens(error) is None, "This error format should NOT be parseable"

        # Archival should fire with a conservative minimum shed
        n_events_after = len(list(agent.event_manager.keys()))
        assert n_events_after < n_events_before, (
            f"Archival should fire even when token count is unparseable: "
            f"{n_events_after} >= {n_events_before}. "
            f"Summary events: {len(summary_events)}"
        )
        assert len(summary_events) >= 1, "Archival should emit Summary events"


class TestContextWindowErrorDetection:
    """Provider context-window errors must be recognized for fallback archival."""

    def test_azure_context_length_exceeded_format(self):
        """Azure context-window messages must be recognized for fallback archival."""
        from nooa.runtime.actor import _is_context_window_error

        exc = Exception(
            "litellm.BadRequestError: AzureException BadRequestError - "
            '{\n  "error": {\n'
            '    "message": "Your input exceeds the context window of this model. '
            'Please adjust your input and try again.",\n'
            '    "code": "context_length_exceeded"\n'
            "  }\n}"
        )

        assert _is_context_window_error(exc)

    def test_context_length_exceeded_code_only(self):
        """Azure errors with only context_length_exceeded code must be recognized."""
        from nooa.runtime.actor import _is_context_window_error

        exc = Exception('{"code": "context_length_exceeded", "message": "Bad request"}')

        assert _is_context_window_error(exc)

    def test_litellm_typed_context_window_error(self):
        """LiteLLM typed context-window errors are caught before provider text fallback."""
        from litellm.exceptions import ContextWindowExceededError

        from nooa.runtime.actor import _is_context_window_error

        exc = ContextWindowExceededError(
            message="provider-specific wording not listed in our substring fallback",
            model="test-model",
            llm_provider="test-provider",
        )

        assert _is_context_window_error(exc)

    def test_litellm_typed_context_window_error_in_cause_chain(self):
        """Wrapped LiteLLM context-window errors are recognized through exception chaining."""
        from litellm.exceptions import ContextWindowExceededError

        from nooa.runtime.actor import _is_context_window_error

        cause = ContextWindowExceededError(
            message="provider-specific wording not listed in our substring fallback",
            model="test-model",
            llm_provider="test-provider",
        )
        exc = RuntimeError("outer wrapper")
        exc.__cause__ = cause

        assert _is_context_window_error(exc)


class TestAnthropicPromptTooLongFormat:
    """Anthropic/Azure gateway 'prompt is too long' overflow must be recognized + parsed.

    Reproduces session d2a3557e: the gateway returned
        litellm.BadRequestError: ... {"type":"invalid_request_error",
        "message":"prompt is too long: 1017198 tokens > 1000000 maximum"}
    which is NOT a typed ContextWindowExceededError and whose wording matched
    neither _is_context_window_error's substrings nor _PROMPT_TOKENS_RE. The
    context-too-long fallback therefore never fired.
    """

    def test_detect_prompt_is_too_long(self):
        """'prompt is too long' must be recognized as a context-window error."""
        from nooa.runtime.actor import _is_context_window_error

        exc = Exception(
            "litellm.BadRequestError: OpenAIException - litellm.BadRequestError: "
            'Azure_aiException - {"type":"error","error":{"type":"invalid_request_error",'
            '"message":"prompt is too long: 1017198 tokens > 1000000 maximum"}}'
        )
        assert _is_context_window_error(exc)

    def test_parse_prompt_is_too_long(self):
        """'prompt is too long: N tokens > M maximum' must yield N."""
        exc = Exception("prompt is too long: 1017198 tokens > 1000000 maximum")
        assert _parse_prompt_tokens(exc) == 1017198

    def test_parse_prompt_is_too_long_wrapped(self):
        """Wrapped litellm/azure payload must still yield the prompt token count."""
        exc = Exception(
            "litellm.BadRequestError: OpenAIException - litellm.BadRequestError: "
            'Azure_aiException - {"type":"error","error":{"type":"invalid_request_error",'
            '"message":"prompt is too long: 1017198 tokens > 1000000 maximum"}}'
        )
        assert _parse_prompt_tokens(exc) == 1017198

    @pytest.mark.asyncio
    async def test_archival_fires_on_anthropic_prompt_too_long(self):
        """End-to-end: the Anthropic 'prompt is too long' overflow now triggers
        the context-too-long fallback (archival), instead of re-raising untouched.

        This is the regression from session d2a3557e where the fallback never fired.
        """
        from unittest.mock import patch

        from nooa import Agent
        from nooa.events import Message
        from nooa.runtime.actor import _current_llm_var, _current_method_var

        llm = _mk_llm(1_000_000)

        class A(Agent, llm=llm):
            async def respond(self, prompt: str) -> str:
                """Respond to {prompt}."""
                ...

        agent = A()
        for i in range(20):
            agent.event_manager.add(Message(content=f"message {i} " * 50))

        n_events_before = len(list(agent.event_manager.keys()))

        error = _ContextWindowExceededError(
            "litellm.BadRequestError: OpenAIException - litellm.BadRequestError: "
            'Azure_aiException - {"type":"error","error":{"type":"invalid_request_error",'
            '"message":"prompt is too long: 1017198 tokens > 1000000 maximum"}}'
        )

        summary_events = []
        agent.event_manager.on("Summary", lambda ev: summary_events.append(ev))

        method = type(agent).respond
        llm_token = _current_llm_var.set(llm)
        method_token = _current_method_var.set(method)
        try:
            with patch.object(llm, "acall", side_effect=error):
                with pytest.raises(_ContextWindowExceededError):
                    await agent.runtime.generate(tools=[], max_tokens=None)
        finally:
            _current_llm_var.reset(llm_token)
            _current_method_var.reset(method_token)

        n_events_after = len(list(agent.event_manager.keys()))
        assert n_events_after < n_events_before, (
            f"Archival should fire for 'prompt is too long': {n_events_after} >= {n_events_before}. "
            f"Summary events: {len(summary_events)}"
        )
        assert len(summary_events) >= 1, "Archival should emit Summary events"
