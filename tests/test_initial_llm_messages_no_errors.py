# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ensure initial messages sent to the LLM do not contain error leakage.

Prevents "system prompt bugs" where exceptions, tracebacks, or error strings
from context building (e.g. _system_prompt(), doc(self), DynamicContext evaluation)
end up in the system or other messages sent to the LLM.

Uses FakeLLMClient to capture messages; asserts that no message content
contains common error indicators (Traceback, Exception:, etc.).
"""

import json

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

# Patterns that indicate an error or traceback leaked into prompt content.
# Keep these specific enough to avoid false positives (e.g. "error" in "rounding error").
_INITIAL_MESSAGE_ERROR_PATTERNS = [
    "Traceback (most recent call last)",
    '  File "',
    "Exception:",
    "Error:",
    "NameError:",
    "AttributeError:",
    "TypeError:",
    "KeyError:",
    "ValueError:",
    "RuntimeError:",
    "raise ",
    "  File '<",
]


def _message_content_as_text(msg: dict) -> str:
    """Extract all text from a message for checking. Handles string or list content."""
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def assert_no_error_patterns_in_messages(
    messages: list[dict],
    error_patterns: list[str] | None = None,
) -> None:
    """Raise AssertionError if any message content contains error/traceback patterns.

    Used to ensure system prompt and initial messages are free of leaked exceptions.
    """
    patterns = error_patterns or _INITIAL_MESSAGE_ERROR_PATTERNS
    for i, msg in enumerate(messages):
        text = _message_content_as_text(msg)
        role = msg.get("role", "?")
        for pattern in patterns:
            assert pattern not in text, (
                f"Message {i} (role={role}) contains error-like pattern {pattern!r}. "
                "This usually means an exception or traceback leaked into the prompt.\n"
                f"First 500 chars: {text[:500]!r}"
            )


def _return_result_tool(result: object, call_id: str = "call_return") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="return_result",
        arguments=json.dumps({"result": result}),
    )


class TestInitialLlmMessagesNoErrors:
    """Initial messages sent to the LLM must not contain error leakage."""

    @pytest.mark.asyncio
    async def test_first_llm_call_messages_contain_no_errors(self):
        """Messages from the first LLM call (system + task) must not contain error patterns."""
        fake_llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content="",
                    tool_calls=[_return_result_tool(42)],
                    finish_reason="tool_calls",
                    assistant_message={"role": "assistant", "content": ""},
                ),
            ]
        )

        class SimpleAgent(Agent, llm=fake_llm):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def answer(self) -> int:
                """Return the answer."""
                ...

        agent = SimpleAgent()
        await agent.answer()

        assert fake_llm.call_count >= 1, "Expected at least one LLM call"
        assert_no_error_patterns_in_messages(fake_llm.last_messages)

    @pytest.mark.asyncio
    async def test_assertion_fails_when_error_in_messages(self):
        """Prove that assert_no_error_patterns_in_messages would catch error leakage."""
        messages_with_error = [
            {
                "role": "system",
                "content": 'You are a helpful agent.\n\nTraceback (most recent call last):\n  File "<stdin>", line 1',
            },
            {"role": "user", "content": "Hello"},
        ]
        with pytest.raises(AssertionError, match="Traceback"):
            assert_no_error_patterns_in_messages(messages_with_error)

    @pytest.mark.asyncio
    async def test_assertion_fails_on_exception_line_in_system(self):
        """Prove that Exception: in system content is detected."""
        messages_with_exception = [
            {"role": "system", "content": "You are TestAgent.\n\nException: something went wrong"},
        ]
        with pytest.raises(AssertionError, match="Exception:"):
            assert_no_error_patterns_in_messages(messages_with_exception)
