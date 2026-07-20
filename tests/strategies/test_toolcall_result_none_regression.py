# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ToolCallEvent.result=None bug fix (!224).

When _handle_return_result raises GenerationError (validation fails AND
session exhausted), the ToolCallEvent was previously left with result=None
in the DB.  The next session's context render would produce tool_use without
a matching tool_result, corrupting the conversation.

Three-part fix verified here:
1. Catch GenerationError in _process_tool_calls return_result path →
   update ToolCallEvent with error result before re-raising.
2. Catch GenerationError in _handle_execute_python inline return_result
   path → same pattern.
3. Defensive check in formatter: if ToolCallEvent.result is None,
   emit a placeholder tool_result instead of skipping.
"""

import json
from typing import Any

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.context_blocks.events import ToolCallEvent, ToolResult
from nooa.context_blocks.formatter import (
    XMLBlockFormatter,
    _event_block_to_messages,
)
from nooa.context_blocks.models import (
    ResolvedBlock,
    Role,
)
from nooa.errors import GenerationError
from nooa.events import ResultStatus
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(content: str, tool_calls: list | None = None) -> LLMResponse:
    """Create a test LLM response."""
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


def _tool_call(code: str, call_id: str = "call_1") -> ToolCall:
    """Create an execute_python tool call."""
    return ToolCall(
        id=call_id,
        name="execute_python",
        arguments=json.dumps({"code": code}),
    )


def _return_result(call_id: str = "call_return", result: Any = None) -> ToolCall:
    """Create a return_result tool call."""
    return ToolCall(
        id=call_id,
        name="return_result",
        arguments=json.dumps({"result": result}),
    )


def _tool_call_block(
    *,
    tool_call_id: str = "tc_1",
    name: str = "execute_python",
    arguments: dict | None = None,
    result: ToolResult | None = None,
    key: str = "tc",
) -> ResolvedBlock:
    """Helper: ResolvedBlock carrying a ToolCallEvent."""
    event = ToolCallEvent(
        tool_call_id=tool_call_id,
        name=name,
        arguments=arguments or {"code": "pass"},
        result=result,
    )
    return ResolvedBlock(key=key, content="", role=Role.ASSISTANT, event=event)


_TEST_LLM = FakeLLMClient()


# ---------------------------------------------------------------------------
# 1. return_result tool call path — GenerationError sets result != None
# ---------------------------------------------------------------------------


class TestReturnResultToolCallPath:
    """When the LLM calls return_result as a tool and validation fails with
    session exhausted, the ToolCallEvent must have result != None."""

    @pytest.mark.asyncio
    async def test_return_result_generation_error_sets_result(self):
        """return_result validation fail + exhausted → ToolCallEvent has error result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=1)))
            async def get_number(self) -> int:
                """Return an integer."""
                ...

        # LLM returns a string instead of int — fails validation.
        # max_retries=1, so the first error exhausts the session → GenerationError.
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(call_id="call_bad", result="not_an_int")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError):
            await agent_instance.get_number()

        # Find the ToolCallEvent for "call_bad"
        events = agent_instance.event_manager.values()
        tool_call_events = [
            e for e in events if e.event_type == "ToolCallEvent" and e.tool_call_id == "call_bad"
        ]
        assert len(tool_call_events) == 1, f"Expected 1 ToolCallEvent, got {len(tool_call_events)}"

        tc_event = tool_call_events[0]
        assert tc_event.result is not None, (
            "BUG REGRESSION: ToolCallEvent.result should not be None when "
            "GenerationError is raised — this would corrupt the next session's context."
        )
        assert tc_event.result.result_status == ResultStatus.ERROR
        assert (
            "session exhausted" in tc_event.result.content.lower()
            or "validation failed" in tc_event.result.content.lower()
        )


# ---------------------------------------------------------------------------
# 2. Inline return_result path (from within execute_python)
# ---------------------------------------------------------------------------


class TestInlineReturnResultPath:
    """When return_result() is called inline within execute_python code and
    validation fails with session exhausted, ToolCallEvent must have result != None."""

    @pytest.mark.asyncio
    async def test_inline_return_result_generation_error_sets_result(self):
        """Inline return_result(bad_value) + exhausted → ToolCallEvent has error result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=1)))
            async def get_number(self) -> int:
                """Return an integer."""
                ...

        # LLM calls return_result("not_an_int") from within execute_python code.
        # max_retries=1, so the first validation error exhausts session.
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call(
                            'return_result("not_an_int")',
                            call_id="call_inline",
                        )
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError):
            await agent_instance.get_number()

        events = agent_instance.event_manager.values()
        tool_call_events = [
            e for e in events if e.event_type == "ToolCallEvent" and e.tool_call_id == "call_inline"
        ]
        assert len(tool_call_events) == 1

        tc_event = tool_call_events[0]
        assert tc_event.result is not None, (
            "BUG REGRESSION: inline return_result path must set ToolCallEvent.result "
            "on GenerationError to prevent context corruption."
        )
        assert tc_event.result.result_status == ResultStatus.ERROR


# ---------------------------------------------------------------------------
# 3. Formatter safety net — ToolCallEvent.result=None still produces tool_result
# ---------------------------------------------------------------------------


class TestFormatterSafetyNet:
    """_event_block_to_messages must emit a tool_result even when
    ToolCallEvent.result is None (the belt-and-suspenders defensive check)."""

    def test_none_result_produces_placeholder_tool_result(self):
        """ToolCallEvent with result=None → placeholder RenderedMessage(role=TOOL)."""
        block = _tool_call_block(
            tool_call_id="tc_orphan",
            name="execute_python",
            arguments={"code": "1 + 1"},
            result=None,  # Simulates the bug
        )

        messages = _event_block_to_messages(block, wrap_content=None)

        # Should produce 2 messages: assistant tool_call + tool result
        assert len(messages) == 2, (
            f"Expected 2 messages (tool_call + placeholder tool_result), got {len(messages)}"
        )

        tool_call_msg = messages[0]
        assert tool_call_msg.role == Role.ASSISTANT
        assert tool_call_msg.tool_call is not None
        assert tool_call_msg.tool_call.id == "tc_orphan"

        result_msg = messages[1]
        assert result_msg.role == Role.TOOL
        assert result_msg.tool_call_id == "tc_orphan"
        assert result_msg.content is not None
        assert "(no result recorded)" in result_msg.content

    def test_normal_result_still_works(self):
        """Sanity check: ToolCallEvent with a proper result still works normally."""
        block = _tool_call_block(
            tool_call_id="tc_ok",
            name="execute_python",
            arguments={"code": "1 + 1"},
            result=ToolResult(
                tool_call_id="tc_ok",
                content="status: complete",
                result_status=ResultStatus.COMPLETE,
            ),
        )

        messages = _event_block_to_messages(block, wrap_content=None)

        assert len(messages) == 2
        assert messages[1].role == Role.TOOL
        assert messages[1].content == "status: complete"
        assert messages[1].tool_call_id == "tc_ok"


# ---------------------------------------------------------------------------
# 4. End-to-end: all tool_use messages have matching tool_result
# ---------------------------------------------------------------------------


class TestEndToEndRenderedMessageIntegrity:
    """Start a full CodeAct generation that exhausts retries and verify
    the RENDERED messages have matching tool_use/tool_result pairs."""

    @pytest.mark.asyncio
    async def test_exhausted_retries_all_tool_calls_have_results(self):
        """After exhausting retries via return_result, every ToolCallEvent has result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=1)))
            async def get_number(self) -> int:
                """Return an integer."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # LLM tries return_result with wrong type
                _resp("", tool_calls=[_return_result(call_id="call_r1", result="bad")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError):
            await agent_instance.get_number()

        # Verify every ToolCallEvent has a non-None result
        events = agent_instance.event_manager.values()
        for ev in events:
            if ev.event_type == "ToolCallEvent":
                assert ev.result is not None, (
                    f"ToolCallEvent {ev.tool_call_id!r} has result=None — "
                    "this would corrupt the conversation on the next session."
                )

    @pytest.mark.asyncio
    async def test_rendered_messages_have_matched_pairs(self):
        """Rendered messages from exhausted session must pair all tool_use with tool_result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=1)))
            async def get_number(self) -> int:
                """Return an integer."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(call_id="call_r2", result="bad")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError):
            await agent_instance.get_number()

        # Re-render events through the formatter
        events = agent_instance.event_manager.values()
        tool_use_ids = set()
        tool_result_ids = set()

        for ev in events:
            if ev.event_type == "ToolCallEvent":
                block = ResolvedBlock(
                    key=f"ev_{ev.tool_call_id}",
                    content="",
                    role=Role.ASSISTANT,
                    event=ev,
                )
                messages = _event_block_to_messages(block, wrap_content=None)
                for msg in messages:
                    if msg.tool_call is not None:
                        tool_use_ids.add(msg.tool_call.id)
                    if msg.role == Role.TOOL and msg.tool_call_id:
                        tool_result_ids.add(msg.tool_call_id)

        assert tool_use_ids == tool_result_ids, (
            f"Mismatched tool_use/tool_result pairs!\n"
            f"tool_use IDs: {tool_use_ids}\ntool_result IDs: {tool_result_ids}"
        )


# ---------------------------------------------------------------------------
# 5. Multiple tool calls — second is return_result that causes GenerationError
# ---------------------------------------------------------------------------


class TestMultipleToolCalls:
    """When the LLM returns multiple tool calls in one response and the
    second one is return_result that causes GenerationError, the first
    tool call's event must also be safe."""

    @pytest.mark.asyncio
    async def test_first_tool_call_result_safe_when_second_fails(self):
        """execute_python + return_result(bad) in one response — first TC has result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=1)))
            async def get_number(self) -> int:
                """Return an integer."""
                ...

        # LLM sends two tool calls: execute_python then return_result with bad value
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call("x = 42", call_id="call_exec"),
                        _return_result(call_id="call_bad_return", result="not_int"),
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError):
            await agent_instance.get_number()

        events = agent_instance.event_manager.values()
        tool_call_events = [e for e in events if e.event_type == "ToolCallEvent"]

        # The execute_python call should have completed successfully
        exec_events = [e for e in tool_call_events if e.tool_call_id == "call_exec"]
        assert len(exec_events) == 1
        assert exec_events[0].result is not None, (
            "execute_python ToolCallEvent should have result set even when "
            "subsequent return_result fails."
        )

        # The return_result call should also have result set (error)
        return_events = [e for e in tool_call_events if e.tool_call_id == "call_bad_return"]
        assert len(return_events) == 1
        assert return_events[0].result is not None, (
            "return_result ToolCallEvent should have error result set on GenerationError."
        )
        assert return_events[0].result.result_status == ResultStatus.ERROR


# ---------------------------------------------------------------------------
# 6. Non-GenerationError from _handle_return_result
# ---------------------------------------------------------------------------


class TestNonGenerationErrorSafety:
    """When validation fails but session is NOT exhausted, the error path
    should update ToolCallEvent result normally (not raise GenerationError)."""

    @pytest.mark.asyncio
    async def test_validation_error_with_retries_remaining_updates_result(self):
        """Validation error with retries left → ToolCallEvent gets error result, no raise."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=3)))
            async def get_number(self) -> int:
                """Return an integer."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # First: bad result (retries left → error result, continues)
                _resp("", tool_calls=[_return_result(call_id="call_bad1", result="not_int")]),
                # Second: valid result
                _resp("", tool_calls=[_return_result(call_id="call_good", result=42)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.get_number()
        assert result == 42

        events = agent_instance.event_manager.values()
        bad_events = [
            e for e in events if e.event_type == "ToolCallEvent" and e.tool_call_id == "call_bad1"
        ]
        assert len(bad_events) == 1
        assert bad_events[0].result is not None
        assert bad_events[0].result.result_status == ResultStatus.ERROR


# ---------------------------------------------------------------------------
# 7. Translated tool call path (unknown tool → execute_python)
# ---------------------------------------------------------------------------


class TestTranslatedToolCallPath:
    """When translate_tool_calls converts an unknown tool to execute_python,
    the ToolCallEvent should still be safe if execution fails."""

    @pytest.mark.asyncio
    async def test_unknown_tool_translated_has_result(self):
        """Unknown tool → translate to execute_python → ToolCallEvent has result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def add(self, a: int, b: int) -> int:
                return a + b

            @strategy(
                CodeActStrategy(config=CodeActConfig(max_retries=1, translate_tool_calls=True))
            )
            async def compute(self) -> int:
                """Compute something."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # LLM calls "add" directly as a tool (will be translated)
                LLMResponse(
                    raw_response=None,
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_translated",
                            name="add",
                            arguments=json.dumps({"a": 1, "b": 2}),
                        )
                    ],
                    finish_reason="tool_calls",
                    assistant_message={"role": "assistant", "content": ""},
                ),
                # Then return the result
                _resp("", tool_calls=[_return_result(call_id="call_ret", result=3)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()
        assert result == 3

        events = agent_instance.event_manager.values()
        translated_events = [
            e
            for e in events
            if e.event_type == "ToolCallEvent" and e.tool_call_id == "call_translated"
        ]
        assert len(translated_events) == 1
        assert translated_events[0].result is not None, (
            "Translated tool call should have result set."
        )

    @pytest.mark.asyncio
    async def test_untranslatable_unknown_tool_has_error_result(self):
        """Truly unknown, untranslatable tool → ToolCallEvent has error result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(
                CodeActStrategy(config=CodeActConfig(max_retries=2, translate_tool_calls=True))
            )
            async def compute(self) -> int:
                """Compute something."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_unknown",
                            name="totally_unknown_tool",
                            arguments=json.dumps({"x": 1}),
                        )
                    ],
                    finish_reason="tool_calls",
                    assistant_message={"role": "assistant", "content": ""},
                ),
                # Then return valid result
                _resp("", tool_calls=[_return_result(call_id="call_ok", result=42)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()
        assert result == 42

        events = agent_instance.event_manager.values()
        unknown_events = [
            e
            for e in events
            if e.event_type == "ToolCallEvent" and e.tool_call_id == "call_unknown"
        ]
        assert len(unknown_events) == 1
        assert unknown_events[0].result is not None
        assert unknown_events[0].result.result_status == ResultStatus.ERROR
        assert "unknown tool" in unknown_events[0].result.content.lower()


# ---------------------------------------------------------------------------
# 8. stop_to_return_result path — synthetic return_result fails → exhausted
# ---------------------------------------------------------------------------


class TestStopToReturnResultPath:
    """When a text-only stop response is converted to synthetic return_result
    and validation fails with session exhausted, ToolCallEvent must be safe."""

    @pytest.mark.asyncio
    async def test_stop_to_synthetic_return_result_exhausted(self):
        """stop + text content → synthetic return_result → validation fails → exhausted.

        For a method returning int, a text-only stop with "hello" should be
        routed through return_result("hello") → validation fails → exhausted.
        The synthetic ToolCallEvent must have result != None.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=1)))
            async def get_number(self) -> int:
                """Return an integer."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # LLM returns text with stop (will be converted to synthetic return_result)
                _resp("hello world"),  # finish_reason="stop"
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError):
            await agent_instance.get_number()

        events = agent_instance.event_manager.values()
        # Find the synthetic return_result ToolCallEvent
        tool_call_events = [e for e in events if e.event_type == "ToolCallEvent"]

        for tc_event in tool_call_events:
            assert tc_event.result is not None, (
                f"Synthetic return_result ToolCallEvent {tc_event.tool_call_id!r} has "
                f"result=None — would corrupt the next session."
            )

    @pytest.mark.asyncio
    async def test_stop_empty_content_none_return_type(self):
        """stop + no content for -> None method should succeed without corruption."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=1)))
            async def do_something(self) -> None:
                """Do something."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Empty stop → synthetic return_result(None) → should succeed for -> None
                LLMResponse(
                    raw_response=None,
                    content="",
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": ""},
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.do_something()
        assert result is None

        events = agent_instance.event_manager.values()
        tool_call_events = [e for e in events if e.event_type == "ToolCallEvent"]
        for tc_event in tool_call_events:
            assert tc_event.result is not None


# ---------------------------------------------------------------------------
# 9. Formatter full-pipeline test (XMLBlockFormatter)
# ---------------------------------------------------------------------------


class TestFormatterFullPipeline:
    """Verify that the full XMLBlockFormatter pipeline handles
    ToolCallEvent.result=None correctly."""

    def test_xml_formatter_handles_none_result(self):
        """XMLBlockFormatter.format() with result=None block → valid messages."""
        block = _tool_call_block(
            tool_call_id="tc_null",
            name="return_result",
            arguments={"result": "bad_value"},
            result=None,
        )

        formatter = XMLBlockFormatter()
        messages = formatter.format([block])

        # Should include both the assistant tool_call and the tool result
        tool_messages = [m for m in messages if m.role == Role.TOOL]
        assert len(tool_messages) >= 1, (
            "XMLBlockFormatter should emit a TOOL result even for result=None"
        )
        assert tool_messages[0].tool_call_id == "tc_null"

    def test_xml_formatter_mixed_none_and_normal_results(self):
        """Mix of result=None and normal results → all tool_use paired."""
        normal_block = _tool_call_block(
            key="tc1",
            tool_call_id="tc_normal",
            name="execute_python",
            arguments={"code": "x = 1"},
            result=ToolResult(
                tool_call_id="tc_normal",
                content="status: complete",
                result_status=ResultStatus.COMPLETE,
            ),
        )
        none_block = _tool_call_block(
            key="tc2",
            tool_call_id="tc_none",
            name="return_result",
            arguments={"result": "bad"},
            result=None,
        )

        formatter = XMLBlockFormatter()
        messages = formatter.format([normal_block, none_block])

        tool_use_ids = {m.tool_call.id for m in messages if m.tool_call is not None}
        tool_result_ids = {
            m.tool_call_id for m in messages if m.role == Role.TOOL and m.tool_call_id
        }

        assert tool_use_ids == tool_result_ids, (
            f"Mismatched pairs: tool_use={tool_use_ids}, tool_result={tool_result_ids}"
        )
