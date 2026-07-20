# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for cross-session event visibility.

Verifies that both ToolCallEvents (execute_python code) and PythonOutput events
from session 1 appear in session 2's LLM context when calling a codeact method
multiple times on the same agent instance.

Session structure (two calls to agent.respond()):
- Session 1 (call 1 + 2 on FakeLLM): LLM runs execute_python with marker code, returns result
- Session 2 (call 3 on FakeLLM): single call, return_result immediately

After session 2, `fake_llm.last_messages` holds exactly what the LLM received for that
one call — the full multi-session message history built by OpenAIProviderFormatter.

Background: investigated after trace 20260409_144053_2102fe64 showed an agent
rediscovering the same bug 49 times. The rediscovery was initially suspected to be
caused by ToolCallEvents not propagating; these tests confirm they DO propagate
with in-memory storage. With SQLite storage the tests FAIL because ToolCallEvent
is not registered in SQLiteEventBackend._CORE_TYPES — it deserializes as Metadata
(Role.METADATA), which the formatter silently skips.

The actual rediscovery cause is that `self.message()` uses record=False, so the
content the agent communicated to the user is absent from the event log (though
the code that called message() is still visible via the ToolCallEvent).
"""

import json
from typing import Any

import pytest

from nooa import Agent
from nooa.storage import SQLiteStorageManager
from nooa.storage.in_memory import InMemoryStorageManager
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

# A distinctive string we can grep for in the serialized session 2 message list.
_MARKER = "CROSS_SESSION_TOOLCALL_MARKER_d4e5f6"

_TEST_LLM = FakeLLMClient()  # module-level placeholder required for class definition


def _resp(content: str = "", tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


def _execute_python(code: str, call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))


def _return_result(result: Any, call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name="return_result", arguments=json.dumps({"result": result}))


@pytest.fixture(params=["memory", "sqlite"])
def storage(request):
    """Parametrized storage fixture.

    memory: in-memory backend — ToolCallEvents never serialized, tests pass.
    sqlite: SQLite backend — ToolCallEvents serialize/deserialize; tests FAIL
            until ToolCallEvent is registered in SQLiteEventBackend._CORE_TYPES.
    """
    if request.param == "memory":
        yield InMemoryStorageManager()
    else:
        sm = SQLiteStorageManager(":memory:")
        yield sm
        sm.close()


class TestCrossSessionEventVisibility:
    """Verify event propagation across codeact session boundaries.

    Each test is parametrized via the `storage` fixture (memory / sqlite).
    The sqlite variant exercises the SQLite serialization path and will FAIL
    until ToolCallEvent is registered in SQLiteEventBackend._CORE_TYPES.
    """

    @pytest.mark.asyncio
    async def test_execute_python_code_visible_in_next_session(self, storage):
        """Session 1's execute_python ToolCallEvent is visible in session 2's LLM context.

        The OpenAIProviderFormatter renders ToolCallEvents as proper assistant tool_use
        messages. All events from the shared event_manager are included in subsequent
        sessions — the LLM sees the full execution history across calls.

        With SQLite storage: FAILS because ToolCallEvent is not in _CORE_TYPES, so it
        deserializes as Metadata (Role.METADATA) and the formatter skips it silently.
        """

        class SimpleAgent(Agent, llm=_TEST_LLM):
            async def respond(self, user_message: str) -> str:
                """Respond to: {user_message}"""
                ...

        session1_code = f"result = 42  # {_MARKER}"

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # --- Session 1 ---
                _resp(tool_calls=[_execute_python(session1_code, call_id="s1c1")]),
                _resp(tool_calls=[_return_result("session1_done", call_id="s1c2")]),
                # --- Session 2: single call ---
                _resp(tool_calls=[_return_result("session2_done", call_id="s2c1")]),
            ]
        )

        agent = SimpleAgent(llm=fake_llm, storage=storage)

        result1 = await agent.respond("first message")
        assert result1 == "session1_done"
        assert fake_llm.call_count == 2

        # Session 2 makes exactly one LLM call → last_messages = session 2's full context
        result2 = await agent.respond("second message")
        assert result2 == "session2_done"
        assert fake_llm.call_count == 3

        serialized = json.dumps(fake_llm.last_messages)

        # Session 1's execute_python code must appear as an assistant tool_use message
        assert _MARKER in serialized, (
            "Session 1's execute_python code is NOT visible in session 2's LLM context.\n"
            f"Expected '{_MARKER}' in a tool_use message, but it's absent."
        )

        # Verify it appears specifically as a tool_use (assistant role), not just in user content
        assistant_tool_uses = [
            m
            for m in fake_llm.last_messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        execute_python_calls = [
            tc
            for m in assistant_tool_uses
            for tc in m["tool_calls"]
            if tc.get("function", {}).get("name") == "execute_python"
            and _MARKER in tc.get("function", {}).get("arguments", "")
        ]
        assert execute_python_calls, (
            "Session 1's execute_python ToolCallEvent did not appear as an assistant "
            "tool_use message in session 2's context."
        )

    @pytest.mark.asyncio
    async def test_python_output_visible_in_next_session(self, storage):
        """Session 1's PythonOutput (stdout) is visible in session 2's LLM context.

        PythonOutput events (record=True, Role.USER) are stored in the event log and
        rendered as context for subsequent sessions.

        This should pass for both memory and sqlite storage since PythonOutput IS
        registered in _CORE_TYPES.
        """

        class SimpleAgent(Agent, llm=_TEST_LLM):
            async def respond(self, user_message: str) -> str:
                """Respond to: {user_message}"""
                ...

        stdout_marker = f"STDOUT_{_MARKER}"
        session1_code = f"print('{stdout_marker}')"

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(tool_calls=[_execute_python(session1_code, call_id="s1c1")]),
                _resp(tool_calls=[_return_result("s1_done", call_id="s1c2")]),
                _resp(tool_calls=[_return_result("s2_done", call_id="s2c1")]),
            ]
        )

        agent = SimpleAgent(llm=fake_llm, storage=storage)

        result1 = await agent.respond("first")
        assert result1 == "s1_done"
        assert fake_llm.call_count == 2

        result2 = await agent.respond("second")
        assert result2 == "s2_done"
        assert fake_llm.call_count == 3

        serialized = json.dumps(fake_llm.last_messages)

        assert stdout_marker in serialized, (
            "Session 1's stdout output is NOT visible in session 2's LLM context.\n"
            f"Expected '{stdout_marker}' to appear in a PythonOutput event."
        )
