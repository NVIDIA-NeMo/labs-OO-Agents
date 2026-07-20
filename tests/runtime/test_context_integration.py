# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for context-blocks refactor.

Tests the dict-like ContextApi API and integration with Agent.
"""

import pytest

from nooa.agent import Agent
from nooa.context_blocks import DynamicContext
from nooa.unifiedllm import FakeLLMClient


class TestContextManager:
    """Test dict-like ContextManager API on Agent."""

    def test_agent_has_context(self):
        """Agent instances have a _context_manager attribute (ContextManager)."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        assert hasattr(agent, "context_manager")

    def test_set_static_method(self):
        """``set_static()`` places blocks in the cacheable partition."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()

        agent.context_manager.set_static("key", "value")
        assert agent.context_manager["key"] == "value"
        assert agent.context_manager.is_static("key") is True

        agent.context_manager.set_dynamic("other", "value")
        assert agent.context_manager.is_static("other") is False

    def test_set_static_value(self):
        """self.context['key'] = 'value' stores a static string."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        agent.context_manager["notes"] = "My notes"
        assert agent.context_manager["notes"] == "My notes"

    def test_set_dynamic_value(self):
        """self.context.set_dynamic() stores a dynamic block."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        agent.context_manager.set_dynamic("status", "self.format_status()")
        assert "status" in agent.context_manager

    def test_setitem_accepts_context_type(self):
        """self.context['key'] = Context(...) sets block with correct placement."""
        from nooa.context_blocks import Context

        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        # Literal, suffix (default)
        agent.context_manager["a"] = Context("hello")
        assert agent.context_manager["a"] == "hello"
        assert agent.context_manager.is_static("a") is False

        # Literal, prefix
        agent.context_manager["b"] = Context("world", prefix=True)
        assert agent.context_manager["b"] == "world"
        assert agent.context_manager.is_static("b") is True

        # Expression, suffix
        agent.context_manager["c"] = Context(expr="'dynamic'")
        assert "c" in agent.context_manager

        # Expression, prefix
        agent.context_manager["d"] = Context(expr="'cached'", prefix=True)
        assert "d" in agent.context_manager
        assert agent.context_manager.is_static("d") is True

    def test_setitem_none_suppresses_block(self):
        """self.context['key'] = None suppresses the block."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        agent.context_manager["key"] = "hello"
        agent.context_manager["key"] = None
        assert agent.context_manager.is_disabled("key") is True

    def test_context_api_can_disable_and_enable_protected_blocks(self):
        """The public API can suppress default framework blocks by key."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        agent.context.disable("system_prompt", "self", "state")
        assert agent.context.disabled() == {"system_prompt", "self", "state"}
        assert agent.context.is_enabled("self") is False
        assert "self" in agent.context_manager.protected_keys

        agent.context.enable("self")
        assert agent.context.disabled() == {"system_prompt", "state"}
        assert agent.context.is_enabled("self") is True

    def test_init_context_none_disables_protected_blocks(self):
        """context={key: None} suppresses protected framework blocks instead of erroring."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent(context={"self": None})
        assert agent.context_manager.is_disabled("self") is True
        assert "self" in agent.context_manager.protected_keys

    def test_disabled_context_blocks_round_trip_through_snapshot(self):
        """Disabled block keys survive save/restore snapshot serialization."""
        from nooa.storage.snapshot import AgentSnapshot

        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        agent.context.disable("self", "strategy_prompt")
        snapshot = AgentSnapshot.from_agent(agent)

        restored = TestAgent()
        snapshot.restore(restored)
        assert restored.context.disabled() == {"self", "strategy_prompt"}

    def test_set_dynamic_rejects_invalid_expr(self):
        """set_dynamic() rejects invalid Python expressions."""
        from nooa.context_blocks.exceptions import BlockSyntaxError

        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        with pytest.raises(BlockSyntaxError):
            agent.context_manager.set_dynamic("bad", "this is not valid python!!!")

    def test_setitem_accepts_dynamic_context(self):
        """self.context['key'] = DynamicContext(...) stores as dynamic block."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        agent.context_manager["key"] = DynamicContext("'value'")
        assert "key" in agent.context_manager

    def test_getitem_returns_original_value(self):
        """self.context['key'] returns the original value."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        agent.context_manager["notes"] = "My notes"
        result = agent.context_manager["notes"]
        assert result == "My notes"

    def test_set_non_string_value_returns_original(self):
        """Non-string values are recoverable via __getitem__."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        original = {"timeout": 30, "retries": 3}
        agent.context_manager["config"] = original
        recovered = agent.context_manager["config"]
        assert recovered is original
        assert recovered["timeout"] == 30
        assert recovered["retries"] == 3

    def test_delete_block(self):
        """del self.context['key'] removes the block."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        agent.context_manager["temp"] = "temporary"
        assert "temp" in agent.context_manager
        del agent.context_manager["temp"]
        assert "temp" not in agent.context_manager

    def test_contains(self):
        """'key' in self.context checks membership."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        assert "nonexistent" not in agent.context_manager
        agent.context_manager["test"] = "value"
        assert "test" in agent.context_manager

    def test_iteration(self):
        """Can iterate over context keys."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        agent.context_manager["a"] = "alpha"
        agent.context_manager["b"] = "beta"
        keys = list(agent.context_manager.keys())
        assert "a" in keys
        assert "b" in keys

    def test_key_not_found_raises(self):
        """Accessing missing key raises KeyError."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        with pytest.raises(KeyError):
            _ = agent.context_manager["missing"]

    def test_dynamic_unresolved_raises_error(self):
        """Reading a dynamic block before resolution raises DynamicNotResolvedError."""
        from nooa.context_blocks.exceptions import DynamicNotResolvedError

        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        agent.context_manager.set_dynamic("status", "self.format_status()")
        with pytest.raises(DynamicNotResolvedError, match="status"):
            _ = agent.context_manager["status"]

    def test_delete_nonexistent_key_raises_keyerror(self):
        """del context['nonexistent'] raises KeyError."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        with pytest.raises(KeyError):
            del agent.context_manager["nonexistent"]

    def test_delete_nonexistent_protected_key_raises_keyerror(self):
        """del context['nonexistent'] raises KeyError when the key is not in _blocks."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        # A key that doesn't exist at all should raise KeyError
        assert "nonexistent" not in agent.context_manager._blocks
        with pytest.raises(KeyError):
            del agent.context_manager["nonexistent"]

    def test_delete_existing_protected_key_raises_protected_error(self):
        """del context['system_prompt'] raises ProtectedBlockError when key is protected."""
        from nooa.context_blocks.exceptions import ProtectedBlockError

        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        # system_prompt is now in both protected_keys and _blocks
        assert "system_prompt" in agent.context_manager.protected_keys
        assert "system_prompt" in agent.context_manager._blocks
        with pytest.raises(ProtectedBlockError):
            del agent.context_manager["system_prompt"]

    def test_pop_protected_key_raises_protected_error(self):
        """pop() on an existing protected key raises ProtectedBlockError."""
        from nooa.context_blocks.exceptions import ProtectedBlockError

        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        # Directly inject a block into _blocks AND mark as protected
        agent.context_manager._blocks["guarded"] = "value"
        agent.context_manager.protected_keys.add("guarded")
        with pytest.raises(ProtectedBlockError):
            agent.context_manager.pop("guarded")

    def test_pop_nonexistent_protected_key_raises_keyerror(self):
        """pop() on a key not in _blocks raises KeyError."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        assert "nonexistent" not in agent.context_manager._blocks
        with pytest.raises(KeyError):
            agent.context_manager.pop("nonexistent")

    def test_pop_existing_protected_key_raises_protected_error(self):
        """pop() on a protected key that is in _blocks raises ProtectedBlockError."""
        from nooa.context_blocks.exceptions import ProtectedBlockError

        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        # system_prompt is now in both protected_keys and _blocks
        assert "system_prompt" in agent.context_manager.protected_keys
        assert "system_prompt" in agent.context_manager._blocks
        with pytest.raises(ProtectedBlockError):
            agent.context_manager.pop("system_prompt")


class TestBuildMessages:
    """Test _build_messages integration."""

    def test_build_messages_callable(self):
        """_build_messages should be callable on runtime."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        assert hasattr(agent.runtime, "_build_messages")
        assert callable(agent.runtime._build_messages)

    @pytest.mark.asyncio
    async def test_build_messages_produces_system_message(self):
        """_build_messages should produce at least a system message."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            async def test_method(self):
                """Test task"""
                ...

        agent = TestAgent()
        method = type(agent).test_method
        messages = await agent.runtime._build_messages(method)

        assert len(messages) >= 1
        assert messages[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_build_messages_resolves_dynamic_blocks(self):
        """After _build_messages, dynamic blocks are readable via self.context."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            async def test_method(self):
                """Test task"""
                ...

            def get_status(self) -> str:
                return "Ready"

        agent = TestAgent()
        agent.context_manager.set_dynamic("status", "self.get_status()")

        # Before build: dynamic block not yet resolved — raises error
        from nooa.context_blocks.exceptions import DynamicNotResolvedError

        with pytest.raises(DynamicNotResolvedError):
            _ = agent.context_manager["status"]

        # Build messages resolves dynamic blocks
        method = type(agent).test_method
        await agent.runtime._build_messages(method)

        # After build: dynamic block is resolved
        assert agent.context_manager["status"] == "Ready"


class TestContextManagerGetPop:
    """Tests for get() and pop() methods."""

    def test_get_existing_key(self):
        """get() returns value for existing key."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        agent.context_manager["notes"] = "My notes"
        assert agent.context_manager.get("notes") == "My notes"

    def test_get_missing_key_returns_default(self):
        """get() returns default for missing key."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        assert agent.context_manager.get("missing") is None
        assert agent.context_manager.get("missing", "fallback") == "fallback"

    def test_pop_existing_key(self):
        """pop() removes and returns value."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        agent.context_manager["temp"] = "temporary"
        value = agent.context_manager.pop("temp")
        assert value == "temporary"
        assert "temp" not in agent.context_manager

    def test_pop_missing_key_with_default(self):
        """pop() returns default for missing key."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        assert agent.context_manager.pop("missing", None) is None

    def test_pop_missing_key_without_default_raises(self):
        """pop() raises KeyError for missing key without default."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent()
        with pytest.raises(KeyError):
            agent.context_manager.pop("missing")


class TestAgentContextParam:
    """Tests for Agent(context=...) and class-level context=... parameters."""

    def test_class_level_context_applied(self):
        """Class-level context blocks are applied at instantiation."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm, context={"focus": "security analysis"}):
            pass

        agent = TestAgent()
        assert agent.context_manager["focus"] == "security analysis"

    def test_instance_level_context_applied(self):
        """Instance-level context blocks are applied at instantiation."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            pass

        agent = TestAgent(context={"mode": "strict"})
        assert agent.context_manager["mode"] == "strict"

    def test_instance_context_overrides_class_context(self):
        """Instance context overrides class context for same key."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm, context={"mode": "default"}):
            pass

        agent = TestAgent(context={"mode": "override"})
        assert agent.context_manager["mode"] == "override"

    def test_class_and_instance_context_merge(self):
        """Class and instance context blocks merge (not replace)."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm, context={"a": "from_class"}):
            pass

        agent = TestAgent(context={"b": "from_instance"})
        assert agent.context_manager["a"] == "from_class"
        assert agent.context_manager["b"] == "from_instance"

    def test_none_removes_block(self):
        """Setting context value to None removes the block."""
        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm, context={"a": "keep", "b": "remove"}):
            pass

        agent = TestAgent(context={"b": None})
        assert agent.context_manager["a"] == "keep"
        assert "b" not in agent.context_manager

    def test_dynamic_in_context(self):
        """DynamicContext values in context are stored for later resolution."""
        fake_llm = FakeLLMClient()

        class TestAgent(
            Agent, llm=fake_llm, context={"status": DynamicContext("self.get_status()")}
        ):
            pass

        agent = TestAgent()
        assert "status" in agent.context_manager

    def test_dynamic_importable_from_nooa(self):
        """DynamicContext is importable from nooa top-level."""
        from nooa import DynamicContext as D
        from nooa.context_blocks import DynamicContext as OrigDynamicContext

        assert D is OrigDynamicContext


class TestScopedContextCurrentCallFiltering:
    """End-to-end test: ScopedContext(events=EventQuery.current_call()) must not
    filter out the Task prompt.

    Regression test for the bug where CodeActStrategy mutates call.id to a tag
    number after creation, making it differ from the UUID that event_manager
    injects into metadata["call_id"]. _prepare_context must use the agent call
    stack ID (the UUID) so the Task event passes the filter and the LLM
    receives the task prompt.
    """

    @pytest.mark.asyncio
    async def test_llm_receives_task_prompt_with_scoped_current_call(self):
        """LLM must receive the task prompt even with EventQuery.current_call().

        Uses a real @strategy decorator with ScopedContext and CodeActStrategy,
        then inspects the messages the FakeLLM actually received.
        """
        import json

        from nooa import EventQuery, strategy
        from nooa.context_blocks import ScopedContext
        from nooa.strategies.codeact import CodeActStrategy
        from nooa.unifiedllm import LLMResponse, ToolCall

        # Script a single LLM response: execute_python with inline return_result
        fake_llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="execute_python",
                            arguments=json.dumps(
                                {"code": 'return_result(result="sentiment: positive")'}
                            ),
                        )
                    ],
                    finish_reason="tool_calls",
                    assistant_message={"role": "assistant", "content": ""},
                )
            ]
        )

        class AnalysisAgent(Agent, llm=fake_llm):
            @strategy(
                CodeActStrategy(),
                ScopedContext(events=EventQuery.current_call()),
            )
            async def analyze_feedback(self, text: str) -> str:
                """Analyze customer feedback: {text}"""
                ...

        agent = AnalysisAgent()
        result = await agent.analyze_feedback("Great product, but shipping was slow")

        assert result == "sentiment: positive"

        # Verify the LLM received the task prompt in its messages.
        # last_messages captures what was sent on the final generate() call.
        user_messages = [m for m in fake_llm.last_messages if m.get("role") == "user"]
        assert len(user_messages) >= 1, (
            "LLM must receive at least one user message containing the task prompt. "
            "If this fails, EventQuery.current_call() is filtering out the Task event."
        )
        all_user_content = " ".join(str(m.get("content", "")) for m in user_messages)
        assert "Analyze customer feedback" in all_user_content, (
            f"Task prompt not found in user messages sent to LLM. "
            f"User content was: {all_user_content!r}"
        )


class TestDecoratorEventsIntegration:
    """Integration tests for @strategy decorator with ScopedContext events."""

    @pytest.mark.asyncio
    async def test_decorator_context_parameter_basic(self):
        """@strategy(context=ScopedContext(...)) basic usage works."""
        from nooa import strategy
        from nooa.context_blocks import ScopedContext

        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            async def regular_method(self) -> str:
                return "regular"

            @strategy(context=ScopedContext(context={"focus": "test"}))
            async def method_with_context(self) -> str:
                return "decorated"

        agent = TestAgent()
        result1 = await agent.regular_method()
        result2 = await agent.method_with_context()

        assert result1 == "regular"
        assert result2 == "decorated"

    @pytest.mark.asyncio
    async def test_decorator_context_accepts_plain_dict(self):
        """@strategy(context=...) accepts a plain dict as context overrides."""
        from nooa import strategy

        class TestAgent(Agent, llm=FakeLLMClient()):
            @strategy(context={"focus": "testing"})
            async def good_method(self) -> str: ...

        assert TestAgent.good_method._strategy_context == {"focus": "testing"}

    @pytest.mark.asyncio
    async def test_decorator_context_with_events(self):
        """@strategy(context=ScopedContext(events={...})) sets decorator events."""
        from nooa import strategy
        from nooa.context_blocks import ScopedContext

        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            @strategy(context=ScopedContext(events={"reminder": "Be thorough"}))
            async def decorated_method(self) -> str:
                return "result"

        agent = TestAgent()
        # Verify decorator was applied (method has _strategy_events attribute)
        method = agent.__class__.decorated_method
        assert hasattr(method, "_strategy_events")
        assert method._strategy_events == {"reminder": "Be thorough"}

    @pytest.mark.asyncio
    async def test_decorator_context_with_both_context_and_events(self):
        """@strategy(context=ScopedContext(context={...}, events={...})) works."""
        from nooa import strategy
        from nooa.context_blocks import ScopedContext

        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            @strategy(
                context=ScopedContext(
                    context={"focus": "security"}, events={"reminder": "Check everything"}
                )
            )
            async def comprehensive_method(self) -> str:
                return "result"

        agent = TestAgent()
        # Verify both context and events were set
        method = agent.__class__.comprehensive_method
        assert hasattr(method, "_strategy_context")
        assert hasattr(method, "_strategy_events")
        assert method._strategy_context == {"focus": "security"}
        assert method._strategy_events == {"reminder": "Check everything"}

    @pytest.mark.asyncio
    async def test_decorator_context_none_is_valid(self):
        """@strategy(context=None) is valid - no context override."""
        from nooa import strategy

        fake_llm = FakeLLMClient()

        class TestAgent(Agent, llm=fake_llm):
            @strategy(context=None)
            async def method_with_none_context(self) -> str:
                return "result"

        agent = TestAgent()
        result = await agent.method_with_none_context()
        assert result == "result"
