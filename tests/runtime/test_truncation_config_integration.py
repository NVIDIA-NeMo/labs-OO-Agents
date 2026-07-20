# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for truncation configuration integration with Agent class."""

import pytest

from nooa import Agent
from nooa.config.truncation_config import (
    CaptureConfig,
    FormatConfig,
    MediaCaptureConfig,
    TruncationConfig,
)
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM
_TEST_LLM = FakeLLMClient()


class TestTruncationConfigResolution:
    """Tests for config resolution at class/instance levels."""

    def test_default_config(self):
        """Agents should have default truncation config."""

        class TestAgent(Agent, llm=_TEST_LLM):
            pass

        agent = TestAgent()

        # Should have default config
        assert agent._truncation is not None
        assert agent._truncation.capture.max_stdout == 50_000
        assert agent._truncation.capture.max_stderr == 2_000
        assert agent._truncation.media_capture.max_attachments_per_execution == 5
        assert agent._truncation.event_format.max_length == 200

    def test_class_level_config(self):
        """Class-level config should override defaults."""

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(
                capture=CaptureConfig(max_stdout=100000),
                media_capture=MediaCaptureConfig(max_attachments_per_execution=20),
                event_format=FormatConfig(max_length=100),
            ),
        ):
            pass

        agent = TestAgent()

        # Should have class-level config
        assert agent._truncation.capture.max_stdout == 100_000
        assert agent._truncation.media_capture.max_attachments_per_execution == 20
        assert agent._truncation.event_format.max_length == 100
        # Other defaults preserved
        assert agent._truncation.capture.max_stderr == 2_000

    def test_instance_level_config(self):
        """Instance-level config should override class config."""

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(capture=CaptureConfig(max_stdout=100000)),
        ):
            pass

        agent = TestAgent(
            truncation=TruncationConfig(
                capture=CaptureConfig(max_stdout=200000), event_format=FormatConfig(max_depth=10)
            )
        )

        # Instance config should win
        assert agent._truncation.capture.max_stdout == 200_000
        assert agent._truncation.event_format.max_depth == 10
        # Other class defaults preserved
        assert agent._truncation.event_format.max_length == 200

    def test_config_merge_behavior(self):
        """Configs should merge properly (later overrides earlier)."""

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(
                capture=CaptureConfig(max_stdout=100000), event_format=FormatConfig(max_length=100)
            ),
        ):
            pass

        agent = TestAgent(
            truncation=TruncationConfig(event_format=FormatConfig(max_length=200, max_depth=5))
        )

        # Merged result
        assert agent._truncation.capture.max_stdout == 100_000  # From class
        assert agent._truncation.event_format.max_length == 200  # From instance
        assert agent._truncation.event_format.max_depth == 5  # From instance
        assert agent._truncation.capture.max_stderr == 2_000  # From default

    def test_multiple_agents_independent_configs(self):
        """Multiple agents should have independent configs."""

        class Agent1(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(capture=CaptureConfig(max_stdout=50000)),
        ):
            pass

        class Agent2(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(capture=CaptureConfig(max_stdout=100000)),
        ):
            pass

        agent1 = Agent1()
        agent2 = Agent2()

        # Should be independent
        assert agent1._truncation.capture.max_stdout == 50_000
        assert agent2._truncation.capture.max_stdout == 100_000


class TestTruncationConfigUsage:
    """Tests for config usage in execution."""

    @pytest.mark.asyncio
    async def test_custom_stdout_limit_applied(self):
        """Custom stdout limit should be applied."""

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(capture=CaptureConfig(max_stdout=1000)),  # Very small limit
        ):
            pass

        agent = TestAgent()

        # Generate output larger than limit
        code = """
for i in range(100):
    print("x" * 50)
"""
        result = await agent.runtime.execute_code(code)

        assert result.success
        # Should be truncated at 1000 chars — prose format shows head+tail split
        assert "Output too large" in result.stdout
        assert "500 and last 500 chars" in result.stdout

    @pytest.mark.asyncio
    async def test_custom_stderr_limit_applied(self):
        """Custom stderr limit should be applied."""

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(capture=CaptureConfig(max_stderr=500)),  # Very small limit
        ):
            pass

        agent = TestAgent()

        # This test is simplified due to sandbox restrictions
        # The limit is verified by the unit tests
        assert agent._truncation.capture.max_stderr == 500

    @pytest.mark.asyncio
    async def test_different_agents_different_limits(self):
        """Different agents should use their own limits."""

        class SmallLimitAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(capture=CaptureConfig(max_stdout=500)),
        ):
            pass

        class LargeLimitAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(capture=CaptureConfig(max_stdout=10000)),
        ):
            pass

        small_agent = SmallLimitAgent()
        large_agent = LargeLimitAgent()

        # Same code, different truncation
        code = 'print("x" * 2000)'

        result_small = await small_agent.runtime.execute_code(code)
        result_large = await large_agent.runtime.execute_code(code)

        # Small agent should truncate — prose format
        assert "Output too large" in result_small.stdout
        # Large agent should not truncate (2000 < 10000)
        assert "Output too large" not in result_large.stdout


class TestConfigMergeEdgeCases:
    """Tests for edge cases in config merging."""

    def test_none_config_uses_defaults(self):
        """Passing None should use defaults."""

        class TestAgent(Agent, llm=_TEST_LLM):
            pass

        agent = TestAgent(truncation=None)

        # Should have defaults
        assert agent._truncation.capture.max_stdout == 50_000

    def test_partial_override(self):
        """Partial configs should only override specified fields."""

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(
                capture=CaptureConfig(max_stdout=100000, max_stderr=30000),
                event_format=FormatConfig(max_length=100),
            ),
        ):
            pass

        agent = TestAgent(
            truncation=TruncationConfig(
                event_format=FormatConfig(max_length=200)
            )  # Only override this
        )

        # Only value.max_length should be overridden
        assert agent._truncation.capture.max_stdout == 100_000  # Preserved from class
        assert agent._truncation.capture.max_stderr == 30_000  # Preserved from class
        assert agent._truncation.event_format.max_length == 200  # Overridden


class TestTokenBudgetIntegration:
    """Tests for token budget fields (max_context_tokens, max_event_tokens)."""

    @pytest.mark.asyncio
    async def test_max_context_tokens_does_not_crash_on_generation(self):
        """Agent with max_context_tokens set should not raise ValueError during generation.

        Regression test: render_context() raises ValueError if context_limit is non-None
        but count_tokens is not provided. The actor must pass count_tokens to render_context().
        """
        from nooa.unifiedllm import FakeLLMClient

        # Return a return_result tool call so CodeActStrategy completes successfully
        llm = FakeLLMClient.with_tool_call("return_result", {"result": "done"})

        class TestAgent(
            Agent,
            llm=llm,
            truncation=TruncationConfig(max_context_tokens=100_000),
        ):
            async def answer(self, question: str) -> str: ...

        agent = TestAgent()
        # Must not raise ValueError: "max_context_tokens / max_event_tokens require a token counter"
        result = await agent.answer("hello")
        assert result == "done"

    @pytest.mark.asyncio
    async def test_max_event_tokens_does_not_crash_on_generation(self):
        """Agent with max_event_tokens set should not raise ValueError during generation."""
        from nooa.unifiedllm import FakeLLMClient

        llm = FakeLLMClient.with_tool_call("return_result", {"result": "summary"})

        class TestAgent(
            Agent,
            llm=llm,
            truncation=TruncationConfig(max_event_tokens=50_000),
        ):
            async def summarize(self) -> str: ...

        agent = TestAgent()
        result = await agent.summarize()
        assert result == "summary"

    @pytest.mark.asyncio
    async def test_token_limits_with_llm_that_cannot_count_still_works(self):
        """Agent with token limits and an LLM lacking count_tokens() must NOT
        raise. The runtime sizes eviction with a provider-calibrated
        chars→tokens ratio rather than the LLM tokenizer, so no count_tokens
        method is required (the old RuntimeError contract was removed).
        """
        import json

        from nooa.unifiedllm import LLMResponse, ToolCall

        class NoCountLLM:
            """Minimal fake LLM without count_tokens — returns a return_result."""

            model = "no-count-model"

            async def acall(self, messages, tools=None, output_model=None, **kwargs):
                return LLMResponse(
                    raw_response=None,
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="return_result",
                            arguments=json.dumps({"result": "done"}),
                        )
                    ],
                    finish_reason="tool_calls",
                    assistant_message={"role": "assistant", "content": ""},
                    reasoning=None,
                    usage={"prompt_tokens": 11, "completion_tokens": 3},
                )

        class TestAgent(
            Agent,
            llm=NoCountLLM(),
            truncation=TruncationConfig(max_context_tokens=100_000),
        ):
            async def answer(self, question: str) -> str: ...

        agent = TestAgent()
        result = await agent.answer("hello")
        assert result == "done"


class TestMethodLevelTruncationConfig:
    """Tests for method-level TruncationConfig via @strategy(truncation=...)."""

    def test_strategy_decorator_stores_truncation_attribute(self):
        """@strategy(truncation=...) should store the config as _strategy_truncation."""
        from nooa import strategy

        tc = TruncationConfig(capture=CaptureConfig(max_stdout=1234))

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(truncation=tc)
            async def method(self) -> str:
                """Do something."""
                ...

        # The decorator should store the truncation config on the underlying function
        fn = TestAgent.__dict__["method"]
        # The wrapper exposes it on itself; the underlying func also has it
        # (via setattr on func directly)
        assert getattr(fn, "_strategy_truncation", None) is tc or (
            hasattr(fn, "__func__") and getattr(fn.__func__, "_strategy_truncation", None) is tc
        )

    def test_strategy_decorator_without_truncation_stores_none(self):
        """@strategy() without truncation= should set _strategy_truncation to None."""
        from nooa import strategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy()
            async def method(self) -> str:
                """Do something."""
                ...

        fn = TestAgent.__dict__["method"]
        assert getattr(fn, "_strategy_truncation", None) is None

    @pytest.mark.asyncio
    async def test_method_level_truncation_visible_to_runtime_during_execution(self):
        """During method execution, runtime.truncation_config should reflect method-level override.

        We use a custom strategy that captures the truncation config visible through
        runtime during strategy execution, then verify it is the merged (method-level) config.
        """
        from nooa import strategy
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.current_call import CurrentCall

        captured_tc = {}

        class CapturingStrategy(GenerationStrategy):
            """Strategy that captures runtime.truncation_config and returns a fixed value."""

            name = "CAPTURING"
            traceable = False
            requires_lock = False

            def get_block_overrides(self):
                return {}

            async def execute(self, runtime, call: CurrentCall):
                captured_tc["config"] = runtime.truncation_config
                return "captured"

        method_tc = TruncationConfig(capture=CaptureConfig(max_stdout=12345))

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(
                capture=CaptureConfig(max_stdout=100000), event_format=FormatConfig(max_length=77)
            ),
        ):
            @strategy(CapturingStrategy(), truncation=method_tc)
            async def run(self) -> str:
                """Run."""
                ...

        agent = TestAgent()
        result = await agent.run()
        assert result == "captured"

        # The strategy should have seen the merged config
        assert "config" in captured_tc
        tc = captured_tc["config"]
        # Method-level field wins
        assert tc.capture.max_stdout == 12345
        # Agent-level field preserved (not in method override)
        assert tc.event_format.max_length == 77

    @pytest.mark.asyncio
    async def test_method_level_truncation_applied_to_execute_code(self):
        """execute_code during a method call should use the method-level stdout limit.

        The agent has a large stdout limit (100k), but the method is decorated with
        a small limit (200 chars). Code executed during that method should be truncated
        at 200 chars.
        """
        from nooa import strategy
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.current_call import CurrentCall

        execution_results = {}

        class CodeRunningStrategy(GenerationStrategy):
            """Strategy that runs some code and captures the stdout result."""

            name = "CODE_RUNNING"
            traceable = False
            requires_lock = False

            def get_block_overrides(self):
                return {}

            async def execute(self, runtime, call: CurrentCall):
                # Execute code that produces 5000 chars of output
                result = await runtime.execute_code("print('x' * 5000)")
                execution_results["stdout"] = result.stdout
                return "done"

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(
                capture=CaptureConfig(max_stdout=100000)
            ),  # Large agent-level limit
        ):
            @strategy(
                CodeRunningStrategy(),
                truncation=TruncationConfig(capture=CaptureConfig(max_stdout=200)),
            )
            async def run_code(self) -> str:
                """Run code."""
                ...

        agent = TestAgent()
        result = await agent.run_code()
        assert result == "done"

        # The stdout should be truncated at 200 chars (not 100_000)
        assert "stdout" in execution_results
        stdout = execution_results["stdout"]
        assert "Output too large" in stdout, (
            f"Expected truncation at 200 chars but got: {stdout[:300]!r}"
        )

    @pytest.mark.asyncio
    async def test_agent_level_media_capture_limit_applied_to_execute_code(self):
        """execute_code should use the agent-level media attachment limit."""

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(
                media_capture=MediaCaptureConfig(max_attachments_per_execution=2)
            ),
        ):
            pass

        agent = TestAgent()
        result = await agent.runtime.execute_code(
            """
for i in range(4):
    show(Image.from_bytes(f"img{i}".encode(), media_type="image/png"))
"""
        )

        assert len(result.images) == 2
        assert "limit reached (2)" in result.stdout

    @pytest.mark.asyncio
    async def test_method_level_media_capture_limit_applied_to_execute_code(self):
        """Method-level truncation config should override agent-level media limit."""
        from nooa import strategy
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.current_call import CurrentCall

        execution_results = {}

        class CodeRunningStrategy(GenerationStrategy):
            name = "CODE_RUNNING"
            traceable = False
            requires_lock = False

            def get_block_overrides(self):
                return {}

            async def execute(self, runtime, call: CurrentCall):
                result = await runtime.execute_code(
                    """
for i in range(4):
    show(Image.from_bytes(f"img{i}".encode(), media_type="image/png"))
"""
                )
                execution_results["images"] = result.images
                execution_results["stdout"] = result.stdout
                return "done"

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(
                media_capture=MediaCaptureConfig(max_attachments_per_execution=4)
            ),
        ):
            @strategy(
                CodeRunningStrategy(),
                truncation=TruncationConfig(
                    media_capture=MediaCaptureConfig(max_attachments_per_execution=2)
                ),
            )
            async def run_code(self) -> str:
                """Run code."""
                ...

        agent = TestAgent()
        result = await agent.run_code()

        assert result == "done"
        assert len(execution_results["images"]) == 2
        assert "limit reached (2)" in execution_results["stdout"]

    @pytest.mark.asyncio
    async def test_method_level_truncation_does_not_affect_other_methods(self):
        """A truncation override on one method should not affect another method."""
        from nooa import strategy
        from nooa.strategies.codeact import CodeActStrategy

        llm = FakeLLMClient.with_tool_call("return_result", {"result": "done"})

        class TestAgent(
            Agent,
            llm=llm,
            truncation=TruncationConfig(capture=CaptureConfig(max_stdout=100000)),
        ):
            @strategy(
                CodeActStrategy(),
                truncation=TruncationConfig(capture=CaptureConfig(max_stdout=500)),
            )
            async def small_limit(self) -> str:
                """Small stdout limit."""
                ...

            @strategy(CodeActStrategy())
            async def big_limit(self) -> str:
                """Uses agent-level limit."""
                ...

        agent = TestAgent()

        # Both methods resolve at the agent level — verify the agent config is intact
        assert agent._truncation.capture.max_stdout == 100_000

        # The small_limit method's wrapper should have _strategy_truncation set
        small_fn = TestAgent.__dict__["small_limit"]
        assert getattr(small_fn, "_strategy_truncation", None) is not None

        # The big_limit method's wrapper should NOT have _strategy_truncation set
        big_fn = TestAgent.__dict__["big_limit"]
        assert getattr(big_fn, "_strategy_truncation", None) is None

    def test_truncation_config_merge_at_method_level(self):
        """Method-level config merges with agent config (method fields win)."""
        from nooa import strategy

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(
                capture=CaptureConfig(max_stdout=100000), event_format=FormatConfig(max_length=100)
            ),
        ):
            @strategy(truncation=TruncationConfig(capture=CaptureConfig(max_stdout=500)))
            async def method(self) -> str:
                """Method with partial truncation override."""
                ...

        agent = TestAgent()

        # Simulate what actor does: merge agent config with method-level override
        method_fn = TestAgent.__dict__["method"]
        method_tc = getattr(method_fn, "_strategy_truncation", None)
        assert method_tc is not None

        merged = agent._truncation.merge_with(method_tc)

        # Method field wins
        assert merged.capture.max_stdout == 500
        # Agent field preserved (not in method override)
        assert merged.event_format.max_length == 100

    @pytest.mark.asyncio
    async def test_truncation_config_reverts_after_method_returns(self):
        """After a method call completes, runtime.truncation_config must return
        to the agent-level config, not the method-level override.

        This verifies the context var is properly reset in the finally block.
        """
        from nooa import strategy
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.current_call import CurrentCall

        captured = {}

        class CapturingStrategy(GenerationStrategy):
            name = "CAPTURING"
            traceable = False
            requires_lock = False

            def get_block_overrides(self):
                return {}

            async def execute(self, runtime, call: CurrentCall):
                captured["during"] = runtime.truncation_config
                return "done"

        method_tc = TruncationConfig(capture=CaptureConfig(max_stdout=99999))

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(capture=CaptureConfig(max_stdout=100000)),
        ):
            @strategy(CapturingStrategy(), truncation=method_tc)
            async def run(self) -> str:
                """Run."""
                ...

        agent = TestAgent()

        # Before: runtime reflects agent-level config
        assert agent.runtime.truncation_config.capture.max_stdout == 100_000

        await agent.run()

        # During: strategy saw the method-level override
        assert captured["during"].capture.max_stdout == 99_999

        # After: runtime has reverted to agent-level config
        assert agent.runtime.truncation_config.capture.max_stdout == 100_000

    @pytest.mark.asyncio
    async def test_method_level_max_context_tokens_visible_to_strategy_but_not_render_budget(self):
        """Method-level max_context_tokens is visible via runtime.truncation_config,
        but the whole-context render/eviction budget is AGENT-level by design.

        ``max_context_tokens`` sizes the shared context window; a per-call value
        cannot meaningfully re-scope it (every call sees the same accumulated
        events). So ``_build_messages`` reads ``self.agent._truncation`` for the
        render budget while ``runtime.truncation_config`` still reports the
        method override for code that wants the per-call config (e.g. stdout
        capture). This test pins BOTH facts so the agent-level render behavior
        can't silently regress to method-level.
        """
        from nooa import strategy
        from nooa.runtime.actor import _current_llm_var
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.current_call import CurrentCall

        captured = {}

        class TokenCapturingStrategy(GenerationStrategy):
            name = "TOKEN_CAPTURING"
            traceable = False
            requires_lock = False

            def get_block_overrides(self):
                return {}

            async def execute(self, runtime, call: CurrentCall):
                # The per-call override is still visible to the strategy.
                captured["runtime_max_context_tokens"] = (
                    runtime.truncation_config.max_context_tokens
                )
                # ...but the render budget actually used is agent-level.
                method = getattr(runtime.agent, call.method_name)
                await runtime._build_messages(method)
                stats = runtime._last_context_stats
                captured["render_max_context_tokens"] = stats.max_context_tokens
                return "done"

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(max_context_tokens=20_000),  # agent default
        ):
            @strategy(
                TokenCapturingStrategy(),
                truncation=TruncationConfig(max_context_tokens=5_000),  # method override
            )
            async def run(self) -> str:
                """Run."""
                ...

        agent = TestAgent()
        token = _current_llm_var.set(_TEST_LLM)
        try:
            await agent.run()
        finally:
            _current_llm_var.reset(token)

        # Strategy still sees the method-level override.
        assert captured["runtime_max_context_tokens"] == 5_000
        # But the render/eviction budget is the AGENT-level value, not the method's.
        assert captured["render_max_context_tokens"] == 20_000

    @pytest.mark.asyncio
    async def test_concurrent_method_calls_have_isolated_truncation_configs(self):
        """Concurrent calls to methods with different truncation configs are isolated.

        asyncio context vars are copied into each new Task at creation time, so
        _execute_with_generation's var.set() in one task cannot bleed into another.
        This verifies the contextvars-based isolation holds under actual concurrency.
        """
        import asyncio

        from nooa import strategy
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.current_call import CurrentCall

        captured = {}

        class CapturingStrategy(GenerationStrategy):
            name = "CAPTURING"
            traceable = False
            requires_lock = False

            def get_block_overrides(self):
                return {}

            async def execute(self, runtime, call: CurrentCall):
                # Yield once to allow the other task to interleave
                await asyncio.sleep(0)
                seen = runtime.truncation_config.capture.max_stdout
                captured[call.method_name] = seen
                # Yield again — if the context var leaked, seen would have changed
                await asyncio.sleep(0)
                assert runtime.truncation_config.capture.max_stdout == seen, (
                    "truncation_config changed after yielding — context var leaked between tasks"
                )
                return "done"

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(capture=CaptureConfig(max_stdout=100000)),
        ):
            @strategy(
                CapturingStrategy(),
                truncation=TruncationConfig(capture=CaptureConfig(max_stdout=1000)),
            )
            async def method_a(self) -> str:
                """Method A with small stdout limit."""
                ...

            @strategy(
                CapturingStrategy(),
                truncation=TruncationConfig(capture=CaptureConfig(max_stdout=2000)),
            )
            async def method_b(self) -> str:
                """Method B with medium stdout limit."""
                ...

        agent = TestAgent()
        await asyncio.gather(agent.method_a(), agent.method_b())

        # Each task must have seen exactly its own method-level config
        assert captured["method_a"] == 1_000
        assert captured["method_b"] == 2_000

    @pytest.mark.asyncio
    async def test_nested_method_call_sees_inner_method_config(self):
        """When outer's strategy calls inner(), inner sees its own truncation config.

        _execute_with_generation sets the context var for the duration of each
        method call and resets it in finally.  Nested calls therefore get their
        own method-level config during execution, and the outer config is
        restored once the inner call returns.
        """
        from nooa import strategy
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.current_call import CurrentCall

        captured: dict = {}

        class InnerCapturingStrategy(GenerationStrategy):
            name = "INNER_CAP"
            traceable = False
            requires_lock = False

            def get_block_overrides(self):
                return {}

            async def execute(self, runtime, call: CurrentCall):
                captured["inner"] = runtime.truncation_config.capture.max_stdout
                return "done"

        class OuterCallingStrategy(GenerationStrategy):
            """Strategy that records its config, calls agent.inner(), then records again."""

            name = "OUTER_CALL"
            traceable = False
            requires_lock = False

            def get_block_overrides(self):
                return {}

            async def execute(self, runtime, call: CurrentCall):
                captured["outer_before"] = runtime.truncation_config.capture.max_stdout
                await runtime.agent.inner()
                # Config must be restored to outer's value after inner returns
                captured["outer_after"] = runtime.truncation_config.capture.max_stdout
                return "done"

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(capture=CaptureConfig(max_stdout=100000)),
        ):
            @strategy(
                OuterCallingStrategy(),
                truncation=TruncationConfig(capture=CaptureConfig(max_stdout=1000)),
            )
            async def outer(self) -> str:
                """Outer method."""
                ...

            @strategy(
                InnerCapturingStrategy(),
                truncation=TruncationConfig(capture=CaptureConfig(max_stdout=2000)),
            )
            async def inner(self) -> str:
                """Inner method called from outer's strategy."""
                ...

        agent = TestAgent()
        await agent.outer()

        # Inner sees its own config (2_000), not outer's (1_000)
        assert captured["inner"] == 2_000
        # Outer sees its own config before and after calling inner
        assert captured["outer_before"] == 1_000
        assert captured["outer_after"] == 1_000


class TestPerCallTruncationRendering:
    """Method-level truncation should apply only to events from that call."""

    @pytest.mark.asyncio
    async def test_method_level_event_format_applies_only_to_matching_call_events(self):
        """Verify method-level event_format applies only to events from that method."""
        from nooa import strategy
        from nooa.events import Error
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.current_call import CurrentCall

        class EventAddingStrategy(GenerationStrategy):
            name = "EVENT_ADDING"
            traceable = False
            requires_lock = False

            def get_block_overrides(self):
                return {}

            async def execute(self, runtime, call: CurrentCall):
                marker = call.method_name.upper()
                runtime.event_manager.add(Error(content=f"{marker}_" + marker[0] * 2000))
                return "done"

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(event_format=FormatConfig(max_string=1000)),
        ):
            @strategy(
                EventAddingStrategy(),
                truncation=TruncationConfig(event_format=FormatConfig(max_string=40)),
            )
            async def small_event(self) -> str: ...

            @strategy(EventAddingStrategy())
            async def agent_event(self) -> str: ...

        agent = TestAgent()
        await agent.small_event()
        await agent.agent_event()

        # _prepare_context deliberately does not serialize events; render through
        # _build_messages so the per-call event_format resolver is exercised.
        from nooa.runtime.actor import _current_llm_var

        token = _current_llm_var.set(_TEST_LLM)
        try:
            messages = await agent.runtime._build_messages(agent.small_event)
        finally:
            _current_llm_var.reset(token)
        rendered = "\n".join(str(m.get("content", "")) for m in messages)

        assert "str(len=2012" in rendered
        assert "[:20]='SMALL_EVENT_" in rendered
        assert "[:500]='AGENT_EVENT_" in rendered

    @pytest.mark.asyncio
    async def test_method_level_context_block_format_does_not_rerender_whole_context(self):
        """Verify method-level truncation does not re-render existing context blocks."""
        from nooa import strategy
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.current_call import CurrentCall

        class BuildingStrategy(GenerationStrategy):
            name = "BUILDING"
            traceable = False
            requires_lock = False

            def get_block_overrides(self):
                return {}

            async def execute(self, runtime, call: CurrentCall):
                blocks = await runtime._prepare_context(
                    getattr(runtime.agent, call.method_name), call.args, call.kwargs
                )
                rendered_context = next(b.content for b in blocks if b.key == "payload")
                return rendered_context

        class TestAgent(
            Agent,
            llm=_TEST_LLM,
            truncation=TruncationConfig(
                context_block_format=FormatConfig(max_string=1000, max_length=100, max_depth=4)
            ),
            context={"payload": {"items": list(range(80))}},
        ):
            @strategy(
                BuildingStrategy(),
                truncation=TruncationConfig(
                    context_block_format=FormatConfig(max_string=40, max_length=10, max_depth=2)
                ),
            )
            async def inspect_context(self) -> str: ...

        agent = TestAgent()
        rendered_context = await agent.inspect_context()

        assert "dict(len=1" not in rendered_context
        assert "list(len=80" not in rendered_context
        assert "79" in rendered_context

    @pytest.mark.asyncio
    async def test_event_truncation_format_persists_across_session_resume(self):
        """metadata["truncation_event_format"] is persisted on events and still
        scopes rendering after the events are reloaded from storage (resume).

        The per-call rendering design must NOT depend on runtime-local
        caches surviving a restart: the format travels with the event's metadata.
        This test reloads events through a SQLite round-trip (serializing metadata
        to JSON and back) and asserts the method-level bound is still applied.
        """
        import tempfile
        from pathlib import Path

        from nooa import strategy
        from nooa.events import Error
        from nooa.runtime.actor import _current_llm_var
        from nooa.storage import SQLiteStorageManager
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.current_call import CurrentCall

        class EventAddingStrategy(GenerationStrategy):
            name = "EVENT_ADDING_RESUME"
            traceable = False
            requires_lock = False

            def get_block_overrides(self):
                return {}

            async def execute(self, runtime, call: CurrentCall):
                runtime.event_manager.add(Error(content="R_" + "R" * 2000))
                return "done"

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "session.db"

            class TestAgent(
                Agent,
                llm=_TEST_LLM,
                truncation=TruncationConfig(event_format=FormatConfig(max_string=1000)),
            ):
                @strategy(
                    EventAddingStrategy(),
                    truncation=TruncationConfig(event_format=FormatConfig(max_string=40)),
                )
                async def small_event(self) -> str: ...

            storage = SQLiteStorageManager(str(db_path))
            agent = TestAgent(storage=storage)
            await agent.small_event()

            # Confirm the format was persisted onto the event's metadata.
            error_events = agent.event_manager.filter(type="Error")
            assert error_events, "expected an Error event"
            persisted = error_events[-1].metadata.get("truncation_event_format")
            assert persisted == {"max_string": 40, "max_length": 200, "max_depth": 5}

            # Release the session lock so a fresh manager can reopen the DB.
            storage.close()

            # Simulate resume: fresh agent + storage pointed at the same DB. The
            # new runtime has a cold format cache, so per-call scoping must come
            # entirely from event metadata.
            storage2 = SQLiteStorageManager(str(db_path))
            try:
                agent2 = TestAgent(storage=storage2)
                assert agent2.event_manager.filter(type="Error"), "events did not reload"

                token = _current_llm_var.set(_TEST_LLM)
                try:
                    messages = await agent2.runtime._build_messages(agent2.small_event)
                finally:
                    _current_llm_var.reset(token)
                rendered = "\n".join(str(m.get("content", "")) for m in messages)

                # The reloaded event renders under the method-level bound (40), not
                # the agent-level bound (1000): head window = max_string // 2 = 20.
                assert "str(len=2002" in rendered
                assert "[:20]='R_" in rendered
                assert "[:500]" not in rendered
            finally:
                storage2.close()
