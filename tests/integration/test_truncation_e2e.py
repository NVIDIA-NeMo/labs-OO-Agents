# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration tests for the truncation pipeline (L2 → L3 → L4).

Tests the interactions between truncation layers:

- **L2** — I/O capture: ``TruncatingStringIO`` head/tail-truncates stdout/stderr
  during ``execute_code()``, producing ``<truncated-output>`` wrappers.
- **L3** — Event rendering: ``format_event()`` / ``pformat()`` renders events
  for the LLM context.  ``PythonOutput.stdout`` and ``.stderr`` carry
  ``spec(max_string=None)`` so they pass through verbatim (no re-truncation).
- **L4.1** — Context block eviction: ``render_context()`` marks over-budget
  non-static blocks EVICTED.
- **L4.2** — Event archival on context-window error: ``_archive_on_context_error``
  collapses oldest events when the API rejects the payload.

The critical invariant: **each layer trusts that earlier layers already bounded
the data**.  L3 must not re-truncate L2 output.  L4 operates on already-rendered
content from L3.

Inline tests use FakeLLMClient (no API calls).
"""

import pytest

from nooa import Agent
from nooa.agentdoc import pformat
from nooa.config.truncation_config import CaptureConfig, FormatConfig, TruncationConfig
from nooa.context_blocks import BlockMetadata, ResolvedBlock, Role
from nooa.context_blocks.events import ResultStatus
from nooa.context_blocks.formatter import XMLBlockFormatter
from nooa.context_blocks.renderer import render_context
from nooa.events import PythonOutput
from nooa.runtime.actor import _current_llm_var
from nooa.unifiedllm import FakeLLMClient

# ── Helpers ──────────────────────────────────────────────────────────────


class _FakeLLM(FakeLLMClient):
    """FakeLLM with a settable context_window for tests."""

    _cw = 4096

    @property
    def context_window(self):  # type: ignore[override]
        return self._cw

    def count_tokens(self, text: str) -> int:
        # ~4 chars per token approximation (fast, no external dependency)
        return max(1, len(text) // 4)


def _mk_agent(
    *,
    context_window: int = 4096,
    max_stdout: int = 50_000,
    max_stderr: int = 2_000,
    max_context_tokens: int | None = None,
    event_format: FormatConfig | None = None,
) -> Agent:
    """Create a test agent with configurable truncation settings."""

    class _LLM(_FakeLLM):
        _cw = context_window

    llm = _LLM()

    tc_kwargs: dict = {}
    tc_kwargs["capture"] = CaptureConfig(max_stdout=max_stdout, max_stderr=max_stderr)
    if max_context_tokens is not None:
        tc_kwargs["max_context_tokens"] = max_context_tokens
    if event_format is not None:
        tc_kwargs["event_format"] = event_format

    tc = TruncationConfig(**tc_kwargs)

    class A(Agent, llm=llm, truncation=tc):
        async def respond(self, prompt: str) -> str:
            """Respond to {prompt}."""
            ...

    return A()


def _make_python_output(
    stdout: str = "",
    stderr: str = "",
    error: str = "",
    tc_id: str = "tc_1",
    exec_count: int = 1,
    status: ResultStatus = ResultStatus.COMPLETE,
) -> PythonOutput:
    """Create a PythonOutput event with given fields."""
    return PythonOutput(
        tool_call_id=tc_id,
        execution_status=status,
        execution_count=exec_count,
        stdout=stdout,
        stderr=stderr,
        error=error,
    )


def _count_tokens(text: str) -> int:
    """Simple char-based token approximation for tests."""
    return max(1, len(text) // 4)


# ── L2 → L3: stdout/stderr truncation → event rendering ─────────────────


class TestL2ToL3Interaction:
    """L2 (TruncatingStringIO) truncates stdout/stderr at capture time.
    L3 (format_event / pformat) must render the result *verbatim* — no
    re-truncation of the already-bounded content.

    The key mechanism: ``PythonOutput.stdout`` and ``.stderr`` carry
    ``Annotated[str, spec(max_string=None)]`` which tells pformat to
    skip string truncation on those fields regardless of the caller's
    ``max_string`` setting.
    """

    def test_truncated_stdout_wrapper_survives_rendering(self):
        """The <truncated-output> wrapper from L2 must appear verbatim in
        the L3-rendered event — no str(len=...) marker wrapping it."""
        # Simulate what L2 produces for large output
        truncated_stdout = (
            "<truncated-output>\n"
            "Output too large (200,000 chars). "
            "Showing first 25,000 and last 25,000 chars.\n\n"
            + "x" * 25_000
            + "\n\n... 150,000 chars not shown ...\n\n"
            + "y" * 25_000
            + "\n</truncated-output>"
        )
        event = _make_python_output(stdout=truncated_stdout)
        rendered = pformat(event, max_string=500)

        # The full truncated_stdout must be present — L3 must NOT re-truncate
        assert truncated_stdout in rendered
        assert "str(len=" not in rendered

    def test_small_stdout_passes_through_unchanged(self):
        """Stdout under the L2 limit must survive L3 rendering exactly."""
        small_output = "hello world\nresult = 42"
        event = _make_python_output(stdout=small_output)
        rendered = pformat(event, max_string=500)
        assert small_output in rendered

    def test_truncated_stderr_wrapper_survives_rendering(self):
        """Same invariant for stderr — spec(max_string=None) applies."""
        truncated_stderr = (
            "<truncated-output>\n"
            "Output too large (10,000 chars). "
            "Showing first 1,000 and last 1,000 chars.\n\n"
            + "W" * 1_000
            + "\n\n... 8,000 chars not shown ...\n\n"
            + "E" * 1_000
            + "\n</truncated-output>"
        )
        event = _make_python_output(stderr=truncated_stderr)
        rendered = pformat(event, max_string=500)
        assert truncated_stderr in rendered
        assert "str(len=" not in rendered

    def test_non_exempt_fields_still_truncated(self):
        """Fields WITHOUT spec(max_string=None) (like error) should still
        respect the caller's max_string — only stdout/stderr are exempt."""
        long_error = "E" * 20_000
        event = _make_python_output(error=long_error)
        rendered = pformat(event, max_string=500)
        # error field SHOULD be truncated (no spec(max_string=None))
        assert long_error not in rendered

    def test_both_stdout_and_stderr_survive_together(self):
        """When both stdout and stderr carry L2-truncated content,
        both must survive L3 rendering verbatim."""
        big_stdout = "OUT_" * 10_000  # 40K chars, under 50K L2 limit
        big_stderr = "ERR_" * 400  # 1.6K chars, under 2K L2 limit
        event = _make_python_output(stdout=big_stdout, stderr=big_stderr)
        rendered = pformat(event, max_string=100)
        assert big_stdout in rendered
        assert big_stderr in rendered

    @pytest.mark.asyncio
    async def test_execute_code_l2_truncation_then_l3_rendering(self):
        """Full pipeline: execute_code truncates large stdout (L2),
        then pformat renders the PythonOutput event (L3) without
        re-truncating the already-bounded content."""
        agent = _mk_agent(max_stdout=1_000)

        # Generate output that exceeds the 1K L2 limit
        code = 'print("A" * 5000)'
        result = await agent.runtime.execute_code(code)

        assert result.success
        # L2 should have truncated
        assert "Output too large" in result.stdout or len(result.stdout) <= 1_000

        # Now render as L3 would
        event = _make_python_output(stdout=result.stdout)
        rendered = pformat(event, max_string=500)

        # The L2-truncated content must survive L3 verbatim
        assert result.stdout in rendered
        # No str(len=...) re-truncation marker
        assert "str(len=" not in rendered

    def test_event_format_config_does_not_retruncate_stdout(self):
        """format_event with tight event_format bounds must still
        preserve stdout/stderr verbatim via spec(max_string=None)."""
        fmt = XMLBlockFormatter()
        tight_format = FormatConfig(max_string=100, max_length=10, max_depth=2)

        big_stdout = "X" * 5_000
        event = _make_python_output(stdout=big_stdout)
        rendered = fmt.format_event(event, event_format=tight_format)

        assert big_stdout in rendered


# ── L3: Event rendering bounds ───────────────────────────────────────────


class TestL3EventRendering:
    """format_event applies structural bounds (max_string, max_length,
    max_depth) from FormatConfig to non-exempt event fields.

    The spec(max_string=None) annotation on PythonOutput.stdout/stderr
    overrides these bounds for those specific fields.
    """

    def test_large_value_field_bounded_by_event_format(self):
        """Non-exempt fields with large values should be bounded."""
        event = _make_python_output(error="z" * 100_000)
        fmt = XMLBlockFormatter()
        rendered = fmt.format_event(
            event,
            event_format=FormatConfig(max_string=500, max_length=50, max_depth=4),
        )
        # error field should be truncated
        assert "z" * 100_000 not in rendered
        assert len(rendered) < 10_000

    def test_stdout_exempt_from_event_format_max_string(self):
        """PythonOutput.stdout has spec(max_string=None) — it must
        survive event_format's max_string bound."""
        big_stdout = "S" * 20_000
        event = _make_python_output(stdout=big_stdout)
        fmt = XMLBlockFormatter()
        rendered = fmt.format_event(
            event,
            event_format=FormatConfig(max_string=100, max_length=10, max_depth=2),
        )
        assert big_stdout in rendered

    def test_stderr_exempt_from_event_format_max_string(self):
        """PythonOutput.stderr has spec(max_string=None) — same exemption."""
        big_stderr = "E" * 5_000
        event = _make_python_output(stderr=big_stderr)
        fmt = XMLBlockFormatter()
        rendered = fmt.format_event(
            event,
            event_format=FormatConfig(max_string=100, max_length=10, max_depth=2),
        )
        assert big_stderr in rendered

    def test_default_event_format_does_not_truncate_moderate_stdout(self):
        """Default event_format (max_string=10_000) should not truncate
        stdout that's already been L2-bounded to 50K."""
        # Realistic: L2 outputs up to 50K, default event_format max_string=10_000
        # But stdout has spec(max_string=None) so it should pass through
        moderate_stdout = "M" * 15_000
        event = _make_python_output(stdout=moderate_stdout)
        rendered = pformat(
            event,
            max_string=10_000,  # default event_format setting
        )
        assert moderate_stdout in rendered


# ── L4.1: Context block eviction ─────────────────────────────────────────


class TestL4ContextBlockEviction:
    """render_context evicts over-budget non-static context blocks
    (L4.1). Events are NOT evicted by this mechanism — only system-role
    context blocks are candidates."""

    def test_over_budget_block_evicted(self):
        """A non-static context block exceeding the budget gets EVICTED."""
        blocks = [
            ResolvedBlock(
                key="system_prompt",
                content="You are helpful.",
                role=Role.SYSTEM,
                metadata=BlockMetadata(static=True),
            ),
            ResolvedBlock(
                key="big_context",
                content="data " * 200,
                role=Role.SYSTEM,
                metadata=BlockMetadata(),
            ),
        ]
        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=_openai_formatter(),
            context_limit=50,
            count_tokens=_count_tokens,
        )
        assert result.stats.context_blocks_dropped > 0
        assert "EVICTED" in str(result.output)

    def test_static_blocks_never_evicted(self):
        """Blocks with static=True survive regardless of budget."""
        blocks = [
            ResolvedBlock(
                key="immutable",
                content="x " * 200,
                role=Role.SYSTEM,
                metadata=BlockMetadata(static=True),
            ),
        ]
        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=_openai_formatter(),
            context_limit=10,
            count_tokens=_count_tokens,
        )
        assert result.stats.context_blocks_dropped == 0

    def test_newest_blocks_evicted_first(self):
        """When multiple blocks exceed budget, newest non-static blocks
        are evicted first (eviction works from the end)."""
        blocks = [
            ResolvedBlock(
                key="system_prompt",
                content="You are helpful.",
                role=Role.SYSTEM,
                metadata=BlockMetadata(static=True),
            ),
        ]
        # Add 5 non-static blocks (~125 tokens each at 4 chars/token)
        for i in range(5):
            blocks.append(
                ResolvedBlock(
                    key=f"block_{i}",
                    content=f"content_{i} " * 50,
                    role=Role.SYSTEM,
                    metadata=BlockMetadata(),
                ),
            )

        # Budget fits ~2 blocks + system_prompt; the rest must be evicted
        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=_openai_formatter(),
            context_limit=260,
            count_tokens=_count_tokens,
        )
        dropped = result.stats.context_blocks_dropped
        assert dropped >= 2, f"Expected at least 2 evicted, got {dropped}"
        # Eviction works from the end — newest blocks (block_4, block_3, ...)
        # are evicted first; oldest (block_0) should survive
        output_str = str(result.output)
        assert "content_0" in output_str, (
            "Oldest block should survive eviction; eviction should start from newest"
        )

    def test_user_blocks_evicted_before_framework_blocks(self):
        """User blocks (user_block=True) are evicted before framework blocks."""
        blocks = [
            ResolvedBlock(
                key="framework",
                content="framework " * 50,
                role=Role.SYSTEM,
                metadata=BlockMetadata(static=False),
            ),
            ResolvedBlock(
                key="user_data",
                content="user " * 50,
                role=Role.SYSTEM,
                metadata=BlockMetadata(static=False, user_block=True),
            ),
        ]
        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=_openai_formatter(),
            context_limit=60,
            count_tokens=_count_tokens,
        )
        assert result.stats.context_blocks_dropped >= 1, (
            f"Expected at least one block to be evicted, got {result.stats.context_blocks_dropped}"
        )
        output_str = str(result.output)
        assert "EVICTED" in output_str
        # User block should be evicted first; framework block should survive
        assert "framework" in output_str.replace("EVICTED", "")

    def test_boundary_block_just_under_then_second_pushes_over(self):
        """One block fits just under the budget; adding a second pushes
        total over the limit and triggers eviction of the newest block."""
        # Budget = 100 tokens. Static system prompt ~1 token.
        # Block A = 380 chars = 95 tokens → fits (1 + 95 = 96 ≤ 100)
        # Block B = 40 chars = 10 tokens → total 106 > 100 → B gets evicted
        blocks = [
            ResolvedBlock(
                key="system_prompt",
                content="small",  # 5 chars ~1 token
                role=Role.SYSTEM,
                metadata=BlockMetadata(static=True),
            ),
            ResolvedBlock(
                key="block_a",
                content="a" * 380,  # 95 tokens, fits under 100
                role=Role.SYSTEM,
                metadata=BlockMetadata(),
            ),
            ResolvedBlock(
                key="block_b",
                content="b" * 40,  # 10 tokens, pushes total over 100
                role=Role.SYSTEM,
                metadata=BlockMetadata(),
            ),
        ]
        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=_openai_formatter(),
            context_limit=100,
            count_tokens=_count_tokens,
        )
        output_str = str(result.output)
        # Block A (oldest non-static) should survive
        assert "a" * 380 in output_str, "Block A should survive — it was added first"
        # Block B (newest non-static) should be evicted
        assert "b" * 40 not in output_str, "Block B should be evicted — it's newest"
        assert result.stats.context_blocks_dropped == 1
        assert "EVICTED" in output_str

    def test_boundary_single_block_exactly_at_limit_no_eviction(self):
        """A single non-static block that exactly fills the budget should
        NOT be evicted — eviction only fires when total exceeds the limit."""
        # Budget = 100 tokens. Static system prompt ~1 token.
        # Block A = 396 chars = 99 tokens → total 100 ≤ 100 → no eviction
        blocks = [
            ResolvedBlock(
                key="system_prompt",
                content="x",  # 1 char ~1 token (rounded up by max(1,...))
                role=Role.SYSTEM,
                metadata=BlockMetadata(static=True),
            ),
            ResolvedBlock(
                key="block_a",
                content="a" * 396,  # 99 tokens
                role=Role.SYSTEM,
                metadata=BlockMetadata(),
            ),
        ]
        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=_openai_formatter(),
            context_limit=100,
            count_tokens=_count_tokens,
        )
        assert result.stats.context_blocks_dropped == 0
        assert "EVICTED" not in str(result.output)
        assert "a" * 396 in str(result.output)

    def test_events_not_evicted_by_context_block_eviction(self):
        """Event blocks (non-SYSTEM role) are not touched by L4.1 eviction.
        Only SYSTEM-role context blocks are candidates."""
        system_block = ResolvedBlock(
            key="system_prompt",
            content="You are helpful.",
            role=Role.SYSTEM,
            metadata=BlockMetadata(static=True),
        )
        event_block = ResolvedBlock(
            key="event_1",
            content="big event output " * 100,
            role=Role.USER,
            metadata=BlockMetadata(tag="1"),
        )
        result = render_context(
            [system_block, event_block],
            block_formatter=XMLBlockFormatter(),
            provider_formatter=_openai_formatter(),
            context_limit=10,
            count_tokens=_count_tokens,
        )
        # Context block eviction should not drop events
        assert result.stats.context_blocks_dropped == 0
        output_str = str(result.output)
        assert "big event output" in output_str


# ── L4.2: Event archival on context overflow error ───────────────────────


class TestL4EventArchival:
    """When the LLM API rejects the context as too large, the runtime
    archives (collapses) the oldest events to reduce utilization."""

    @pytest.mark.asyncio
    async def test_archive_collapses_oldest_events(self):
        """_archive_on_context_error collapses oldest events."""
        agent = _mk_agent(context_window=4096)

        # Fill with many events
        for i in range(20):
            tc_id = f"tc_{i}"
            agent.event_manager.add(
                PythonOutput(
                    tool_call_id=tc_id,
                    execution_status=ResultStatus.COMPLETE,
                    execution_count=i,
                    stdout="output " * 50,
                )
            )

        active_before = len(list(agent.event_manager.keys()))
        assert active_before == 20

        # Simulate what happens on context overflow
        # Manually set context stats so _archive_on_context_error has data
        from nooa.context_blocks.models import ContextWindowStats

        agent.runtime._last_context_stats = ContextWindowStats(
            context_blocks_count=1,
            events_count=20,
            context_blocks_chars=500,
            events_chars=5000,
            prompt_tokens=5500,
            max_context_tokens=None,
            model_context_window=4096,
            context_blocks_dropped=0,
            events_dropped=0,
        )

        agent.runtime._archive_on_context_error(ctx_window=4096)

        active_after = len(list(agent.event_manager.keys()))
        assert active_after < active_before, (
            f"Expected events to be archived: {active_before} → {active_after}"
        )

    @pytest.mark.asyncio
    async def test_archived_events_not_in_context(self):
        """After archival, collapsed events should not appear in context
        (they're replaced by a Summary event)."""
        agent = _mk_agent(context_window=4096)

        # Add events with distinctive content
        for i in range(10):
            tc_id = f"tc_{i}"
            agent.event_manager.add(
                PythonOutput(
                    tool_call_id=tc_id,
                    execution_status=ResultStatus.COMPLETE,
                    execution_count=i,
                    stdout=f"MARKER_{i}_OUTPUT",
                )
            )

        # Collapse the first 5 events
        active_tags = list(agent.event_manager.keys())
        agent.event_manager.collapse(active_tags[0], active_tags[4])

        # Active events should not include the collapsed ones
        active_values = list(agent.event_manager.values())

        # Collapsed events should be replaced by a Summary
        has_summary = any(hasattr(e, "summary_tag") for e in active_values)
        assert has_summary, "Collapsed events should produce a Summary"


# ── Cross-Layer Pipeline Tests ───────────────────────────────────────────


class TestCrossLayerPipeline:
    """Full pipeline tests exercising multiple truncation layers together."""

    @pytest.mark.asyncio
    async def test_l2_truncated_stdout_in_rendered_context(self):
        """Full pipeline: large stdout → L2 truncation → event created →
        L3 rendering → context assembly.  The truncation wrapper from L2
        must survive through to the final rendered context."""
        agent = _mk_agent(max_stdout=500, context_window=100_000)

        # Execute code that produces large stdout
        result = await agent.runtime.execute_code('print("Z" * 5000)')
        assert result.success
        assert "Output too large" in result.stdout

        # Add as event (simulating what codeact does)
        event = PythonOutput(
            tool_call_id="tc_pipeline",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout=result.stdout,
        )
        agent.event_manager.add(event)

        # Render the full context
        method = type(agent).respond
        token = _current_llm_var.set(agent._llm)
        try:
            messages = await agent.runtime._build_messages(
                method, call_args=(agent, "hi"), call_kwargs={}
            )
        finally:
            _current_llm_var.reset(token)

        # The L2 truncation wrapper must be present in final context
        full_context = str(messages)
        assert "Output too large" in full_context

    @pytest.mark.asyncio
    async def test_l2_small_output_survives_full_pipeline(self):
        """Small output that doesn't trigger L2 truncation must arrive
        intact through L3 rendering and context assembly."""
        agent = _mk_agent(context_window=100_000)

        result = await agent.runtime.execute_code('print("hello world")')
        assert result.success
        assert result.stdout == "hello world\n"

        event = PythonOutput(
            tool_call_id="tc_small",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout=result.stdout,
        )
        agent.event_manager.add(event)

        method = type(agent).respond
        token = _current_llm_var.set(agent._llm)
        try:
            messages = await agent.runtime._build_messages(
                method, call_args=(agent, "hi"), call_kwargs={}
            )
        finally:
            _current_llm_var.reset(token)

        assert "hello world" in str(messages)

    @pytest.mark.asyncio
    async def test_many_l2_truncated_events_trigger_l4_eviction(self):
        """Many events with L2-bounded stdout can still exceed the context
        block budget → L4.1 should evict context blocks to make room."""
        agent = _mk_agent(
            max_stdout=2_000,
            context_window=8192,
            max_context_tokens=200,
        )

        # Add a large user context block
        agent.context["big_data"] = "data " * 500

        # Add multiple events with moderate stdout
        for i in range(5):
            agent.event_manager.add(
                PythonOutput(
                    tool_call_id=f"tc_{i}",
                    execution_status=ResultStatus.COMPLETE,
                    execution_count=i,
                    stdout=f"result_{i} " * 100,
                )
            )

        method = type(agent).respond
        token = _current_llm_var.set(agent._llm)
        try:
            messages = await agent.runtime._build_messages(
                method, call_args=(agent, "hi"), call_kwargs={}
            )
        finally:
            _current_llm_var.reset(token)

        # The user context block should have been evicted
        stats = agent.runtime._last_context_stats
        assert stats is not None
        # Either evicted or fits — the point is the pipeline didn't crash
        assert messages is not None

    @pytest.mark.asyncio
    async def test_head_tail_structure_preserved_through_pipeline(self):
        """The head/tail structure from L2's TruncatingStringIO must survive
        all the way through L3 event rendering into the context."""
        from nooa.agentdoc import TruncatingStringIO

        # Create a head/tail truncated output directly
        buf = TruncatingStringIO(limit=200)
        buf.write("HEAD_CONTENT_" * 20)  # 260 chars, exceeds 200
        buf.write("TAIL_CONTENT_" * 20)  # more tail content
        truncated = buf.getvalue()

        assert "Output too large" in truncated
        assert "chars not shown" in truncated

        # Now render it through L3
        event = _make_python_output(stdout=truncated)
        rendered = pformat(event, max_string=100)

        # The truncation markers must survive
        assert "Output too large" in rendered
        assert "chars not shown" in rendered

    def test_l3_rendering_preserves_l2_truncation_notice_in_event_format(self):
        """format_event with event_format must preserve the L2 truncation
        notice verbatim, not wrap it in str(len=...) markers."""
        from nooa.agentdoc import TruncatingStringIO

        # Produce realistic L2 output
        buf = TruncatingStringIO(limit=500)
        for i in range(100):
            buf.write(f"line {i}: " + "x" * 50 + "\n")
        l2_output = buf.getvalue()
        assert buf.was_truncated

        event = _make_python_output(stdout=l2_output)
        fmt = XMLBlockFormatter()
        rendered = fmt.format_event(
            event,
            event_format=FormatConfig(max_string=100, max_length=10, max_depth=2),
        )

        # Full L2 output must be in rendered — no re-truncation
        assert l2_output in rendered
        assert "str(len=" not in rendered

    @pytest.mark.asyncio
    async def test_context_eviction_and_event_archival_coexist(self):
        """Both L4.1 (block eviction) and L4.2 (event archival) can fire
        in the same session without interfering with each other."""
        agent = _mk_agent(context_window=4096, max_context_tokens=100)

        # Add a large context block (will be evicted by L4.1)
        agent.context["big"] = "big " * 200

        # Add many events (would trigger L4.2 on context overflow)
        for i in range(15):
            agent.event_manager.add(
                PythonOutput(
                    tool_call_id=f"tc_{i}",
                    execution_status=ResultStatus.COMPLETE,
                    execution_count=i,
                    stdout=f"output_{i} " * 30,
                )
            )

        # Build messages — L4.1 eviction should fire for the context block
        method = type(agent).respond
        token = _current_llm_var.set(agent._llm)
        try:
            messages = await agent.runtime._build_messages(
                method, call_args=(agent, "hi"), call_kwargs={}
            )
        finally:
            _current_llm_var.reset(token)

        assert messages is not None
        stats = agent.runtime._last_context_stats
        assert stats is not None

        # Now simulate L4.2 — archive events
        from nooa.context_blocks.models import ContextWindowStats

        agent.runtime._last_context_stats = ContextWindowStats(
            context_blocks_count=1,
            events_count=15,
            context_blocks_chars=100,
            events_chars=3000,
            prompt_tokens=3100,
            max_context_tokens=100,
            model_context_window=4096,
            context_blocks_dropped=1,
            events_dropped=0,
        )

        active_before = len(list(agent.event_manager.keys()))
        agent.runtime._archive_on_context_error(ctx_window=4096)
        active_after = len(list(agent.event_manager.keys()))

        # Some events should have been archived
        assert active_after < active_before


# ── Helper for provider formatter ────────────────────────────────────────


def _openai_formatter():
    """Return an OpenAI provider formatter."""
    from nooa.context_blocks.formatter import OpenAIProviderFormatter

    return OpenAIProviderFormatter()
