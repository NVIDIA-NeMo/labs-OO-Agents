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
from nooa.context_blocks.events import ResultStatus
from nooa.context_blocks.formatter import XMLBlockFormatter
from nooa.events import PythonOutput
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


# ── L4.2: Event archival on context overflow error ───────────────────────


# ── Cross-Layer Pipeline Tests ───────────────────────────────────────────


# ── Helper for provider formatter ────────────────────────────────────────


def _openai_formatter():
    """Return an OpenAI provider formatter."""
    from nooa.context_blocks.formatter import OpenAIProviderFormatter

    return OpenAIProviderFormatter()
