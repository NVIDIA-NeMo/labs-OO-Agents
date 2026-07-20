# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for context_snapshot OTel hook (on_messages_built)."""

from __future__ import annotations

import tempfile
from typing import Any
from unittest.mock import MagicMock

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from otlp_test_helpers import read_all_otlp_jsonl_spans

from nooa.tracing._hooks_impl import OpenInferenceHooks
from nooa.tracing._otlp_file_exporter import OtlpJsonFileExporter


class TestOnMessagesBuiltProtocol:
    """Test that InstrumentationHooks protocol includes on_messages_built."""

    def test_protocol_has_on_messages_built(self):
        from nooa.runtime.hooks import InstrumentationHooks

        assert hasattr(InstrumentationHooks, "on_messages_built")

    def test_openinference_hooks_has_on_messages_built(self):
        """OpenInferenceHooks must implement on_messages_built."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = OtlpJsonFileExporter(tmpdir)
            provider = TracerProvider(resource=Resource(attributes={}))
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            tracer = provider.get_tracer("test")
            hooks = OpenInferenceHooks(tracer=tracer)
            assert hasattr(hooks, "on_messages_built")
            assert callable(hooks.on_messages_built)


class TestContextSnapshotSpan:
    """Test that on_messages_built creates context_snapshot spans."""

    def _make_hooks_and_exporter(self, tmpdir: str):
        """Create hooks instance with a real tracer writing to tmpdir."""
        exporter = OtlpJsonFileExporter(tmpdir)
        provider = TracerProvider(resource=Resource(attributes={}))
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")
        hooks = OpenInferenceHooks(tracer=tracer)
        return hooks, exporter, provider

    def _read_spans(self, tmpdir: str) -> list[dict[str, Any]]:
        """Read all spans from OTLP JSONL files in tmpdir."""
        return read_all_otlp_jsonl_spans(tmpdir)

    def test_first_call_emits_full_snapshot(self):
        """First on_messages_built call should emit full system message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks, exporter, provider = self._make_hooks_and_exporter(tmpdir)
            agent = MagicMock()
            type(agent).__name__ = "TestAgent"

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]

            call_id = "call-001"
            ctx = hooks.before_agent_call(
                agent=agent,
                method_name="solve",
                args=(),
                kwargs={},
                call_id=call_id,
                parent_call_id=None,
            )

            hooks.on_messages_built(
                agent=agent,
                method_name="solve",
                messages=messages,
                generation_id="gen-001",
            )

            hooks.after_agent_call(
                agent=agent,
                method_name="solve",
                result=None,
                exception=None,
                context=ctx,
            )
            provider.force_flush()

            spans = self._read_spans(tmpdir)
            snapshot_spans = [s for s in spans if s["name"] == "context_snapshot"]
            assert len(snapshot_spans) == 1

            attrs = snapshot_spans[0]["attributes"]
            assert attrs["nooa.system_message"] == "You are a helpful assistant."
            assert attrs["nooa.system_message.is_diff"] is False
            assert attrs["nooa.system_message.turn_index"] == 0

    def test_unchanged_system_message_skips_span(self):
        """Second call with same system message should not emit a span."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks, exporter, provider = self._make_hooks_and_exporter(tmpdir)
            agent = MagicMock()
            type(agent).__name__ = "TestAgent"

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]

            call_id = "call-001"
            ctx = hooks.before_agent_call(
                agent=agent,
                method_name="solve",
                args=(),
                kwargs={},
                call_id=call_id,
                parent_call_id=None,
            )

            hooks.on_messages_built(
                agent=agent,
                method_name="solve",
                messages=messages,
                generation_id="gen-001",
            )
            hooks.on_messages_built(
                agent=agent,
                method_name="solve",
                messages=messages,
                generation_id="gen-002",
            )

            hooks.after_agent_call(
                agent=agent,
                method_name="solve",
                result=None,
                exception=None,
                context=ctx,
            )
            provider.force_flush()

            spans = self._read_spans(tmpdir)
            snapshot_spans = [s for s in spans if s["name"] == "context_snapshot"]
            assert len(snapshot_spans) == 1

    def test_changed_system_message_emits_diff(self):
        """When system message changes, emit a unified diff."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks, exporter, provider = self._make_hooks_and_exporter(tmpdir)
            agent = MagicMock()
            type(agent).__name__ = "TestAgent"

            messages_v1 = [
                {"role": "system", "content": "You are a helpful assistant.\nBe concise."},
                {"role": "user", "content": "Hello"},
            ]
            messages_v2 = [
                {"role": "system", "content": "You are a helpful assistant.\nBe verbose."},
                {"role": "user", "content": "Hello"},
            ]

            call_id = "call-001"
            ctx = hooks.before_agent_call(
                agent=agent,
                method_name="solve",
                args=(),
                kwargs={},
                call_id=call_id,
                parent_call_id=None,
            )

            hooks.on_messages_built(
                agent=agent,
                method_name="solve",
                messages=messages_v1,
                generation_id="gen-001",
            )
            hooks.on_messages_built(
                agent=agent,
                method_name="solve",
                messages=messages_v2,
                generation_id="gen-002",
            )

            hooks.after_agent_call(
                agent=agent,
                method_name="solve",
                result=None,
                exception=None,
                context=ctx,
            )
            provider.force_flush()

            spans = self._read_spans(tmpdir)
            snapshot_spans = [s for s in spans if s["name"] == "context_snapshot"]
            assert len(snapshot_spans) == 2

            # First is full snapshot
            assert snapshot_spans[0]["attributes"]["nooa.system_message.is_diff"] is False
            assert snapshot_spans[0]["attributes"]["nooa.system_message.turn_index"] == 0

            # Second is unified diff
            diff_span = snapshot_spans[1]
            assert diff_span["attributes"]["nooa.system_message.is_diff"] is True
            assert diff_span["attributes"]["nooa.system_message.turn_index"] == 1
            diff_text = diff_span["attributes"]["nooa.system_message"]
            assert "-Be concise." in diff_text
            assert "+Be verbose." in diff_text

    def test_state_resets_between_agent_calls(self):
        """Each agent call should start fresh (full snapshot on first gen)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks, exporter, provider = self._make_hooks_and_exporter(tmpdir)
            agent = MagicMock()
            type(agent).__name__ = "TestAgent"

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]

            # First agent call
            ctx1 = hooks.before_agent_call(
                agent=agent,
                method_name="solve",
                args=(),
                kwargs={},
                call_id="call-001",
                parent_call_id=None,
            )
            hooks.on_messages_built(
                agent=agent,
                method_name="solve",
                messages=messages,
                generation_id="gen-001",
            )
            hooks.after_agent_call(
                agent=agent,
                method_name="solve",
                result=None,
                exception=None,
                context=ctx1,
            )

            # Second agent call — should emit full snapshot again
            ctx2 = hooks.before_agent_call(
                agent=agent,
                method_name="solve",
                args=(),
                kwargs={},
                call_id="call-002",
                parent_call_id=None,
            )
            hooks.on_messages_built(
                agent=agent,
                method_name="solve",
                messages=messages,
                generation_id="gen-002",
            )
            hooks.after_agent_call(
                agent=agent,
                method_name="solve",
                result=None,
                exception=None,
                context=ctx2,
            )

            provider.force_flush()

            spans = self._read_spans(tmpdir)
            snapshot_spans = [s for s in spans if s["name"] == "context_snapshot"]
            assert len(snapshot_spans) == 2
            assert all(
                s["attributes"]["nooa.system_message.is_diff"] is False for s in snapshot_spans
            )

    def test_no_system_message_skips_span(self):
        """If messages[0] is not a system message, skip the span."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks, exporter, provider = self._make_hooks_and_exporter(tmpdir)
            agent = MagicMock()
            type(agent).__name__ = "TestAgent"

            messages = [
                {"role": "user", "content": "Hello"},
            ]

            call_id = "call-001"
            ctx = hooks.before_agent_call(
                agent=agent,
                method_name="solve",
                args=(),
                kwargs={},
                call_id=call_id,
                parent_call_id=None,
            )
            hooks.on_messages_built(
                agent=agent,
                method_name="solve",
                messages=messages,
                generation_id="gen-001",
            )
            hooks.after_agent_call(
                agent=agent,
                method_name="solve",
                result=None,
                exception=None,
                context=ctx,
            )
            provider.force_flush()

            spans = self._read_spans(tmpdir)
            snapshot_spans = [s for s in spans if s["name"] == "context_snapshot"]
            assert len(snapshot_spans) == 0


class TestActorHookCallSite:
    """Test that actor.generate() calls on_messages_built."""

    def test_on_messages_built_is_called(self):
        """Verify call_before_hook('on_messages_built', ...) appears in actor.generate()."""
        import inspect

        from nooa.runtime.actor import ActorRuntime

        source = inspect.getsource(ActorRuntime.generate)
        assert "on_messages_built" in source, "actor.generate() must call on_messages_built hook"
