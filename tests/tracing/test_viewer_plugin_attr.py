# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the nooa.viewer.plugin span attribute."""

from __future__ import annotations

import tempfile
from typing import Any
from unittest.mock import MagicMock

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from otlp_test_helpers import read_all_otlp_jsonl_spans

from nooa.tracing._hooks_impl import (
    VIEWER_PLUGIN_ATTR,
    OpenInferenceHooks,
    ViewerPlugin,
)
from nooa.tracing._otlp_file_exporter import OtlpJsonFileExporter


class TestViewerPluginAttribute:
    """Verify nooa.viewer.plugin is set on all span types."""

    def _make_hooks(self, tmpdir: str):
        exporter = OtlpJsonFileExporter(tmpdir)
        provider = TracerProvider(resource=Resource(attributes={}))
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")
        hooks = OpenInferenceHooks(tracer=tracer)
        return hooks, provider

    def _read_spans(self, tmpdir: str) -> list[dict[str, Any]]:
        return read_all_otlp_jsonl_spans(tmpdir)

    def _make_agent(self):
        agent = MagicMock()
        type(agent).__name__ = "TestAgent"
        return agent

    def test_agent_call_sets_method_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks, provider = self._make_hooks(tmpdir)
            agent = self._make_agent()

            ctx = hooks.before_agent_call(
                agent=agent,
                method_name="solve",
                args=(),
                kwargs={},
                call_id="call-001",
                parent_call_id=None,
            )
            hooks.after_agent_call(
                agent=agent,
                method_name="solve",
                result="done",
                exception=None,
                context=ctx,
            )
            provider.force_flush()

            spans = self._read_spans(tmpdir)
            method_spans = [s for s in spans if s["name"] == "method.solve"]
            assert len(method_spans) == 1
            assert method_spans[0]["attributes"][VIEWER_PLUGIN_ATTR] == ViewerPlugin.METHOD

    def test_generation_sets_generation_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks, provider = self._make_hooks(tmpdir)
            agent = self._make_agent()

            ctx = hooks.before_generation(
                agent=agent,
                method_name="solve",
                strategy="PURE_PYTHON",
                generation_id="gen-001",
                parent_generation_id=None,
            )
            hooks.after_generation(
                agent=agent,
                method_name="solve",
                result="output",
                exception=None,
                context=ctx,
                generation_id="gen-001",
            )
            provider.force_flush()

            spans = self._read_spans(tmpdir)
            gen_spans = [s for s in spans if s["name"] == "generation"]
            assert len(gen_spans) == 1
            assert gen_spans[0]["attributes"][VIEWER_PLUGIN_ATTR] == ViewerPlugin.GENERATION

    def test_code_execution_sets_code_execution_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks, provider = self._make_hooks(tmpdir)
            agent = self._make_agent()

            ctx = hooks.before_code_execution(
                agent=agent,
                code="print('hello')",
                execution_id="exec-001",
            )
            hooks.after_code_execution(
                agent=agent,
                code="print('hello')",
                result=None,
                exception=None,
                context=ctx,
                execution_id="exec-001",
            )
            provider.force_flush()

            spans = self._read_spans(tmpdir)
            code_spans = [s for s in spans if s["name"] == "code_execution"]
            assert len(code_spans) == 1
            assert code_spans[0]["attributes"][VIEWER_PLUGIN_ATTR] == ViewerPlugin.CODE_EXECUTION

    def test_method_invocation_sets_method_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks, provider = self._make_hooks(tmpdir)
            agent = self._make_agent()

            ctx = hooks.before_method_invocation(
                agent=agent,
                method_name="helper",
                args=(),
                kwargs={},
                invocation_id="inv-001",
            )
            hooks.after_method_invocation(
                agent=agent,
                method_name="helper",
                result="ok",
                exception=None,
                context=ctx,
                invocation_id="inv-001",
            )
            provider.force_flush()

            spans = self._read_spans(tmpdir)
            inv_spans = [s for s in spans if s["name"] == "method_call.helper"]
            assert len(inv_spans) == 1
            assert inv_spans[0]["attributes"][VIEWER_PLUGIN_ATTR] == ViewerPlugin.METHOD

    def test_tool_execution_sets_tool_execution_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks, provider = self._make_hooks(tmpdir)
            agent = self._make_agent()

            ctx = hooks.before_tool_execution(
                agent=agent,
                tool_name="return_result",
                arguments={"value": 42},
                execution_id="texec-001",
            )
            hooks.after_tool_execution(
                agent=agent,
                tool_name="return_result",
                arguments={"value": 42},
                result=42,
                exception=None,
                context=ctx,
                execution_id="texec-001",
            )
            provider.force_flush()

            spans = self._read_spans(tmpdir)
            tool_spans = [s for s in spans if s["name"] == "tool_execution.return_result"]
            assert len(tool_spans) == 1
            assert tool_spans[0]["attributes"][VIEWER_PLUGIN_ATTR] == ViewerPlugin.TOOL_EXECUTION

    def test_context_snapshot_has_no_plugin_attr(self):
        """context_snapshot spans should NOT have the plugin attribute (falls back to SpanPlugin)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks, provider = self._make_hooks(tmpdir)
            agent = self._make_agent()

            ctx = hooks.before_agent_call(
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
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ],
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
            assert VIEWER_PLUGIN_ATTR not in snapshot_spans[0]["attributes"]
