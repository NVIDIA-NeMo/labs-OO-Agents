# SPDX-License-Identifier: Apache-2.0
"""JSONL traces retain native LLM lifecycle message attributes."""

from __future__ import annotations

import json
import tempfile

from otlp_test_helpers import read_all_otlp_jsonl_spans

from nooa.runtime.llm_lifecycle import begin_llm_call, end_llm_call
from nooa.tracing import enable_tracing, exporters, flush_traces, set_session
from nooa.unifiedllm import LLMResponse


def test_saved_jsonl_has_input_and_output_messages_on_llm_span():
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "T1_INPUT_MARKER what is 2+2?"},
    ]
    response = LLMResponse(
        content="T1_OUTPUT_MARKER 4",
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": "T1_OUTPUT_MARKER 4"},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        enable_tracing(exporters=[exporters.jsonl(tmpdir)])
        set_session("t1-jsonl-messages")
        call = begin_llm_call("test-model", messages)
        end_llm_call(call, response=response)
        flush_traces()
        spans = read_all_otlp_jsonl_spans(tmpdir)
        llm_spans = [s for s in spans if s["attributes"].get("openinference.span.kind") == "LLM"]
        assert llm_spans
        assert llm_spans[0]["name"] == "llm.call"
        attrs = llm_spans[0]["attributes"]
        assert attrs["nooa.viewer.plugin"] == "llm.call"
        # Missing provider usage remains missing; tracing must not estimate totals.
        assert "llm.token_count.prompt" not in attrs
        assert "llm.token_count.completion" not in attrs
        assert "llm.token_count.total" not in attrs
        assert any(k.startswith("llm.input_messages.") for k in attrs)
        assert any(k.startswith("llm.output_messages.") for k in attrs)
        flat = json.dumps(attrs)
        assert "T1_INPUT_MARKER" in flat
        assert "T1_OUTPUT_MARKER" in flat
