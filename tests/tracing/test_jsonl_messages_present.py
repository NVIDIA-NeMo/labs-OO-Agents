# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T1: a saved OTLP JSONL trace contains the full LLM messages on its spans.

Invariant: a saved JSONL trace is self-contained.  Anyone reading it (a CI
artifact reader, a local file viewer, ``import-traces`` to a fresh DB) must
see the actual input/output messages of every LLM call as
``llm.input_messages.*`` / ``llm.output_messages.*`` attributes on the
corresponding ``LLM`` span -- without needing the journal sideband.

Today this passes because OpenInference's ``LiteLLMInstrumentor`` stamps the
attributes on the span as a side-effect of ``litellm.acompletion``.  The MR
!128 journal protocol is allowed to *strip* these on the HTTP wire (and
reconstruct them on the receiver/download side), but it must not strip them
on the actor side -- otherwise the file path silently loses content.

This test is the regression guard for that invariant.  If a future change
moves the strip up to the actor or above the file exporter, this test fails.
"""

from __future__ import annotations

import json
import tempfile

import pytest
from otlp_test_helpers import read_all_otlp_jsonl_spans

from nooa.tracing import enable_tracing, exporters, flush_traces, set_session


@pytest.mark.asyncio
async def test_saved_jsonl_has_input_and_output_messages_on_llm_span():
    """A direct ``litellm.acompletion`` call with a file exporter must produce
    a JSONL whose LLM span carries the full input + output messages."""
    pytest.importorskip(
        "openinference.instrumentation.litellm",
        reason=(
            "openinference-instrumentation-litellm is required to populate "
            "llm.input_messages.* / llm.output_messages.* on LLM spans. "
            "The actor->file path relies on it; if it's not installed the "
            "file output is incomplete. Make it a hard dep (Phase 2)."
        ),
    )
    import litellm

    with tempfile.TemporaryDirectory() as tmpdir:
        enable_tracing(exporters=[exporters.jsonl(tmpdir)])
        set_session("t1-jsonl-messages")

        # litellm.acompletion with mock_response triggers the OpenInference
        # litellm instrumentor (which writes message attrs to spans) without
        # any network call.
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "T1_INPUT_MARKER what is 2+2?"},
            ],
            mock_response="T1_OUTPUT_MARKER 4",
        )

        # Force flush so SimpleSpanProcessor writes everything.
        flush_traces()

        spans = read_all_otlp_jsonl_spans(tmpdir)
        assert spans, f"no spans written to {tmpdir}"

        llm_spans = [s for s in spans if s["attributes"].get("openinference.span.kind") == "LLM"]
        assert llm_spans, f"no LLM spans in saved JSONL; got names {[s.get('name') for s in spans]}"

        gen = llm_spans[0]
        attr_keys = set(gen["attributes"].keys())
        input_keys = {k for k in attr_keys if k.startswith("llm.input_messages.")}
        output_keys = {k for k in attr_keys if k.startswith("llm.output_messages.")}

        assert input_keys, (
            "saved JSONL is missing llm.input_messages.* on the LLM span; "
            "file traces must be self-contained -- the journal sideband "
            "is an HTTP wire optimization, not a replacement for span "
            "attributes. attrs present: "
            f"{sorted(attr_keys)}"
        )
        assert output_keys, (
            "saved JSONL is missing llm.output_messages.* on the LLM span. "
            f"attrs present: {sorted(attr_keys)}"
        )

        # The actual content -- not just the keys -- must round-trip.
        flat = json.dumps(gen["attributes"])
        assert "T1_INPUT_MARKER" in flat, "input message content stripped from saved JSONL"
        assert "T1_OUTPUT_MARKER" in flat, "output message content stripped from saved JSONL"
