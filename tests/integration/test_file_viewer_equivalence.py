# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T2: file save and viewer download produce equivalent OTLP.

Invariant: journaling is an HTTP wire/storage optimization, invisible to
consumers.  Whether a trace lands on disk via ``exporters.jsonl`` or is
downloaded from the viewer via ``GET /api/trace/export``, the resulting OTLP
must be the same -- in particular, every LLM span carries full
``llm.input_messages.*`` / ``llm.output_messages.*`` content.

If this test fails:
- The wire/sideband stripped messages and the receiver's download path didn't
  reconstruct them; ``export_session_otlp`` needs the same augmentation that
  ``get_session_spans`` already does for the UI render path.
- Or block content never landed on the receiver (missing
  ``/v1/journal/blocks`` endpoint), so reconstruction can't find content for
  the hashes it sees.
"""

from __future__ import annotations

import json
import tempfile
import urllib.request

import pytest
from otlp_test_helpers import read_all_otlp_jsonl_spans

from nooa.tracing import enable_tracing, exporters, set_session


def _gather_otlp_from_jsonl(text: str) -> dict[str, dict]:
    """Return ``{span_id: span_dict}`` from a raw OTLP JSONL string."""
    spans: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for rs in json.loads(line).get("resourceSpans", []):
            for ss in rs.get("scopeSpans", []):
                for sp in ss.get("spans", []):
                    sid = sp.get("spanId")
                    if sid:
                        spans[sid] = sp
    return spans


@pytest.fixture
def live_viewer():
    """Start a real viewer-equivalent HTTP server (FastAPI + trace routes) on
    an ephemeral port, backed by a fresh temp DB.

    Reuses ``HeadlessOtlpBackend`` because it mounts the same trace_router
    the production viewer does and exposes ``/v1/traces``,
    ``/v1/journal/*`` (whatever's registered), and ``/api/trace/export``.
    """
    from eval_pipeline.headless_backend import HeadlessOtlpBackend

    backend = HeadlessOtlpBackend()
    base_url = backend.start()
    try:
        yield base_url
    finally:
        backend.stop()


@pytest.mark.asyncio
async def test_file_save_equals_viewer_download(live_viewer, monkeypatch):
    """One LLM call -> file via jsonl exporter + viewer via journal exporter.
    The downloaded OTLP from the viewer must contain the same span attributes
    (including ``llm.input_messages.*`` / ``llm.output_messages.*``) as the
    saved file.

    The native lifecycle creates the LLM span and journal record once; both
    exporters must preserve equivalent message attributes.
    """
    from nooa.runtime.llm_lifecycle import begin_llm_call, end_llm_call
    from nooa.unifiedllm import LLMResponse

    with tempfile.TemporaryDirectory() as tmpdir:
        # Both sinks active simultaneously: file + journal-to-viewer.
        enable_tracing(
            exporters=[
                exporters.jsonl(tmpdir),
                exporters.journal(endpoint=f"{live_viewer}/v1/traces"),
            ]
        )
        session_id = "t2-file-vs-viewer"
        set_session(session_id)

        messages = [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "T2_INPUT_MARKER hello?"},
        ]
        call = begin_llm_call("test-model", messages)
        outer_span_id_hex = format(call.context["span"].get_span_context().span_id, "016x")
        response = LLMResponse(
            content="T2_OUTPUT_MARKER hi",
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": "T2_OUTPUT_MARKER hi"},
        )
        end_llm_call(call, response=response)

        from nooa.tracing import _provider

        assert _provider is not None
        _provider.force_flush()

        # Sync the viewer's OTLP ingest queue.
        urllib.request.urlopen(
            urllib.request.Request(f"{live_viewer}/v1/sync", method="POST"),
            timeout=10,
        )

        # _post_json fires-and-forgets in daemon threads.  Poll until the
        # journal call record has landed so the export can resolve it.
        import logging
        import time

        log = logging.getLogger(__name__)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{live_viewer}/api/traces/{session_id}/calls",
                    timeout=2,
                ) as r:
                    body = json.loads(r.read().decode())
                if body:
                    break
            except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
                # Transient HTTP / serialization failure during the brief
                # window before the journal POST has been committed.  Log
                # at debug so a real persistent failure is visible
                # instead of silently looping until the deadline.
                log.debug("polling /api/traces/.../calls failed: %s", exc)
            time.sleep(0.05)

        # File side: read the saved JSONL.
        file_spans_flat = read_all_otlp_jsonl_spans(tmpdir)
        file_spans = {s["spanId"]: s for s in file_spans_flat if s.get("spanId")}
        assert file_spans, f"no spans written to {tmpdir}"

        # Viewer side: GET the export.
        with urllib.request.urlopen(
            f"{live_viewer}/api/trace/export?session_id={session_id}",
            timeout=10,
        ) as r:
            viewer_text = r.read().decode()
        viewer_spans = _gather_otlp_from_jsonl(viewer_text)

        # The user-facing invariant is "the LLM span I called the model on
        # has the same message attrs in the file and in the viewer
        # download."  In production a real call is wrapped by exactly one
        # OpenInference instrumentor span per call, so this is unambiguous.
        # The native lifecycle span id correlates the file and journal record.
        file_flat_by_id = {s["spanId"]: s for s in file_spans_flat if s.get("spanId")}
        common_ids = {outer_span_id_hex} & set(file_spans) & set(viewer_spans)
        assert common_ids, (
            f"outer LLM span {outer_span_id_hex!r} missing from one side.\n"
            f"file ids:   {sorted(file_spans)}\n"
            f"viewer ids: {sorted(viewer_spans)}"
        )

        for sid in sorted(common_ids):
            file_attrs = file_flat_by_id[sid]["attributes"]
            viewer_attrs = {
                a["key"]: a.get("value") for a in viewer_spans[sid].get("attributes", [])
            }

            # The whole point of this test: every llm.input_messages.* and
            # llm.output_messages.* the file has, the viewer download must
            # also have, with byte-equivalent content.
            file_msg_keys = {
                k
                for k in file_attrs
                if k.startswith("llm.input_messages.") or k.startswith("llm.output_messages.")
            }
            viewer_msg_keys = {
                k
                for k in viewer_attrs
                if k.startswith("llm.input_messages.") or k.startswith("llm.output_messages.")
            }
            assert file_msg_keys == viewer_msg_keys, (
                f"message-attribute key sets differ on span {sid} "
                f"(name={file_flat_by_id[sid].get('name')}).\n"
                f"  in file only: {sorted(file_msg_keys - viewer_msg_keys)}\n"
                f"  in viewer only: {sorted(viewer_msg_keys - file_msg_keys)}\n"
                f"This indicates the wire/journal layer stripped messages "
                f"and the viewer's /api/trace/export did not reconstruct them. "
                f"export_session_otlp must call the same augmentation that "
                f"get_session_spans uses for the UI render path."
            )
            for k in file_msg_keys:
                # File helper returns python-typed values; viewer returns
                # OTLP AnyValue dicts.  Compare via JSON round-trip.
                file_val = file_attrs[k]
                viewer_val = viewer_attrs[k]
                # Extract scalar from OTLP AnyValue.
                if isinstance(viewer_val, dict):
                    if "stringValue" in viewer_val:
                        viewer_scalar = viewer_val["stringValue"]
                    elif "intValue" in viewer_val:
                        viewer_scalar = int(viewer_val["intValue"])
                    elif "doubleValue" in viewer_val:
                        viewer_scalar = viewer_val["doubleValue"]
                    elif "boolValue" in viewer_val:
                        viewer_scalar = viewer_val["boolValue"]
                    else:
                        viewer_scalar = viewer_val
                else:
                    viewer_scalar = viewer_val
                assert file_val == viewer_scalar, (
                    f"span {sid} attribute {k} differs:\n"
                    f"  file:   {file_val!r}\n"
                    f"  viewer: {viewer_scalar!r}"
                )

        # Sanity: marker content ended up in the viewer download.
        assert "T2_INPUT_MARKER" in viewer_text, (
            "input message content missing from viewer download"
        )
        assert "T2_OUTPUT_MARKER" in viewer_text, (
            "output message content missing from viewer download"
        )
