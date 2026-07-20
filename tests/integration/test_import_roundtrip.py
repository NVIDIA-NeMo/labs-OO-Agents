# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T6: save -> import-traces -> download is the identity transform.

The user-visible invariant is:

    "I should be able to save a jsonl directly while doing OTLP, then
    upload that jsonl to a fresh viewer with import-traces, then download
    it from the viewer, and the result should be the same as what I
    saved.  Journaling is an implementation detail of the wire."

This test pins exactly that round-trip:

  1.  enable_tracing(file + journal-to-viewer-A) -> one litellm.acompletion
      -> the file on disk has full messages on its LLM span (T1 covers
      that file is complete).
  2.  Spin up an *empty* viewer-B.
  3.  ``nooa import-traces <saved.jsonl> --endpoint <viewer-B>``
  4.  ``GET /api/trace/export?session_id=...`` from viewer-B.
  5.  Compare span-by-span with the original file: same span ids, same
      ``llm.input_messages.*`` / ``llm.output_messages.*``.

Today this fails for any session that originated from the journal-wire
path (because viewer-A's download is broken -- T2/T5).  But for files
written directly by ``exporters.jsonl``, this round-trip should work
end-to-end with no special handling, because the file already has full
messages on spans.

This test stays focused on the file -> import -> download edge so it
exercises ``import-traces`` plumbing specifically, not the full journal
chain.
"""

from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path

import pytest


@pytest.fixture
def fresh_viewer():
    """A real viewer-equivalent HTTP server on an ephemeral port + temp DB.
    Reuses ``HeadlessOtlpBackend`` for the same reason T2 does: same
    routes, lifecycle, isolation."""
    from eval_pipeline.headless_backend import HeadlessOtlpBackend

    backend = HeadlessOtlpBackend()
    base_url = backend.start()
    try:
        yield base_url
    finally:
        backend.stop()


@pytest.mark.asyncio
async def test_save_then_import_then_download_preserves_messages(fresh_viewer):
    """A JSONL written by the file exporter, then re-imported into a fresh
    viewer, must round-trip its full message content."""
    pytest.importorskip("openinference.instrumentation.litellm")
    import litellm

    from nooa.tracing import enable_tracing, exporters, set_session

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        enable_tracing(exporters=[exporters.jsonl(str(tmpdir))])

        session_id = "t6-roundtrip"
        set_session(session_id)

        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "T6_INPUT what is 2+2?"},
            ],
            mock_response="T6_OUTPUT 4",
        )

        from nooa.tracing import _provider

        assert _provider is not None
        _provider.force_flush()

        # 1. Locate the file the exporter wrote.
        files = list(tmpdir.glob("*.jsonl"))
        assert files, f"file exporter wrote nothing into {tmpdir}"
        saved = files[0]

        # 2. Capture the file's spans (truth source).
        file_bodies = [json.loads(line) for line in saved.read_text().splitlines() if line.strip()]
        file_spans = {
            sp.get("spanId"): sp
            for body in file_bodies
            for rs in body.get("resourceSpans", [])
            for ss in rs.get("scopeSpans", [])
            for sp in ss.get("spans", [])
            if sp.get("spanId")
        }
        assert file_spans, f"saved file has no spans: {saved}"
        llm_span_id = next(
            sid
            for sid, sp in file_spans.items()
            if any(
                a["key"] == "openinference.span.kind"
                and a.get("value", {}).get("stringValue") == "LLM"
                for a in sp.get("attributes", [])
            )
        )
        file_attrs = {
            a["key"]: a.get("value", {}).get("stringValue")
            for a in file_spans[llm_span_id].get("attributes", [])
        }
        assert "llm.input_messages.0.message.content" in file_attrs, (
            "T1 sanity precondition: saved file must have message attrs"
        )

        # 3. Import the file into a fresh viewer via the actual CLI command.
        from click.testing import CliRunner
        from nooa_cli.commands.import_traces import command

        # The file's basename becomes the session_id during import; rename
        # to a known session id we can query.
        target = tmpdir / f"{session_id}.jsonl"
        saved.rename(target)

        runner = CliRunner()
        result = runner.invoke(
            command,
            [str(target), "--endpoint", fresh_viewer],
        )
        assert result.exit_code == 0, (
            f"import-traces failed (exit={result.exit_code}):\nstdout:\n{result.output}"
        )

        # Sync the viewer so all queued spans are written before we query.
        urllib.request.urlopen(
            urllib.request.Request(f"{fresh_viewer}/v1/sync", method="POST"),
            timeout=10,
        )

        # 4. Download from the viewer.
        with urllib.request.urlopen(
            f"{fresh_viewer}/api/trace/export?session_id={session_id}",
            timeout=10,
        ) as r:
            downloaded_text = r.read().decode()

        downloaded_spans = {
            sp.get("spanId"): sp
            for line in downloaded_text.splitlines()
            if line.strip()
            for body in [json.loads(line)]
            for rs in body.get("resourceSpans", [])
            for ss in rs.get("scopeSpans", [])
            for sp in ss.get("spans", [])
            if sp.get("spanId")
        }
        assert llm_span_id in downloaded_spans, (
            f"viewer download is missing span {llm_span_id}; got: {sorted(downloaded_spans)}"
        )
        downloaded_attrs = {
            a["key"]: a.get("value", {}).get("stringValue")
            for a in downloaded_spans[llm_span_id].get("attributes", [])
        }

        # 5. The contract.  Every llm.input_messages.* / llm.output_messages.*
        # in the saved file must be present and equal in the download.
        msg_keys = [
            k
            for k in file_attrs
            if k.startswith("llm.input_messages.") or k.startswith("llm.output_messages.")
        ]
        assert msg_keys, "test precondition: file must have message attrs"
        for k in msg_keys:
            assert downloaded_attrs.get(k) == file_attrs[k], (
                f"round-trip lost or changed {k}:\n"
                f"  file:       {file_attrs[k]!r}\n"
                f"  downloaded: {downloaded_attrs.get(k)!r}\n"
                f"This means import-traces -> /v1/traces -> /api/trace/export "
                f"is not the identity transform on message attrs."
            )

        assert "T6_INPUT" in downloaded_text
        assert "T6_OUTPUT" in downloaded_text
