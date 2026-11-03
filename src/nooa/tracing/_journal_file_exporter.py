# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable, append-only file exporter for NOOA's message journal.

Each ``<session>.nooa.jsonl`` file contains standard OTLP ``TracesData``
envelopes with large LLM message attributes removed, plus typed NOOA journal
records containing content-addressed blocks and LLM call skeletons.  The file
is intentionally line-oriented: a process crash can lose at most the line
being written, and importers can stream artifacts without loading them whole.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from nooa.tracing._otlp_serialize import build_resource_spans

if TYPE_CHECKING:
    from nooa.tracing._litellm_journal import FileMessageJournalCallback

log = logging.getLogger(__name__)

JOURNAL_ENVELOPE_KEY = "nooaJournal"
JOURNAL_FORMAT = "nooa.message_journal"
JOURNAL_VERSION = 1


class JournalFileWriter:
    """Thread-safe writer shared by the span exporter and LiteLLM callback."""

    def __init__(self, trace_dir: str | Path) -> None:
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialized_files: set[Path] = set()
        self._default_session: str | None = None

    def path_for_session(self, session_id: str | None) -> Path:
        if not session_id:
            with self._lock:
                if self._default_session is None:
                    self._default_session = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                session_id = self._default_session
        return self.trace_dir / f"{session_id}.nooa.jsonl"

    def append(self, session_id: str | None, record: dict[str, Any]) -> None:
        path = self.path_for_session(session_id)
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False, default=str)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path not in self._initialized_files:
                if not path.exists() or path.stat().st_size == 0:
                    manifest = {
                        JOURNAL_ENVELOPE_KEY: {
                            "format": JOURNAL_FORMAT,
                            "version": JOURNAL_VERSION,
                            "type": "manifest",
                            "session_id": session_id or "",
                        }
                    }
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(manifest, separators=(",", ":")) + "\n")
                self._initialized_files.add(path)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def append_blocks(self, session_id: str, blocks: list[dict[str, str]]) -> None:
        if blocks:
            self.append(
                session_id,
                {
                    JOURNAL_ENVELOPE_KEY: {
                        "format": JOURNAL_FORMAT,
                        "version": JOURNAL_VERSION,
                        "type": "blocks",
                        "session_id": session_id,
                        "blocks": blocks,
                    }
                },
            )

    def append_call(self, session_id: str, call: dict[str, Any]) -> None:
        self.append(
            session_id,
            {
                JOURNAL_ENVELOPE_KEY: {
                    "format": JOURNAL_FORMAT,
                    "version": JOURNAL_VERSION,
                    "type": "call",
                    "session_id": session_id,
                    "call": call,
                }
            },
        )


class JournalFileExporter(SpanExporter):
    """Write stripped OTLP spans and their message journal to one JSONL file."""

    synchronous = True
    _LLM_MESSAGE_PREFIXES = ("llm.input_messages.", "llm.output_messages.")
    _LLM_VALUE_KEYS = ("input.value", "output.value")

    def __init__(self, trace_dir: str | Path) -> None:
        self.trace_dir = Path(trace_dir)
        self._writer = JournalFileWriter(trace_dir)
        self._callback = self._install_callback()

    def _install_callback(self) -> FileMessageJournalCallback:
        import litellm

        from nooa.tracing._litellm_journal import _INSTALL_LOCK, FileMessageJournalCallback

        with _INSTALL_LOCK:
            for callback in litellm.callbacks:
                if (
                    isinstance(callback, FileMessageJournalCallback)
                    and type(callback) is FileMessageJournalCallback
                ):
                    callback.add_writer(self._writer)
                    return callback
            callback = FileMessageJournalCallback(self._writer)
            litellm.callbacks.append(callback)
            return callback

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            by_session: dict[str | None, list[ReadableSpan]] = {}
            for span in spans:
                value = (span.attributes or {}).get("session.id")
                session_id = str(value) if value else None
                by_session.setdefault(session_id, []).append(span)

            for session_id, session_spans in by_session.items():
                override = {"session.id": session_id} if session_id else None
                resource_spans = build_resource_spans(
                    session_spans,
                    resource_attrs_override=override,
                    exclude_attr_prefixes=self._LLM_MESSAGE_PREFIXES,
                    exclude_attr_prefixes_llm_only=self._LLM_VALUE_KEYS,
                )
                self._writer.append(session_id, {"resourceSpans": resource_spans})
        except Exception as exc:
            log.warning("Error exporting journal file spans: %s", exc, exc_info=True)
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        import litellm

        from nooa.tracing._litellm_journal import _INSTALL_LOCK

        with _INSTALL_LOCK:
            self._callback.remove_writer(self._writer)
            if not self._callback.has_writers():
                litellm.callbacks = [c for c in litellm.callbacks if c is not self._callback]

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    def describe(self, session_id: str | None = None) -> str:
        name = f"{session_id}.nooa.jsonl" if session_id else "*.nooa.jsonl"
        return f"journal-file:{self.trace_dir / name}"
