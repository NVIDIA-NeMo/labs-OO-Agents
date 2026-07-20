# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OTLP JSON file exporter.

Writes spans as standard OTLP JSON (``TracesData``) objects, one per line, to
JSONL files.  This follows the `OTLP File Exporter specification
<https://opentelemetry.io/docs/specs/otel/protocol/file-exporter/>`_.

Session-based file routing: spans are written to
``{trace_dir}/{session_id}.jsonl`` based on the ``session.id`` span attribute
(stamped by :class:`SessionSpanProcessor`).
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from nooa.tracing._otlp_serialize import build_resource_spans

log = logging.getLogger(__name__)


class OtlpJsonFileExporter(SpanExporter):
    """Export spans as OTLP JSONL files.

    Each line in the output file is a complete ``TracesData`` JSON object::

        {"resourceSpans":[{"resource":{...},"scopeSpans":[...]}]}

    File routing uses the ``session.id`` span attribute:

    - Spans with ``session.id`` are written to ``{trace_dir}/{session_id}.jsonl``.
    - Spans without one fall back to a timestamp-based default file.
    """

    def __init__(self, trace_dir: str | Path):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._initialized_files: set[Path] = set()
        self._default_file: Path | None = None

    @property
    def default_file(self) -> Path | None:
        """Path to the timestamp-based file used for spans without session.id."""
        return self._default_file

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans, routing each to the file matching its ``session.id``."""
        try:
            by_file: dict[Path, list[ReadableSpan]] = {}
            for span in spans:
                target = self._target_file(span)
                by_file.setdefault(target, []).append(span)

            for target_file, file_spans in by_file.items():
                self._ensure_file(target_file)
                resource_spans = build_resource_spans(file_spans)
                payload = {"resourceSpans": resource_spans}
                line = json.dumps(payload, separators=(",", ":"), default=str) + "\n"
                with self._lock, open(target_file, "a") as fh:
                    fh.write(line)
        except Exception as e:
            log.warning("Error exporting spans: %s", e, exc_info=True)
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def _target_file(self, span: ReadableSpan) -> Path:
        """Determine the output file based on the span's ``session.id``."""
        session_id = (span.attributes or {}).get("session.id")
        if session_id:
            return self.trace_dir / f"{session_id}.jsonl"
        # Lazy-create a fallback file for spans without session.id
        if self._default_file is None:
            with self._lock:
                if self._default_file is None:
                    from datetime import UTC

                    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                    self._default_file = self.trace_dir / f"{timestamp}.jsonl"
        return self._default_file

    def _ensure_file(self, path: Path) -> None:
        """Create the file if it doesn't exist yet (thread-safe)."""
        with self._lock:
            if path not in self._initialized_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                self._initialized_files.add(path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def describe(self, session_id: str | None = None) -> str:
        """Human-readable description of this exporter."""
        name = f"{session_id}.jsonl" if session_id else "*.jsonl"
        return f"otlp-file:{self.trace_dir / name}"

    def shutdown(self) -> None:
        with self._lock:
            self._initialized_files.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        # No-op: export() opens files in append mode and closes them immediately,
        # so all data is flushed to disk on each export() call.
        return True
