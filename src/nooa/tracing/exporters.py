# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exporter factory functions.

Each factory is a pure function that returns a ``SpanExporter`` instance.
No global state, no TracerProvider setup -- that's :func:`enable_tracing`'s job.

Usage::

    from openinference_instrumentation_nooa import enable_tracing, exporters

    enable_tracing(exporters=[exporters.jsonl("./traces")])
    enable_tracing(exporters=[exporters.otlp("http://localhost:4318/v1/traces")])
    enable_tracing(exporters=[exporters.jsonl(), exporters.langfuse()])
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from opentelemetry.sdk.trace.export import SpanExporter


def jsonl(trace_dir: str | Path | None = None) -> SpanExporter:
    """OTLP JSONL file exporter.

    Writes standard OTLP JSON (``TracesData``) objects, one per line, following
    the `OTLP File Exporter specification`_.  Session-based file routing writes
    to ``{trace_dir}/{session_id}.jsonl``.

    Auto-detects *trace_dir* when ``None``:
    ``TRACE_DIR`` env var → ``./traces/``.

    .. _OTLP File Exporter specification:
       https://opentelemetry.io/docs/specs/otel/protocol/file-exporter/
    """
    from nooa.tracing._otlp_file_exporter import OtlpJsonFileExporter

    if trace_dir is None:
        trace_dir = _auto_detect_trace_dir()
    return OtlpJsonFileExporter(trace_dir)


def journal_file(trace_dir: str | Path | None = None) -> SpanExporter:
    """Portable file exporter using NOOA's content-addressed message journal.

    Writes ``<session_id>.nooa.jsonl`` with message-stripped standard OTLP span
    envelopes plus typed journal block/call records.  The artifact is accepted
    by ``nooa import-traces`` and ``nooa import-harbor`` and reconstructs full
    OpenInference messages in the viewer.  This exporter is opt-in and does not
    alter the behavior of :func:`jsonl` or the default live viewer exporter.
    """
    from nooa.tracing._journal_file_exporter import JournalFileExporter

    if trace_dir is None:
        trace_dir = _auto_detect_trace_dir()
    return JournalFileExporter(trace_dir)


def otlp(
    endpoint: str,
    headers: dict[str, str] | None = None,
) -> SpanExporter:
    """OTLP HTTP exporter for third-party collectors (Jaeger, Tempo, Phoenix, etc.).

    Raises :class:`ImportError` with install instructions if the optional
    ``opentelemetry-exporter-otlp-proto-http`` package is missing.
    """
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError as err:
        raise ImportError(
            "OTLP HTTP exporter not installed.  Install with: pip install opentelemetry-exporter-otlp-proto-http"
        ) from err

    return OTLPSpanExporter(endpoint=endpoint, headers=headers or {})


def langfuse(
    host: str | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
) -> SpanExporter:
    """OTLP exporter pre-configured for Langfuse.

    Reads ``LANGFUSE_HOST``, ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``
    env vars when the corresponding arguments are ``None``.
    """
    import base64

    host = host or os.getenv("LANGFUSE_HOST")
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")

    if not all([host, public_key, secret_key]):
        raise ValueError(
            "Langfuse requires host, public_key, and secret_key. "
            "Provide as arguments or set LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY env vars."
        )

    assert host is not None and public_key is not None and secret_key is not None
    endpoint = f"{host.rstrip('/')}/api/public/otel/v1/traces"
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return otlp(endpoint=endpoint, headers={"Authorization": f"Basic {auth}"})


def local_otlp(
    endpoint: str | None = None,
) -> SpanExporter:
    """Lightweight OTLP JSON/HTTP exporter for the local nooa viewer.

    Zero external dependencies (uses ``urllib``).  Defaults to
    ``OTLP_ENDPOINT`` env var or ``http://localhost:5001/v1/traces``.
    """
    from nooa.tracing._otlp_http_exporter import OtlpJsonHttpExporter

    endpoint = endpoint or os.getenv("OTLP_ENDPOINT", "http://localhost:5001/v1/traces")
    return OtlpJsonHttpExporter(endpoint=endpoint)


def journal(
    endpoint: str | None = None,
) -> SpanExporter:
    """Content-addressed journal exporter for the local nooa viewer.

    Sends spans to the viewer with ``llm.input_messages.*`` /
    ``llm.output_messages.*`` attributes stripped, and installs a litellm
    ``CustomLogger`` that posts only new (delta) messages to the viewer's
    journal endpoints.  The viewer reconstructs full messages from journal
    hashes, reducing storage from O(N²) to O(delta).

    This is the **default** exporter for the local viewer path.  Use
    :func:`local_otlp` instead if you want traditional full-message spans
    (e.g. when sending to Phoenix or Langfuse alongside the viewer).

    Defaults to ``OTLP_ENDPOINT`` env var or ``http://localhost:5001``.
    """
    from nooa.tracing._journal_exporter import JournalExporter

    base = endpoint or os.getenv("OTLP_ENDPOINT", "http://localhost:5001/v1/traces")
    parsed = urlparse(base)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    return JournalExporter(base_url)


def console() -> SpanExporter:
    """Console exporter -- prints spans to stdout. Useful for debugging."""
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter

    return ConsoleSpanExporter()


# NOTE: The old OTel-based ``exporters.atif(...)`` factory was removed when
# ATIF export became event-subscriber-based; use
# :func:`nooa.atif.install_atif` or :func:`nooa.atif.atif_scope`.


def _auto_detect_trace_dir() -> Path:
    """Auto-detect trace directory.

    Resolution order: ``TRACE_DIR`` env var → ``./traces/``.
    """
    env_dir = os.getenv("TRACE_DIR")
    if env_dir:
        return Path(env_dir)

    return Path("./traces")
