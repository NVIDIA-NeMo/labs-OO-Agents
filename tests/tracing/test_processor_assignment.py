# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for span processor assignment in _add_exporters().

Every exporter is wrapped in a ``SecretScrubSpanProcessor`` (to redact secrets
before export); ``_processors`` unwraps it to assert the underlying contract:
  - OtlpJsonHttpExporter  -> BatchSpanProcessor  (non-blocking, batched)
  - OtlpJsonFileExporter  -> SimpleSpanProcessor (in-process I/O, synchronous)
  - ConsoleSpanExporter   -> SimpleSpanProcessor (in-process I/O, synchronous)
"""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from nooa.tracing import _add_exporters
from nooa.tracing._otlp_file_exporter import OtlpJsonFileExporter
from nooa.tracing._otlp_http_exporter import OtlpJsonHttpExporter
from nooa.tracing._secret_scrubber import SecretScrubSpanProcessor


def _processors(provider: TracerProvider):
    """Return the inner span processors, unwrapping the secret scrubber.

    Every attached processor must be a ``SecretScrubSpanProcessor``; this
    asserts that invariant and returns the wrapped (inner) processors so the
    exporter -> processor-type contract can be checked.
    """
    attached = list(provider._active_span_processor._span_processors)
    inner = []
    for p in attached:
        assert isinstance(p, SecretScrubSpanProcessor), (
            f"every processor must be wrapped in SecretScrubSpanProcessor, got {type(p).__name__}"
        )
        inner.append(p.inner)
    return inner


class TestProcessorAssignment:
    def test_http_exporter_gets_batch_processor(self):
        """OtlpJsonHttpExporter must use BatchSpanProcessor (non-blocking)."""
        provider = TracerProvider()
        exp = OtlpJsonHttpExporter()
        _add_exporters(provider, [exp])

        procs = _processors(provider)
        assert len(procs) == 1
        assert isinstance(procs[0], BatchSpanProcessor), (
            f"OtlpJsonHttpExporter must be wrapped in BatchSpanProcessor, got {type(procs[0]).__name__}"
        )

    def test_file_exporter_gets_simple_processor(self, tmp_path):
        """OtlpJsonFileExporter must use SimpleSpanProcessor (in-process I/O)."""
        provider = TracerProvider()
        exp = OtlpJsonFileExporter(tmp_path)
        _add_exporters(provider, [exp])

        procs = _processors(provider)
        assert len(procs) == 1
        assert isinstance(procs[0], SimpleSpanProcessor)
        assert procs[0].span_exporter is exp

    def test_console_exporter_gets_simple_processor(self):
        """ConsoleSpanExporter must use SimpleSpanProcessor."""
        provider = TracerProvider()
        exp = ConsoleSpanExporter()
        _add_exporters(provider, [exp])

        procs = _processors(provider)
        assert len(procs) == 1
        assert isinstance(procs[0], SimpleSpanProcessor)
        assert procs[0].span_exporter is exp

    def test_mixed_exporters_get_correct_processors(self, tmp_path):
        """A mix of exporter types each gets its appropriate processor."""
        provider = TracerProvider()
        http_exp = OtlpJsonHttpExporter()
        file_exp = OtlpJsonFileExporter(tmp_path)
        _add_exporters(provider, [http_exp, file_exp])

        procs = _processors(provider)
        assert len(procs) == 2

        batch_procs = [p for p in procs if isinstance(p, BatchSpanProcessor)]
        simple_procs = [p for p in procs if isinstance(p, SimpleSpanProcessor)]
        assert len(batch_procs) == 1
        assert len(simple_procs) == 1
        assert simple_procs[0].span_exporter is file_exp
