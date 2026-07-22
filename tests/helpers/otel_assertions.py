# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry assertion helpers - STUB.

OpenTelemetry tracing has been moved to an external instrumentation package
(openinference-instrumentation-nemo-oo-agents). This module is a stub that marks
tests as skipped until the instrumentation package is integrated.
"""

import pytest

# Flag to skip tests that depend on OTel
OTEL_TRACING_AVAILABLE = False


def _skip_otel_test():
    """Skip test because OTel tracing is not available."""
    pytest.skip("OTel tracing moved to external instrumentation package")


def setup_test_tracing():
    """Setup test tracing - skips test."""
    _skip_otel_test()


def get_test_spans(exporter):
    """Get test spans - skips test."""
    _skip_otel_test()


def clear_test_spans(exporter):
    """Clear test spans - no-op."""
    pass  # No-op for fixture compatibility


def count_spans_by_name(spans, name):
    """Count spans - skips test."""
    _skip_otel_test()


def get_span_by_name(spans, name):
    """Get span - skips test."""
    _skip_otel_test()


def get_spans_by_name(spans, name):
    """Get spans - skips test."""
    _skip_otel_test()


def assert_span_count(spans, name, expected_count):
    """Assert span count - skips test."""
    _skip_otel_test()


def assert_span_attribute(span, key, expected_value):
    """Assert span attribute - skips test."""
    _skip_otel_test()


def assert_span_has_attribute(span, key):
    """Assert span has attribute - skips test."""
    _skip_otel_test()


def get_span_events(span, event_name=None):
    """Get span events - skips test."""
    _skip_otel_test()


def assert_parent_child_relationship(parent, child):
    """Assert parent-child - skips test."""
    _skip_otel_test()
