# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PurePythonStrategy._strip_wrappers metric emission.

The nested_wrapper_iteration metric should fire ONLY on genuine nesting,
not on single-wrapper inputs (R3 bug fix).
"""

import pytest

from nooa.runtime.harness_metrics import (
    _NULL_METRICS,
    _harness_metrics_var,
    start_harness_metrics,
)
from nooa.strategies.pure_python import PurePythonStrategy


@pytest.fixture(autouse=True)
def _clean_harness_metrics():
    token = _harness_metrics_var.set(_NULL_METRICS)
    yield
    _harness_metrics_var.reset(token)


class TestStripWrappersMetric:
    """Verify nested_wrapper_iteration fires only on genuine nesting."""

    def test_single_fence_wrapper_does_not_record_nested(self):
        """A plain `lang\\ncode\\n` is NOT nesting — metric must NOT fire."""
        strategy = PurePythonStrategy()
        hm, _ = start_harness_metrics()
        try:
            strategy._strip_wrappers("```python\nx = 1\n```")
            assert hm.nested_wrapper_iterations == 0
        finally:
            pass

    def test_single_xml_wrapper_does_not_record_nested(self):
        """A plain <tag>code</tag> is NOT nesting — metric must NOT fire."""
        strategy = PurePythonStrategy()
        hm, _ = start_harness_metrics()
        try:
            strategy._strip_wrappers("<tool_code>x = 1</tool_code>")
            assert hm.nested_wrapper_iterations == 0
        finally:
            pass

    def test_plain_code_does_not_record_nested(self):
        strategy = PurePythonStrategy()
        hm, _ = start_harness_metrics()
        try:
            strategy._strip_wrappers("x = 1")
            assert hm.nested_wrapper_iterations == 0
        finally:
            pass

    def test_xml_wrapping_fence_records_nested(self):
        """XML-wrapped fence requires 2 strips — metric MUST fire."""
        strategy = PurePythonStrategy()
        hm, _ = start_harness_metrics()
        try:
            strategy._strip_wrappers("<tool_code>```python\nx = 1\n```</tool_code>")
            assert hm.nested_wrapper_iterations >= 2
        finally:
            pass
