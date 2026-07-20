# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TDD: gl-74 — _safe_serialize_execution_result caps returned_value before JSON.

Change 4 of truncation-2.0: Use truncating_pformat(rv, max_chars=50_000) and remove
the post-JSON string slicing that produces invalid JSON on large return values.
"""

import json

from nooa.tracing._hooks_impl import OpenInferenceHooks


class TestGl74ReturnedValueCap:
    """_safe_serialize_execution_result must cap returned_value, produce valid JSON."""

    def setup_method(self):
        self.hooks = OpenInferenceHooks.__new__(OpenInferenceHooks)

    def _make_result(self, returned_value):
        from nooa.events import ExecutionResult

        return ExecutionResult(stdout="", stderr="", returned_value=returned_value)

    def test_small_return_value_serializes_to_valid_json(self):
        result = self._make_result({"answer": 42})
        serialized = self.hooks._safe_serialize_execution_result(result)
        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)

    def test_large_string_return_value_passes_through_verbatim(self):
        """Block-level string truncation removed; large strings flow through.
        Wire-protocol concerns (OTLP payload size) are a follow-up."""
        huge = "x" * 200_000
        result = self._make_result(huge)
        serialized = self.hooks._safe_serialize_execution_result(result)
        parsed = json.loads(serialized)  # Must still be valid JSON
        rv = parsed.get("returned_value", "")
        assert rv == huge

    def test_large_list_return_value_is_capped(self):
        """Non-string return values still get the OOM-safety net."""
        big = list(range(100_000))
        result = self._make_result(big)
        serialized = self.hooks._safe_serialize_execution_result(result)
        parsed = json.loads(serialized)
        rv = parsed.get("returned_value", "")
        # Non-string render is bounded by TruncatingStringIO at max_chars
        assert len(rv) <= 55_000

    def test_post_json_slicing_gone_output_is_valid_json(self):
        # TDD: will fail until Change 4 is implemented
        # Old code did s[:50000] on the JSON string — breaks JSON syntax.
        # New code caps returned_value before serialization → always valid JSON.
        huge = list(range(500_000))
        result = self._make_result(huge)
        serialized = self.hooks._safe_serialize_execution_result(result)
        # Must parse without error
        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)

    def test_none_returned_value_not_in_output(self):
        from nooa.events import ExecutionResult

        result = ExecutionResult(stdout="", stderr="")
        serialized = self.hooks._safe_serialize_execution_result(result)
        parsed = json.loads(serialized)
        assert "returned_value" not in parsed or parsed["returned_value"] is None
