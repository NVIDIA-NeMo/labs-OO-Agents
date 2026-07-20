# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for _safe_serialize — bounded span attribute serialization."""

from nooa.tracing._hooks_impl import OpenInferenceHooks


class TestSafeSerialize:
    """_safe_serialize delegates to safe_pformat with a hard cap."""

    def test_small_string_preserved(self):
        result = OpenInferenceHooks._safe_serialize("hello")
        assert result == "hello"

    def test_small_int_preserved(self):
        result = OpenInferenceHooks._safe_serialize(42)
        assert result == "42"

    def test_none_serialized(self):
        result = OpenInferenceHooks._safe_serialize(None)
        assert result == "None"

    def test_dict_serialized(self):
        result = OpenInferenceHooks._safe_serialize({"key": "value"})
        assert "key" in result
        assert "value" in result

    def test_large_string_passes_through_verbatim(self):
        """Block-level string truncation has been removed. Strings now pass
        through ``_safe_serialize`` verbatim — wire-protocol concerns (OTLP
        size limits) are a separate follow-up."""
        large = "x" * 200_000
        result = OpenInferenceHooks._safe_serialize(large, max_chars=1000)
        assert result == large

    def test_large_object_truncated(self):
        """Non-strings still get the OOM-safety net via TruncatingStringIO."""
        big_list = list(range(100_000))
        result = OpenInferenceHooks._safe_serialize(big_list, max_chars=5000)
        assert len(result) < 10_000  # cap + notice overhead

    def test_large_nested_trace_payload_is_structurally_bounded(self):
        """Trace attrs bound container breadth + string size before the global cap.

        Without trace-specific ``max_length``/``max_string`` options, this kind
        of payload is fully walked and materialized into a multi-MB string before
        the final 50 KB cap is applied.
        """
        obj = {"files": [{"name": f"f{i}", "content": "x" * 100_000} for i in range(40)]}

        result = OpenInferenceHooks._safe_serialize(obj, max_chars=50_000)

        assert "list(len=40" in result
        assert "str(len=100000" in result
        assert "f0" in result
        assert "f39" in result
        assert len(result) < 50_000

    def test_default_cap_is_50k(self):
        """Values under 50K are not truncated by default."""
        medium = "y" * 40_000
        result = OpenInferenceHooks._safe_serialize(medium)
        assert result == medium

    def test_broken_repr_does_not_crash(self):
        """Objects with broken __repr__ are handled gracefully."""

        class Bomb:
            def __repr__(self):
                raise RuntimeError("boom")

        result = OpenInferenceHooks._safe_serialize(Bomb())
        assert isinstance(result, str)  # doesn't crash

    def test_deeply_nested_object_does_not_overflow_stack(self):
        """Deeply nested objects are depth-limited to prevent stack overflow (segfault)."""
        # Build a 500-level deep nested dict — well beyond any reasonable stack
        obj: dict = {}
        current = obj
        for _i in range(500):
            current["nested"] = {}
            current = current["nested"]
        current["leaf"] = "value"

        result = OpenInferenceHooks._safe_serialize(obj)
        assert isinstance(result, str)
        # Should not contain 500 levels — depth is capped
        from nooa.tracing._hooks_impl import _TRACE_MAX_DEPTH

        assert result.count("nested") <= _TRACE_MAX_DEPTH

    def test_pydantic_model(self):
        """Pydantic models are serialized via safe_pformat."""
        from nooa.events import ExecutionResult

        result_obj = ExecutionResult(stdout="hello", stderr="", returned_value=42)
        serialized = OpenInferenceHooks._safe_serialize(result_obj)
        assert "hello" in serialized
        assert "42" in serialized
