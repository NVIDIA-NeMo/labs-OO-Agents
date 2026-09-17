# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for extract_and_parse_json robustness."""

import json

import pytest

from nooa.unifiedllm import extract_and_parse_json
from nooa.unifiedllm.unifiedllm import _llm_metrics_callback


class TestExtractAndParseJson:
    def test_plain_json(self):
        """Clean JSON is parsed without modification."""
        result = extract_and_parse_json('{"answer": "42", "confidence": 0.9}')
        assert result == {"answer": "42", "confidence": 0.9}

    def test_full_response_json_fence(self):
        """A full-response JSON fence is extracted, parsed, and recorded."""
        text = '```json\n{"answer": "42"}\n```'
        events = []
        token = _llm_metrics_callback.set(lambda event, detail: events.append(event))
        try:
            result = extract_and_parse_json(text)
        finally:
            _llm_metrics_callback.reset(token)

        assert result == {"answer": "42"}
        assert events == ["json_fence_removed"]

    def test_full_response_bare_fence(self):
        """A full-response bare fence is extracted and parsed."""
        text = '```\n{"answer": "42"}\n```'
        result = extract_and_parse_json(text)
        assert result == {"answer": "42"}

    def test_bash_fence_does_not_mask_later_json(self):
        """A non-JSON fence in prose does not replace the complete response."""
        text = """Run the example:

```bash
uv run python scripts/run_local.py
```

Result: {"answer": "42"}
"""
        result = extract_and_parse_json(text)
        assert result == {"answer": "42"}

    def test_full_response_bash_fence_raises_for_complete_input(self):
        """A non-JSON fence is not mistaken for a JSON wrapper."""
        text = """```bash
uv run python scripts/run_local.py
```"""
        with pytest.raises(json.JSONDecodeError) as exc_info:
            extract_and_parse_json(text)

        assert exc_info.value.doc.startswith("```bash")
        assert "uv run python scripts/run_local.py" in exc_info.value.doc

    def test_markdown_bold_prefix(self):
        """LLMs sometimes wrap JSON in markdown bold markers."""
        text = '**{"answer": "42", "confidence": 0.9}**'
        result = extract_and_parse_json(text)
        assert result == {"answer": "42", "confidence": 0.9}

    def test_markdown_bold_prefix_only(self):
        """Double-star bold wrapping is stripped before parsing."""
        text = '**{"answer": "hello"}**'
        result = extract_and_parse_json(text)
        assert result == {"answer": "hello"}

    def test_single_star_prefix(self):
        """Single-star italic wrapping is stripped before parsing."""
        text = '*{"answer": "hello"}*'
        result = extract_and_parse_json(text)
        assert result == {"answer": "hello"}

    def test_bold_with_whitespace(self):
        """Bold markers with surrounding whitespace are stripped."""
        text = '** {"answer": "hello"} **'
        result = extract_and_parse_json(text)
        assert result == {"answer": "hello"}

    def test_nested_json_extraction(self):
        """JSON embedded in prose is extracted via nested-object regex."""
        text = 'Here is the answer: {"answer": "42"} and more text'
        result = extract_and_parse_json(text)
        assert result == {"answer": "42"}

    def test_empty_text_raises(self):
        """Empty input raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            extract_and_parse_json("")

    def test_unparseable_raises(self):
        """Non-JSON text raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            extract_and_parse_json("this is not json at all")
