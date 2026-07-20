# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for extract_and_parse_json robustness."""

import json

import pytest

from nooa.unifiedllm import extract_and_parse_json


class TestExtractAndParseJson:
    def test_plain_json(self):
        """Clean JSON is parsed without modification."""
        result = extract_and_parse_json('{"answer": "42", "confidence": 0.9}')
        assert result == {"answer": "42", "confidence": 0.9}

    def test_markdown_fenced_json(self):
        """JSON inside markdown code fences is extracted and parsed."""
        text = '```json\n{"answer": "42"}\n```'
        result = extract_and_parse_json(text)
        assert result == {"answer": "42"}

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
