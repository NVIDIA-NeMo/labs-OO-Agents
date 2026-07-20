# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for function/lambda default rendering in doc()."""

from nooa.agentdoc import doc


def _my_validator(x: int) -> bool:
    return x > 0


def _build_default() -> str:
    return "hello"


class MyService:
    """A service with function and lambda defaults."""

    def process(self, value: int, validator=_my_validator) -> str:
        """Process value using validator."""
        return str(value)

    def transform(self, text: str, normalizer=_build_default) -> str:
        """Transform text."""
        return text

    def filter_items(self, items: list, key=lambda x: x) -> list:
        """Filter items using key function."""
        return items

    def with_none_default(self, x: int = 0) -> int:
        """Method with a plain int default (sanity check — not a function)."""
        return x


# ── named function defaults ───────────────────────────────────────────────────


def test_named_function_default_shows_name():
    result = doc(MyService())
    assert "_my_validator" in result


def test_named_function_default_not_ellipsis():
    result = doc(MyService())
    lines = [ln for ln in result.splitlines() if "def process" in ln]
    assert lines, "process method not found"
    assert "..." not in lines[0].split("=")[-1]


def test_second_named_function_default_shows_name():
    result = doc(MyService())
    assert "_build_default" in result


# ── lambda defaults ───────────────────────────────────────────────────────────


def test_lambda_default_shows_lambda_label():
    result = doc(MyService())
    lines = [ln for ln in result.splitlines() if "def filter_items" in ln]
    assert lines, "filter_items method not found"
    # Lambda should render as '<lambda>' not as '...'
    assert "<lambda>" in lines[0]


# ── non-function defaults unaffected ─────────────────────────────────────────


def test_plain_default_unaffected():
    result = doc(MyService())
    lines = [ln for ln in result.splitlines() if "def with_none_default" in ln]
    assert lines, "with_none_default method not found"
    assert "= 0" in lines[0]
