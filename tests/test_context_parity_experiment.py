# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the deterministic context-parity comparator."""

from copy import deepcopy

from experiments.context_parity.run import canonicalize, compare_captures


def _capture(content: str) -> dict:
    return {
        "schema_version": 1,
        "scenarios": {
            "case": {
                "result": {"value": "ok"},
                "requests": [
                    {
                        "messages": [{"role": "system", "content": content}],
                        "tools": None,
                        "output_schema": None,
                        "options": {},
                    }
                ],
            }
        },
    }


def test_comparator_rejects_content_changes():
    result = compare_captures(_capture("policy A"), _capture("policy B"))
    assert not result["passed"]
    assert result["diff"]


def test_comparator_accepts_only_declared_incidental_differences():
    baseline = _capture("id=123e4567-e89b-42d3-a456-426614174000  \r\nnext")
    candidate = _capture("id=987e6543-e21b-42d3-a456-426614174999\nnext")
    result = compare_captures(baseline, candidate)
    assert result["passed"]
    assert not result["exact_bytes_equal"]


def test_comparator_accepts_legacy_context_envelope_formatting():
    baseline = _capture(
        "<context>\n<state>\nready\n</state>\n<dynamic>\nyes\n</dynamic>\n</context>"
    )
    candidate = _capture("<state>\nready\n</state>\n\n<dynamic>\nyes\n</dynamic>")
    assert compare_captures(baseline, candidate)["passed"]


def test_comparator_rejects_content_change_inside_legacy_envelope():
    baseline = _capture("<context>\n<state>\nready\n</state>\n</context>")
    candidate = _capture("<state>\nchanged\n</state>")
    assert not compare_captures(baseline, candidate)["passed"]


def test_canonicalize_does_not_mutate_capture():
    capture = _capture("123e4567-e89b-42d3-a456-426614174000")
    original = deepcopy(capture)
    canonicalize(capture)
    assert capture == original
