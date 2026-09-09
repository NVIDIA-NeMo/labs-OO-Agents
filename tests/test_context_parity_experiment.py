# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the deterministic context-parity comparator."""

import importlib
from copy import deepcopy

import experiments.context_parity.capture as capture_module
import nooa.agent as agent_module
import nooa.runtime.method_wrapper as method_wrapper
from experiments.context_parity.capture import _request_messages, _request_options
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


def test_capture_removes_only_exact_transport_metadata():
    messages = [
        {
            "role": "user",
            "content": {"_nooa_cache_boundary": "user data"},
            "_nooa_cache_boundary": True,
        }
    ]
    options = {
        "cache_control_injection_points": [],
        "user_cache_control_injection_points": ["keep"],
        "metadata": {"cache_control_injection_points": "keep"},
    }

    assert _request_options({"cache_control_injection_points": [{"role": "system"}]}) == {
        "cache_control_injection_points": [{"role": "system"}]
    }

    assert _request_messages(messages) == [
        {"role": "user", "content": {"_nooa_cache_boundary": "user data"}}
    ]
    assert _request_options(options) == {
        "user_cache_control_injection_points": ["keep"],
        "metadata": {"cache_control_injection_points": "keep"},
    }


def test_importing_capture_does_not_mutate_framework_state(monkeypatch):
    sentinel_flush = object()
    monkeypatch.setattr(agent_module, "_auto_tracing_attempted", False)
    monkeypatch.setattr(method_wrapper, "_flush_litellm_journal", sentinel_flush)

    importlib.reload(capture_module)

    assert agent_module._auto_tracing_attempted is False
    assert method_wrapper._flush_litellm_journal is sentinel_flush
