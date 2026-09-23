# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CodeAct output-token continuation policy."""

from nooa.strategies.codeact import decide_length_continuation


def test_continue_when_length_with_nonempty_text():
    assert (
        decide_length_continuation(
            finish_reason="length",
            content="partial answer",
            has_tool_calls=False,
            continuation_count=0,
            max_continuations=3,
        )
        == "continue"
    )


def test_no_continue_empty_content():
    assert (
        decide_length_continuation(
            finish_reason="length",
            content="",
            has_tool_calls=False,
            continuation_count=0,
            max_continuations=3,
        )
        == "fail_empty"
    )


def test_no_continue_whitespace_only_content():
    assert (
        decide_length_continuation(
            finish_reason="length",
            content=" \t\n",
            has_tool_calls=False,
            continuation_count=0,
            max_continuations=3,
        )
        == "fail_empty"
    )


def test_no_continue_tool_calls_only():
    assert (
        decide_length_continuation(
            finish_reason="length",
            content="",
            has_tool_calls=True,
            continuation_count=0,
            max_continuations=3,
        )
        == "fail_tools"
    )


def test_no_continue_length_with_text_and_tool_calls():
    assert (
        decide_length_continuation(
            finish_reason="length",
            content="also a tool call",
            has_tool_calls=True,
            continuation_count=0,
            max_continuations=3,
        )
        == "fail_tools"
    )


def test_bound_blocks_fourth_consecutive_length():
    kwargs = {
        "finish_reason": "length",
        "content": "still going",
        "has_tool_calls": False,
        "max_continuations": 3,
    }
    assert decide_length_continuation(continuation_count=2, **kwargs) == "continue"
    assert decide_length_continuation(continuation_count=3, **kwargs) == "fail_bound"


def test_zero_bound_disables_continuation():
    assert (
        decide_length_continuation(
            finish_reason="length",
            content="partial",
            has_tool_calls=False,
            continuation_count=0,
            max_continuations=0,
        )
        == "fail_bound"
    )


def test_counter_resets_on_stop():
    assert (
        decide_length_continuation(
            finish_reason="stop",
            content="done",
            has_tool_calls=False,
            continuation_count=2,
            max_continuations=3,
        )
        == "reset"
    )


def test_counter_resets_on_tool_call_progress():
    assert (
        decide_length_continuation(
            finish_reason="tool_calls",
            content="",
            has_tool_calls=True,
            continuation_count=2,
            max_continuations=3,
        )
        == "reset"
    )
