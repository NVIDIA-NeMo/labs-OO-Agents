# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for bounded trace inputs in the programmatic explorer."""

import json

from nooa.trace_explorer.explorer import _io_json_field


def _attributes() -> dict:
    return {
        "input.value": json.dumps(
            {
                "$nooa": {
                    "kind": "truncated-json",
                    "limit_chars": 50_000,
                    "preview_chars": 17,
                },
                "preview": '{"args": [[1, 2,',
            }
        )
    }


def test_truncated_method_input_keeps_preview_visible():
    attributes = _attributes()

    args = _io_json_field(attributes, "input.value", "args", default="[]")
    kwargs = _io_json_field(attributes, "input.value", "kwargs", default="{}")

    assert args == [
        "<trace input truncated at 50000 serialized characters>\nSerialized JSON prefix:\n"
        '{"args": [[1, 2,'
    ]
    assert kwargs == {}


def test_truncated_code_input_keeps_preview_visible():
    code = _io_json_field(_attributes(), "input.value", "code", default="")

    assert code.startswith("<trace input truncated at 50000 serialized characters>")
    assert code.endswith('{"args": [[1, 2,')
