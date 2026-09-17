# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ordinary bounded JSON inputs in the programmatic explorer."""

import json

from nooa.trace_explorer.explorer import _io_decoded_value, _io_json_field


def _attributes() -> dict:
    return {
        "input.value": json.dumps({"args": [[1, 2]], "kwargs": {}, "code": "print('x')"}),
        "input.mime_type": "application/json",
        "nooa.input.preview.incomplete": True,
        "nooa.input.preview.paths": ["/args/0"],
    }


def test_bounded_method_input_remains_structured():
    attributes = _attributes()

    args = _io_json_field(attributes, "input.value", "args", default="[]")
    kwargs = _io_json_field(attributes, "input.value", "kwargs", default="{}")

    assert args == [[1, 2]]
    assert kwargs == {}


def test_bounded_code_input_remains_structured():
    code = _io_json_field(_attributes(), "input.value", "code", default="")

    assert code == "print('x')"


def test_old_envelope_shape_is_not_treated_as_control_data():
    attributes = {
        "input.value": json.dumps(
            {"$nooa": {"kind": "truncated-json"}, "preview": "user data", "args": [1]}
        )
    }

    assert _io_json_field(attributes, "input.value", "args", default=[]) == [1]


def test_io_decoding_obeys_mime_and_decodes_only_once():
    json_string = {
        "output.value": json.dumps('{"stdout": "literal"}'),
        "output.mime_type": "application/json",
    }
    plain_text = {
        "output.value": '{"stdout": "plain"}',
        "output.mime_type": "text/plain",
    }
    legacy = {"output.value": '{"stdout": "legacy"}'}

    assert _io_decoded_value(json_string, "output") == '{"stdout": "literal"}'
    assert _io_decoded_value(plain_text, "output") == '{"stdout": "plain"}'
    assert _io_decoded_value(legacy, "output") == {"stdout": "legacy"}
