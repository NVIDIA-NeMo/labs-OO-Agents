# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for immutable provider-native JSON."""

import copy
import json
import sys

import pytest
from pydantic import ValidationError

from nooa._immutable_json import freeze, json_containers
from nooa.llm_types import AssistantText


@pytest.mark.parametrize(
    "native",
    [
        {1: "integer key", "1": "string key"},
        {"nested": {2: "integer key"}},
        {"items": [{"valid": True}, {False: "boolean key"}]},
    ],
    ids=["colliding-root-key", "nested-key", "key-inside-array"],
)
def test_native_json_rejects_non_string_object_keys(native):
    """Every object key must be a string before native state is accepted."""
    with pytest.raises(ValidationError, match="object keys must be strings"):
        AssistantText(text="answer", native=native)


@pytest.mark.parametrize(
    "native",
    [
        {"score": float("nan")},
        {"nested": {"score": float("inf")}},
        {"items": [0, float("-inf")]},
    ],
    ids=["nan", "positive-infinity", "nested-negative-infinity"],
)
def test_native_json_rejects_non_finite_numbers(native):
    """Non-finite floats cannot silently turn into JSON null values."""
    with pytest.raises(ValidationError, match="finite JSON numbers"):
        AssistantText(text="answer", native=native)


def test_valid_native_json_round_trips_without_copying_large_strings():
    """Valid nested JSON stays lossless, immutable, and scalar-sharing."""
    large = "opaque-provider-state-" * 10_000
    native = {
        "none": None,
        "boolean": True,
        "integer": 42,
        "floats": [0.0, -0.0, 5e-324, sys.float_info.max],
        "text": large,
        "nested": {"items": ("a", 2, False)},
    }
    expected = {
        **native,
        "nested": {"items": ["a", 2, False]},
    }
    model = AssistantText(text="answer", native=native)

    projected = json_containers(model.native)
    restored = AssistantText.model_validate_json(model.model_dump_json())
    assert projected == expected
    assert model.model_dump(mode="json")["native"] == expected
    assert json.loads(model.model_dump_json())["native"] == expected
    assert json_containers(restored.native) == expected
    assert model.native["text"] is large
    assert projected["text"] is large


def test_valid_native_json_preserves_immutability_and_deepcopy_identity():
    """Validation does not weaken immutable storage or deep-copy sharing."""
    model = AssistantText(text="answer", native={"nested": [{"value": 1}]})

    with pytest.raises(TypeError):
        model.native["new"] = "value"
    with pytest.raises(TypeError):
        model.native["nested"][0]["value"] = 2

    assert freeze(model.native) is model.native
    assert copy.deepcopy(model).native is model.native


@pytest.mark.parametrize("bad", [{1, 2}, b"bytes", object()])
def test_existing_unsupported_leaf_types_remain_rejected(bad):
    """The stricter checks retain rejection of other non-JSON leaves."""
    with pytest.raises(TypeError, match="JSON values only"):
        freeze({"bad": bad})
