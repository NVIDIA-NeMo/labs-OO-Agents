# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from pydantic import BaseModel, Field, ValidationError
import pytest

from nooa._immutable_json import NativeJSON, freeze, json_containers


class SampleModel(BaseModel):
    text: str = "test"
    native: NativeJSON | None = Field(default=None, repr=False)


@pytest.mark.parametrize(
    "bad_key",
    [1, 2.5, True, None, (1, 2)],
)
def test_freeze_rejects_non_string_keys_at_root(bad_key):
    with pytest.raises(TypeError, match="object keys must be strings"):
        freeze({bad_key: "value"})


@pytest.mark.parametrize(
    "bad_key",
    [1, 2.5, False, None, (1, 2)],
)
def test_freeze_rejects_non_string_keys_nested_in_dict(bad_key):
    with pytest.raises(TypeError, match="object keys must be strings"):
        freeze({"outer": {bad_key: "nested_value"}})


@pytest.mark.parametrize(
    "bad_key",
    [1, 2.5, True, None, (1, 2)],
)
def test_freeze_rejects_non_string_keys_nested_in_list(bad_key):
    with pytest.raises(TypeError, match="object keys must be strings"):
        freeze([{"outer": [1, {bad_key: "nested_value"}]}])


def test_model_construction_rejects_non_string_keys():
    with pytest.raises(TypeError, match="object keys must be strings"):
        SampleModel(text="answer", native={1: "integer key", "1": "string key"})

    with pytest.raises(TypeError, match="object keys must be strings"):
        SampleModel(text="answer", native={"nested": {42: "value"}})


@pytest.mark.parametrize(
    "non_finite",
    [float("nan"), float("inf"), float("-inf")],
)
def test_freeze_rejects_non_finite_floats_at_root(non_finite):
    with pytest.raises(ValueError, match="non-finite float"):
        freeze({"score": non_finite})


@pytest.mark.parametrize(
    "non_finite",
    [float("nan"), float("inf"), float("-inf")],
)
def test_freeze_rejects_non_finite_floats_nested(non_finite):
    with pytest.raises(ValueError, match="non-finite float"):
        freeze({"nested": {"scores": [1.0, 2.0, non_finite]}})


@pytest.mark.parametrize(
    "non_finite",
    [float("nan"), float("inf"), float("-inf")],
)
def test_model_construction_rejects_non_finite_floats(non_finite):
    with pytest.raises(ValidationError, match="non-finite float"):
        SampleModel(text="answer", native={"score": non_finite})

    with pytest.raises(ValidationError, match="non-finite float"):
        SampleModel(text="answer", native={"metrics": [non_finite]})


def test_valid_nested_json_round_trips_losslessly():
    data = {
        "str": "hello",
        "int": 42,
        "float": 3.14159,
        "bool": True,
        "null": None,
        "list": [1, "two", 3.0, False, None, {"nested_key": "val"}],
        "dict": {
            "sub": {"deep": "leaf"},
            "empty_list": [],
            "empty_dict": {},
        },
    }
    model = SampleModel(text="answer", native=data)
    dumped = model.model_dump(mode="json")["native"]
    assert dumped == data

    # Verify model_dump_json preserves values
    json_str = model.model_dump_json()
    assert '"str":"hello"' in json_str
    assert '"int":42' in json_str
    assert '"bool":true' in json_str
    assert '"null":null' in json_str


def test_large_string_leaves_remain_shared_not_copied():
    large_string = "x" * 100_000
    data = {"large": large_string, "nested": [large_string]}
    model = SampleModel(text="answer", native=data)

    # In frozen object, scalar strings are borrowed
    assert model.native["large"] is large_string
    assert model.native["nested"][0] is large_string

    # Containers allocation also preserves scalar leaves
    containers = json_containers(model.native)
    assert containers["large"] is large_string
    assert containers["nested"][0] is large_string


def test_deep_immutability_and_copy_behavior():
    data = {"k": "v", "arr": [1, 2], "nested": {"a": 1}}
    model = SampleModel(text="answer", native=data)

    with pytest.raises(TypeError):
        model.native["k"] = "modified"

    with pytest.raises(TypeError):
        model.native["nested"]["a"] = 2

    # model_copy(deep=True) shares the immutable native mapping
    copied = model.model_copy(deep=True)
    assert copied.native is model.native
