# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for bounded JSON previews used by tracing."""

from __future__ import annotations

import json
import math
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer

from nooa import hidden, spec
from nooa.tracing._hooks_impl import OpenInferenceHooks
from nooa.tracing._trace_json import Limits, trace_fields, trace_json


def _parsed(value: Any, **limits: int) -> tuple[Any, tuple[str, ...], str]:
    result = trace_json(value, limits=Limits(**limits))
    return json.loads(result.text), result.incomplete_paths, result.text


def test_fitting_native_value_matches_standard_json() -> None:
    value = {"text": "hello\nworld", "items": [1, True, None, 1.5]}

    result = trace_json(value)

    assert result.text == json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(", ", ": "),
        sort_keys=False,
    )
    assert result.incomplete_paths == ()


def test_framework_fields_and_types_survive_truncation() -> None:
    result = trace_fields(
        args=(["x" * 100_000],),
        kwargs={"later": "y" * 100_000},
        limits=Limits(max_chars=200, max_nodes=50, max_depth=16),
    )

    parsed = json.loads(result.text)
    assert set(parsed) == {"args", "kwargs"}
    assert isinstance(parsed["args"], list)
    assert isinstance(parsed["kwargs"], dict)
    assert len(result.text) <= 200
    assert result.incomplete_paths


def test_nested_pydantic_huge_string_is_prefix_bounded() -> None:
    class Inner(BaseModel):
        text: str

    class Outer(BaseModel):
        inner: Inner

    result = trace_json(
        Outer(inner=Inner(text="x" * 10_000_000)),
        limits=Limits(max_chars=500, max_nodes=50, max_depth=16),
    )

    parsed = json.loads(result.text)
    assert parsed["inner"]["text"]
    assert len(parsed["inner"]["text"]) < 500
    assert "/inner/text" in result.incomplete_paths
    assert len(result.text) <= 500


def test_unknown_objects_never_call_user_rendering_hooks() -> None:
    class Bomb:
        def __repr__(self) -> str:
            raise AssertionError("repr called")

        def __str__(self) -> str:
            raise AssertionError("str called")

        def __bool__(self) -> bool:
            raise AssertionError("bool called")

        def __iter__(self):
            raise AssertionError("iter called")

    parsed, paths, _text = _parsed({"value": Bomb()})

    assert parsed["value"].startswith("<opaque: ")
    assert paths == ("/value",)


def test_container_subclasses_are_opaque_without_iteration() -> None:
    class HostileList(list):
        def __iter__(self):
            raise AssertionError("iter called")

    parsed, paths, _text = _parsed(HostileList([1, 2, 3]))

    assert isinstance(parsed, str)
    assert paths == ("",)


def test_type_classification_does_not_call_metaclass_hooks() -> None:
    calls: list[str] = []

    class BombMeta(type):
        @property
        def __mro__(cls) -> tuple[type, ...]:
            calls.append("mro")
            return (cls, object)

        @property
        def __name__(cls) -> str:
            calls.append("name")
            return "hooked"

        @property
        def __dict__(cls) -> dict[str, Any]:
            calls.append("dict")
            return {}

        def __hash__(cls) -> int:
            raise AssertionError("metaclass hash called")

        def __eq__(cls, other: object) -> bool:
            raise AssertionError("metaclass equality called")

        def __instancecheck__(cls, instance: object) -> bool:
            raise AssertionError("instance check called")

        def __subclasscheck__(cls, subclass: type) -> bool:
            raise AssertionError("subclass check called")

    class Unknown(metaclass=BombMeta):
        pass

    parsed, paths, _text = _parsed(Unknown())

    assert parsed == "<opaque: Unknown>"
    assert paths == ("",)
    assert calls == []


def test_pydantic_instance_attribute_hooks_are_not_called() -> None:
    armed = False

    class Model(BaseModel):
        value: str

        def __getattribute__(self, name: str) -> Any:
            if armed:
                raise AssertionError(f"attribute hook called for {name}")
            return super().__getattribute__(name)

    value = Model(value="visible")
    armed = True

    assert json.loads(trace_json(value).text) == {"value": "visible"}


def test_pydantic_dict_descriptor_override_is_not_called() -> None:
    calls: list[str] = []

    class Model(BaseModel):
        value: str

        @property
        def __dict__(self) -> dict[str, Any]:
            calls.append("dict")
            raise AssertionError("dict descriptor called")

    value = object.__new__(Model)
    BaseModel.__dict__["__dict__"].__set__(value, {"value": "visible"})

    assert json.loads(trace_json(value).text) == {"value": "visible"}
    assert calls == []


def test_malformed_pydantic_field_metadata_does_not_call_equality_hooks() -> None:
    calls: list[str] = []

    class Evil:
        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            return False

    class Model(BaseModel):
        value: str

    Model.model_fields["value"].exclude = Evil()  # type: ignore[assignment]
    result = trace_json(Model(value="secret"))

    assert json.loads(result.text) == {}
    assert result.incomplete_paths == ("",)
    assert calls == []


def test_hostile_key_in_pydantic_storage_is_rejected_without_equality() -> None:
    calls: list[str] = []

    class EvilKey:
        def __hash__(self) -> int:
            return hash("_agentdoc_fields_docs")

        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            return False

    class Model(BaseModel):
        value: str

    value = Model(value="secret")
    dict.__setitem__(object.__getattribute__(value, "__dict__"), EvilKey(), "extra")
    calls.clear()

    result = trace_json(value)

    assert json.loads(result.text) == {}
    assert result.incomplete_paths == ("",)
    assert calls == []


def test_escape_heavy_scalar_stays_within_exact_limit() -> None:
    result = trace_json(
        '"\\\n' * 1_000_000,
        limits=Limits(max_chars=257, max_nodes=5, max_depth=2),
    )

    assert len(result.text) <= 257
    assert json.loads(result.text)
    assert result.incomplete_paths == ("",)


@pytest.mark.parametrize("key", ['"' * 100_000, "\\" * 100_000, "\n" * 100_000])
def test_escape_heavy_dictionary_key_stays_within_exact_limit(key: str) -> None:
    result = trace_json(
        {key: "unreachable", "later": "also unreachable"},
        limits=Limits(max_chars=127, max_nodes=10, max_depth=3),
    )

    assert json.loads(result.text) == {}
    assert len(result.text) <= 127
    assert result.incomplete_paths == ("",)


def test_nested_container_with_one_character_left_uses_fitting_fallback() -> None:
    result = trace_json(
        {"x": [[]]},
        limits=Limits(max_chars=10, max_nodes=10, max_depth=5),
    )

    json.loads(result.text)
    assert len(result.text) <= 10
    assert result.incomplete_paths


def test_exact_accounting_holds_across_small_budgets() -> None:
    value = {
        '"\\\n': ["😀" * 30, {"deep": [1, 2, 3]}, None],
        "tail": "x" * 100,
    }

    for max_chars in range(4, 160):
        result = trace_json(
            value,
            limits=Limits(max_chars=max_chars, max_nodes=30, max_depth=5),
        )
        json.loads(result.text)
        assert len(result.text) <= max_chars


def test_work_budget_stops_wide_container() -> None:
    parsed, paths, text = _parsed(
        [None] * 100_000,
        max_chars=50_000,
        max_nodes=5,
        max_depth=16,
    )

    assert len(parsed) < 10
    assert paths == ("",)
    assert len(text) <= 50_000


def test_cycles_are_incomplete_but_shared_values_are_not_cycles() -> None:
    cycle: list[Any] = []
    cycle.append(cycle)
    cyclic = trace_json(cycle)
    assert cyclic.incomplete_paths == ("/0",)
    json.loads(cyclic.text)

    shared = [1, 2]
    repeated = trace_json([shared, shared])
    assert json.loads(repeated.text) == [[1, 2], [1, 2]]
    assert repeated.incomplete_paths == ()


def test_incomplete_paths_are_escaped_and_metadata_is_bounded() -> None:
    escaped = trace_json({"a/b~c": float("nan")})
    assert escaped.incomplete_paths == ("/a~1b~0c",)

    collapsed = trace_json({str(index): float("nan") for index in range(100)})
    assert collapsed.incomplete_paths == ("",)


def test_nonfinite_and_giant_numbers_are_valid_bounded_json() -> None:
    value = {"nan": math.nan, "inf": math.inf, "huge": 1 << 1_000_000}

    result = trace_json(value, limits=Limits(max_chars=500, max_nodes=20, max_depth=5))
    parsed = json.loads(result.text)

    assert set(parsed) == {"nan", "inf", "huge"}
    assert result.incomplete_paths == ("/nan", "/inf", "/huge")


def test_pydantic_visibility_is_fail_closed_and_does_not_run_serializers() -> None:
    private_value = "must-not-appear"

    class VisibleModel(BaseModel):
        model_config = ConfigDict(extra="allow")

        public: str = "visible"
        excluded: str = Field(default=private_value, exclude=True)
        no_repr: str = Field(default=private_value, repr=False)
        annotated_hidden: Annotated[str, hidden] = private_value

        @computed_field
        @property
        def computed(self) -> str:
            raise AssertionError("computed property called")

        @field_serializer("public")
        def serialize_public(self, value: str) -> str:
            raise AssertionError("serializer called")

    value = VisibleModel(runtime_extra=private_value)
    result = trace_json(value)
    parsed = json.loads(result.text)

    assert parsed == {"public": "visible"}
    assert private_value not in result.text
    assert result.incomplete_paths == ()


def test_pydantic_exclusions_beat_visibility_opt_in() -> None:
    class Model(BaseModel):
        excluded: str = Field(default="secret", exclude=True)
        no_repr: str = Field(default="secret", repr=False)

    value = Model()
    spec(value, "excluded", hidden=False)
    spec(value, "no_repr", hidden=False)

    assert json.loads(trace_json(value).text) == {}


def test_instance_visibility_override_beats_class_and_annotation() -> None:
    class Model(BaseModel):
        value: Annotated[str, hidden] = "visible"

    value = Model()
    spec(Model, "value", hidden=True)
    spec(value, "value", hidden=False)

    assert json.loads(trace_json(value).text) == {"value": "visible"}


def test_span_metadata_reports_incomplete_paths_out_of_band() -> None:
    class RecordingSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, Any] = {}

        def set_attribute(self, key: str, value: Any) -> None:
            self.attributes[key] = value

    span = RecordingSpan()
    preview = trace_json("x" * 100, limits=Limits(max_chars=10))

    OpenInferenceHooks._set_json_preview(span, preview, direction="output")  # type: ignore[arg-type]

    assert span.attributes["output.value"] == preview.text
    assert span.attributes["output.mime_type"] == "application/json"
    assert span.attributes["nooa.output.preview.version"] == 1
    assert span.attributes["nooa.output.preview.incomplete"] is True
    assert span.attributes["nooa.output.preview.paths"] == ("",)


def test_invalid_limits_are_rejected_before_inspection() -> None:
    class Bomb:
        def __repr__(self) -> str:
            raise AssertionError("repr called")

    with pytest.raises(ValueError):
        trace_json(Bomb(), limits=Limits(max_chars=0, max_nodes=1, max_depth=1))
    with pytest.raises(ValueError):
        trace_json(Bomb(), limits=Limits(max_chars=10, max_nodes=0, max_depth=1))
    with pytest.raises(ValueError):
        trace_fields(
            args=(Bomb(),),
            kwargs={},
            limits=Limits(max_chars=5, max_nodes=10, max_depth=1),
        )


def test_user_data_resembling_old_envelope_is_ordinary_data() -> None:
    value = {
        "$nooa": {"kind": "truncated-json", "limit_chars": 5, "preview_chars": 1},
        "preview": "x",
    }

    result = trace_json(value)

    assert json.loads(result.text) == value
    assert result.incomplete_paths == ()
