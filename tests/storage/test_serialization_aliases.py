# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snapshots contain canonical Pydantic field names, not external input aliases."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, ClassVar, Literal

import pytest
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import TypedDict

from nooa.errors.storage import DeserializationError
from nooa.storage.markers import snapshotable
from nooa.storage.serialization import deserialize, serialize


class _RequiredAlias(BaseModel):
    """A required external alias must not make its saved field disappear."""

    value: int = Field(alias="apiValue")


class _DefaultAlias(BaseModel):
    """A default must not silently replace a previously saved value."""

    value: int = Field(default=0, alias="apiValue")


class _ValidationAlias(BaseModel):
    """Validation-only aliases do not rename durable fields."""

    value: int = Field(validation_alias="apiValue")


class _PathAlias(BaseModel):
    """Nested input paths are unrelated to the flat snapshot field layout."""

    value: int = Field(validation_alias=AliasPath("response", 0, "value"))


class _ChoiceAlias(BaseModel):
    """Alternative external inputs must not change canonical persisted state."""

    value: int = Field(validation_alias=AliasChoices("firstChoice", "secondChoice"))


class _CrossingAliases(BaseModel):
    """Aliases that equal sibling field names must not exchange their values."""

    first: int = Field(alias="second")
    second: int = Field(alias="first")


class _DuplicateAliases(BaseModel):
    """Distinct field values can share the same external input alias."""

    first: int = Field(alias="input")
    second: int = Field(alias="input")


class _GeneratedAlias(BaseModel):
    """Alias generators likewise leave the snapshot's canonical field names intact."""

    model_config = ConfigDict(alias_generator=str.upper)
    value: int


class _FrozenAlias(BaseModel):
    """Restoration validates a new immutable model rather than mutating a live class."""

    model_config = ConfigDict(frozen=True)
    value: int = Field(alias="apiValue")


def _duplicate_alias_value():
    """Use valid independent Python field values after external input parsing."""
    value = _DuplicateAliases(input=1)
    value.first = 7
    value.second = 9
    return value


_VALUES = [
    _RequiredAlias(apiValue=7),
    _DefaultAlias(apiValue=7),
    _ValidationAlias(apiValue=7),
    _PathAlias(response=[{"value": 7}]),
    _ChoiceAlias(secondChoice=7),
    _CrossingAliases(second=11, first=22),
    _duplicate_alias_value(),
    _GeneratedAlias(VALUE=7),
    _FrozenAlias(apiValue=7),
]


def _roundtrip(value):
    """Restore the public codec's output after a real JSON encoding boundary."""
    blob, allowlist = serialize(value)
    return deserialize(json.loads(json.dumps(blob)), allowlist)


@pytest.mark.parametrize("value", _VALUES, ids=lambda value: type(value).__name__)
def test_codec_preserves_aliased_field_values(value):
    """All input-alias forms preserve both concrete type and stored values."""
    restored = _roundtrip(value)
    assert type(restored) is type(value)
    assert restored == value


class _ValidatedAlias(BaseModel):
    """Field, before-model and after-model validators must still run on resume."""

    value: int = Field(alias="apiValue")
    calls: ClassVar[list[str]] = []

    @model_validator(mode="before")
    @classmethod
    def check_input(cls, data):
        """Keep the mapping input contract for model-before validators."""
        cls.calls.append("before")
        assert isinstance(data, dict)
        return data

    @field_validator("value")
    @classmethod
    def positive(cls, value):
        """Reject invalid stored values rather than using unchecked construction."""
        cls.calls.append("field")
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @model_validator(mode="after")
    def check_model(self):
        """Preserve model-after validation and exact execution count."""
        self.calls.append("after")
        return self


def test_restoration_runs_all_validators_once():
    """Alias repair must keep normal validation and not construct an unchecked model."""
    value = _ValidatedAlias(apiValue=7)
    _ValidatedAlias.calls = []
    assert _roundtrip(value) == value
    assert _ValidatedAlias.calls == ["before", "field", "after"]


def test_invalid_stored_value_still_fails_validation():
    """Snapshot data is not trusted merely because it passed validation when saved."""
    blob, allowlist = serialize(_ValidatedAlias(apiValue=7))
    blob["data"]["value"] = -1
    with pytest.raises(ValidationError, match="value must be positive"):
        deserialize(blob, allowlist)


def test_model_schema_config_and_external_input_are_unchanged():
    """Do not mutate class config or rebind the application's live validator."""
    schema = _RequiredAlias.__pydantic_core_schema__
    validator = _RequiredAlias.__pydantic_validator__
    config = dict(_RequiredAlias.model_config)
    assert _roundtrip(_RequiredAlias(apiValue=7)).value == 7
    assert _RequiredAlias.__pydantic_core_schema__ is schema
    assert _RequiredAlias.__pydantic_validator__ is validator
    assert _RequiredAlias.model_config == config
    assert _RequiredAlias(apiValue=8).value == 8
    with pytest.raises(ValidationError):
        _RequiredAlias.model_validate({"value": 7})


def test_concurrent_restores_do_not_change_external_alias_validation():
    """Canonical restore and external construction can safely share model classes."""

    def restore_and_validate(index):
        """Exercise canonical snapshots and normal alias inputs in the same thread."""
        value = _CrossingAliases(second=index, first=-index)
        assert _roundtrip(value) == value
        assert _RequiredAlias(apiValue=index).value == index
        with pytest.raises(ValidationError):
            _RequiredAlias.model_validate({"value": index})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(restore_and_validate, range(128)))


@snapshotable
class _NestedRecord:
    """Typed nested extra value used to check allowlist protection."""

    def __init__(self, name):
        """Initialize durable record content."""
        self.name = name


class _ExtraAlias(BaseModel):
    """Canonical fields and allowed extras retain their existing separate roles."""

    model_config = ConfigDict(extra="allow")
    value: int = Field(alias="apiValue")


def test_allowed_extras_including_alias_spelling_are_retained():
    """An extra with the external alias spelling must not overwrite the saved field."""
    value = _ExtraAlias(apiValue=7, record=_NestedRecord("kept"))
    value.model_extra["apiValue"] = 99
    restored = _roundtrip(value)
    assert restored.value == 7
    assert restored.model_extra["apiValue"] == 99
    assert isinstance(restored.model_extra["record"], _NestedRecord)
    assert restored.model_extra["record"].name == "kept"


def test_nested_extra_still_requires_allowlist():
    """Changing field lookup does not weaken typed nested-value allowlists."""
    value = _ExtraAlias(apiValue=7, record=_NestedRecord("kept"))
    blob, allowlist = serialize(value)
    with pytest.raises(DeserializationError, match="not in the allowlist"):
        deserialize(blob, allowlist - {f"{__name__}._NestedRecord"})


class _HookAlias(BaseModel):
    """Explicit snapshot/restore hooks continue to own custom persisted schemas."""

    value: int = Field(alias="apiValue")

    def __snapshot_data__(self):
        """Write an intentionally non-canonical custom hook schema."""
        return {"custom": self.value}

    @classmethod
    def __restore_snapshot__(cls, data):
        """Restore via the application hook rather than the normal model path."""
        return cls(apiValue=data["custom"])


def test_custom_snapshot_hooks_still_take_precedence():
    """Do not apply canonical field lookup to a hook-owned format."""
    assert _roundtrip(_HookAlias(apiValue=7)).value == 7


class _NestedAlias(BaseModel):
    """Nested models have their own typed envelopes and canonical fields."""

    child: _RequiredAlias = Field(alias="result")


class _AliasedTree(BaseModel):
    """Recursive model instances retain their aliased fields and child types."""

    value: int = Field(alias="apiValue")
    children: list[_AliasedTree] = Field(default_factory=list, alias="nodes")


class _RevalidatedAlias(BaseModel):
    """A parent may request that an already restored child be validated again."""

    model_config = ConfigDict(revalidate_instances="always")
    value: int = Field(default=0, alias="apiValue")


class _RevalidatingParent(BaseModel):
    """Unaliased parent holding a child with explicit instance revalidation."""

    child: _RevalidatedAlias


class _Cat(BaseModel):
    """One branch of an aliased discriminator union."""

    kind: Literal["cat"] = Field(alias="type")
    value: int = Field(alias="apiValue")


class _Dog(BaseModel):
    """Another branch of an aliased discriminator union."""

    kind: Literal["dog"] = Field(alias="type")
    value: int = Field(alias="apiValue")


class _DiscriminatedParent(BaseModel):
    """Preserve discrimination when children are already restored Python objects."""

    child: Annotated[_Cat | _Dog, Field(discriminator="kind")]


@pytest.mark.parametrize(
    "value",
    [
        _NestedAlias(result={"apiValue": 7}),
        _AliasedTree(apiValue=7, nodes=[{"apiValue": 8}]),
        _RevalidatingParent(child={"apiValue": 7}),
        _DiscriminatedParent(child={"type": "cat", "apiValue": 7}),
        _DiscriminatedParent(child={"type": "dog", "apiValue": 8}),
    ],
    ids=lambda value: type(value).__name__,
)
def test_nested_and_recursive_alias_models(value):
    """Keep typed children, recursive schema references and discriminators intact."""
    restored = _roundtrip(value)
    assert type(restored) is type(value)
    assert restored == value


class _TypedRecord(TypedDict):
    """TypedDict fields also store canonical names after input validation."""

    value: Annotated[int, Field(alias="apiValue")]


class _TypedDictParent(BaseModel):
    """Unlike models, TypedDict values remain plain dictionaries in snapshots."""

    record: _TypedRecord


def test_typed_dictionary_child_uses_canonical_saved_keys():
    """Preserve TypedDict values when parent validation revisits a stored plain dict."""
    value = _TypedDictParent(record={"apiValue": 7})
    assert value.record == {"value": 7}
    assert _roundtrip(value) == value


class _CustomValidationEntry(BaseModel):
    """Applications may own the validation entry point, not just schema validators."""

    value: int
    calls: ClassVar[int] = 0

    @classmethod
    def model_validate(cls, data):
        """Keep the pre-existing custom entry point and its accepted call signature."""
        cls.calls += 1
        return super().model_validate(data)


def test_custom_model_validation_entry_point_remains_in_control():
    """Do not bypass custom entry points or require new keywords from overrides."""
    value = _CustomValidationEntry(value=7)
    _CustomValidationEntry.calls = 0
    assert _roundtrip(value) == value
    assert _CustomValidationEntry.calls == 1


class _GenericAlias[T](BaseModel):
    """Concrete generic specializations retain their aliased typed values."""

    value: T = Field(alias="apiValue")


@pytest.mark.parametrize(
    "value", [_GenericAlias[int](apiValue=7), _GenericAlias[str](apiValue="kept")]
)
def test_generic_alias_model_roundtrip(value):
    """Generic class names and type-specific validation remain intact."""
    assert _roundtrip(value) == value


class _WrapValidatedAlias(BaseModel):
    """Wrapping model validators remain around the complete validation pipeline."""

    value: int = Field(alias="apiValue")
    calls: ClassVar[list[str]] = []

    @model_validator(mode="wrap")
    @classmethod
    def check_wrapped(cls, data, handler):
        """Invoke the original model validator exactly once from its wrapper."""
        cls.calls.append("enter")
        result = handler(data)
        cls.calls.append("exit")
        return result


class _DefaultWrapValidatedAlias(_WrapValidatedAlias):
    """A dropped runtime flag must not silently replace a saved field with its default."""

    value: int = Field(default=0, alias="apiValue")


@pytest.mark.parametrize(
    "value", [_WrapValidatedAlias(apiValue=7), _DefaultWrapValidatedAlias(apiValue=7)]
)
def test_model_wrap_validator_remains_in_place(value):
    """Required and defaulted fields retain saved values without bypassing wrappers."""
    type(value).calls = []
    assert _roundtrip(value) == value
    assert type(value).calls == ["enter", "exit"]


class _WrappedRevalidatingParent(BaseModel):
    """A model wrap boundary must preserve revalidated child aliases."""

    child: _RevalidatedAlias

    @model_validator(mode="wrap")
    @classmethod
    def check_wrapped(cls, data, handler):
        """Retain the original wrapper around child validation."""
        return handler(data)


class _FieldWrappedRevalidatingParent(BaseModel):
    """A field wrap boundary must likewise forward canonical validation flags."""

    child: _RevalidatedAlias

    @field_validator("child", mode="wrap")
    @classmethod
    def check_wrapped(cls, value, handler):
        """Retain the original field wrapper during child revalidation."""
        return handler(value)


@pytest.mark.parametrize("model", [_WrappedRevalidatingParent, _FieldWrappedRevalidatingParent])
def test_wrapped_parent_preserves_revalidated_child_value(model):
    """Saved child fields must not silently reset after either wrap boundary."""
    value = model(child={"apiValue": 8})
    restored = _roundtrip(value)
    assert type(restored) is model
    assert type(restored.child) is _RevalidatedAlias
    assert restored.child.value == 8
    assert restored == value


class _CustomAliasedValidationEntry(_CustomValidationEntry):
    """A passthrough override explicitly retains external input-alias rules."""

    value: int = Field(alias="apiValue")


def test_passthrough_override_retains_its_alias_contract():
    """Document the opt-out instead of bypassing an application's validation entry point."""
    blob, allowlist = serialize(_CustomAliasedValidationEntry(apiValue=7))
    with pytest.raises(ValidationError, match="apiValue"):
        deserialize(blob, allowlist)


class _RenamedExtraAlias(BaseModel):
    """External alias choices do not implicitly migrate canonical snapshot fields."""

    model_config = ConfigDict(extra="allow")
    current: int = Field(default=0, validation_alias=AliasChoices("current", "legacy"))


def test_historical_alias_choice_key_remains_an_extra():
    """Renamed fields need a restoration hook; legacy data remains available as extras."""
    blob, allowlist = serialize(_RenamedExtraAlias(current=7))
    blob["data"] = {"legacy": 7}
    restored = deserialize(blob, allowlist)
    assert restored.current == 0
    assert restored.model_extra == {"legacy": 7}
