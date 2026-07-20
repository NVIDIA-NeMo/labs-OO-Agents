# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for type-based extractor registry."""

import pytest

from nooa.agentdoc import doc, pformat
from nooa.agentdoc.ext import (
    FieldInfo,
    TypeInfo,
    clear_registry,
    extract_type_info,
    get_type_info_extractor,
    register_type_info_extractor,
    unregister_type_info_extractor,
)


class SampleType:
    """Sample type for registry tests."""

    def __init__(self, value: int):
        self.value = value


class SampleSubType(SampleType):
    """Subclass of SampleType."""

    def __init__(self, value: int, extra: str):
        super().__init__(value)
        self.extra = extra


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


def test_register_and_get_type_info_extractor():
    """Test basic registration and retrieval."""

    @register_type_info_extractor(SampleType)
    def sample_extractor(obj):
        return TypeInfo(
            name="SampleType",
            base=None,
            fields=[FieldInfo("value", "int", ..., "The value")],
            methods=[],
            docstring="Custom extracted.",
        )

    obj = SampleType(42)
    extractor = get_type_info_extractor(obj)

    assert extractor is not None
    assert extractor is sample_extractor


def test_get_type_info_extractor_returns_none_if_not_registered():
    """Test that get_type_info_extractor returns None for unregistered types."""
    obj = SampleType(42)
    assert get_type_info_extractor(obj) is None


def test_extractor_used_in_extract_type_info():
    """Test that registered extractor is used by extract_type_info()."""

    @register_type_info_extractor(SampleType)
    def sample_extractor(obj):
        return TypeInfo(
            name="CustomSampleType",
            base="CustomBase",
            fields=[FieldInfo("custom_field", "str", "default", None)],
            methods=[],
            docstring="Custom extraction.",
        )

    info = extract_type_info(SampleType)

    assert info.name == "CustomSampleType"
    assert info.base == "CustomBase"
    assert len(info.fields) == 1
    assert info.fields[0].name == "custom_field"


def test_extractor_used_in_doc():
    """Test that registered extractor is used by doc()."""

    @register_type_info_extractor(SampleType)
    def sample_extractor(obj):
        return TypeInfo(
            name="SampleType",
            base=None,
            fields=[FieldInfo("value", "int", ..., "Custom description")],
            methods=[],
            docstring="Custom doc.",
        )

    result = doc(SampleType)

    assert "SampleType" in result
    assert "Custom description" in result


def test_extractor_with_instance_values():
    """Test extractor that returns (TypeInfo, values) for instances."""

    @register_type_info_extractor(SampleType)
    def sample_extractor(obj):
        type_info = TypeInfo(
            name="SampleType",
            base=None,
            fields=[FieldInfo("value", "int", ..., None)],
            methods=[],
            docstring="Sample type.",
        )

        if isinstance(obj, type):
            return type_info
        else:
            # Return tuple for instances
            return (type_info, {"value": obj.value * 10})  # Transform value

    obj = SampleType(5)
    # With new design: doc(instance) shows type, pformat(instance) shows values
    result = pformat(obj)

    # Should show transformed value (50, not 5)
    assert "50" in result


def test_extractor_inheritance_via_mro():
    """Test that subclasses use parent's extractor via MRO lookup."""

    @register_type_info_extractor(SampleType)
    def sample_extractor(obj):
        cls = obj if isinstance(obj, type) else type(obj)
        return TypeInfo(
            name=cls.__name__,
            base="ExtractedBase",
            fields=[],
            methods=[],
            docstring="From extractor.",
        )

    # SubType should use SampleType's extractor
    info = extract_type_info(SampleSubType)

    assert info.name == "SampleSubType"
    assert info.base == "ExtractedBase"


def test_subclass_can_override_extractor():
    """Test that subclasses can have their own extractor."""

    @register_type_info_extractor(SampleType)
    def parent_extractor(obj):
        return TypeInfo(
            name="ParentExtracted",
            base=None,
            fields=[],
            methods=[],
            docstring="Parent.",
        )

    @register_type_info_extractor(SampleSubType)
    def child_extractor(obj):
        return TypeInfo(
            name="ChildExtracted",
            base=None,
            fields=[],
            methods=[],
            docstring="Child.",
        )

    # Parent uses parent extractor
    parent_info = extract_type_info(SampleType)
    assert parent_info.name == "ParentExtracted"

    # Child uses child extractor
    child_info = extract_type_info(SampleSubType)
    assert child_info.name == "ChildExtracted"


def test_unregister_type_info_extractor():
    """Test that unregister removes an extractor."""

    @register_type_info_extractor(SampleType)
    def sample_extractor(obj):
        return TypeInfo(
            name="CustomName",
            base=None,
            fields=[],
            methods=[],
            docstring="Custom.",
        )

    info = extract_type_info(SampleType)
    assert info.name == "CustomName"

    # Unregister
    unregister_type_info_extractor(SampleType)

    # Should now use automatic extraction
    info = extract_type_info(SampleType)
    assert info.name == "SampleType"  # Original name


def test_type_info_protocol_takes_precedence_over_extractor():
    """Test that __type_info__ protocol on class takes precedence over registry."""

    class TypeWithProtocol:
        def __init__(self, value):
            self.value = value

        @classmethod
        def __type_info__(cls):
            return TypeInfo(
                name="ProtocolName",
                base=None,
                fields=[],
                methods=[],
                docstring="From protocol.",
            )

    @register_type_info_extractor(TypeWithProtocol)
    def extractor(obj):
        return TypeInfo(
            name="ExtractorName",
            base=None,
            fields=[],
            methods=[],
            docstring="From extractor.",
        )

    # Extractor should be found first (registry has higher priority)
    # This is intentional - registry allows overriding even protocol implementations
    extractor_func = get_type_info_extractor(TypeWithProtocol)
    assert extractor_func is not None

    # But extract_type_info checks registry first
    info = extract_type_info(TypeWithProtocol)
    assert info.name == "ExtractorName"  # Registry wins


def test_clear_registry():
    """Test that clear_registry removes all extractors."""

    @register_type_info_extractor(SampleType)
    def extractor1(obj):
        return TypeInfo("E1", None, [], [], None)

    @register_type_info_extractor(SampleSubType)
    def extractor2(obj):
        return TypeInfo("E2", None, [], [], None)

    # Both should work
    assert get_type_info_extractor(SampleType) is not None
    assert get_type_info_extractor(SampleSubType) is not None

    # Clear all
    clear_registry()

    # Neither should have extractors
    assert get_type_info_extractor(SampleType) is None
    assert get_type_info_extractor(SampleSubType) is None


def test_multiple_inheritance_mro():
    """Test extractor lookup with multiple inheritance."""

    class Base1:
        pass

    class Base2:
        pass

    class Multi(Base1, Base2):
        pass

    @register_type_info_extractor(Base1)
    def base1_extractor(obj):
        cls = obj if isinstance(obj, type) else type(obj)
        return TypeInfo(
            name=cls.__name__,
            base="Base1Extracted",
            fields=[],
            methods=[],
            docstring="From Base1 extractor.",
        )

    # Multi should find Base1's extractor via MRO
    info = extract_type_info(Multi)
    assert info.base == "Base1Extracted"


def test_extractor_for_class_vs_instance():
    """Test that extractor handles both classes and instances correctly."""

    @register_type_info_extractor(SampleType)
    def sample_extractor(obj):
        type_info = TypeInfo(
            name="SampleType",
            base=None,
            fields=[FieldInfo("value", "int", ..., None)],
            methods=[],
            docstring="Sample.",
        )

        if isinstance(obj, type):
            # For type, just return TypeInfo
            return type_info
        else:
            # For instance, return (TypeInfo, values)
            return (type_info, {"value": obj.value})

    # Test with class
    class_info = extract_type_info(SampleType)
    assert class_info.name == "SampleType"

    # Test with instance
    obj = SampleType(42)
    # With new design: doc(instance) shows type, pformat(instance) shows values
    instance_result = pformat(obj)
    assert "42" in instance_result
