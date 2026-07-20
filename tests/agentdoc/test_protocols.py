# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for extraction protocol support."""

from nooa.agentdoc import doc, pformat
from nooa.agentdoc.ext import (
    CallableInfo,
    FieldInfo,
    SupportsInstanceValues,
    SupportsTypeInfo,
    TypeInfo,
    extract_type_info,
)


class CustomTypeInfoClass:
    """Class that implements __type_info__ protocol."""

    def __init__(self):
        self.value = 42
        self._internal = "hidden"

    @classmethod
    def __type_info__(cls) -> TypeInfo:
        """Custom type info that hides internals."""
        return TypeInfo(
            name="CustomTypeInfoClass",
            base=None,
            fields=[
                FieldInfo(name="value", type="int", default=..., description="The main value"),
            ],
            methods=[
                CallableInfo(
                    name="get_value",
                    signature="(self)",
                    return_type="int",
                    docstring="Get the value.",
                    is_async=False,
                ),
            ],
            docstring="A custom class with controlled type info.",
        )

    def __instance_values__(self) -> dict:
        """Return only the public value."""
        return {"value": self.value}

    def get_value(self) -> int:
        """Get the value."""
        return self.value


class PartialCustomClass:
    """Class that only implements __type_info__, not __instance_values__."""

    def __init__(self):
        self.data = [1, 2, 3]
        self.name = "test"

    @classmethod
    def __type_info__(cls) -> TypeInfo:
        """Custom type info."""
        return TypeInfo(
            name="PartialCustomClass",
            base=None,
            fields=[
                FieldInfo(name="data", type="list[int]", default=..., description=None),
                FieldInfo(name="name", type="str", default=..., description=None),
            ],
            methods=[],
            docstring="Partial custom class.",
        )

    def process(self):
        """A regular method."""
        pass


class TestProtocolDetection:
    """Tests for protocol detection functions."""

    def test_has_type_info_true(self):
        """Test SupportsTypeInfo detects __type_info__."""
        assert isinstance(CustomTypeInfoClass, SupportsTypeInfo) is True

    def test_has_type_info_false(self):
        """Test SupportsTypeInfo returns False for non-implementing classes."""

        class PlainClass:
            pass

        assert isinstance(PlainClass, SupportsTypeInfo) is False

    def test_has_instance_values_true(self):
        """Test SupportsInstanceValues detects __instance_values__."""
        obj = CustomTypeInfoClass()
        assert isinstance(obj, SupportsInstanceValues) is True

    def test_has_instance_values_false(self):
        """Test SupportsInstanceValues returns False for non-implementing objects."""
        obj = object()
        assert isinstance(obj, SupportsInstanceValues) is False


class TestCustomTypeInfo:
    """Tests for custom __type_info__ implementation."""

    def test_custom_type_info_is_used(self):
        """Test extract_type_info() uses __type_info__ when available."""
        info = extract_type_info(CustomTypeInfoClass)

        assert info.name == "CustomTypeInfoClass"
        assert len(info.fields) == 1
        assert info.fields[0].name == "value"
        assert info.fields[0].description == "The main value"
        assert len(info.methods) == 1
        assert info.methods[0].name == "get_value"

    def test_custom_type_info_used_in_doc(self):
        """Test doc() uses custom type info."""
        result = doc(CustomTypeInfoClass)

        assert "CustomTypeInfoClass" in result
        assert "value: int" in result
        assert "The main value" in result
        assert "get_value" in result
        # Should NOT include _internal (filtered by __type_info__)
        assert "_internal" not in result


class TestCustomInstanceValues:
    """Tests for custom __instance_values__ implementation."""

    def test_custom_instance_values_is_used(self):
        """Test pformat uses __instance_values__ when formatting instances."""
        obj = CustomTypeInfoClass()
        obj.value = 99
        # With new design: doc(instance) shows type, pformat(instance) shows values
        result = pformat(obj)

        # Should show the custom value
        assert "99" in result
        # Should NOT show _internal (filtered by __instance_values__)
        assert "hidden" not in result

    def test_partial_custom_uses_automatic_values(self):
        """Test object with only __type_info__ uses automatic value extraction."""
        obj = PartialCustomClass()
        result = doc(obj)

        # Should show the data value
        assert "data" in result
        assert "name" in result


class TestProtocolPriority:
    """Tests for protocol method priority over automatic introspection."""

    def test_type_info_protocol_precedence(self):
        """Test __type_info__ takes precedence over automatic extraction."""
        info = extract_type_info(CustomTypeInfoClass)

        # Should have only 1 field (from custom __type_info__)
        # not include _internal from automatic extraction
        assert len(info.fields) == 1
        assert info.fields[0].name == "value"

    def test_instance_values_protocol_precedence(self):
        """Test __instance_values__ takes precedence over automatic extraction."""
        obj = CustomTypeInfoClass()
        obj._internal = "should be hidden"
        result = doc(obj)

        # Should not include _internal
        assert "should be hidden" not in result


class TestProtocolWithRealObjects:
    """Tests protocol with more realistic object examples."""

    def test_database_client_example(self):
        """Test protocol with database client example."""

        class DatabaseClient:
            def __init__(self, url: str, connected: bool = False):
                self.url = url
                self._connection = None
                self.connected = connected

            @classmethod
            def __type_info__(cls) -> TypeInfo:
                return TypeInfo(
                    name="DatabaseClient",
                    base=None,
                    fields=[
                        FieldInfo("url", "str", ..., "Database connection URL"),
                        FieldInfo("connected", "bool", False, "Connection status"),
                    ],
                    methods=[
                        CallableInfo(
                            "query", "(self, sql: str)", "list[Row]", "Execute a SQL query.", False
                        ),
                        CallableInfo("close", "(self)", "None", "Close the connection.", False),
                    ],
                    docstring="A database client.",
                )

            def __instance_values__(self) -> dict:
                return {
                    "url": self.url,
                    "connected": self.connected,
                    # Hide _connection
                }

            def query(self, sql: str):
                """Execute a SQL query."""
                pass

            def close(self):
                """Close the connection."""
                pass

        db = DatabaseClient("postgresql://localhost/mydb", connected=True)

        # Test doc shows type structure
        doc_result = doc(db)
        assert "DatabaseClient" in doc_result
        assert "Database connection URL" in doc_result
        assert "Connection status" in doc_result

        # Test pformat shows instance values
        pformat_result = pformat(db)
        assert "DatabaseClient" in pformat_result
        assert "postgresql://localhost/mydb" in pformat_result
        assert "True" in pformat_result  # connected=True
        assert "_connection" not in pformat_result  # Should be hidden
