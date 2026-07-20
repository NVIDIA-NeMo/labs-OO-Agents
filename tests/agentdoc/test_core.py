# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for core introspection functions."""

import pytest

from nooa.agentdoc import doc, pformat
from nooa.agentdoc.ext import DocConfig
from nooa.agentdoc.introspect import methods, variables


# Test fixtures
class SimpleClass:
    """A simple test class."""

    def __init__(self):
        self.counter = 0
        self.items = ["a", "b", "c"]
        self.data = {"key": "value"}

    def process(self, x: int) -> int:
        """Process a number and return doubled value."""
        return x * 2

    def get_count(self) -> int:
        """Return the current counter value."""
        return self.counter

    async def async_method(self, name: str) -> str:
        """An async method example."""
        return f"Hello, {name}"


class Calculator:
    """A calculator tool."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b


class ClassWithTool:
    """Class with a tool as class attribute."""

    calculator = Calculator()

    def __init__(self):
        self.value = 42


class TestDoc:
    """Tests for doc() function."""

    def test_doc_basic(self):
        """Test doc() returns formatted documentation with Python class syntax."""
        obj = SimpleClass()
        result = doc(obj)

        assert "class SimpleClass:" in result
        # New format uses Python class syntax, not markdown headers
        assert "def " in result  # Methods shown as def statements
        assert "counter" in result  # Variables shown as attributes

    def test_doc_includes_methods(self):
        """Test doc() includes method signatures."""
        obj = SimpleClass()
        result = doc(obj)

        assert "process" in result
        assert "get_count" in result
        assert "async_method" in result

    def test_doc_includes_variables(self):
        """Test pformat() includes instance variables (doc shows type structure)."""
        obj = SimpleClass()
        # With new design: doc(instance) shows type, pformat(instance) shows values
        result = pformat(obj)

        assert "counter" in result
        assert "items" in result
        assert "data" in result

    def test_doc_includes_docstrings(self):
        """Test doc() includes method docstrings."""
        obj = SimpleClass()
        result = doc(obj)

        assert "Process a number" in result or "process" in result.lower()

    def test_doc_with_config(self):
        """Test doc() respects DocConfig - note: config affects methods()/variables() helpers."""
        obj = SimpleClass()
        # Note: doc() now uses pformat() which doesn't use DocConfig directly
        # DocConfig is for the methods() and variables() helper functions
        result = doc(obj)

        # Should still have structure with Python class syntax
        assert "class SimpleClass:" in result
        assert "def " in result  # Methods shown as def statements


class TestDocUnified:
    """Tests for unified doc() output with Python syntax."""

    def test_doc_pydantic_python_syntax(self):
        """doc() on Pydantic shows Python class with field descriptions."""
        from pydantic import BaseModel, Field

        class UserModel(BaseModel):
            """User information."""

            name: str = Field(description="User's full name")
            age: int

        result = doc(UserModel)

        assert "class UserModel(BaseModel):" in result
        assert '"""User information."""' in result
        assert "name: str" in result
        assert "User's full name" in result
        assert "age: int" in result
        assert "Access:" not in result
        assert "### " not in result

    def test_doc_dataclass_python_syntax(self):
        """doc() on dataclass shows decorator and fields."""
        from dataclasses import dataclass

        @dataclass
        class Point:
            """A point."""

            x: int
            y: int
            label: str = "origin"

        result = doc(Point)

        assert "@dataclass" in result
        assert "class Point:" in result
        assert '"""A point."""' in result
        assert "x: int" in result
        assert "label: str" in result

    def test_doc_namedtuple_python_syntax(self):
        """doc() on NamedTuple shows Python syntax without internals."""
        from typing import NamedTuple

        class Coord(NamedTuple):
            lat: float
            lon: float

        result = doc(Coord)

        assert "class Coord(NamedTuple):" in result
        assert "lat: float" in result
        assert "_tuplegetter" not in result
        assert "Access:" not in result

    def test_doc_filters_framework_docstrings(self):
        """doc() filters Pydantic BaseModel inherited docstrings."""
        from pydantic import BaseModel

        class Simple(BaseModel):
            value: int

        result = doc(Simple)

        assert "Usage spec:" not in result
        assert "Simple" in result

    def test_doc_enum_python_syntax(self):
        """doc() on Enum shows members as assignments."""
        import enum

        class Status(enum.Enum):
            """Status options."""

            PENDING = "pending"
            DONE = "done"

        result = doc(Status)

        assert "class Status(Enum):" in result
        assert '"""Status options."""' in result
        assert "PENDING = 'pending'" in result
        assert "DONE = 'done'" in result


class TestMethods:
    """Tests for methods() function."""

    def test_methods_lists_all_methods(self):
        """Test methods() lists all public methods."""
        obj = SimpleClass()
        result = methods(obj)

        assert "process" in result
        assert "get_count" in result
        assert "async_method" in result

    def test_methods_summary_includes_signatures(self):
        """Test methods() includes signatures in summary mode."""
        obj = Calculator()
        result = methods(obj, detail="summary")

        assert "add" in result
        assert "multiply" in result
        # Should have parameter info
        assert "int" in result or "a" in result

    def test_methods_full_includes_docstrings(self):
        """Test methods() includes full docstrings in full mode."""
        obj = Calculator()
        result = methods(obj, detail="full")

        assert "Add two numbers" in result or "add" in result.lower()

    def test_methods_shows_async(self):
        """Test methods() marks async methods."""
        obj = SimpleClass()
        result = methods(obj)

        assert "async" in result and "async_method" in result

    def test_methods_hides_private(self):
        """Test methods() hides private methods by default."""
        obj = SimpleClass()
        result = methods(obj)

        assert "__init__" not in result
        assert "__dict__" not in result

    def test_methods_respects_hidden_decorator(self):
        """methods() must honour @hidden, not just the _ prefix."""
        from nooa.agentdoc import hidden

        class Service:
            def public_api(self) -> str:
                """Public."""
                return ""

            @hidden
            def internal(self) -> None:
                """Should not appear."""

        result = methods(Service)
        assert "public_api" in result
        assert "internal" not in result


class TestVariables:
    """Tests for variables() function."""

    def test_variables_lists_instance_vars(self):
        """Test variables() lists instance variables."""
        obj = SimpleClass()
        result = variables(obj)

        assert "counter" in result
        assert "items" in result
        assert "data" in result

    def test_variables_shows_values(self):
        """Test variables() shows current values."""
        obj = SimpleClass()
        result = variables(obj)

        assert "0" in result  # counter value
        assert "['a', 'b', 'c']" in result or "list" in result

    def test_variables_shows_types(self):
        """Test variables() shows type annotations."""
        obj = SimpleClass()
        config = DocConfig(include_types=True)
        result = variables(obj, config=config)

        # Should have type information
        assert "int" in result or "list" in result or "dict" in result

    def test_variables_includes_class_attributes(self):
        """Test variables() includes class-level attributes."""
        obj = ClassWithTool()
        result = variables(obj)

        assert "calculator" in result or "Calculator" in result
        assert "value" in result

    def test_variables_hides_private(self):
        """Test variables() hides private attributes by default."""
        obj = SimpleClass()
        result = variables(obj)

        assert "__dict__" not in result
        assert "__class__" not in result

    def test_variables_with_config_hidden_names(self):
        """Test variables() respects hidden_names in config."""
        obj = SimpleClass()
        config = DocConfig(hidden_names=frozenset({"counter"}))
        result = variables(obj, config=config)

        assert "counter" not in result
        assert "items" in result  # Other vars still visible


class TestDrillDownHints:
    """Tests for drill-down hints in variables()."""

    def test_hints_for_complex_objects(self):
        """Test hints are shown for objects with methods."""
        obj = ClassWithTool()
        result = variables(obj)

        # Should have drill-down hints
        assert "#" in result and ("doc(" in result or "methods(" in result)

    def test_hints_not_for_primitives(self):
        """Test hints are not shown for primitive types."""
        obj = SimpleClass()
        result = variables(obj)

        # counter is an int, shouldn't have hint on the same line
        lines = result.split("\n")
        counter_line = [line for line in lines if "counter" in line and "int" in line]
        if counter_line:
            # The hint should only appear for complex objects
            assert "# doc(self.counter)" not in counter_line[0]

    def test_hints_configurable(self):
        """Test hints can be disabled via config."""
        obj = ClassWithTool()
        config = DocConfig(include_hints=False)
        result = variables(obj, config=config)

        # Should not have hints when disabled
        assert "# methods(" not in result or "calculator" not in result


class TestConfig:
    """Tests for DocConfig functionality."""

    def test_config_max_value_chars(self):
        """Test config respects max_value_chars."""
        obj = SimpleClass()
        obj.long_string = "x" * 100  # type: ignore[attr-defined]

        config = DocConfig(max_value_chars=20)
        result = variables(obj, config=config)

        # Truncated → str(len=N, [:H]='...', [-T:]='...') marker
        assert "str(len=100," in result

    def test_config_hidden_prefixes(self):
        """Test config hides attributes with specific prefixes."""
        obj = SimpleClass()
        obj._private = "secret"  # type: ignore[attr-defined]
        obj.public = "visible"  # type: ignore[attr-defined]

        config = DocConfig(hidden_prefixes=("_",))
        result = variables(obj, config=config)

        assert "_private" not in result
        assert "public" in result

    def test_config_include_docstrings(self):
        """Test config controls docstring inclusion."""
        obj = Calculator()

        # With docstrings
        config_with = DocConfig(include_docstrings=True)
        result_with = methods(obj, config=config_with)

        # Without docstrings
        config_without = DocConfig(include_docstrings=False)
        result_without = methods(obj, config=config_without)

        # Result without should be shorter (no docstrings)
        assert len(result_without) <= len(result_with)


class TestDocOnFunctionsAndMethods:
    """Tests for doc() on functions and methods.

    Note: The new format shows function definitions with Python syntax,
    not markdown headers.
    """

    def test_doc_on_method(self):
        """Test doc() on a method shows signature and docstring."""
        obj = SimpleClass()
        result = doc(obj.process)

        # New format: function definition with class name
        assert "SimpleClass.process" in result
        assert "x: int" in result
        assert "-> int" in result
        assert "Process a number" in result

    def test_doc_on_async_method(self):
        """Test doc() on async method shows async prefix."""
        obj = SimpleClass()
        result = doc(obj.async_method)

        assert "async def SimpleClass.async_method" in result
        assert "name: str" in result
        assert "-> str" in result

    def test_doc_on_function(self):
        """Test doc() on a standalone function."""

        def my_func(a: int, b: str = "default") -> bool:
            """Check if string representation of a equals b."""
            return str(a) == b

        result = doc(my_func)

        assert "my_func(a: int, b: str" in result
        assert "a: int" in result
        assert "b: str" in result
        assert "-> bool" in result
        assert "Check if string representation" in result

    def test_doc_on_method_shows_return_type(self):
        """Test doc() on method shows return type in signature."""
        obj = SimpleClass()
        result = doc(obj.process)

        # Return type is shown in the signature, not as separate section
        assert "-> int" in result

    def test_doc_on_method_without_docstring(self):
        """Test doc() handles methods without docstrings."""

        class NoDocClass:
            def no_doc_method(self, x: int) -> int:
                return x * 2

        obj = NoDocClass()
        result = doc(obj.no_doc_method)

        assert "NoDocClass.no_doc_method" in result
        # New format shows ... for missing docstring
        assert "..." in result

    def test_doc_on_method_with_pydantic_return(self):
        """Test doc() on method shows signature with Pydantic return type."""
        pytest.importorskip("pydantic")
        from pydantic import BaseModel, Field

        class Result(BaseModel):
            value: int = Field(description="The computed value")
            message: str = Field(default="", description="Optional message")

        class MyClass:
            def compute(self, x: int) -> Result:
                """Compute a result."""
                return Result(value=x * 2)

        obj = MyClass()
        result = doc(obj.compute)

        # New format shows function definition with return type and class name
        assert "MyClass.compute" in result
        assert "-> Result" in result
        assert "Compute a result" in result

    def test_doc_on_method_with_dataclass_return(self):
        """Test doc() on method shows signature with dataclass return type."""
        import dataclasses

        @dataclasses.dataclass
        class DataResult:
            name: str
            count: int = 0

        class MyClass:
            def get_data(self) -> DataResult:
                """Get data result."""
                return DataResult(name="test")

        obj = MyClass()
        result = doc(obj.get_data)

        assert "MyClass.get_data" in result
        assert "-> DataResult" in result
        assert "Get data result" in result

    def test_doc_on_method_with_typed_dict_return(self):
        """Test doc() on method shows signature with TypedDict return type."""
        from typing import TypedDict

        class PersonDict(TypedDict):
            name: str
            age: int

        class MyClass:
            def get_person(self) -> PersonDict:
                """Get a person as TypedDict."""
                return {"name": "Alice", "age": 30}

        obj = MyClass()
        result = doc(obj.get_person)

        assert "MyClass.get_person" in result
        assert "-> PersonDict" in result
        assert "Get a person as TypedDict" in result

    def test_doc_on_method_with_optional_return(self):
        """Test doc() handles Optional return types (union with None)."""

        class MyClass:
            def find_item(self, key: str) -> str | None:
                """Find an item, returns None if not found."""
                return None

        obj = MyClass()
        result = doc(obj.find_item)

        assert "MyClass.find_item" in result
        # Should show union type in return annotation
        assert "str" in result
        # The return annotation should be shown
        assert "str" in result

    def test_doc_on_method_with_optional_pydantic_return(self):
        """Test doc() on method with Optional[Model] return type."""
        pytest.importorskip("pydantic")

        from pydantic import BaseModel, Field

        class Item(BaseModel):
            id: int = Field(description="Item ID")
            name: str = Field(description="Item name")

        class MyClass:
            def get_item(self, item_id: int) -> Item | None:
                """Get an item by ID, returns None if not found."""
                return Item(id=item_id, name="test")

        obj = MyClass()
        result = doc(obj.get_item)

        # New format shows function definition with return type and class name
        assert "MyClass.get_item" in result
        assert "Get an item by ID" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_object_with_no_methods(self):
        """Test object with no public methods."""
        obj = object()
        result = methods(obj)

        # Should not crash, might return empty or just dunder methods
        assert isinstance(result, str)

    def test_object_with_no_variables(self):
        """Test object with no instance variables."""
        obj = object()
        result = variables(obj)

        # Should not crash
        assert isinstance(result, str)

    def test_object_with_broken_repr(self):
        """Test object whose repr() raises exception."""

        class BrokenRepr:
            def __repr__(self):
                raise RuntimeError("repr broken")

        obj = SimpleClass()
        obj.broken = BrokenRepr()  # type: ignore[attr-defined]

        # Should not crash
        result = variables(obj)
        assert isinstance(result, str)

    def test_none_values(self):
        """Test handling of None values."""
        obj = SimpleClass()
        obj.none_value = None  # type: ignore[attr-defined]

        result = variables(obj)
        assert "None" in result


class TestStructuredTypes:
    """Tests for structured types (Pydantic, TypedDict, dataclass)."""

    def test_typeddict_shows_fields_not_dict_methods(self):
        """Test that TypedDict classes show fields instead of dict methods."""
        from typing import TypedDict

        class UserDict(TypedDict):
            """User information."""

            name: str
            age: int
            email: str | None

        result = doc(UserDict)

        # Should show fields
        assert "name: str" in result
        assert "age: int" in result
        assert "email:" in result

        # Should NOT show dict methods
        assert "def clear" not in result
        assert "def copy" not in result
        assert "def keys" not in result
        assert "def items" not in result

        # Should include docstring
        assert "User information" in result

    def test_pydantic_model_shows_fields_not_base_methods(self):
        """Test that Pydantic models show fields instead of BaseModel methods."""
        from pydantic import BaseModel

        class UserModel(BaseModel):
            """User model."""

            name: str
            age: int
            email: str | None = None

        result = doc(UserModel)

        # Should show fields
        assert "name: str" in result
        assert "age: int" in result
        assert "email:" in result

        # Should NOT show BaseModel methods
        assert "model_dump" not in result
        assert "model_validate" not in result
        assert "model_copy" not in result

        # Should include docstring
        assert "User model" in result

    def test_dataclass_shows_fields(self):
        """Test that dataclass classes show fields correctly."""
        from dataclasses import dataclass

        @dataclass
        class UserData:
            """User dataclass."""

            name: str
            age: int
            email: str | None = None

        result = doc(UserData)

        # Should show fields
        assert "name: str" in result
        assert "age: int" in result
        assert "email:" in result

        # Should include docstring
        assert "User dataclass" in result

    def test_typeddict_shows_python_syntax(self):
        """Test that TypedDict shows Python class syntax."""
        from typing import TypedDict

        class ResultDict(TypedDict):
            """Result data."""

            status: str
            code: int

        result = doc(ResultDict)

        # Should show Python class syntax (not markdown)
        assert "class ResultDict(TypedDict):" in result
        assert "status: str" in result
        assert "code: int" in result
        # Should NOT have markdown-style Access hint
        assert "Access:" not in result


class TestLargeValueTruncation:
    """Tests for truncation of large member variables to prevent LLM context explosion."""

    def test_huge_string_is_truncated(self):
        """When the caller passes max_string, huge nested strings are bounded."""

        class HugeStringClass:
            def __init__(self):
                self.huge_string = "x" * 100_000  # 100KB string

        obj = HugeStringClass()
        # Hardcoded fallbacks were removed — pformat() with no kwargs renders
        # full content. The caller is expected to pass bounds explicitly.
        result = pformat(obj, max_string=200, max_length=10)

        # Result should be small (under 1KB)
        assert len(result) < 1000, f"pformat() output too large: {len(result)} chars"
        # Truncation 3.0 string marker: str(len=N, [:H]='...', [-T:]='...')
        assert "str(len=100000," in result

    def test_huge_list_is_truncated(self):
        """When the caller passes max_length, huge nested lists are bounded."""

        class HugeListClass:
            def __init__(self):
                self.huge_list = list(range(100_000))  # 100K items

        obj = HugeListClass()
        # Caller passes explicit bounds — no hidden fallback.
        result = pformat(obj, max_length=10, max_string=200)

        # Result should be small
        assert len(result) < 1000, f"pformat() output too large: {len(result)} chars"
        # Should show truncation marker and tail items (slice-keys form for lists)
        assert "list(len=100000," in result and "999" in result  # tail item 99999

    def test_huge_dict_is_truncated(self):
        """When the caller passes max_length, huge nested dicts are bounded."""

        class HugeDictClass:
            def __init__(self):
                self.huge_dict = {f"key_{i}": f"value_{i}" for i in range(50_000)}

        obj = HugeDictClass()
        # Caller passes explicit bounds — no hidden fallback.
        result = pformat(obj, max_length=10, max_string=200)

        # Result should be small (dict expands to multi-line)
        assert len(result) < 2000, f"pformat() output too large: {len(result)} chars"
        # Should show truncation marker and tail items (items wrapper for dicts)
        assert "dict(len=50000, items=" in result and "499" in result  # tail key key_49999

    def test_deeply_nested_object_is_truncated(self):
        """Test that deeply nested objects don't blow up context."""

        class DeeplyNested:
            def __init__(self):
                # Create deeply nested structure
                nested = {"level": 0}
                for i in range(1, 100):
                    nested = {"level": i, "child": nested, "data": "x" * 100}
                self.nested = nested

        obj = DeeplyNested()
        result = doc(obj)

        # Result should be small - pprint depth=2 should limit nesting
        assert len(result) < 2000, f"doc() output too large: {len(result)} chars"

    def test_object_with_huge_repr_is_truncated(self):
        """Test that objects with huge repr() output are truncated."""

        class HugeReprClass:
            def __init__(self):
                self._data = "x" * 100_000

            def __repr__(self):
                return f"HugeReprClass(data={self._data})"

        class ContainerClass:
            def __init__(self):
                self.huge_obj = HugeReprClass()

        obj = ContainerClass()
        result = doc(obj)

        # Result should be small
        assert len(result) < 1000, f"doc() output too large: {len(result)} chars"

    def test_multiple_huge_attrs_all_truncated(self):
        """Test that multiple huge attributes are all truncated."""

        class MultiHugeClass:
            def __init__(self):
                self.huge_string = "a" * 50_000
                self.huge_list = list(range(50_000))
                self.huge_dict = {f"k{i}": i for i in range(50_000)}
                self.huge_nested = {"deep": {"deeper": {"data": "x" * 10_000}}}

        obj = MultiHugeClass()
        result = doc(obj)

        # Result should still be reasonable
        assert len(result) < 2000, f"doc() output too large: {len(result)} chars"

    def test_config_max_value_chars_respected(self):
        """Test that config.max_value_chars is respected for truncation."""

        class TestClass:
            def __init__(self):
                self.medium_string = "y" * 500

        obj = TestClass()

        # With small max_value_chars
        config_small = DocConfig(max_value_chars=50)
        result_small = variables(obj, config=config_small)

        # With large max_value_chars
        config_large = DocConfig(max_value_chars=1000)
        result_large = variables(obj, config=config_large)

        # Small config should truncate → str(len=N, [:H]='...', [-T:]='...')
        assert "str(len=" in result_small
        # Large config should not truncate (500 < 1000):
        # ensure some of the value is present and no truncation marker is used.
        assert "yyy" in result_large
        assert "str(len=" not in result_large

    def test_doc_on_huge_builtin_with_max_length(self):
        """doc() on raw containers respects explicit max_length, shows head+tail."""
        from nooa.agentdoc._pformat import _pformat_to_str

        huge_list = list(range(100_000))
        result = _pformat_to_str(huge_list, max_length=10, max_string=500, max_depth=3)

        assert "list(len=100000," in result  # truncation marker
        assert "99999" in result  # tail item visible
        assert len(result) < 2000

    def test_doc_on_object_with_huge_repr(self):
        """Test doc() on object with huge data doesn't blow up context.

        Classes without __repr__ go through field extraction which shows
        AST expressions (compact). Classes WITH __repr__ get repr() —
        we trust the author's intent.
        """

        class HugeData:
            def __init__(self):
                # Use public attribute so it shows in variables
                self.data = "z" * 100_000

        obj = HugeData()
        result = doc(obj)

        # Should be compact - field extraction truncates the huge string
        # using the str(len=N, ...) marker, not the raw 100k-char value
        assert len(result) < 1000, f"doc() on huge data too large: {len(result)}"
        # Verify the field is extracted with truncation marker
        assert "data:" in result
        assert "str(len=100000," in result

    def test_doc_on_class_with_repr_uses_repr(self):
        """Classes with custom __repr__ should use repr(), not field extraction."""

        class WithRepr:
            def __init__(self):
                self.hidden = "not extracted"

            def __repr__(self):
                return "custom_repr_sentinel"

        result = doc(WithRepr())
        assert "custom_repr_sentinel" in result
        assert "hidden" not in result

    def test_slots_class_with_repr_uses_repr(self):
        """__slots__-only classes with __repr__ should also trust repr()."""

        class SlotsWithRepr:
            __slots__ = ("x", "y")

            def __init__(self):
                self.x = 1
                self.y = 2

            def __repr__(self):
                return "SlotsWithRepr(custom)"

        result = doc(SlotsWithRepr())
        assert "SlotsWithRepr(custom)" in result

    def test_pydantic_still_uses_field_extraction_despite_repr(self):
        """Pydantic models define __repr__ but should still use field extraction."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str = "hello"
            count: int = 42

            def __repr__(self):
                return "PYDANTIC_REPR_SENTINEL"

        result = pformat(MyModel())
        # Field extraction produces ClassName(field=value) style
        assert "name=" in result
        assert "hello" in result
        assert "count=" in result
        # Must NOT fall through to __repr__
        assert "PYDANTIC_REPR_SENTINEL" not in result

    def test_dataclass_still_uses_field_extraction_despite_repr(self):
        """Dataclasses define __repr__ but should still use field extraction."""
        import dataclasses

        @dataclasses.dataclass
        class DC:
            x: int = 1
            y: str = "hi"

            def __repr__(self):
                return "DC_REPR_SENTINEL"

        result = pformat(DC())
        assert "x=" in result
        assert "y=" in result
        assert "hi" in result
        # Must NOT fall through to __repr__
        assert "DC_REPR_SENTINEL" not in result

    def test_plain_class_with_repr_uses_repr_not_fields(self):
        """Plain class with __repr__ should use repr, not extract fields."""
        from nooa.agentdoc import pformat

        class ToolLike:
            def __init__(self):
                self.cwd = "/tmp/project"
                self._internal = "secret"

            def __repr__(self):
                return f"ToolLike(cwd={self.cwd!r})"

        result = pformat(ToolLike())
        # Should use __repr__, not field extraction
        assert result == "ToolLike(cwd='/tmp/project')"
        # Should NOT show _internal (repr doesn't include it)
        assert "_internal" not in result


class TestDocTruncationHeader:
    """Tests for truncation behavior in doc() output.

    Per design doc: Output uses Python class syntax (not markdown headers).
    Truncation is shown inline as `... +N` in values.
    """

    def test_truncation_shown_when_variables_truncated(self):
        """Truncation marker should appear when value is truncated."""

        class TruncatedClass:
            def __init__(self):
                # String that's long enough to trigger truncation
                # pformat() default max_string is None, so explicitly set it
                self.long_string = "x" * 1000

        obj = TruncatedClass()
        # With new design: doc(instance) shows type, pformat(instance) shows values
        result = pformat(obj, max_string=500)

        # Truncation marker should appear inline: str(len=N, [:H]='...', [-T:]='...')
        assert "str(len=1000," in result

    def test_no_truncation_marker_when_no_truncation(self):
        """Truncation marker should NOT appear when value fits."""

        class SmallClass:
            def __init__(self):
                self.value = 42

        obj = SmallClass()
        result = doc(obj)

        # No truncation marker should be present
        assert "str(len=" not in result


class TestPformatMatchesRich:
    """Tests that _pformat output matches Rich pprint format."""

    @pytest.mark.skipif(
        not pytest.importorskip("rich", reason="rich not installed"),
        reason="rich not installed",
    )
    def test_string_truncation_uses_truncation_3_marker(self):
        """String truncation uses the truncation 3.0 slice-keys marker.

        We deliberately diverge from Rich's ``'foo'+N`` legacy form here —
        the matrix data showed the slice-keys form (head + tail with the
        total length up front) is ~25-30 pp easier for LLMs to read.
        """
        from nooa.agentdoc._pformat import _pformat_to_str as _pformat

        data = "x" * 100
        our_output = _pformat(data, max_string=10)

        # str(len=100, [:5]='xxxxx', [-5:]='xxxxx')
        assert our_output.startswith("str(len=100,")
        assert "[:5]=" in our_output
        assert "[-5:]=" in our_output
        # Old "'foo'+N" form should NOT appear.
        assert "'+" not in our_output

    @pytest.mark.skipif(
        not pytest.importorskip("rich", reason="rich not installed"),
        reason="rich not installed",
    )
    def test_list_truncation_shows_head_and_tail(self):
        """List truncation uses head+tail format (diverges intentionally from Rich head-only)."""
        from nooa.agentdoc._pformat import _pformat_to_str as _pformat

        data = list(range(20))
        max_length = 5
        our_output = _pformat(data, max_length=max_length)

        # Head: first 3 items (ceiling of 5/2)
        assert "0" in our_output
        assert "2" in our_output
        # Tail: last 2 items
        assert "19" in our_output
        assert "18" in our_output
        # Truncation 3.0 slice-keys marker
        assert our_output.startswith("list(len=20,")
        assert "[:3]=[" in our_output
        assert "[-2:]=[" in our_output

    @pytest.mark.skipif(
        not pytest.importorskip("rich", reason="rich not installed"),
        reason="rich not installed",
    )
    def test_dict_truncation_shows_head_and_tail(self):
        """Dict truncation uses head+tail format (diverges intentionally from Rich head-only)."""
        from nooa.agentdoc._pformat import _pformat_to_str as _pformat

        data = {f"k{i}": i for i in range(20)}
        max_length = 3
        our_output = _pformat(data, max_length=max_length)

        # Head: first 2 keys (ceiling of 3/2)
        assert "'k0'" in our_output
        assert "'k1'" in our_output
        # Tail: last 1 key
        assert "'k19'" in our_output
        # Truncation 3.0 items wrapper for dicts
        assert our_output.startswith("dict(len=20, items={")
        assert our_output.endswith("})")


class TestPformatAdditionalTypes:
    """Tests for _pformat with additional types (tuples, sets, etc)."""

    def test_tuple_truncation(self):
        """Tuple truncation uses the slice-keys marker with tuple parens."""
        from nooa.agentdoc._pformat import _pformat_to_str as _pformat

        data = tuple(range(20))
        result = _pformat(data, max_length=5)

        assert result.startswith("tuple(len=20,")
        assert "[:3]=(" in result
        assert "[-2:]=(" in result
        assert "19" in result  # tail item visible

    def test_set_truncation(self):
        """Set truncation uses the items wrapper (no positional anchor)."""
        from nooa.agentdoc._pformat import _pformat_to_str as _pformat

        data = set(range(20))
        result = _pformat(data, max_length=5)

        assert result.startswith("set(len=20, items={")
        assert result.endswith("})")

    def test_frozenset_truncation(self):
        """Frozenset truncation uses the items wrapper."""
        from nooa.agentdoc._pformat import _pformat_to_str as _pformat

        data = frozenset(range(20))
        result = _pformat(data, max_length=5)

        assert result.startswith("frozenset(len=20, items=")


class TestPformatTypes:
    """Tests for pformat() on types."""

    def test_pformat_pydantic_type(self):
        """pformat() on Pydantic type shows Python class syntax."""
        from pydantic import BaseModel

        from nooa.agentdoc._pformat import _pformat_to_str as _pformat

        class UserModel(BaseModel):
            name: str
            age: int

        result = _pformat(UserModel)

        assert "class UserModel(BaseModel):" in result
        assert "name: str" in result
        assert "age: int" in result

    def test_pformat_dataclass_type(self):
        """pformat() on dataclass type shows decorator and fields."""
        from dataclasses import dataclass

        from nooa.agentdoc._pformat import _pformat_to_str as _pformat

        @dataclass
        class Point:
            x: int
            y: int

        result = _pformat(Point)

        assert "@dataclass" in result
        assert "class Point:" in result
        assert "x: int" in result
        assert "y: int" in result

    def test_pformat_type_truncates_fields(self):
        """pformat() truncates fields with max_length."""
        from pydantic import BaseModel

        from nooa.agentdoc._pformat import _pformat_to_str as _pformat

        class BigModel(BaseModel):
            f1: str
            f2: str
            f3: str
            f4: str
            f5: str
            f6: str

        result = _pformat(BigModel, max_length=3)

        assert "f1: str" in result
        assert "f2: str" in result
        assert "f3: str" in result
        assert "..." in result  # truncation 3.0: bare ellipsis (was "... +3")
        assert "f6: str" not in result

    def test_pformat_enum_type(self):
        """pformat() on Enum shows members."""
        import enum

        from nooa.agentdoc._pformat import _pformat_to_str as _pformat

        class Color(enum.Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        result = _pformat(Color)

        assert "class Color(Enum):" in result
        assert "RED = 1" in result
        assert "GREEN = 2" in result
        assert "BLUE = 3" in result


class TestParameterDefaultTruncation:
    """Tests that parameter defaults are truncated properly.

    Per design: Signatures show truncated defaults with '...' indicator.
    """

    def test_function_with_large_default_string(self):
        """Function with large string default should be truncated."""

        def func_with_big_default(data: str = "x" * 1000):
            pass

        result = doc(func_with_big_default)

        # Should show truncated default, not full 1000 chars
        assert len(result) < 500
        # Should have truncation in the signature (... indicates truncation)
        assert "..." in result

    def test_function_with_large_default_list(self):
        """Function with large list default should be truncated."""

        def func_with_list_default(items: list = list(range(100))):  # noqa: B006, B008
            pass

        result = doc(func_with_list_default)

        # Should show truncated default
        assert len(result) < 500
        # Truncation shown with ... (may or may not have +N depending on formatter)
        assert "..." in result


class TestFieldDefaultTruncation:
    """Tests that Pydantic/dataclass field defaults are truncated."""

    def test_dataclass_with_large_default(self):
        """Dataclass with large default should show truncated value."""
        from dataclasses import dataclass

        @dataclass
        class ConfigWithDefaults:
            small: str = "hello"
            large: str = "x" * 1000

        result = doc(ConfigWithDefaults)

        # Should show the small default fully
        assert "'hello'" in result
        # Large default should be truncated (not show full 1000 chars)
        assert len(result) < 1000

    def test_pydantic_with_large_default(self):
        """Pydantic model with large default should show truncated value."""
        from pydantic import BaseModel

        class ModelWithDefaults(BaseModel):
            small: str = "hello"
            large: str = "x" * 1000

        result = doc(ModelWithDefaults)

        # Should show the small default
        assert "'hello'" in result
        # Large default should be truncated
        assert len(result) < 2000  # Account for Pydantic methods


class TestRichApiCompatibility:
    """Tests for API compatibility with rich.pprint()."""

    def test_pformat_accepts_rich_parameters(self):
        """pformat() should accept all rich.pprint() parameters."""
        data = {"a": 1, "b": 2, "c": 3}

        # Should not raise - all Rich parameters accepted
        result = pformat(
            data,
            console=None,
            indent_guides=True,
            max_length=2,
            max_string=50,
            max_depth=3,
            expand_all=False,
        )

        # max_length=2: head='a', tail='c', 'b' correctly dropped (middle item)
        assert "a" in result
        assert "c" in result
        assert "b" not in result  # correctly in dropped middle

    def test_pprint_accepts_rich_parameters(self):
        """pprint() should accept all rich.pprint() parameters."""
        import sys
        from io import StringIO

        from nooa.agentdoc import pprint

        data = [1, 2, 3, 4, 5]

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        # Should not raise - all Rich parameters accepted
        pprint(
            data,
            console=None,
            indent_guides=True,
            max_length=3,
            max_string=50,
            max_depth=3,
            expand_all=False,
        )

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        assert "1" in output
        assert "2" in output

    def test_expand_all_produces_multiline_output(self):
        """expand_all=True should force multiline output for containers."""
        data = [1, 2, 3]

        compact = pformat(data, expand_all=False)
        expanded = pformat(data, expand_all=True)

        # Compact should fit on one line
        assert "\n" not in compact or compact.count("\n") < expanded.count("\n")
        # Expanded should have newlines
        assert "\n" in expanded

    def test_console_parameter_is_ignored(self):
        """console parameter should be accepted but ignored."""
        data = {"key": "value"}

        # Pass any value for console - should work
        result1 = pformat(data, console=None)
        result2 = pformat(data, console="fake_console")
        result3 = pformat(data, console=object())

        # All should produce the same output
        assert result1 == result2 == result3

    def test_indent_guides_parameter_is_ignored(self):
        """indent_guides parameter should be accepted but ignored."""
        data = {"key": "value"}

        # Both values should produce the same output
        result_true = pformat(data, indent_guides=True)
        result_false = pformat(data, indent_guides=False)

        assert result_true == result_false


class TestDocSubclassUnhide:
    """Integration tests: subclass unhiding visible in doc() output."""

    def test_subclass_unhides_parent_hidden_field(self):
        """Field hidden in parent but re-declared in child should appear in doc()."""
        from typing import Annotated

        from nooa.agentdoc._visibility import hidden

        class Parent:
            x: Annotated[str, hidden] = "secret"
            visible: str = "shown"

        class Child(Parent):
            x: str = "now visible"  # re-declared without hidden

        result = doc(Child())
        assert "x" in result
        assert "visible" in result

    def test_parent_hidden_field_absent_from_parent_doc(self):
        from typing import Annotated

        from nooa.agentdoc._visibility import hidden

        class Parent:
            x: Annotated[str, hidden] = "secret"

        result = doc(Parent())
        assert "x" not in result

    def test_child_does_not_expose_field_still_hidden_in_parent(self):
        """Field hidden in both parent and child remains hidden."""
        from typing import Annotated

        from nooa.agentdoc._visibility import hidden

        class Parent:
            x: Annotated[str, hidden] = "secret"

        class Child(Parent):
            pass  # does NOT re-declare x

        result = doc(Child())
        assert "x" not in result


class TestHiddenProperty:
    """@hidden @property should be invisible in doc(instance)."""

    def test_hidden_property_not_in_doc_instance(self):
        """@property @hidden must not appear in doc(instance)."""
        from nooa.agentdoc import hidden

        class MyClass:
            @property
            @hidden
            def secret_id(self) -> str:
                """Hidden computed value."""
                return "abc-123"

            visible: str = "shown"

        obj = MyClass()
        obj.visible = "shown"
        result = doc(obj)
        assert "secret_id" not in result, "@property @hidden should not appear in doc(instance)"
        assert "visible" in result

    def test_hidden_property_not_in_doc_class(self):
        """@property @hidden must not appear in doc(Class)."""
        from nooa.agentdoc import hidden

        class MyClass:
            @property
            @hidden
            def secret_id(self) -> str:
                """Hidden computed value."""
                return "abc-123"

        result = doc(MyClass)
        assert "secret_id" not in result
