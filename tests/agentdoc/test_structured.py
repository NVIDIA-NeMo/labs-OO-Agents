# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TypeInfo extraction."""

# Path must be set before other imports so "from fixtures.init_field_classes import ..." works
# ruff: noqa: E402
import sys
from pathlib import Path

_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from dataclasses import dataclass
from typing import NamedTuple, TypedDict

from nooa.agentdoc import doc
from nooa.agentdoc._info import REQUIRED
from nooa.agentdoc._structured import extract_type_info


class TestExtractStructuredType:
    """Tests for extract_type_info() on various class kinds."""

    def test_extract_pydantic_model(self):
        """Extract fields from Pydantic model."""
        from pydantic import BaseModel, Field

        class UserModel(BaseModel):
            """User information."""

            name: str = Field(description="User's full name")
            age: int

        result = extract_type_info(UserModel)

        assert result.name == "UserModel"
        assert result.base == "BaseModel"
        assert len(result.fields) == 2
        # Fields are now FieldInfo objects
        assert result.fields[0].name == "name"
        assert result.fields[0].type == "str"
        assert result.fields[0].default is REQUIRED
        assert result.fields[0].description == "User's full name"
        assert result.fields[1].name == "age"
        assert result.fields[1].type == "int"
        assert result.fields[1].default is REQUIRED
        assert result.fields[1].description is None
        assert result.docstring == "User information."

    def test_extract_dataclass(self):
        """Extract fields from dataclass."""

        @dataclass
        class Point:
            """A 2D point."""

            x: int
            y: int
            label: str = "origin"

        result = extract_type_info(Point)

        assert result.name == "Point"
        assert result.base == "@dataclass"
        assert len(result.fields) == 3
        assert result.fields[0].name == "x"
        assert result.fields[0].type == "int"
        assert result.fields[0].default is REQUIRED
        assert result.fields[1].name == "y"
        assert result.fields[1].type == "int"
        assert result.fields[1].default is REQUIRED
        assert result.fields[2].name == "label"
        assert result.fields[2].type == "str"
        assert result.fields[2].default == "origin"

    def test_extract_namedtuple(self):
        """Extract fields from NamedTuple."""

        class Coord(NamedTuple):
            lat: float
            lon: float

        result = extract_type_info(Coord)

        assert result.name == "Coord"
        assert result.base == "NamedTuple"
        assert len(result.fields) == 2
        assert result.fields[0].name == "lat"
        assert result.fields[0].type == "float"
        assert result.fields[0].default is REQUIRED
        assert result.fields[1].name == "lon"
        assert result.fields[1].type == "float"
        assert result.fields[1].default is REQUIRED
        # Should NOT contain _tuplegetter
        assert "_tuplegetter" not in str(result)

    def test_extract_typeddict(self):
        """Extract fields from TypedDict."""

        class Config(TypedDict):
            host: str
            port: int

        result = extract_type_info(Config)

        assert result.name == "Config"
        assert result.base == "TypedDict"
        assert len(result.fields) == 2
        # Required fields should have no description
        for field in result.fields:
            assert field.description is None

    def test_extract_typeddict_optional_fields(self):
        """Test extracting TypedDict with optional fields (total=False)."""

        class Config(TypedDict, total=False):
            debug: bool
            log_level: str
            max_retries: int

        result = extract_type_info(Config)

        assert result.name == "Config"
        assert result.base == "TypedDict"
        assert len(result.fields) == 3

        # All fields should be marked as optional
        for field in result.fields:
            assert field.default is REQUIRED  # No default value (not None!)
            assert field.description == "optional"  # Marked as optional

        # Test rendering
        from nooa.agentdoc import doc

        output = doc(Config)
        assert "debug: bool  # optional" in output
        assert "log_level: str  # optional" in output
        assert "max_retries: int  # optional" in output
        # Should NOT show = None (which would be semantically incorrect)
        assert "= None" not in output

    def test_extract_plain_class(self):
        """Extract from plain class with annotations."""

        class Calculator:
            """A calculator."""

            value: float = 0.0

            def add(self, x: float) -> float:
                return self.value + x

            def multiply(self, x: float) -> float:
                return self.value * x

        result = extract_type_info(Calculator)

        assert result.name == "Calculator"
        assert result.base is None
        # Check field as FieldInfo
        value_field = next((f for f in result.fields if f.name == "value"), None)
        assert value_field is not None
        assert value_field.type == "float"
        assert value_field.default == 0.0
        # Methods are now CallableInfo objects with qualified names
        method_names = [m.name for m in result.methods]
        assert any("add" in name for name in method_names)
        assert any("multiply" in name for name in method_names)
        assert result.docstring == "A calculator."

    def test_extract_filters_dunder_methods(self):
        """Should not include __init__, __repr__, etc in methods."""

        @dataclass
        class Simple:
            x: int

        result = extract_type_info(Simple)

        # Dataclass generates __init__, __repr__, __eq__ etc
        method_names = [m.name for m in result.methods]
        assert "__init__" not in method_names
        assert "__repr__" not in method_names
        assert "__eq__" not in method_names

    def test_extract_attrs_class(self):
        """Extract fields from attrs class."""
        pytest = __import__("pytest")
        attr = pytest.importorskip("attr")

        @attr.s
        class Person:
            """A person."""

            name: str = attr.ib()
            age: int = attr.ib(default=0)

        result = extract_type_info(Person)

        assert result.name == "Person"
        assert result.base == "@attrs"
        assert len(result.fields) == 2
        assert result.fields[0].name == "name"
        assert result.fields[1].name == "age"
        assert result.fields[1].default == 0

    def test_extract_enum(self):
        """Extract members from Enum class."""
        import enum

        class Color(enum.Enum):
            """Color options."""

            RED = 1
            GREEN = 2
            BLUE = 3

        result = extract_type_info(Color)

        assert result.name == "Color"
        assert result.base == "Enum"
        assert len(result.fields) == 3
        # Enum "fields" are actually members
        assert result.fields[0].name == "RED"
        assert result.fields[0].type == "int"
        assert result.fields[0].default == 1
        assert result.fields[1].name == "GREEN"
        assert result.fields[1].type == "int"
        assert result.fields[1].default == 2
        assert result.fields[2].name == "BLUE"
        assert result.fields[2].type == "int"
        assert result.fields[2].default == 3
        assert result.docstring == "Color options."

    def test_extract_str_enum(self):
        """Extract members from StrEnum class."""
        import enum

        class Status(enum.StrEnum):
            PENDING = "pending"
            ACTIVE = "active"
            DONE = "done"

        result = extract_type_info(Status)

        assert result.name == "Status"
        assert result.base == "Enum"
        assert len(result.fields) == 3
        assert result.fields[0].name == "PENDING"
        assert result.fields[0].type == "str"
        assert result.fields[0].default == "pending"


class TestInitFieldExtraction:
    """Tests for extracting fields defined in __init__.

    Note: These tests use classes from fixture files because inspect.getsource()
    doesn't work for dynamically defined classes (in tests/REPL).
    """

    def test_extract_init_fields_from_real_file(self):
        """Extract __init__ fields from a real file-based class.

        This tests the core functionality using a fixture file since
        inspect.getsource() requires actual source files.
        """
        from fixtures.init_field_classes import SimpleCounter

        from nooa.agentdoc._structured import _extract_init_fields

        fields = _extract_init_fields(SimpleCounter)

        field_names = [f.name for f in fields]
        assert "count" in field_names
        assert "name" in field_names

        count_field = next(f for f in fields if f.name == "count")
        assert count_field.type == "int"
        assert count_field.default == 0

        name_field = next(f for f in fields if f.name == "name")
        assert name_field.type == "str"
        assert name_field.default == "default"

    def test_extract_init_annotated_fields_from_real_file(self):
        """Extract Annotated fields with descriptions from a real file."""
        from fixtures.init_field_classes import AnnotatedTool

        from nooa.agentdoc._structured import _extract_init_fields

        fields = _extract_init_fields(AnnotatedTool)

        field_names = [f.name for f in fields]
        assert "connection_string" in field_names
        assert "query_count" in field_names

        conn_field = next(f for f in fields if f.name == "connection_string")
        assert conn_field.type == "str"
        assert conn_field.description == "Database connection string"
        # Default is omitted when value comes from __init__ parameter
        assert conn_field.default is REQUIRED

        count_field = next(f for f in fields if f.name == "query_count")
        assert count_field.type == "int"
        assert count_field.description == "Total queries executed"
        assert count_field.default == 0

    def test_extract_init_skips_private_fields_from_real_file(self):
        """Should not extract private fields from __init__."""
        from fixtures.init_field_classes import SecretClass

        from nooa.agentdoc._structured import _extract_init_fields

        fields = _extract_init_fields(SecretClass)

        field_names = [f.name for f in fields]
        assert "public" in field_names
        assert "_private" not in field_names
        assert "_SecretClass__dunder" not in field_names

    def test_full_extraction_with_init_fields(self):
        """Test that extract_type_info includes __init__ fields."""
        from fixtures.init_field_classes import SimpleCounter

        result = extract_type_info(SimpleCounter)

        assert result.name == "SimpleCounter"
        field_names = [f.name for f in result.fields]
        assert "count" in field_names
        assert "name" in field_names

    def test_class_level_fields_not_duplicated(self):
        """Fields at class level should not be duplicated from __init__."""
        from fixtures.init_field_classes import MixedFields

        result = extract_type_info(MixedFields)

        field_names = [f.name for f in result.fields]
        # count is defined at class level, should appear only once
        assert field_names.count("count") == 1
        # name is only in __init__, should be present
        assert "name" in field_names


class TestInitFieldExtractionMRO:
    """Tests for MRO-based field extraction from parent __init__ methods.

    Verifies that _extract_init_fields walks the class hierarchy so child
    classes inherit parent fields, just like normal Python attribute inheritance.
    """

    def test_child_inherits_parent_init_fields(self):
        """Child with own __init__ should see both its fields and parent's."""
        from fixtures.init_field_classes import ChildWithOwnInit

        from nooa.agentdoc._structured import _extract_init_fields

        fields = _extract_init_fields(ChildWithOwnInit)
        field_names = [f.name for f in fields]

        # Child's own fields
        assert "child_field" in field_names
        # Parent's fields (inherited via MRO)
        assert "parent_field" in field_names
        # Shared field exists (child version wins)
        assert "shared_field" in field_names
        # Private parent fields should be excluded
        assert "_private_parent" not in field_names

    def test_child_overrides_parent_field(self):
        """When child redefines a parent field, child's version wins."""
        from fixtures.init_field_classes import ChildWithOwnInit

        from nooa.agentdoc._structured import _extract_init_fields

        fields = _extract_init_fields(ChildWithOwnInit)
        shared = next(f for f in fields if f.name == "shared_field")

        # Child sets shared_field = 42, parent sets it to 1
        assert shared.default == 42

    def test_child_without_init_inherits_all_parent_fields(self):
        """Child with no __init__ should get all parent's fields."""
        from fixtures.init_field_classes import ChildWithoutInit

        from nooa.agentdoc._structured import _extract_init_fields

        fields = _extract_init_fields(ChildWithoutInit)
        field_names = [f.name for f in fields]

        assert "parent_field" in field_names
        assert "shared_field" in field_names
        # No child-specific fields since it has no __init__
        assert "child_field" not in field_names

    def test_grandchild_inherits_through_chain(self):
        """Grandchild should see fields from all ancestors."""
        from fixtures.init_field_classes import GrandchildClass

        from nooa.agentdoc._structured import _extract_init_fields

        fields = _extract_init_fields(GrandchildClass)
        field_names = [f.name for f in fields]

        # Grandchild's own field
        assert "grandchild_field" in field_names
        # From ChildWithOwnInit
        assert "child_field" in field_names
        # From ParentWithFields (via MRO)
        assert "parent_field" in field_names
        # Shared field should be present
        assert "shared_field" in field_names

    def test_no_duplicate_fields_from_mro(self):
        """Each field name should appear exactly once."""
        from fixtures.init_field_classes import GrandchildClass

        from nooa.agentdoc._structured import _extract_init_fields

        fields = _extract_init_fields(GrandchildClass)
        field_names = [f.name for f in fields]

        # No duplicates
        assert len(field_names) == len(set(field_names))

    def test_child_fields_come_first(self):
        """Child fields should appear before parent fields (MRO order)."""
        from fixtures.init_field_classes import ChildWithOwnInit

        from nooa.agentdoc._structured import _extract_init_fields

        fields = _extract_init_fields(ChildWithOwnInit)
        field_names = [f.name for f in fields]

        # child_field and shared_field are from ChildWithOwnInit
        # parent_field is from ParentWithFields
        child_idx = field_names.index("child_field")
        parent_idx = field_names.index("parent_field")
        assert child_idx < parent_idx, "Child fields should appear before parent fields"

    def test_extract_type_info_includes_inherited_fields(self):
        """extract_type_info should include inherited fields from parent __init__."""
        from fixtures.init_field_classes import ChildWithOwnInit

        result = extract_type_info(ChildWithOwnInit)
        field_names = [f.name for f in result.fields]

        assert "child_field" in field_names
        assert "parent_field" in field_names
        assert "shared_field" in field_names

    def test_doc_output_includes_inherited_fields(self):
        """doc() should show inherited fields from parent __init__."""
        from fixtures.init_field_classes import ChildWithoutInit

        result = doc(ChildWithoutInit)

        assert "parent_field" in result
        assert "shared_field" in result


class TestFormatType:
    """Tests for format_type() utility."""

    def test_format_simple_types(self):
        from nooa.agentdoc._structured import format_type

        assert format_type(str) == "str"
        assert format_type(int) == "int"
        assert format_type(float) == "float"
        assert format_type(bool) == "bool"
        assert format_type(type(None)) == "None"

    def test_format_generic_types(self):
        from nooa.agentdoc._structured import format_type

        assert format_type(list[str]) == "list[str]"
        assert format_type(dict[str, int]) == "dict[str, int]"
        # Test modern union syntax
        assert "None" in format_type(str | None)

    def test_format_union_types(self):
        from nooa.agentdoc._structured import format_type

        # Test modern union syntax
        result = format_type(str | int)
        assert "str" in result
        assert "int" in result


class TestTypedDictAnnotatedDescriptions:
    def test_optional_field_with_annotated_description(self):
        """Optional TypedDict field with Annotated description must show the description, not just 'optional'."""
        from typing import Annotated

        from typing_extensions import TypedDict

        class Config(TypedDict, total=False):
            timeout: Annotated[int, "Request timeout in seconds"]

        info = extract_type_info(Config)
        field = next(f for f in info.fields if f.name == "timeout")
        assert field.description == "Request timeout in seconds"

    def test_optional_field_without_description_still_shows_optional(self):
        """Optional TypedDict field with no Annotated description still shows 'optional'."""
        from typing_extensions import TypedDict

        class Config(TypedDict, total=False):
            debug: bool

        info = extract_type_info(Config)
        field = next(f for f in info.fields if f.name == "debug")
        assert field.description == "optional"

    def test_required_field_with_annotated_description(self):
        """Required TypedDict field shows its description normally."""
        from typing import Annotated

        from typing_extensions import TypedDict

        class Config(TypedDict):
            host: Annotated[str, "Database hostname"]

        info = extract_type_info(Config)
        field = next(f for f in info.fields if f.name == "host")
        assert field.description == "Database hostname"


class TestPydanticDefaultNone:
    def test_field_with_default_none_not_required(self):
        """Pydantic Field(default=None) must render as = None, not as a required field."""
        try:
            from pydantic import BaseModel, Field
        except ImportError:
            import pytest

            pytest.skip("pydantic not installed")

        class MyModel(BaseModel):
            value: str | None = Field(default=None)

        info = extract_type_info(MyModel)
        field = next(f for f in info.fields if f.name == "value")
        assert field.default is not REQUIRED
        assert field.default is None


class TestSlotsDocRendering:
    def test_slots_class_no_member_descriptor_in_doc(self):
        """doc() on a __slots__ class must not render member_descriptor as a default."""
        from nooa.agentdoc import doc

        class Vec:
            __slots__ = ("x", "y")
            x: float
            y: float

        result = doc(Vec)
        assert "member_descriptor" not in result
        assert "member 'x'" not in result
        assert "member 'y'" not in result

    def test_slots_class_with_default_renders_default(self):
        """__slots__ dataclass with defaults renders the default, not member_descriptor."""
        import dataclasses

        from nooa.agentdoc import doc

        @dataclasses.dataclass(slots=True)
        class Point:
            x: float = 0.0
            y: float = 0.0

        result = doc(Point)
        assert "0.0" in result
        assert "member_descriptor" not in result

    def test_slots_class_required_fields_shown_without_default(self):
        """__slots__ fields without defaults are shown as required (no default displayed)."""
        from nooa.agentdoc._structured import REQUIRED, extract_type_info

        class Vec:
            __slots__ = ("x", "y")
            x: float
            y: float

        info = extract_type_info(Vec)
        for field in info.fields:
            assert field.default is REQUIRED or not str(field.default).startswith("<member")


class TestEnumDocFormatting:
    """Tests for enum member value formatting in doc()."""

    def test_enum_with_long_dict_value_is_truncated(self):
        """Enum members with long dict values must be truncated in doc() output."""
        import enum

        class Config(enum.Enum):
            """Config options."""

            DEFAULTS = {
                "alpha": "value_one",
                "beta": "value_two",
                "gamma": "value_three",
                "delta": "value_four",
                "epsilon": "value_five",
                "zeta": "value_six",
            }

        full_repr = repr(Config.DEFAULTS.value)
        assert len(full_repr) > 60, "Test precondition: repr must be longer than 60 chars"

        result = doc(Config)
        assert len(result) < len(full_repr) + 100  # output must be shorter than raw repr + overhead
        assert full_repr not in result

    def test_enum_with_simple_value_unchanged(self):
        """Enum members with simple integer values render as-is."""
        import enum

        class Status(enum.Enum):
            X = 1

        result = doc(Status)
        assert "X = 1" in result


class TestAugmentDocstringProseParameters:
    """Regression: 'parameters' in prose must not suppress Annotated Args augmentation."""

    def test_parameters_in_prose_does_not_block_args_section(self):
        """doc() must add Args from Annotated even when 'parameters' appears in prose."""
        from typing import Annotated

        def fn(
            x: Annotated[int, "The input value"],
            y: Annotated[str, "The label"],
        ) -> str:
            """Do something.

            The parameters are accepted but ignored in some modes.

            Returns:
                A string.
            """
            ...

        result = doc(fn)
        assert "Args:" in result
        assert "x: The input value" in result
        assert "y: The label" in result

    def test_parameters_as_section_header_still_blocks(self):
        """A proper 'Parameters:' section header still suppresses augmentation."""
        from typing import Annotated

        def fn(x: Annotated[int, "The input"]) -> None:
            """Do something.

            Parameters:
                x: Already documented here.
            """
            ...

        result = doc(fn)
        # Should NOT add a duplicate Args section
        assert result.count("Args:") == 0
        assert "Already documented here" in result


class TestFormatModuleInfoInaccessibleSymbol:
    """Tests for _format_module_info fallback when __all__ lists an inaccessible symbol."""

    def test_inaccessible_symbol_renders_comment(self):
        """A symbol listed in ordered_names but missing from all maps renders a comment."""
        from nooa.agentdoc import pformat
        from nooa.agentdoc._info import ModuleInfo

        info = ModuleInfo(
            name="fake_module",
            docstring=None,
            functions=[],
            classes=[],
            values=[("KNOWN", "'hello'")],
            ordered_names=["KNOWN", "MISSING_SYMBOL"],
        )

        result = pformat(info)
        assert "# MISSING_SYMBOL: (not accessible)" in result
        assert "KNOWN = 'hello'" in result

    def test_accessible_symbols_render_normally(self):
        """Symbols that are accessible are rendered without fallback comment."""
        from nooa.agentdoc import pformat
        from nooa.agentdoc._info import ModuleInfo

        info = ModuleInfo(
            name="fake_module",
            docstring=None,
            functions=[],
            classes=[("MyClass", "A simple class")],
            values=[("MY_CONST", "42")],
            ordered_names=["MyClass", "MY_CONST"],
        )

        result = pformat(info)
        assert "class MyClass" in result
        assert "MY_CONST = 42" in result
        assert "(not accessible)" not in result


class TestCollectItemsNarrowExcept:
    """Tests for pformat() — narrow exception handling for errant properties."""

    def test_property_raising_value_error_is_skipped(self):
        """A property that raises ValueError is skipped gracefully by pformat()."""
        from nooa.agentdoc import pformat

        class Problematic:
            x: int = 1

            @property
            def broken(self) -> int:  # type: ignore[return]
                raise ValueError("not available")

        obj = Problematic.__new__(Problematic)
        # Should not raise; broken property is silently skipped
        result = pformat(obj)
        assert "broken" not in result

    def test_property_raising_runtime_error_is_skipped(self):
        """A property that raises RuntimeError is skipped gracefully by pformat()."""
        from nooa.agentdoc import pformat

        class Problematic:
            value: str = "ok"

            @property
            def unstable(self) -> str:  # type: ignore[return]
                raise RuntimeError("not ready")

        obj = Problematic.__new__(Problematic)
        result = pformat(obj)
        assert "unstable" not in result


class TestPropertyExtraction:
    """Properties should appear as fields in doc() output."""

    def test_property_with_return_annotation_appears_as_field(self):
        """A @property with a return annotation shows as a typed field."""

        class Agent:
            @property
            def status(self) -> str:
                return "ok"

        result = doc(Agent)
        assert "status" in result
        assert "str" in result

    def test_property_docstring_shown_as_description(self):
        """First line of property docstring appears as a # comment."""

        class Agent:
            @property
            def status(self) -> str:
                """Current operational status."""
                return "ok"

        result = doc(Agent)
        assert "status" in result
        assert "Current operational status" in result

    def test_property_without_return_annotation(self):
        """A @property with no return annotation defaults to Any."""

        class Agent:
            @property
            def data(self):
                return {}

        result = doc(Agent)
        assert "data" in result

    def test_property_appears_in_output(self):
        """Properties appear as fields in doc() output."""

        class Agent:
            @property
            def count(self) -> int:
                return 0

        result = doc(Agent)
        assert "count: int" in result

    def test_property_hidden_via_spec(self):
        """spec(hidden=True) on a property excludes it from doc()."""
        from nooa.agentdoc import spec

        class Agent:
            @property
            def secret(self) -> str:
                return "shh"

            @property
            def visible(self) -> str:
                return "hi"

        spec(Agent, "secret", hidden=True)

        result = doc(Agent)
        assert "visible" in result
        assert "secret" not in result

    def test_private_property_excluded(self):
        """Properties prefixed with _ are excluded by default."""

        class Agent:
            @property
            def public(self) -> str:
                return "yes"

            @property
            def _private(self) -> str:
                return "no"

        result = doc(Agent)
        assert "public" in result
        assert "_private" not in result

    def test_property_inherited_from_parent(self):
        """Properties defined on a parent class appear in doc() of child."""

        class Base:
            @property
            def name(self) -> str:
                """The name."""
                return ""

        class Child(Base):
            pass

        result = doc(Child)
        assert "name" in result

    def test_property_no_default_shown(self):
        """Properties have no default value — no '= ...' in output."""

        class Agent:
            @property
            def count(self) -> int:
                return 0

        result = doc(Agent)
        # Should show  count: int  — no = something
        assert "count" in result
        lines = [line for line in result.splitlines() if "count" in line]
        assert lines, "count should appear in output"
        assert "=" not in lines[0]

    def test_property_alongside_regular_fields(self):
        """Properties and regular annotated fields coexist correctly."""

        class Agent:
            name: str = "agent"

            @property
            def upper_name(self) -> str:
                """Uppercased name."""
                return self.name.upper()

        result = doc(Agent)
        assert "name" in result
        assert "upper_name" in result
