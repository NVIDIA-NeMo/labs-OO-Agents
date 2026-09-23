# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for multi-type doc() functionality and inline_depth parameter."""

import types

import pytest
from pydantic import BaseModel

from nooa.agentdoc import doc, pformat
from nooa.agentdoc._discover import discover_referenced_types
from nooa.agentdoc._info import CallableInfo, FieldInfo, ModuleInfo, TypeInfo
from nooa.agentdoc._metadata import set_docs_metadata, set_field_metadata
from nooa.agentdoc.registry import (
    register_module_info_extractor,
    register_type_info_extractor,
    unregister_module_info_extractor,
    unregister_type_info_extractor,
)


# Test fixtures - types with overlapping referenced types
class Address(BaseModel):
    """A street address."""

    street: str
    city: str
    zip_code: str


class Customer(BaseModel):
    """A customer with an address."""

    name: str
    address: Address


class Order(BaseModel):
    """An order placed by a customer."""

    order_id: int
    customer: Customer


class Product(BaseModel):
    """A product in the catalog."""

    product_id: int
    name: str
    price: float


class Invoice(BaseModel):
    """An invoice for an order."""

    invoice_id: int
    order: Order
    customer: Customer  # Shared with Order


class Category(BaseModel):
    """A product category."""

    name: str
    description: str


class CategorizedProduct(BaseModel):
    """A product with category."""

    product: Product
    category: Category


# Simple types without references
class SimpleA(BaseModel):
    """Simple type A."""

    value_a: str


class SimpleB(BaseModel):
    """Simple type B."""

    value_b: int


class TestMultiTypeDoc:
    """Test multi-type doc() calls."""

    def test_doc_single_type_backward_compatible(self):
        """Single type calls work as before."""
        result = doc(Customer)

        assert "class Customer" in result
        assert "name: str" in result
        assert "address: Address" in result

    def test_doc_multiple_types_varargs(self):
        """doc(Type1, Type2) documents both types."""
        result = doc(SimpleA, SimpleB)

        assert "class SimpleA" in result
        assert "class SimpleB" in result
        assert "value_a: str" in result
        assert "value_b: int" in result

    def test_doc_multiple_instances_preserves_referenced_types(self):
        """Multi-instance docs discover the same contract types as type docs."""

        class Inner(BaseModel):
            value: str

        class Outer(BaseModel):
            inner: Inner

        class RuntimeDetail:
            detail: str = "runtime"

        outer = Outer(inner=Inner(value="x"))
        outer.__pydantic_extra__ = {"detail": RuntimeDetail()}
        result = doc(outer, SimpleA(value_a="a"), inline_depth=1)

        assert "class Outer(BaseModel):" in result
        assert "class SimpleA(BaseModel):" in result
        assert "## Referenced Types" in result
        assert result.count("class Inner(BaseModel):") == 1
        assert result.count("class RuntimeDetail:") == 1

    def test_doc_multiple_types_list(self):
        """doc([Type1, Type2]) flattens and documents both."""
        result = doc([SimpleA, SimpleB])

        assert "class SimpleA" in result
        assert "class SimpleB" in result

    def test_doc_multiple_types_tuple(self):
        """doc((Type1, Type2)) flattens and documents both."""
        result = doc((SimpleA, SimpleB))

        assert "class SimpleA" in result
        assert "class SimpleB" in result

    def test_doc_deduplication_of_referenced_types(self):
        """Referenced types appear only once across multiple primary types."""
        # Both Invoice and Order reference Customer
        result = doc(Invoice, Order, inline_depth=1)

        # Primary types should appear
        assert "class Invoice" in result
        assert "class Order" in result

        # Customer should appear only once in Referenced Types
        customer_count = result.count("class Customer")
        assert customer_count == 1, f"Customer appeared {customer_count} times, expected 1"

    def test_doc_no_duplicate_primary_types_in_references(self):
        """Primary types should not appear in Referenced Types section."""
        result = doc(Customer, Address, inline_depth=1)

        # Both should appear as primary types
        assert "class Customer" in result
        assert "class Address" in result

        # Address is referenced by Customer, but since it's a primary type,
        # it should NOT appear in Referenced Types section
        # There should be no "## Referenced Types" section since Address
        # is already a primary type
        lines = result.split("\n")
        in_ref_section = False
        for line in lines:
            if "## Referenced Types" in line:
                in_ref_section = True
            if in_ref_section and "class Address" in line:
                pytest.fail("Address appeared in Referenced Types even though it's a primary type")

    def test_doc_requires_at_least_one_object(self):
        """doc() with no arguments raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one object"):
            doc()

    def test_doc_empty_list_documented_as_value(self):
        """doc([]) documents empty list as a value (not flattened)."""
        result = doc([])

        # Empty list should be documented as a value
        assert "[]" in result


class TestTypeDepthParameter:
    """Test inline_depth parameter controlling reference recursion."""

    def test_inline_depth_zero_no_references(self):
        """inline_depth=0 shows no referenced types."""
        result = doc(Customer, inline_depth=0)

        assert "class Customer" in result
        assert "## Referenced Types" not in result

    def test_inline_depth_one_direct_references(self):
        """inline_depth=1 includes direct references but not their references."""
        # Order -> Customer -> Address
        result = doc(Order, inline_depth=1)

        assert "class Order" in result
        assert "## Referenced Types" in result
        assert "class Customer" in result
        assert "class Address" not in result

    def test_inline_depth_two_transitive_references(self):
        """inline_depth=2 shows transitive referenced types."""
        # Order -> Customer -> Address
        result = doc(Order, inline_depth=2)

        assert "class Order" in result
        assert "## Referenced Types" in result
        assert "class Customer" in result
        assert "class Address" in result  # Transitive through Customer

    def test_inline_depth_bounds_multi_object_references(self):
        """Multi-object docs use the same direct-versus-transitive semantics."""
        direct = doc(Order, SimpleA, inline_depth=1)
        transitive = doc(Order, SimpleA, inline_depth=2)

        assert "class Customer" in direct
        assert "class Address" not in direct
        assert "class Address" in transitive

    def test_inline_depth_default_with_concise_false(self):
        """Default inline_depth=1 when concise=False."""
        result = doc(Customer)  # concise=False is default

        assert "## Referenced Types" in result
        assert "class Address" in result

    def test_inline_depth_default_with_concise_true(self):
        """Default inline_depth=1 is independent of concise docstrings."""
        result = doc(Customer, concise=True)

        assert "## Referenced Types" in result
        assert "class Address" in result

    @pytest.mark.parametrize("invalid_depth", [None, -1, 1.5, "1", True])
    def test_inline_depth_rejects_non_nonnegative_integers(self, invalid_depth):
        """inline_depth accepts only non-negative integers."""
        error = ValueError if invalid_depth == -1 else TypeError
        with pytest.raises(error, match="inline_depth must be a non-negative integer"):
            doc(Customer, inline_depth=invalid_depth)

    def test_inline_depth_override_with_concise_true(self):
        """Explicit inline_depth overrides concise=True default."""
        result = doc(Customer, concise=True, inline_depth=1)

        assert "## Referenced Types" in result
        assert "class Address" in result

    def test_inline_depth_override_with_concise_false(self):
        """Explicit inline_depth=0 overrides concise=False default."""
        result = doc(Customer, concise=False, inline_depth=0)

        assert "## Referenced Types" not in result


class TestDiscoverWithSeen:
    """Test discover_referenced_types with seen parameter."""

    def test_seen_excludes_types(self):
        """Types in seen set are excluded from results."""
        # Customer references Address
        seen = {Address}
        result = discover_referenced_types(Customer, seen=seen)

        assert Address not in result

    def test_seen_empty_includes_all(self):
        """Empty seen set includes all referenced types."""
        seen = set()
        result = discover_referenced_types(Customer, seen=seen)

        assert Address in result

    def test_seen_none_includes_all(self):
        """seen=None includes all referenced types."""
        result = discover_referenced_types(Customer, seen=None)

        assert Address in result


class TestDataListNotFlattened:
    """Test that data lists are not flattened."""

    def test_list_of_ints_not_flattened(self):
        """doc([1, 2, 3]) treats it as a value, not multiple objects."""
        result = doc([1, 2, 3])

        # Should show list representation, not individual ints
        assert "[1, 2, 3]" in result or "1, 2, 3" in result

    def test_list_of_strings_not_flattened(self):
        """doc(['a', 'b']) treats it as a value."""
        result = doc(["a", "b"])

        # Should show list representation
        assert "'a'" in result
        assert "'b'" in result

    def test_list_of_types_is_flattened(self):
        """doc([Type1, Type2]) flattens to doc(Type1, Type2)."""
        result = doc([SimpleA, SimpleB])

        # Both types should be documented
        assert "class SimpleA" in result
        assert "class SimpleB" in result


class TestMultiTypeOutput:
    """Test output format for multi-type documentation."""

    def test_primary_types_before_references(self):
        """Primary types appear before Referenced Types section."""
        result = doc(Order, Product, inline_depth=1)
        lines = result.split("\n")

        order_idx = None
        product_idx = None
        ref_idx = None

        for i, line in enumerate(lines):
            if "class Order" in line and order_idx is None:
                order_idx = i
            if "class Product" in line and product_idx is None:
                product_idx = i
            if "## Referenced Types" in line:
                ref_idx = i
                break

        assert order_idx is not None, "Order not found"
        assert product_idx is not None, "Product not found"
        assert ref_idx is not None, "Referenced Types section not found"
        assert order_idx < ref_idx, "Order should appear before Referenced Types"
        assert product_idx < ref_idx, "Product should appear before Referenced Types"

    def test_multi_type_with_functions(self):
        """doc() works with functions as well as types."""

        def my_function(x: Customer) -> Order:
            """Convert customer to order."""
            return Order(order_id=1, customer=x)

        result = doc(my_function, SimpleA)

        assert "def my_function" in result
        assert "class SimpleA" in result


class TestUnifiedMultiDocumentPolicy:
    """Regression coverage for policy parity between one and many primaries."""

    def test_expand_false_is_honored_at_every_graph_level(self):
        class Hidden(BaseModel):
            value: int

        class Middle(BaseModel):
            hidden: Hidden

        class Root(BaseModel):
            middle: Middle

        set_docs_metadata(Hidden, expand=False)
        result = doc(Root, SimpleA, inline_depth=2)
        assert "class Middle(BaseModel):" in result
        assert "class Hidden(BaseModel):" not in result

    def test_referenced_members_are_not_capped_at_fifty(self):
        Wide = type(
            "Wide",
            (),
            {"__annotations__": {f"field_{index:02}": int for index in range(55)}},
        )

        class Root:
            wide: Wide

        result = doc(Root, SimpleA)
        assert "field_54: int" in result
        assert "# ..." not in result

    def test_references_are_globally_sorted_not_breadth_first(self):
        class ADeep(BaseModel):
            value: int

        class ZDirect(BaseModel):
            deep: ADeep

        class Root(BaseModel):
            direct: ZDirect

        result = doc(Root, SimpleA, inline_depth=2)
        assert result.index("class ADeep(BaseModel):") < result.index("class ZDirect(BaseModel):")

    def test_duplicate_primaries_are_identity_deduplicated(self):
        obj = SimpleA(value_a="same")
        result = doc(obj, obj)
        assert result.count("class SimpleA(BaseModel):") == 1

    def test_info_objects_are_primaries_not_discovered_dataclasses(self):
        type_info = TypeInfo(
            name="Described",
            base=None,
            fields=[FieldInfo("value", "int")],
            methods=[],
            docstring=None,
        )
        callable_info = CallableInfo("work", "()", "None", None)
        result = doc(type_info, callable_info)
        assert "class Described:" in result
        assert "def work() -> None:" in result
        assert "class TypeInfo" not in result
        assert "class CallableInfo" not in result
        assert "## Referenced Types" not in result


class TestIdentityAndNormalizedSubjects:
    def test_equal_hash_colliding_types_remain_distinct_and_sort_by_module(self):
        class EqualMeta(type):
            def __eq__(cls, other):
                return isinstance(other, EqualMeta)

            def __hash__(cls):
                return 1

        first = EqualMeta("Same", (), {"__module__": "zzz", "__doc__": "Z module."})
        second = EqualMeta("Same", (), {"__module__": "aaa", "__doc__": "A module."})
        root = type(
            "CollisionRoot",
            (),
            {"__module__": __name__, "__annotations__": {"first": first, "second": second}},
        )

        result = doc(root, SimpleA)
        assert result.count("class Same:") == 2
        assert result.index("A module.") < result.index("Z module.")

    def test_unhashable_metaclass_types_can_be_discovered(self):
        class UnhashableMeta(type):
            __hash__ = None

        referenced = UnhashableMeta("UnhashableReference", (), {"__module__": __name__})
        root = type(
            "UnhashableRoot",
            (),
            {"__module__": __name__, "__annotations__": {"reference": referenced}},
        )

        result = doc(root, SimpleA)
        assert "class UnhashableReference:" in result

    def test_custom_instance_extractor_runs_once_and_its_values_seed_references(self):
        class RuntimeOnly:
            pass

        class Custom:
            pass

        calls = 0

        @register_type_info_extractor(Custom)
        def extract(obj):
            nonlocal calls
            calls += 1
            return (
                TypeInfo(
                    name="Custom",
                    base=None,
                    fields=[FieldInfo("runtime", "RuntimeOnly")],
                    methods=[],
                    docstring=None,
                ),
                {"runtime": RuntimeOnly()},
            )

        try:
            result = doc(Custom(), SimpleA)
        finally:
            unregister_type_info_extractor(Custom)

        assert calls == 1
        assert "runtime: RuntimeOnly" in result
        assert result.count("class RuntimeOnly:") == 1

    def test_supports_instance_values_runtime_only_reference_matches_single_doc(self):
        class RuntimeOnly:
            pass

        class ProtocolValues:
            declared: str

            def __instance_values__(self):
                return {"declared": "yes", "runtime": RuntimeOnly()}

        value = ProtocolValues()
        single = doc(value)
        multiple = doc(value, SimpleA)
        assert "runtime: RuntimeOnly" in single
        assert "runtime: RuntimeOnly" in multiple
        assert "class RuntimeOnly:" in single
        assert "class RuntimeOnly:" in multiple

    def test_mixed_function_module_and_value_with_no_expansion(self):
        def work(value: Customer) -> Order:
            return Order(order_id=1, customer=value)

        module = types.ModuleType("sample_module", "Sample module.")
        result = doc(work, module, 42, inline_depth=0)
        assert "def work" in result
        assert "# sample_module" in result
        assert "42" in result
        assert "## Referenced Types" not in result


class TestDocumentPipelineCoverage:
    """Focused branch and parity coverage for the shared document pipeline."""

    def test_class_and_referenced_extractors_each_run_once(self):
        class Referenced:
            marker: int

        class Root:
            child: Referenced

        root_calls = 0
        referenced_calls = 0

        @register_type_info_extractor(Root)
        def extract_root(obj):
            nonlocal root_calls
            root_calls += 1
            return TypeInfo("Root", None, [FieldInfo("child", "Referenced")], [], None)

        @register_type_info_extractor(Referenced)
        def extract_referenced(obj):
            nonlocal referenced_calls
            referenced_calls += 1
            return TypeInfo("Referenced", None, [FieldInfo("marker", "int")], [], None)

        try:
            result = doc(Root)
        finally:
            unregister_type_info_extractor(Root)
            unregister_type_info_extractor(Referenced)

        assert root_calls == 1
        assert referenced_calls == 1
        assert result.count("class Referenced:") == 1

    def test_instance_extractor_returning_type_info_runs_once(self):
        class Custom:
            def __init__(self):
                self.value = 17

        calls = 0

        @register_type_info_extractor(Custom)
        def extract(obj):
            nonlocal calls
            calls += 1
            return TypeInfo("Custom", None, [FieldInfo("value", "int")], [], None)

        try:
            result = doc(Custom())
        finally:
            unregister_type_info_extractor(Custom)

        assert calls == 1
        assert "value: int = 17" in result

    def test_cycle_and_diamond_are_deduplicated_and_exclude_primary(self):
        class Root:
            pass

        class Shared:
            pass

        class Left:
            pass

        class Right:
            pass

        Root.__annotations__ = {"left": Left, "right": Right}
        Left.__annotations__ = {"root": Root, "shared": Shared}
        Right.__annotations__ = {"root": Root, "shared": Shared}

        result = doc(Root, inline_depth=3)

        assert result.count("class Root:") == 1
        assert result.count("class Left:") == 1
        assert result.count("class Right:") == 1
        assert result.count("class Shared:") == 1
        assert result.count("## Referenced Types") == 1

    @pytest.mark.parametrize("depth", [1, 2])
    def test_single_and_multi_reference_suffixes_match(self, depth):
        def suffix(text):
            return text.split("## Referenced Types", 1)[1]

        single = doc(Order, inline_depth=depth)
        multiple = doc(Order, SimpleA, inline_depth=depth)

        assert suffix(single) == suffix(multiple)

    def test_function_and_bound_method_follow_reference_depth(self):
        def work(customer: Customer) -> Order:
            return Order(order_id=1, customer=customer)

        class Service:
            def work(self, customer: Customer) -> Order:
                return Order(order_id=1, customer=customer)

        for callable_obj in (work, Service().work):
            direct = doc(callable_obj, 42, inline_depth=1)
            transitive = doc(callable_obj, 42, inline_depth=2)
            assert direct.count("class Customer(BaseModel):") == 1
            assert direct.count("class Order(BaseModel):") == 1
            assert "class Address(BaseModel):" not in direct
            assert transitive.count("class Address(BaseModel):") == 1

    def test_structured_info_primaries_preserve_order_and_do_not_discover_internals(self):
        type_info = TypeInfo("Described", None, [FieldInfo("value", "int")], [], None)
        callable_info = CallableInfo("work", "()", "None", None)
        module_info = ModuleInfo(
            "curated",
            "Curated module.",
            [CallableInfo("run", "()", "None", "Run it.")],
        )

        result = doc(type_info, callable_info, module_info)

        assert result.index("class Described:") < result.index("def work() -> None:")
        assert result.index("def work() -> None:") < result.index("# curated")
        assert "def run() -> None:" in result
        assert "## Referenced Types" not in result
        assert "class TypeInfo" not in result
        assert "class ModuleInfo" not in result

    def test_distinct_equal_primary_instances_are_not_deduplicated(self):
        class EqualValue:
            marker: str

            def __init__(self, marker: str):
                self.marker = marker

            def __eq__(self, other):
                return isinstance(other, EqualValue)

            def __hash__(self):
                return 1

        result = doc(EqualValue("first"), EqualValue("second"), inline_depth=0)

        assert "marker: str = 'first'" in result
        assert "marker: str = 'second'" in result

    def test_hidden_declared_and_runtime_fields_do_not_seed_references(self):
        class HiddenDeclared:
            pass

        class HiddenRuntime:
            pass

        class VisibleRuntime:
            pass

        class Subject:
            declared: HiddenDeclared

            def __init__(self):
                self.declared = HiddenDeclared()
                self.secret_runtime = HiddenRuntime()
                self.visible_runtime = VisibleRuntime()

        value = Subject()
        set_field_metadata(value, "declared", hidden=True)
        set_field_metadata(value, "secret_runtime", hidden=True)

        for result in (doc(value), doc(value, SimpleA)):
            assert "declared:" not in result
            assert "secret_runtime:" not in result
            assert "class HiddenDeclared:" not in result
            assert "class HiddenRuntime:" not in result
            assert "visible_runtime: VisibleRuntime" in result
            assert result.count("class VisibleRuntime:") == 1

    def test_same_name_and_module_references_sort_by_qualname(self):
        later = type("Same", (), {"__module__": "shared", "__doc__": "Later marker."})
        earlier = type("Same", (), {"__module__": "shared", "__doc__": "Earlier marker."})
        later.__qualname__ = "Zulu.Same"
        earlier.__qualname__ = "Alpha.Same"

        class Root:
            pass

        Root.__annotations__ = {"later": later, "earlier": earlier}
        result = doc(Root)

        assert result.count("class Same:") == 2
        assert result.index("Earlier marker.") < result.index("Later marker.")

    def test_registered_module_extractor_runs_once_in_mixed_document(self):
        module = types.ModuleType("curated_module")
        calls = 0

        @register_module_info_extractor(module)
        def extract(mod):
            nonlocal calls
            calls += 1
            return ModuleInfo(
                "curated_module",
                "Curated module.",
                [CallableInfo("curated", "()", "None", "Curated callable.")],
            )

        try:
            result = doc(module, 42)
        finally:
            unregister_module_info_extractor(module)

        assert calls == 1
        assert result.count("# curated_module") == 1
        assert result.count("def curated() -> None:") == 1
        assert result.rstrip().endswith("42")

    def test_direct_callable_pformat_preserves_reference_spacing(self):
        class Alpha:
            pass

        class Beta:
            pass

        def work(alpha: Alpha) -> Beta:
            return Beta()

        result = pformat(work)

        assert result == (
            "def work(alpha: Alpha) -> Beta:\n"
            "    ...\n\n"
            "## Referenced Types\n"
            "class Alpha:\n\n"
            "class Beta:"
        )
