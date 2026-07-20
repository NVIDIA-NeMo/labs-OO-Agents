# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for multi-type doc() functionality and inline_depth parameter."""

import pytest
from pydantic import BaseModel

from nooa.agentdoc import doc
from nooa.agentdoc._discover import discover_referenced_types


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
        """inline_depth=1 shows direct referenced types."""
        result = doc(Customer, inline_depth=1)

        assert "class Customer" in result
        assert "## Referenced Types" in result
        assert "class Address" in result

    def test_inline_depth_two_transitive_references(self):
        """inline_depth=2 shows transitive referenced types."""
        # Order -> Customer -> Address
        result = doc(Order, inline_depth=2)

        assert "class Order" in result
        assert "## Referenced Types" in result
        assert "class Customer" in result
        assert "class Address" in result  # Transitive through Customer

    def test_inline_depth_default_with_concise_false(self):
        """Default inline_depth=1 when concise=False."""
        result = doc(Customer)  # concise=False is default

        assert "## Referenced Types" in result
        assert "class Address" in result

    def test_inline_depth_default_with_concise_true(self):
        """Default inline_depth=0 when concise=True."""
        result = doc(Customer, concise=True)

        assert "## Referenced Types" not in result

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
