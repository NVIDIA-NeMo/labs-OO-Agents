# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for advanced type validation in ReturnValueValidator.

Tests for typing constructs:
- Union types (int | float, Union[int, str])
- Literal types (Literal["a", "b"])
- Annotated types (Annotated[int, ...])
- Heterogeneous tuples (tuple[int, str, float])
- Homogeneous tuples with ellipsis (tuple[int, ...])
"""

from typing import Annotated, Literal, Union

import pytest

from nooa import Agent, strategy
from nooa.strategies.generated_code import ReturnValueValidator
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient


class TestIsInstanceOf:
    """Unit tests for ReturnValueValidator._is_instance_of()."""

    @pytest.fixture
    def validator(self):
        return ReturnValueValidator()

    # === Union types ===

    def test_union_pipe_syntax_int(self, validator):
        """int | float should match int."""
        assert validator._is_instance_of(42, int | float) is True

    def test_union_pipe_syntax_float(self, validator):
        """int | float should match float."""
        assert validator._is_instance_of(3.14, int | float) is True

    def test_union_pipe_syntax_reject(self, validator):
        """int | float should reject str."""
        assert validator._is_instance_of("hello", int | float) is False

    def test_union_typing_syntax(self, validator):
        """typing.Union[int, str] syntax should also work (not just int | str)."""
        # Explicitly test typing.Union syntax - intentionally not using int | str
        union_type = Union[int, str]  # noqa: UP007
        assert validator._is_instance_of(42, union_type) is True
        assert validator._is_instance_of("hello", union_type) is True
        assert validator._is_instance_of(3.14, union_type) is False

    def test_optional_type(self, validator):
        """int | None should match int and None."""
        assert validator._is_instance_of(42, int | None) is True
        assert validator._is_instance_of(None, int | None) is True
        assert validator._is_instance_of("x", int | None) is False

    # === Literal types ===

    def test_literal_string_match(self, validator):
        """Literal["a", "b"] should match "a" or "b"."""
        assert validator._is_instance_of("a", Literal["a", "b"]) is True
        assert validator._is_instance_of("b", Literal["a", "b"]) is True

    def test_literal_string_reject(self, validator):
        """Literal["a", "b"] should reject "c"."""
        assert validator._is_instance_of("c", Literal["a", "b"]) is False

    def test_literal_int_match(self, validator):
        """Literal[1, 2, 3] should match 1, 2, or 3."""
        assert validator._is_instance_of(1, Literal[1, 2, 3]) is True
        assert validator._is_instance_of(2, Literal[1, 2, 3]) is True

    def test_literal_int_reject(self, validator):
        """Literal[1, 2, 3] should reject 4."""
        assert validator._is_instance_of(4, Literal[1, 2, 3]) is False

    def test_literal_bool_match(self, validator):
        """Literal[True] should match True."""
        assert validator._is_instance_of(True, Literal[True]) is True
        assert validator._is_instance_of(False, Literal[True]) is False

    # === Annotated types ===

    def test_annotated_int(self, validator):
        """Annotated[int, ...] should match int."""
        assert validator._is_instance_of(42, Annotated[int, "metadata"]) is True
        assert validator._is_instance_of("x", Annotated[int, "metadata"]) is False

    def test_annotated_str(self, validator):
        """Annotated[str, ...] should match str."""
        assert validator._is_instance_of("hello", Annotated[str, "field"]) is True
        assert validator._is_instance_of(42, Annotated[str, "field"]) is False

    def test_annotated_with_multiple_metadata(self, validator):
        """Annotated with multiple metadata should still work."""
        assert validator._is_instance_of(42, Annotated[int, "a", "b", "c"]) is True


class TestValidateGenericElements:
    """Unit tests for ReturnValueValidator._validate_generic_elements()."""

    @pytest.fixture
    def validator(self):
        return ReturnValueValidator()

    # === list with union elements ===

    def test_list_with_union_elements(self, validator):
        """list[int | str] should accept mixed int/str elements."""
        result = validator._validate_generic_elements(
            [1, "two", 3, "four"], list, (int | str,), "test_method"
        )
        assert result == [1, "two", 3, "four"]

    def test_list_with_union_rejects_invalid(self, validator):
        """list[int | str] should reject float elements."""
        with pytest.raises(TypeError) as exc_info:
            validator._validate_generic_elements([1, 2.5, 3], list, (int | str,), "test_method")
        assert "wrong type" in str(exc_info.value)

    # === list with Literal elements ===

    def test_list_with_literal_elements(self, validator):
        """list[Literal["a", "b"]] should accept "a" and "b"."""
        result = validator._validate_generic_elements(
            ["a", "b", "a"], list, (Literal["a", "b"],), "test_method"
        )
        assert result == ["a", "b", "a"]

    def test_list_with_literal_rejects_invalid(self, validator):
        """list[Literal["a", "b"]] should reject "c"."""
        with pytest.raises(TypeError) as exc_info:
            validator._validate_generic_elements(
                ["a", "c"], list, (Literal["a", "b"],), "test_method"
            )
        assert "wrong type" in str(exc_info.value)

    # === list with Annotated elements ===

    def test_list_with_annotated_elements(self, validator):
        """list[Annotated[int, ...]] should accept int elements."""
        result = validator._validate_generic_elements(
            [1, 2, 3], list, (Annotated[int, "meta"],), "test_method"
        )
        assert result == [1, 2, 3]

    def test_list_with_annotated_rejects_invalid(self, validator):
        """list[Annotated[int, ...]] should reject str elements."""
        with pytest.raises(TypeError) as exc_info:
            validator._validate_generic_elements(
                [1, "two"], list, (Annotated[int, "meta"],), "test_method"
            )
        assert "wrong type" in str(exc_info.value)

    # === Heterogeneous tuples ===

    def test_heterogeneous_tuple_valid(self, validator):
        """tuple[int, str, float] should accept (1, "x", 3.0)."""
        result = validator._validate_generic_elements(
            (1, "x", 3.0), tuple, (int, str, float), "test_method"
        )
        assert result == (1, "x", 3.0)

    def test_heterogeneous_tuple_wrong_type(self, validator):
        """tuple[int, str, float] should reject wrong element types."""
        with pytest.raises(TypeError) as exc_info:
            validator._validate_generic_elements(
                ("wrong", "x", 3.0), tuple, (int, str, float), "test_method"
            )
        assert "element 0" in str(exc_info.value).lower()

    def test_heterogeneous_tuple_wrong_length(self, validator):
        """tuple[int, str] should reject wrong length."""
        with pytest.raises(TypeError) as exc_info:
            validator._validate_generic_elements((1, "x", 3.0), tuple, (int, str), "test_method")
        assert "2 elements" in str(exc_info.value)

    # === Homogeneous tuples with ellipsis ===

    def test_homogeneous_tuple_valid(self, validator):
        """tuple[int, ...] should accept any length of ints."""
        result = validator._validate_generic_elements(
            (1, 2, 3, 4, 5), tuple, (int, ...), "test_method"
        )
        assert result == (1, 2, 3, 4, 5)

    def test_homogeneous_tuple_empty(self, validator):
        """tuple[int, ...] should accept empty tuple."""
        result = validator._validate_generic_elements((), tuple, (int, ...), "test_method")
        assert result == ()

    def test_homogeneous_tuple_rejects_invalid(self, validator):
        """tuple[int, ...] should reject str elements."""
        with pytest.raises(TypeError) as exc_info:
            validator._validate_generic_elements((1, 2, "three"), tuple, (int, ...), "test_method")
        assert "wrong type" in str(exc_info.value)

    # === dict with union values ===

    def test_dict_with_union_values(self, validator):
        """dict[str, int | float] should accept mixed int/float values."""
        result = validator._validate_generic_elements(
            {"a": 1, "b": 2.5, "c": 3}, dict, (str, int | float), "test_method"
        )
        assert result == {"a": 1, "b": 2.5, "c": 3}

    def test_dict_with_union_rejects_invalid_value(self, validator):
        """dict[str, int | float] should reject str values."""
        with pytest.raises(TypeError) as exc_info:
            validator._validate_generic_elements(
                {"a": 1, "b": "two"}, dict, (str, int | float), "test_method"
            )
        assert "wrong type" in str(exc_info.value)


class TestIntegrationWithAgent:
    """Integration tests for advanced type validation with agent execution."""

    @pytest.mark.asyncio
    async def test_union_return_type_int(self):
        """Agent method with int | float return should accept int."""
        llm = FakeLLMClient.with_code_responses(["return 42"])

        class TestAgent(Agent, llm=llm):
            @strategy(PurePythonStrategy())
            async def calculate(self) -> int | float:
                """Return a number."""
                ...

        result = await TestAgent().calculate()
        assert result == 42

    @pytest.mark.asyncio
    async def test_union_return_type_float(self):
        """Agent method with int | float return should accept float."""
        llm = FakeLLMClient.with_code_responses(["return 3.14"])

        class TestAgent(Agent, llm=llm):
            @strategy(PurePythonStrategy())
            async def calculate(self) -> int | float:
                """Return a number."""
                ...

        result = await TestAgent().calculate()
        assert result == 3.14

    @pytest.mark.asyncio
    async def test_list_with_union_elements(self):
        """Agent method with list[int | float] return should work."""
        llm = FakeLLMClient.with_code_responses(["return [1, 2.5, 3, 4.0]"])

        class TestAgent(Agent, llm=llm):
            @strategy(PurePythonStrategy())
            async def get_numbers(self) -> list[int | float]:
                """Return a list of numbers."""
                ...

        result = await TestAgent().get_numbers()
        assert result == [1, 2.5, 3, 4.0]

    @pytest.mark.asyncio
    async def test_literal_return_type(self):
        """Agent method with Literal return should work."""
        llm = FakeLLMClient.with_code_responses(['return "success"'])

        class TestAgent(Agent, llm=llm):
            @strategy(PurePythonStrategy())
            async def get_status(self) -> Literal["success", "failure", "pending"]:
                """Return a status."""
                ...

        result = await TestAgent().get_status()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_tuple_heterogeneous_return(self):
        """Agent method with tuple[int, str, bool] return should work."""
        llm = FakeLLMClient.with_code_responses(['return (42, "hello", True)'])

        class TestAgent(Agent, llm=llm):
            @strategy(PurePythonStrategy())
            async def get_data(self) -> tuple[int, str, bool]:
                """Return structured data."""
                ...

        result = await TestAgent().get_data()
        assert result == (42, "hello", True)
