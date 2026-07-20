# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ArgumentValidator class.

Tests comprehensive argument validation (arity + types) for agent method calls.
"""

import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from nooa.strategies.generated_code import ArgumentValidator


class SampleModel(BaseModel):
    """Pydantic model for testing."""

    name: str
    value: int


class SampleTypedDict(TypedDict):
    """TypedDict for testing."""

    key: str
    count: int


# --- Test Functions (used by ArgumentValidator) ---


def no_args_func():
    """Function with no arguments."""
    pass


def required_args_func(a: str, b: int):
    """Function with required arguments."""
    pass


def with_defaults_func(a: str, b: int = 10, c: str = "default"):
    """Function with default arguments."""
    pass


def optional_arg_func(a: str, b: int | None = None):
    """Function with Optional argument."""
    pass


def union_arg_func(a: str | int):
    """Function with Union type argument."""
    pass


def list_arg_func(values: list[float]):
    """Function with generic list argument."""
    pass


def dict_arg_func(data: dict[str, int]):
    """Function with generic dict argument."""
    pass


def pydantic_arg_func(model: SampleModel):
    """Function with Pydantic model argument."""
    pass


def typeddict_arg_func(data: SampleTypedDict):
    """Function with TypedDict argument."""
    pass


def self_func(self, a: str, b: int):
    """Method with self parameter."""
    pass


def no_type_hints_func(a, b):
    """Function without type hints."""
    pass


def mixed_hints_func(a: str, b):
    """Function with partial type hints."""
    pass


def kwargs_func(a: str, **kwargs):
    """Function with **kwargs."""
    pass


def args_func(*args: str):
    """Function with *args."""
    pass


# --- Tests ---


class TestArgumentValidatorArity:
    """Tests for arity (required args) validation."""

    def test_no_args_succeeds(self):
        """Function with no args should accept no args."""
        validator = ArgumentValidator()
        validator.validate(no_args_func, (), {})

    def test_missing_required_arg_raises(self):
        """Missing required argument should raise TypeError."""
        validator = ArgumentValidator()
        with pytest.raises(TypeError) as exc_info:
            validator.validate(required_args_func, ("hello",), {})

        error_msg = str(exc_info.value)
        assert "required_args_func" in error_msg
        assert "missing" in error_msg.lower() or "required" in error_msg.lower()

    def test_all_required_args_provided(self):
        """Providing all required args should succeed."""
        validator = ArgumentValidator()
        validator.validate(required_args_func, ("hello", 42), {})

    def test_extra_positional_arg_raises(self):
        """Extra positional argument should raise TypeError."""
        validator = ArgumentValidator()
        with pytest.raises(TypeError) as exc_info:
            validator.validate(required_args_func, ("hello", 42, "extra"), {})

        error_msg = str(exc_info.value)
        assert "required_args_func" in error_msg

    def test_unexpected_kwarg_raises(self):
        """Unexpected keyword argument should raise TypeError."""
        validator = ArgumentValidator()
        with pytest.raises(TypeError) as exc_info:
            validator.validate(required_args_func, ("hello", 42), {"unexpected": "value"})

        error_msg = str(exc_info.value)
        assert "required_args_func" in error_msg
        assert "unexpected" in error_msg.lower()

    def test_default_args_can_be_omitted(self):
        """Arguments with defaults can be omitted."""
        validator = ArgumentValidator()
        # Only provide required arg 'a', omit 'b' and 'c' which have defaults
        validator.validate(with_defaults_func, ("hello",), {})

    def test_default_args_can_be_provided(self):
        """Arguments with defaults can be explicitly provided."""
        validator = ArgumentValidator()
        validator.validate(with_defaults_func, ("hello", 20, "custom"), {})

    def test_kwargs_passed_correctly(self):
        """Keyword arguments should work."""
        validator = ArgumentValidator()
        validator.validate(required_args_func, (), {"a": "hello", "b": 42})

    def test_mixed_positional_and_kwargs(self):
        """Mix of positional and keyword args should work."""
        validator = ArgumentValidator()
        validator.validate(required_args_func, ("hello",), {"b": 42})


class TestArgumentValidatorTypes:
    """Tests for type validation using Pydantic."""

    def test_correct_types_succeed(self):
        """Correct types should pass validation."""
        validator = ArgumentValidator()
        validator.validate(required_args_func, ("hello", 42), {})

    def test_wrong_type_raises(self):
        """Wrong type should raise TypeError with clear message."""
        validator = ArgumentValidator()
        with pytest.raises(TypeError) as exc_info:
            validator.validate(required_args_func, ("hello", "not_an_int"), {})

        error_msg = str(exc_info.value)
        assert "required_args_func" in error_msg
        assert "wrong type" in error_msg.lower() or "int" in error_msg.lower()
        assert "b" in error_msg  # Parameter name should be mentioned

    def test_optional_none_succeeds(self):
        """None for Optional argument should succeed."""
        validator = ArgumentValidator()
        validator.validate(optional_arg_func, ("hello", None), {})

    def test_optional_with_value_succeeds(self):
        """Providing value for Optional argument should succeed."""
        validator = ArgumentValidator()
        validator.validate(optional_arg_func, ("hello", 42), {})

    def test_union_first_type_succeeds(self):
        """First type in Union should succeed."""
        validator = ArgumentValidator()
        validator.validate(union_arg_func, ("hello",), {})

    def test_union_second_type_succeeds(self):
        """Second type in Union should succeed."""
        validator = ArgumentValidator()
        validator.validate(union_arg_func, (42,), {})

    def test_union_wrong_type_raises(self):
        """Type not in Union should raise TypeError."""
        validator = ArgumentValidator()
        with pytest.raises(TypeError) as exc_info:
            validator.validate(union_arg_func, ([1, 2, 3],), {})

        error_msg = str(exc_info.value)
        assert "union_arg_func" in error_msg

    def test_list_generic_succeeds(self):
        """Correct list[float] should succeed."""
        validator = ArgumentValidator()
        validator.validate(list_arg_func, ([1.0, 2.5, 3.0],), {})

    def test_list_wrong_element_type_raises(self):
        """List with wrong element types should raise TypeError."""
        validator = ArgumentValidator()
        with pytest.raises(TypeError) as exc_info:
            validator.validate(list_arg_func, (["a", "b", "c"],), {})

        error_msg = str(exc_info.value)
        assert "list_arg_func" in error_msg

    def test_dict_generic_succeeds(self):
        """Correct dict[str, int] should succeed."""
        validator = ArgumentValidator()
        validator.validate(dict_arg_func, ({"a": 1, "b": 2},), {})

    def test_dict_wrong_value_type_raises(self):
        """Dict with wrong value types should raise TypeError."""
        validator = ArgumentValidator()
        with pytest.raises(TypeError) as exc_info:
            validator.validate(dict_arg_func, ({"a": "not_int"},), {})

        error_msg = str(exc_info.value)
        assert "dict_arg_func" in error_msg


class TestArgumentValidatorPydanticModels:
    """Tests for Pydantic model argument validation."""

    def test_pydantic_model_instance_succeeds(self):
        """Passing Pydantic model instance should succeed."""
        validator = ArgumentValidator()
        model = SampleModel(name="test", value=42)
        validator.validate(pydantic_arg_func, (model,), {})

    def test_pydantic_model_dict_coerced(self):
        """Passing dict that matches Pydantic model should succeed (Pydantic coerces)."""
        validator = ArgumentValidator()
        # Pydantic accepts dicts for model types and coerces them to model instances
        validator.validate(pydantic_arg_func, ({"name": "test", "value": 42},), {})

    def test_pydantic_model_wrong_type_raises(self):
        """Passing wrong type for Pydantic model arg should raise TypeError."""
        validator = ArgumentValidator()
        with pytest.raises(TypeError) as exc_info:
            validator.validate(pydantic_arg_func, ("not_a_model",), {})

        error_msg = str(exc_info.value)
        assert "pydantic_arg_func" in error_msg


class TestArgumentValidatorTypedDict:
    """Tests for TypedDict argument validation."""

    def test_typeddict_correct_dict_succeeds(self):
        """Dict matching TypedDict structure should succeed."""
        validator = ArgumentValidator()
        validator.validate(typeddict_arg_func, ({"key": "test", "count": 42},), {})

    def test_typeddict_missing_key_raises(self):
        """Dict missing required TypedDict key should raise TypeError."""
        validator = ArgumentValidator()
        with pytest.raises(TypeError) as exc_info:
            validator.validate(typeddict_arg_func, ({"key": "test"},), {})

        error_msg = str(exc_info.value)
        assert "typeddict_arg_func" in error_msg

    def test_typeddict_wrong_value_type_raises(self):
        """Dict with wrong value type should raise TypeError."""
        validator = ArgumentValidator()
        with pytest.raises(TypeError) as exc_info:
            validator.validate(typeddict_arg_func, ({"key": "test", "count": "not_int"},), {})

        error_msg = str(exc_info.value)
        assert "typeddict_arg_func" in error_msg


class TestArgumentValidatorSelfParameter:
    """Tests for methods with 'self' parameter."""

    def test_self_is_skipped(self):
        """'self' parameter should be excluded from validation."""
        validator = ArgumentValidator()
        # Pass args without self - validator should handle this
        validator.validate(self_func, ("hello", 42), {})

    def test_self_missing_required_raises(self):
        """Missing required args (after self) should raise TypeError."""
        validator = ArgumentValidator()
        with pytest.raises(TypeError) as exc_info:
            validator.validate(self_func, ("hello",), {})

        error_msg = str(exc_info.value)
        assert "self_func" in error_msg


class TestArgumentValidatorEdgeCases:
    """Tests for edge cases."""

    def test_no_type_hints_skips_type_validation(self):
        """Functions without type hints should skip type validation."""
        validator = ArgumentValidator()
        # Any types should pass since there are no hints
        validator.validate(no_type_hints_func, ("hello", 42), {})
        validator.validate(no_type_hints_func, (123, [1, 2, 3]), {})

    def test_partial_type_hints(self):
        """Functions with partial hints should validate only hinted params."""
        validator = ArgumentValidator()
        # 'a' has hint (str), 'b' does not
        validator.validate(mixed_hints_func, ("hello", 42), {})
        validator.validate(mixed_hints_func, ("hello", "anything"), {})

        # Wrong type for 'a' should fail
        with pytest.raises(TypeError):
            validator.validate(mixed_hints_func, (123, "anything"), {})

    def test_kwargs_function(self):
        """Functions with **kwargs should accept extra kwargs."""
        validator = ArgumentValidator()
        validator.validate(kwargs_func, ("hello",), {"extra": "value", "another": 123})

    def test_args_function(self):
        """Functions with *args should accept extra positional args."""
        validator = ArgumentValidator()
        validator.validate(args_func, ("a", "b", "c", "d"), {})

    def test_error_message_includes_signature(self):
        """Error messages should include the method signature."""
        validator = ArgumentValidator()
        with pytest.raises(TypeError) as exc_info:
            validator.validate(required_args_func, (), {})

        error_msg = str(exc_info.value)
        assert "Signature:" in error_msg
        assert "a: str" in error_msg or "a:" in error_msg

    def test_error_message_includes_provided_args(self):
        """Error messages should show what arguments were provided."""
        validator = ArgumentValidator()
        with pytest.raises(TypeError) as exc_info:
            validator.validate(required_args_func, ("hello",), {})

        error_msg = str(exc_info.value)
        assert "Provided:" in error_msg
        assert "hello" in error_msg


class TestArgumentValidatorWithWrappedFunctions:
    """Tests for functions that have been wrapped (have _original attribute)."""

    def test_wrapped_function_uses_original(self):
        """Validator should use _original attribute if present."""

        def original(a: str, b: int):
            pass

        def wrapper(*args, **kwargs):
            pass

        wrapper._original = original

        validator = ArgumentValidator()
        # Should validate against original's signature
        validator.validate(wrapper, ("hello", 42), {})

        with pytest.raises(TypeError):
            validator.validate(wrapper, ("hello",), {})
