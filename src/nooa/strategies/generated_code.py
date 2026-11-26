# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for validating and managing LLM-generated Python code.

This module is intentionally strategy-agnostic: it contains no event management logic
and instead returns structured issues that strategies can convert into feedback.

Key invariants:
- Keep runtime-level safety validation in `validate_planning_code()` (imports, exec/eval/compile).
- Helper functions are compiled by the cell engine itself
  (`nooa.runtime.execution_namespace`), never pre-compiled here.
"""

import ast
import inspect
import types
from typing import Any, get_type_hints

from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError
from pydantic.errors import PydanticSchemaGenerationError

from nooa.agentdoc import pformat
from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG, TruncationConfig
from nooa.strategies.current_call import bind_call_arguments


def _pformat(value: Any, tc: TruncationConfig) -> str:
    """Format a value using truncation config's value-render settings."""
    return pformat(value, **tc.event_format.model_dump())


class GeneratedCodeValidator:
    """REPL-policy validation (not security)."""

    def validate(self, code: str, agent: Any) -> list[str]:
        """Return a list of human-readable validation error messages."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Let execution handle syntax errors (keeps Python traceback formatting).
            return []

        errors: list[str] = []

        await_errors = self._missing_await_errors(tree, agent)
        errors.extend(await_errors)

        return errors

    def _missing_await_errors(self, tree: ast.AST, agent: Any) -> list[str]:
        async_method_names = self._collect_async_method_names(agent)
        if not async_method_names:
            return []

        parent_map = self._build_parent_map(tree)

        errors: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                continue

            method_name = node.func.attr
            if method_name not in async_method_names:
                continue

            if not self._is_in_await_or_comprehension(node, parent_map):
                errors.append(
                    f"Method `{method_name}` is async and must be called with `await`.\n"
                    f"   Example: result = await self.{method_name}(...)"
                )

        return errors

    def _collect_async_method_names(self, agent: Any) -> set[str]:
        from nooa.agentdoc.visibility import is_hidden_method

        names: set[str] = set()

        # 1) Class-level async methods
        for attr_name in dir(agent.__class__):
            if attr_name.startswith("__"):
                continue
            try:
                attr = getattr(agent.__class__, attr_name, None)
            except Exception:
                continue
            if attr and inspect.iscoroutinefunction(attr):
                if not is_hidden_method(attr):
                    names.add(attr_name)

        # 2) Instance-level async callables (important for instance-bound helpers)
        for attr_name in dir(agent):
            if attr_name.startswith("__"):
                continue
            try:
                attr = getattr(agent, attr_name)
            except Exception:
                continue
            if callable(attr) and inspect.iscoroutinefunction(attr):
                if not is_hidden_method(attr):
                    names.add(attr_name)

        return names

    def _build_parent_map(self, tree: ast.AST) -> dict[ast.AST, ast.AST]:
        parent_map: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent
        return parent_map

    def _is_in_await_or_comprehension(
        self, node: ast.AST, parent_map: dict[ast.AST, ast.AST]
    ) -> bool:
        current = node
        while current in parent_map:
            parent = parent_map[current]

            if isinstance(parent, ast.Await) and getattr(parent, "value", None) == current:
                return True

            # Allow common gather patterns:
            #   tasks = [self.async_method(x) for x in items]
            #   tasks = (self.async_method(x) for x in items)
            if isinstance(parent, (ast.ListComp, ast.GeneratorExp)):
                return True

            current = parent

        return False


def _type_name(t: Any) -> str:
    """Get a readable name for a type, handling generics."""
    from typing import get_args, get_origin

    origin = get_origin(t)
    if origin is not None:
        args = get_args(t)
        origin_name = getattr(origin, "__name__", str(origin))
        if args:
            args_str = ", ".join(_type_name(a) for a in args)
            return f"{origin_name}[{args_str}]"
        return origin_name

    if hasattr(t, "__name__"):
        return str(t.__name__)
    return str(t)


class ReturnValueValidator:
    """Validate/coerce values returned from REPL-style execution against method annotations."""

    # Basic types that can be validated directly
    _BASIC_TYPES = (str, int, float, bool, list, dict, tuple, set, bytes)

    def validate(self, value: Any, runtime: Any, method_name: str) -> Any:
        method = getattr(runtime.agent, method_name, None)
        if not method:
            return value

        try:
            hints = get_type_hints(method, include_extras=True)
            return_type = hints.get("return", inspect.Parameter.empty)
        except (NameError, TypeError, AttributeError):
            try:
                sig = inspect.signature(method)
                return_type = sig.return_annotation
            except (ValueError, TypeError):
                return value

        if return_type == inspect.Signature.empty:
            return value

        # Check if None is allowed BEFORE unwrapping Optional
        none_allowed = self._allows_none(return_type)

        # Unwrap Optional for further type validation
        unwrapped_type = self._unwrap_optional(return_type)

        if value is None:
            if none_allowed:
                return value
            # None is not allowed by this return type
            raise TypeError(
                f"Return value type mismatch for method `{method_name}`.\n"
                f"Expected: {_type_name(return_type)}\n"
                f"Received: None\n"
                f"Hint: Return a value of the expected type, or change the return type to "
                f"`{_type_name(unwrapped_type)} | None` if None is a valid return value."
            )

        if self._is_pydantic_model(unwrapped_type):
            return self._validate_pydantic(value, unwrapped_type, method_name)

        # Handle both plain types (list, dict) and parameterized generics (list[str], dict[str, int])
        origin = self._get_origin(unwrapped_type)
        if origin in self._BASIC_TYPES or unwrapped_type in self._BASIC_TYPES:
            return self._validate_basic_type(
                value, unwrapped_type, method_name, runtime.truncation_config
            )

        return value

    def _get_origin(self, return_type: Any) -> Any:
        """Get the origin type of a generic alias (e.g., list from list[str])."""
        from typing import get_origin

        origin = get_origin(return_type)
        return origin if origin is not None else return_type

    def _allows_none(self, return_type: Any) -> bool:
        """Check if a return type allows None as a valid value.

        Returns True for:
        - None (return type is None itself)
        - type(None) / NoneType
        - Optional[T] (Union[T, None])
        - T | None (UnionType with None)

        Returns False for all other types.
        """
        import types as _types
        from typing import Union, get_args, get_origin

        # None or NoneType itself allows None
        if return_type is None or return_type is type(None):
            return True

        origin = get_origin(return_type)
        args = get_args(return_type)

        # Check if it's a Union/UnionType containing None
        if origin in (Union, _types.UnionType):
            none_type = type(None)
            return any(t is none_type for t in args)

        return False

    def _unwrap_optional(self, return_type: Any) -> Any:
        import types as _types
        from typing import Union, get_args, get_origin

        origin = get_origin(return_type)
        args = get_args(return_type)

        if origin in (Union, _types.UnionType):
            none_type = type(None)
            non_none_types = [t for t in args if t is not none_type]
            if len(non_none_types) == 1:
                return non_none_types[0]

        return return_type

    def _is_pydantic_model(self, return_type: Any) -> bool:
        from pydantic import BaseModel

        return isinstance(return_type, type) and issubclass(return_type, BaseModel)

    def _validate_pydantic(self, value: Any, return_type: Any, method_name: str) -> Any:
        from pydantic import ValidationError as PydanticValidationError

        if isinstance(value, return_type):
            return value

        if isinstance(value, dict):
            try:
                return return_type(**value)
            except PydanticValidationError as e:
                error_details = [
                    f"  - {'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                    for err in e.errors()
                ]
                raise TypeError(
                    f"Return value validation failed for method `{method_name}`.\n"
                    f"Expected type: {return_type.__name__}\n"
                    f"Received: dict\n"
                    f"Validation errors:\n" + "\n".join(error_details) + "\n"
                    f"Hint: Return a dict with keys: {', '.join(return_type.model_fields.keys())}"
                ) from e

        raise TypeError(
            f"Return value type mismatch for method `{method_name}`.\n"
            f"Expected: {return_type.__name__}\n"
            f"Received: {type(value).__name__}\n"
            f"Hint: Return a dict with keys matching {return_type.__name__} fields."
        )

    def _validate_basic_type(
        self,
        value: Any,
        return_type: Any,
        method_name: str,
        truncation_config: TruncationConfig = DEFAULT_TRUNCATION_CONFIG,
    ) -> Any:
        from typing import get_args, get_origin

        origin = get_origin(return_type)
        base_type = origin if origin is not None else return_type

        # Check container type first
        if not isinstance(value, base_type):
            # Coercion for str type
            if base_type is str:
                try:
                    return str(value)
                except Exception:
                    pass

            type_name = getattr(return_type, "__name__", str(return_type))
            raise TypeError(
                f"Return value type mismatch for method `{method_name}`.\n"
                f"Expected: {type_name}\n"
                f"Received: {type(value).__name__}"
            )

        # For parameterized generics, validate element types
        type_args = get_args(return_type)
        if type_args and origin is not None:
            return self._validate_generic_elements(
                value, origin, type_args, method_name, truncation_config
            )

        return value

    def _validate_generic_elements(
        self,
        value: Any,
        origin: type,
        type_args: tuple[Any, ...],
        method_name: str,
        truncation_config: TruncationConfig = DEFAULT_TRUNCATION_CONFIG,
    ) -> Any:
        """Validate elements of parameterized generic types like list[str], dict[str, int]."""

        def fmt(v: Any) -> str:
            return _pformat(v, truncation_config)

        if origin is list or origin is set:
            element_type = type_args[0]
            for i, elem in enumerate(value):
                if not self._is_instance_of(elem, element_type):
                    raise TypeError(
                        f"Return value type mismatch for method `{method_name}`.\n"
                        f"Expected: {origin.__name__}[{_type_name(element_type)}]\n"
                        f"Element at index {i} has wrong type: {type(elem).__name__}\n"
                        f"Value: {fmt(elem)}"
                    )

        elif origin is tuple:
            # Handle tuple[int, ...] (homogeneous, variable-length)
            if len(type_args) == 2 and type_args[1] is ...:
                element_type = type_args[0]
                for i, elem in enumerate(value):
                    if not self._is_instance_of(elem, element_type):
                        raise TypeError(
                            f"Return value type mismatch for method `{method_name}`.\n"
                            f"Expected: tuple[{_type_name(element_type)}, ...]\n"
                            f"Element at index {i} has wrong type: {type(elem).__name__}\n"
                            f"Value: {fmt(elem)}"
                        )
            # Handle tuple[int, str, float] (heterogeneous, fixed-length)
            elif type_args:
                if len(value) != len(type_args):
                    raise TypeError(
                        f"Return value type mismatch for method `{method_name}`.\n"
                        f"Expected tuple with {len(type_args)} elements, got {len(value)}"
                    )
                for i, (elem, expected_type) in enumerate(zip(value, type_args, strict=True)):
                    if not self._is_instance_of(elem, expected_type):
                        raise TypeError(
                            f"Return value type mismatch for method `{method_name}`.\n"
                            f"Expected element {i} to be {_type_name(expected_type)}\n"
                            f"Got: {type(elem).__name__}\n"
                            f"Value: {fmt(elem)}"
                        )

        elif origin is dict:
            key_type = type_args[0] if len(type_args) > 0 else Any
            val_type = type_args[1] if len(type_args) > 1 else Any
            for k, v in value.items():
                if not self._is_instance_of(k, key_type):
                    raise TypeError(
                        f"Return value type mismatch for method `{method_name}`.\n"
                        f"Expected dict key type: {_type_name(key_type)}\n"
                        f"Key has wrong type: {type(k).__name__}\n"
                        f"Key: {fmt(k)}"
                    )
                if not self._is_instance_of(v, val_type):
                    raise TypeError(
                        f"Return value type mismatch for method `{method_name}`.\n"
                        f"Expected dict value type: {_type_name(val_type)}\n"
                        f"Value for key {fmt(k)} has wrong type: {type(v).__name__}\n"
                        f"Value: {fmt(v)}"
                    )

        return value

    def _is_instance_of(self, value: Any, expected_type: Any) -> bool:
        """Check if value is an instance of expected_type, handling typing generics."""
        import typing
        from typing import Any as TypingAny
        from typing import Union, get_args, get_origin

        # Any matches anything
        if expected_type is TypingAny:
            return True

        # Handle generic types by checking origin
        origin = get_origin(expected_type)
        if origin is not None:
            # Union types (int | float or Union[int, float]) - check if value matches any member
            if origin is Union or origin is types.UnionType:
                return any(self._is_instance_of(value, arg) for arg in get_args(expected_type))
            # Literal types - check if value is one of the allowed literals
            if origin is typing.Literal:
                return value in get_args(expected_type)
            # Annotated types - unwrap and check the actual type (first arg)
            if origin is typing.Annotated:
                args = get_args(expected_type)
                # Annotated always has at least one type arg; the fallback is a safety net.
                return self._is_instance_of(value, args[0]) if args else True
            # Other generic types (list, dict, etc.) - check origin
            return isinstance(value, origin)

        # Plain type check
        try:
            return isinstance(value, expected_type)
        except TypeError:
            # Some typing constructs can't be used with isinstance
            return True


class ArgumentValidator:
    """Validate method call arguments against signature and type hints using Pydantic.

    This validator ensures that:
    1. All required arguments are provided (arity check via sig.bind())
    2. Arguments match their type hints (type check via Pydantic TypeAdapter)

    Used by the metaclass and @strategy decorator wrappers to catch invalid calls
    early, before dispatching to the LLM for code generation.
    """

    def validate(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        truncation_config: TruncationConfig = DEFAULT_TRUNCATION_CONFIG,
    ) -> None:
        """Validate arguments against function signature and type hints.

        Args:
            func: The function/method being called (used to extract signature and type hints)
            args: Positional arguments passed to the method (excluding 'self')
            kwargs: Keyword arguments passed to the method
            truncation_config: Truncation settings for value display in error messages.

        Raises:
            TypeError: If required arguments are missing, unexpected arguments are provided,
                      or argument types don't match their type hints.
        """

        # Get the original function if this is a wrapper
        original_func = getattr(func, "_original", func)

        try:
            sig = inspect.signature(original_func)
        except (ValueError, TypeError):
            # Can't inspect signature, skip validation
            return

        # Get parameter list excluding 'self'
        params = list(sig.parameters.items())
        if params and params[0][0] == "self":
            params = params[1:]

        # Build a signature without 'self' for binding
        params_without_self = [p for name, p in params]
        sig_without_self = sig.replace(parameters=params_without_self)

        # Step 1: Arity check via bind()
        try:
            bound = bind_call_arguments(sig_without_self, args, kwargs)
        except TypeError as e:
            # Extract method name for error message
            method_name = getattr(original_func, "__name__", str(original_func))
            signature_str = str(sig)

            # Provide detailed error message
            provided_args = self._format_provided_args(args, kwargs, params, truncation_config)
            raise TypeError(
                f"Invalid call to {method_name}():\n"
                f"  {e}\n\n"
                f"  Signature: {method_name}{signature_str}\n"
                f"  Provided: {provided_args}"
            ) from None

        # Step 2: Type check via Pydantic TypeAdapter
        try:
            hints = get_type_hints(original_func)
        except Exception:
            # Can't get type hints (forward refs, etc.), skip type validation
            return

        method_name = getattr(original_func, "__name__", str(original_func))
        signature_str = str(sig)

        # Build a dict of param name -> param kind for VAR_POSITIONAL/VAR_KEYWORD detection
        param_kinds = {name: p.kind for name, p in sig_without_self.parameters.items()}

        for param_name, value in bound.arguments.items():
            if param_name not in hints:
                continue  # No type hint for this parameter

            # Skip VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs) - their annotations
            # apply to individual elements, not the collected tuple/dict
            param_kind = param_kinds.get(param_name)
            if param_kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            expected_type = hints[param_name]
            self._validate_type(
                param_name, value, expected_type, method_name, signature_str, truncation_config
            )

    def _validate_type(
        self,
        param_name: str,
        value: Any,
        expected_type: Any,
        method_name: str,
        signature_str: str,
        truncation_config: TruncationConfig = DEFAULT_TRUNCATION_CONFIG,
    ) -> None:
        """Validate a single argument against its expected type using Pydantic."""

        try:
            adapter = TypeAdapter(expected_type)
        except PydanticSchemaGenerationError:
            # Pydantic can't handle this type (e.g., Protocol classes, some ABCs)
            # Skip validation for this parameter
            return

        try:
            adapter.validate_python(value, strict=True)
        except PydanticValidationError as e:
            # Format Pydantic errors into a clear message
            error_details = []
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"]) if err["loc"] else param_name
                error_details.append(f"    - {loc}: {err['msg']}")

            # Get type name for error message
            type_name = _type_name(expected_type)
            value_type = type(value).__name__
            value_repr = _pformat(value, truncation_config)

            raise TypeError(
                f"Invalid call to {method_name}():\n"
                f"  Argument '{param_name}' has wrong type: expected {type_name}, got {value_type}\n"
                f"  Value: {value_repr}\n"
                f"  Validation errors:\n" + "\n".join(error_details) + "\n\n"
                f"  Signature: {method_name}{signature_str}"
            ) from None

    def _format_provided_args(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        params: list[tuple[str, Any]],
        truncation_config: TruncationConfig = DEFAULT_TRUNCATION_CONFIG,
    ) -> str:
        """Format provided arguments for error message."""
        parts = []

        # Map positional args to parameter names
        for i, value in enumerate(args):
            fmt = _pformat(value, truncation_config)
            if i < len(params):
                param_name = params[i][0]
                parts.append(f"{param_name}={fmt}")
            else:
                parts.append(fmt)

        # Add keyword arguments
        for key, value in kwargs.items():
            parts.append(f"{key}={_pformat(value, truncation_config)}")

        return ", ".join(parts) if parts else "(no arguments)"
