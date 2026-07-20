# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill plugins for strategies.

Prefill runs before the main generation loop, injecting a synthetic
"first turn" that executes code through the normal strategy path.
Results persist in session_locals for subsequent turns.

Usage:
    @strategy(CodeActStrategy(prefill=InspectInputsPrefill()))
    async def my_method(self, data: list[str]) -> Result:
        '''...'''
        ...
"""

import types
import typing
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from nooa.strategies.current_call import CurrentCall


def _is_media(value: Any) -> bool:
    """Check if a value is a Media instance (lazy import to avoid circular deps)."""
    try:
        from nooa.media import Media

        return isinstance(value, Media)
    except ImportError:
        return False


def _get_complex_type(annotation: Any) -> type | None:
    """Extract a complex type from an annotation that should be documented.

    Returns the type if it's a non-simple type that would benefit from doc(),
    or None if it's a simple type.

    Handles:
    - Plain types: MyModel -> MyModel
    - Optional[MyModel] -> MyModel
    - Union[MyModel, None] -> MyModel
    - list, dict, str, int, etc. -> None (simple types)
    """
    simple_types = (str, int, float, bool, type(None), list, dict, tuple, set)

    # Handle None
    if annotation is None or annotation is type(None):
        return None

    # Handle typing constructs (Optional, Union, X | Y unions, etc.)
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        # Get non-None args
        args = typing.get_args(annotation)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            # Optional[X] or Union[X, None] - recurse on X
            return _get_complex_type(non_none_args[0])
        # Multiple types in union - don't document
        return None

    # Handle generic types like list[str], dict[str, int]
    if origin is not None:
        # It's a generic - check if the origin is simple
        if origin in simple_types:
            return None
        # Complex generic like MyGeneric[T] - return origin
        if isinstance(origin, type):
            return origin
        return None

    # Plain type
    if isinstance(annotation, type):
        if annotation in simple_types:
            return None
        return annotation

    return None


class Prefill(Protocol):
    """Protocol for prefill plugins.

    Prefill generates code to execute before the main generation loop.
    The code runs through the strategy's normal execution path (validation,
    helpers, namespace) and results persist in session_locals.
    """

    def get_code(self, call: "CurrentCall", config: Any = None) -> str | None:
        """Generate prefill code for the given call.

        Args:
            call: CurrentCall with method details and arguments
            config: Optional truncation config (for parameter inspection)

        Returns:
            Python code to execute, or None to skip prefill
        """
        ...


class InspectInputsPrefill:
    """Prefill that inspects input parameters using pprint().

    Generates code that prints the call signature and pprint of each parameter,
    helping the LLM understand the inputs before generating code.

    Uses pprint() with truncation config limits to prevent huge parameters
    from overwhelming the context window.

    Example output in events:
        Current call: def analyze(self, data: list[str]) -> AnalysisResult
        data (list):
        [
            'item1',
            'item2',
            ...
        ]
        Return type: AnalysisResult { field1: str, field2: int }

    Note: show_return_type is now always True (hardcoded for simplicity).
    """

    def get_code(self, call: "CurrentCall", config: Any = None) -> str | None:
        """Generate exploration code for the given call.

        Args:
            call: CurrentCall with method details and arguments
            config: TruncationConfig with max_length, max_string, max_depth settings

        Returns:
            Python code to execute, or None if no parameters to inspect
        """
        param_names = list(call.kwargs.keys())

        if not param_names:
            return None

        method_name = call.method_name

        # Use config values (literals so LLM sees what was called)
        if config is not None:
            max_length = config.prefill_format.max_length
            max_string = config.prefill_format.max_string
            max_depth = config.prefill_format.max_depth
        else:
            # Defaults if no config provided
            max_length = 50
            max_string = 500
            max_depth = 4

        code_lines = [
            f"# Inspecting inputs for {method_name}().",
            f'print(f"Task: {method_name}()")',
        ]

        for param in param_names:
            code_lines.append(f'print(f"\\n{param} ({{type({param}).__name__}}):")')
            # Auto-show Media parameters (images, audio, files) so the LLM perceives them
            if _is_media(call.kwargs.get(param)):
                code_lines.append(f"show({param})")
            else:
                # Honor per-parameter spec() overrides from Annotated metadata.
                p_spec = call.param_specs.get(param, {})
                p_ml = p_spec.get("max_length", max_length)
                p_ms = p_spec.get("max_string", max_string)
                p_md = p_spec.get("max_depth", max_depth)
                code_lines.append(
                    f"pprint({param}, max_length={p_ml}, max_string={p_ms}, max_depth={p_md})"
                )

        # Always show return type for complex types (show_return_type hardcoded to True)
        if call.return_type:
            complex_type = _get_complex_type(call.return_type)
            if complex_type is not None:
                # inline_depth=1 expands one level of referenced/nested types (e.g. for
                # `Type2 {t1: Type1}` it also shows Type1's fields). Without it, concise
                # mode hides nested type structure and the model can't construct the
                # value from the prefill alone. See KDD-cup feedback.
                code_lines.append(
                    'print(f"\\nReturn type: {doc(_call.return_type, concise=True, inline_depth=1)}")'
                )

        return "\n".join(code_lines)
