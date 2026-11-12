# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral tool metadata and callable parameter models."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class Tool:
    """Standardized tool representation across all LLM APIs

    Clean design: Either provide parameters_model (Pydantic) or it's auto-generated from callable.

    - parameters_model: Pydantic model defining parameter schema (recommended)
    - If None, auto-generates from callable's signature
    - Preserves all type information (Union, nested models, TypedDict, etc.)
    """

    name: str
    description: str
    callable: Callable
    parameters_model: type["BaseModel"] | None = None  # Pydantic model for parameters

    def get_parameter_schema(self, *, strict: bool = False) -> dict[str, Any]:
        """Get the JSON schema for parameters.

        If parameters_model is provided, use its schema.
        Otherwise, auto-generate from callable signature.

        The returned schema has $ref resolved inline and Pydantic noise
        (title, default) stripped — clean for any LLM provider.

        Args:
            strict: When True, apply the adapter's strict schema conversion.
        """
        # Compatibility API: wire-specific conversion belongs to the adapters.
        from nooa.unifiedllm.adapters.schema import _clean_schema

        if self.parameters_model is not None:
            schema = self.parameters_model.model_json_schema()
        else:
            schema = self._auto_generate_raw_schema()

        return _clean_schema(schema, strict=strict)

    def _auto_generate_raw_schema(self) -> dict[str, Any]:
        """Auto-generate parameter schema from callable signature."""
        from pydantic import create_model

        sig = inspect.signature(self.callable)
        field_definitions = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            default = ... if param.default == inspect.Parameter.empty else param.default

            field_definitions[param_name] = (param_type, default)

        if not field_definitions:
            # No parameters
            return {"type": "object", "properties": {}, "required": []}

        # Create temporary Pydantic model
        TempModel = create_model(f"{self.name}_params", **field_definitions)
        return TempModel.model_json_schema()


def create_tool_from_callable(tool_callable: Callable) -> Tool:
    """Extract Tool metadata from a Python function

    Creates a Tool with auto-generated parameter schema from the callable's signature.
    """
    docstring = tool_callable.__doc__ or f"Call the {tool_callable.__name__} function"

    # Let Tool auto-generate the schema from the callable
    return Tool(
        name=tool_callable.__name__,
        description=docstring,
        callable=tool_callable,
        parameters_model=None,  # Will auto-generate from signature
    )
