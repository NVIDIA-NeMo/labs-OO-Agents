# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit route-local structured-output carrier, never inferred from a model ID."""

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel

StructuredOutputMode = Literal["auto", "anthropic"] | None


def validate_structured_output_mode(value: object) -> StructuredOutputMode:
    if value is not None and value not in ("auto", "anthropic"):
        raise ValueError("structured_output_mode must be 'auto', 'anthropic', or None")
    return value


def anthropic_output_format(output_model: type[BaseModel]) -> dict:
    """Compile an object contract to Anthropic's output_config.format.

    Preserve schema keywords and required fields, including references. Close
    model objects (as the default strict carrier does), but never turn a typed
    dictionary into a closed empty object. Validation always uses the original
    Pydantic model, not this wire schema. Provider schema-subset acceptance is
    deliberately not guessed or recovered here.
    """
    schema = deepcopy(output_model.model_json_schema())
    if schema.get("type") != "object":
        raise ValueError("Anthropic structured output requires an object-root output_model")

    def close_objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node.setdefault("additionalProperties", False)
            for child in node.values():
                close_objects(child)
        elif isinstance(node, list):
            for child in node:
                close_objects(child)

    close_objects(schema)
    return {"type": "json_schema", "schema": schema}
