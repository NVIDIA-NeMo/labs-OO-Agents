# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Choose how a chat request asks the model for structured (JSON) output.

When a call passes ``output_model``, the request must tell the provider which
JSON schema the reply has to follow. The *structured output mode* of a model
entry selects the request field that carries that schema:

- ``None`` or ``"auto"``: the default, LiteLLM's ``response_format`` field.
- ``"anthropic"``: Anthropic's ``output_config.format`` field.

The mode is set explicitly on the model entry (``structured_output_mode``) and
applies only to that model entry; it is never guessed from the model name.
"""

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel

StructuredOutputMode = Literal["auto", "anthropic"] | None


def validate_structured_output_mode(value: object) -> StructuredOutputMode:
    """Return ``value`` if it is a valid mode, else raise ``ValueError``."""
    if value is not None and value not in ("auto", "anthropic"):
        raise ValueError("structured_output_mode must be 'auto', 'anthropic', or None")
    return value


def anthropic_output_format(output_model: type[BaseModel]) -> dict:
    """Build the ``output_config.format`` value for Anthropic structured output.

    The JSON schema of ``output_model`` is sent unchanged (keywords, required
    fields and ``$ref`` references are kept), except that every object with
    declared ``properties`` gets ``additionalProperties: false`` unless it
    already sets it, as the default strict ``response_format`` path does.
    Objects without ``properties`` (for example ``dict[str, int]``) are left
    open, so they do not become objects that accept no keys.

    The reply is still validated against ``output_model`` itself, not against
    this schema. This function does not check whether the provider supports
    every schema keyword; an unsupported schema fails at the provider.

    Raises:
        ValueError: If the schema's root is not an object.
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
