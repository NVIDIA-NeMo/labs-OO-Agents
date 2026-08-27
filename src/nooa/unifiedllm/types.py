# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight LLM protocol types.

This module intentionally avoids importing LiteLLM. Strategies need these
types at class-definition time, while provider clients only need LiteLLM when a
real client is constructed or called.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel


def _resolve_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $ref references by inlining $defs definitions."""
    defs = schema.get("$defs", {})

    def _inline(node, resolving: set[str] | None = None):
        if resolving is None:
            resolving = set()
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref = node["$ref"]
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.split("/")[-1]
                if name in resolving:
                    return {"type": "object"}
                if name in defs:
                    resolving.add(name)
                    resolved = _inline(defs[name], resolving)
                    resolving.remove(name)
                    return resolved
            return {"type": "object"}
        result = {}
        for k, v in node.items():
            if k == "$defs":
                continue
            if k == "properties" and isinstance(v, dict):
                result[k] = {pk: _inline(pv, resolving) for pk, pv in v.items()}
            elif k in ("items", "additionalProperties") and isinstance(v, dict):
                result[k] = _inline(v, resolving)
            elif k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
                result[k] = [_inline(i, resolving) if isinstance(i, dict) else i for i in v]
            else:
                result[k] = v
        if "allOf" in result and len(result["allOf"]) == 1:
            merged = dict(result["allOf"][0])
            del result["allOf"]
            for k, v in result.items():
                if k not in merged:
                    merged[k] = v
            result = merged
        return result

    return _inline(schema)


def _strip_schema_noise(schema: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """Strip Pydantic noise from a JSON schema recursively."""
    strip_keys = frozenset({"title", "default"})
    strict_allowed = frozenset(
        {
            "type",
            "description",
            "enum",
            "const",
            "properties",
            "required",
            "items",
            "additionalProperties",
            "anyOf",
            "oneOf",
            "allOf",
        }
    )

    def _strip(node):
        if not isinstance(node, dict):
            return node
        if strict:
            cleaned = {k: v for k, v in node.items() if k in strict_allowed}
        else:
            cleaned = {k: v for k, v in node.items() if k not in strip_keys}
        if "properties" in cleaned:
            cleaned["properties"] = {k: _strip(v) for k, v in cleaned["properties"].items()}
        if "items" in cleaned and isinstance(cleaned["items"], dict):
            cleaned["items"] = _strip(cleaned["items"])
        if "additionalProperties" in cleaned and isinstance(cleaned["additionalProperties"], dict):
            cleaned["additionalProperties"] = _strip(cleaned["additionalProperties"])
        for key in ("anyOf", "oneOf", "allOf"):
            if key in cleaned and isinstance(cleaned[key], list):
                cleaned[key] = [_strip(i) if isinstance(i, dict) else i for i in cleaned[key]]
        if strict and cleaned.get("type") == "object":
            cleaned["additionalProperties"] = False
            cleaned.setdefault("properties", {})
            cleaned["required"] = list(cleaned["properties"].keys())
        return cleaned

    return _strip(schema)


def _clean_schema(schema: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """Clean a Pydantic-generated JSON schema for LLM tool use."""
    resolved = _resolve_schema_refs(schema)
    cleaned = _strip_schema_noise(resolved, strict=strict)
    result = {
        "type": cleaned.get("type", "object"),
        "properties": cleaned.get("properties", {}),
        "required": cleaned.get("required", []),
    }
    if strict:
        result["additionalProperties"] = False
    return result


def _strict_schema_valid(schema: dict[str, Any]) -> bool:
    """Check whether a schema satisfies OpenAI strict-mode requirements."""

    def _check(node: dict[str, Any]) -> bool:
        if not isinstance(node, dict):
            return True
        if (
            "type" not in node
            and "anyOf" not in node
            and "oneOf" not in node
            and "allOf" not in node
        ):
            return False
        if (
            node.get("type") == "object"
            and "properties" in node
            and set(node.get("required", [])) != set(node["properties"].keys())
        ):
            return False
        if node.get("type") == "array":
            items = node.get("items")
            if not isinstance(items, dict) or not any(
                k in items for k in ("type", "anyOf", "oneOf", "allOf", "enum", "const")
            ):
                return False
        for v in node.get("properties", {}).values():
            if isinstance(v, dict) and not _check(v):
                return False
        if isinstance(node.get("items"), dict) and not _check(node["items"]):
            return False
        for key in ("anyOf", "oneOf", "allOf"):
            for item in node.get(key, []):
                if isinstance(item, dict) and not _check(item):
                    return False
        return True

    return _check(schema)


@dataclass
class Tool:
    """Standardized tool representation across all LLM APIs."""

    name: str
    description: str
    callable: Callable
    parameters_model: type[BaseModel] | None = None

    def get_parameter_schema(self, *, strict: bool = False) -> dict[str, Any]:
        """Get the JSON schema for parameters."""
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
            return {"type": "object", "properties": {}, "required": []}

        temp_model = create_model(f"{self.name}_params", **field_definitions)
        return temp_model.model_json_schema()


def create_tool_from_callable(tool_callable: Callable) -> Tool:
    """Extract Tool metadata from a Python function."""
    docstring = tool_callable.__doc__ or f"Call the {tool_callable.__name__} function"
    return Tool(
        name=tool_callable.__name__,
        description=docstring,
        callable=tool_callable,
        parameters_model=None,
    )


@dataclass
class ToolCall:
    """Standardized tool call representation across all LLM APIs."""

    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    """Standardized response from any LLM API."""

    raw_response: Any
    content: str | BaseModel
    tool_calls: list[ToolCall]
    finish_reason: Literal["stop", "tool_calls", "length", "error"]
    assistant_message: dict[str, Any]
    reasoning: str | None = None
    usage: dict[str, int] | None = None

    @property
    def message(self) -> str | BaseModel | None:
        """Backward-compatible alias for content."""
        return self.content


__all__ = [
    "LLMResponse",
    "Tool",
    "ToolCall",
    "create_tool_from_callable",
]
