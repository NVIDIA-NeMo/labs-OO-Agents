# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit guards for the strict / non-strict response_format routing (issue 232).

Strict OpenAI structured outputs cannot express free-form objects (``dict``),
untyped arrays (bare ``list``), heterogeneous tuples, or unique arrays (``set``).
``_maybe_sanitize_response_format`` must route those to a non-strict json_schema
(with unsupported keywords stripped) while leaving strict-compatible types — scalars,
typed lists, Pydantic models, Literal/Optional/Union/Enum — on the default strict path.

These are pure-Python schema-shape checks (no network); they pin the behavior the
live ``tests/integration/test_predict_return_types_live.py`` matrix exercises end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel

from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.predict import PredictStrategy
from nooa.unifiedllm.unifiedllm import (
    _loose_response_schema,
    _maybe_sanitize_response_format,
    _resolve_schema_refs,
    _responses_output_params,
    _schema_strict_compatible,
    _strict_schema_valid,
)

_OPENAI = "openai/azure/openai/gpt-5-mini"
_BEDROCK = "aws/anthropic/bedrock-claude-opus-4-7"


class _Point(BaseModel):
    x: int
    y: int


@dataclass
class _Cluster:
    theme: str


class _Color(StrEnum):
    RED = "red"
    GREEN = "green"


def _model_for(return_type):
    return PredictStrategy()._create_response_model(return_type, "meth")


def _is_strict_path(model, payload) -> bool:
    """Strict path returns the Pydantic model unchanged (litellm builds strict schema)."""
    return payload is model


# ── strict-compatible types stay on the default (strict) path ───────────────
class TestStrictCompatibleTypesUnchanged:
    def _assert_strict(self, return_type):
        model = _model_for(return_type)
        payload = _maybe_sanitize_response_format(_OPENAI, model)
        assert _is_strict_path(model, payload), f"{return_type!r} should stay strict"
        assert _schema_strict_compatible(_resolve_schema_refs(model.model_json_schema()))

    def test_int(self):
        self._assert_strict(int)

    def test_str(self):
        self._assert_strict(str)

    def test_bool(self):
        self._assert_strict(bool)

    def test_list_str(self):
        self._assert_strict(list[str])

    def test_list_model(self):
        self._assert_strict(list[_Point])

    def test_literal(self):
        self._assert_strict(Literal["a", "b"])

    def test_optional(self):
        self._assert_strict(int | None)

    def test_union(self):
        self._assert_strict(int | str)

    def test_enum(self):
        self._assert_strict(_Color)

    def test_pydantic_model(self):
        self._assert_strict(_Point)

    def test_dataclass(self):
        self._assert_strict(_Cluster)


# ── strict-incompatible types fall back to a non-strict json_schema ─────────
class TestStrictIncompatibleTypesFallBack:
    def _assert_nonstrict(self, return_type):
        model = _model_for(return_type)
        payload = _maybe_sanitize_response_format(_OPENAI, model)
        assert isinstance(payload, dict), f"{return_type!r} should fall back to a dict payload"
        assert payload["type"] == "json_schema"
        assert payload["json_schema"]["strict"] is False
        assert not _schema_strict_compatible(_resolve_schema_refs(model.model_json_schema()))
        return payload["json_schema"]["schema"]

    def test_bare_dict(self):
        self._assert_nonstrict(dict)

    def test_typed_dict(self):
        self._assert_nonstrict(dict[str, int])

    def test_bare_list(self):
        self._assert_nonstrict(list)

    def test_tuple(self):
        self._assert_nonstrict(tuple[int, str])

    def test_set(self):
        schema = self._assert_nonstrict(set[str])
        # uniqueItems is unsupported by OpenAI even non-strict — must be stripped.
        assert "uniqueItems" not in _flatten_keys(schema)

    def test_tuple_strips_prefixitems(self):
        schema = self._assert_nonstrict(tuple[int, str])
        assert "prefixItems" not in _flatten_keys(schema)

    # issue 148 patterns: Any (empty schema), dict value-unions, list[dict]
    def test_any(self):
        self._assert_nonstrict(Any)

    def test_dict_value_union(self):
        self._assert_nonstrict(dict[str, int | bool])

    def test_list_dict(self):
        self._assert_nonstrict(list[dict])


def _flatten_keys(node) -> set[str]:
    keys: set[str] = set()

    def _walk(n):
        if isinstance(n, dict):
            keys.update(n.keys())
            for v in n.values():
                _walk(v)
        elif isinstance(n, list):
            for v in n:
                _walk(v)

    _walk(node)
    return keys


# ── Bedrock path: strict json_schema dict, with unsupported keywords stripped ─
class TestBedrock:
    def test_bedrock_model_returns_strict_dict(self):
        model = _model_for(_Point)
        payload = _maybe_sanitize_response_format(_BEDROCK, model)
        assert isinstance(payload, dict)
        assert payload["json_schema"]["strict"] is True

    def test_bedrock_strips_prefixitems_for_tuple(self):
        # Bedrock rejects prefixItems ("not supported"); it must be stripped (issue 232).
        model = _model_for(tuple[int, str])
        payload = _maybe_sanitize_response_format(_BEDROCK, model)
        assert "prefixItems" not in _flatten_keys(payload["json_schema"]["schema"])

    def test_bedrock_strips_uniqueitems_for_set(self):
        # Bedrock rejects uniqueItems ("not supported"); it must be stripped (issue 232).
        model = _model_for(set[str])
        payload = _maybe_sanitize_response_format(_BEDROCK, model)
        assert "uniqueItems" not in _flatten_keys(payload["json_schema"]["schema"])


# ── Responses API path (litellm.responses text_format / text) ───────────────
class TestResponsesOutputParams:
    """Mirror of the completion path for the Responses API (issue 232 follow-up)."""

    def _assert_strict(self, return_type):
        params = _responses_output_params(_model_for(return_type))
        assert set(params) == {"text_format"}, f"{return_type!r} should use text_format"

    def _assert_nonstrict(self, return_type):
        params = _responses_output_params(_model_for(return_type))
        assert "text_format" not in params, f"{return_type!r} should not use text_format"
        fmt = params["text"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["strict"] is False
        return fmt["schema"]

    def test_scalar_uses_text_format(self):
        self._assert_strict(int)

    def test_typed_list_uses_text_format(self):
        self._assert_strict(list[str])

    def test_model_uses_text_format(self):
        self._assert_strict(_Point)

    def test_dict_uses_nonstrict_text(self):
        self._assert_nonstrict(dict[str, int])

    def test_bare_list_uses_nonstrict_text(self):
        self._assert_nonstrict(list)

    def test_tuple_uses_nonstrict_text(self):
        schema = self._assert_nonstrict(tuple[int, str])
        assert "prefixItems" not in _flatten_keys(schema)

    def test_set_uses_nonstrict_text(self):
        schema = self._assert_nonstrict(set[str])
        assert "uniqueItems" not in _flatten_keys(schema)


# ── _schema_strict_compatible direct checks ─────────────────────────────────
class TestSchemaStrictCompatible:
    def test_free_form_object_incompatible(self):
        assert not _schema_strict_compatible(
            {"type": "object", "additionalProperties": {"type": "integer"}}
        )

    def test_object_without_properties_incompatible(self):
        assert not _schema_strict_compatible({"type": "object"})

    def test_untyped_array_incompatible(self):
        assert not _schema_strict_compatible({"type": "array", "items": {}})

    def test_tuple_incompatible(self):
        assert not _schema_strict_compatible(
            {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}]}
        )

    def test_unique_array_incompatible(self):
        assert not _schema_strict_compatible(
            {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
        )

    def test_typed_object_compatible(self):
        assert _schema_strict_compatible(
            {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            }
        )

    def test_typed_array_compatible(self):
        assert _schema_strict_compatible({"type": "array", "items": {"type": "string"}})

    def test_nested_freeform_incompatible(self):
        assert not _schema_strict_compatible(
            {
                "type": "object",
                "properties": {
                    "data": {"type": "object", "additionalProperties": {"type": "integer"}}
                },
                "required": ["data"],
            }
        )

    def test_untyped_node_incompatible(self):
        # `Any` / untyped members are emitted as empty `{}` — strict needs a type (issue 148).
        assert not _schema_strict_compatible(
            {"type": "object", "properties": {"v": {}}, "required": ["v"]}
        )


# ── _strict_schema_valid array-items requirement (CodeAct return_result tool path) ──
class TestStrictSchemaValidArrays:
    """Strict tool schemas need typed array `items`; tuples lose theirs after cleaning."""

    def test_array_without_items_invalid(self):
        # Strict cleaning drops a tuple's prefixItems, leaving `{type: array}` with no items.
        assert not _strict_schema_valid({"type": "array"})

    def test_array_empty_items_invalid(self):
        assert not _strict_schema_valid({"type": "array", "items": {}})

    def test_array_typed_items_valid(self):
        assert _strict_schema_valid({"type": "array", "items": {"type": "string"}})

    def test_object_with_typed_array_valid(self):
        assert _strict_schema_valid(
            {
                "type": "object",
                "properties": {"xs": {"type": "array", "items": {"type": "integer"}}},
                "required": ["xs"],
                "additionalProperties": False,
            }
        )


# ── _loose_response_schema normalization (Responses API safety) ─────────────
class TestLooseResponseSchema:
    def test_strips_unsupported_and_adds_defaults_for_tuple(self):
        # tuple -> array with prefixItems/minItems/maxItems; must become a plain typed array.
        loose = _loose_response_schema(
            {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "array",
                        "prefixItems": [{"type": "integer"}, {"type": "string"}],
                        "minItems": 2,
                        "maxItems": 2,
                    }
                },
                "required": ["result"],
            }
        )
        result = loose["properties"]["result"]
        assert "prefixItems" not in result
        assert result.get("items") == {}  # array gets an items default
        assert result["type"] == "array"

    def test_adds_properties_default_for_freeform_object(self):
        loose = _loose_response_schema({"type": "object", "additionalProperties": True})
        assert loose["properties"] == {}  # Azure Responses requires `properties`


# ── ResponsesClient tool schema: strict-incompatible return types fall back safely ──
class TestResponsesToolSchemaFallback:
    """The return_result tool for tuple/set must yield a Responses-API-safe schema."""

    def _convert(self, return_type):
        from nooa.unifiedllm.unifiedllm import ResponsesClient

        tool = CodeActStrategy()._build_return_result_tool(return_type, "meth")
        # _convert_tool_to_schema is a plain method using only `self`-free helpers.
        return ResponsesClient._convert_tool_to_schema(object.__new__(ResponsesClient), tool)

    def test_tuple_tool_schema_has_items_and_no_prefixitems(self):
        out = self._convert(tuple[int, str])
        keys = _flatten_keys(out["parameters"])
        assert "prefixItems" not in keys
        assert out["strict"] is False
        # the result array must have an items key (Azure rejects "array missing items")
        result = out["parameters"]["properties"]["result"]
        assert "items" in result

    def test_set_tool_schema_strips_uniqueitems(self):
        out = self._convert(set[str])
        assert "uniqueItems" not in _flatten_keys(out["parameters"])

    def test_scalar_tool_schema_stays_strict(self):
        out = self._convert(int)
        assert out["strict"] is True
