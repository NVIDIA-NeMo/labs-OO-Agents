# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import inspect
import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import BaseModel, RootModel

from ._anyllm import AnyLLMTransport, close_async_client_sync
from .http_config import HttpConfig
from .retry import EmptyContentError, sync_retry, with_retry
from .retry_config import RetryConfig
from .types import LLMChunk, LLMToolCallChunk, LLMUsage

logger = logging.getLogger(__name__)

# Optional integration with nooa debug handler for LLM call tracking
# This allows the debug signal handler to show pending LLM calls
try:
    from nooa.runtime.debug_handler import llm_call_context as _llm_call_context

    _HAS_DEBUG_HANDLER = True
except ImportError:
    _HAS_DEBUG_HANDLER = False
    _llm_call_context = None


# Optional harness metrics callback — set by the agent framework via ContextVar.
# No reverse import needed: the callback is injected by actor.py at session start.
_llm_metrics_callback: ContextVar[Callable[[str, Any], None] | None] = ContextVar(
    "llm_metrics_callback", default=None
)


def _record_llm_metric(event: str, detail: Any = None) -> None:
    """Fire-and-forget metric recording. No-op if no callback is set.

    Swallows any exception from the callback — instrumentation must never
    break the LLM call flow.
    """
    cb = _llm_metrics_callback.get()
    if cb is not None:
        try:
            cb(event, detail)
        except Exception as e:  # noqa: BLE001
            logger.debug("Metric callback failed for event %r: %s", event, e)


@contextmanager
def _track_llm_call(model: str, endpoint: str | None = None, prompt_tokens: int | None = None):
    """Track LLM call for debug purposes (if nooa debug handler is available)."""
    if _HAS_DEBUG_HANDLER and _llm_call_context:
        with _llm_call_context(model=model, endpoint=endpoint, prompt_tokens=prompt_tokens):
            yield
    else:
        yield


# ============================================================================
# Per-client HTTP transport
# ============================================================================
# Previously this module monkey-patched httpx.AsyncClient globally at import
# time to force max_keepalive_connections=0 (prevents CLOSE_WAIT hangs). That
# affected *every* httpx client in the host process — user code and unrelated
# libraries included — and its config lived in a module global, so the most
# recently constructed client silently won (see GitLab #329).
#
# Instead, each UnifiedLLM client now owns its own httpx client(s), built from
# its HttpConfig, and passes them to AnyLLM per call via AnyLLM's
# caller-provided-client support. No global state, no monkey-patch, and two
# clients with different HttpConfigs stay fully independent.


class _ClientHttp:
    """Per-client HTTP transport owned and closed by UnifiedLLM."""

    def __init__(self, http_config: HttpConfig):
        import httpx

        limits = http_config.to_httpx_limits()
        self.limits = limits
        timeout = http_config.to_httpx_timeout()
        self.httpx_sync = httpx.Client(limits=limits, timeout=timeout, follow_redirects=True)
        self.httpx_async = httpx.AsyncClient(limits=limits, timeout=timeout, follow_redirects=True)
        # Compatibility names used by the request builders. AnyLLM receives the
        # underlying httpx client, never a provider SDK object.
        self.sync_client = self.httpx_sync
        self.async_client = self.httpx_async
        self._sync_closed = False
        self._async_closed = False

    @classmethod
    def for_completion(
        cls, model: str, config: dict[str, Any], http_config: HttpConfig
    ) -> "_ClientHttp":
        return cls(http_config)

    @classmethod
    def for_responses(
        cls, model: str, config: dict[str, Any], http_config: HttpConfig
    ) -> "_ClientHttp":
        return cls(http_config)

    def close(self) -> None:
        """Close all resources for synchronous callers.

        AnyLLM's sync facade runs its async provider on its dedicated runner
        loop, so close the injected AsyncClient on that same loop. Calling the
        sync close API from an active event loop is an error; async callers must
        use ``await aclose()``.
        """
        if self._sync_closed and self._async_closed:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("close() cannot run inside an event loop; use 'await aclose()'")
        if not self._sync_closed:
            self._sync_closed = True
            self.httpx_sync.close()
        if not self._async_closed:
            close_async_client_sync(self.httpx_async)
            self._async_closed = True

    async def aclose(self) -> None:
        if not self._sync_closed:
            self._sync_closed = True
            self.httpx_sync.close()
        if not self._async_closed:
            self._async_closed = True
            await self.httpx_async.aclose()


def _recursively_parse_json_strings(obj: Any) -> Any:
    """Recursively parse any string values that are valid JSON objects/arrays.

    Some models double-encode nested JSON, e.g., {"value": '{"key": "val"}'}.
    This function detects and parses such strings.
    """
    if isinstance(obj, str):
        # Try to parse as JSON if it looks like an object or array
        stripped = obj.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
                _record_llm_metric("json_double_decoded")
                # Recursively process the parsed result
                return _recursively_parse_json_strings(parsed)
            except json.JSONDecodeError:
                pass
        return obj
    elif isinstance(obj, dict):
        return {k: _recursively_parse_json_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_recursively_parse_json_strings(item) for item in obj]
    return obj


# ── Response cleanup: JSON parsing ────────────────────────────────────
# Intercept point: JSON extraction and cleanup for structured output.
# Handles fence removal, control char cleanup, escape fixing, nested
# extraction. Consider making this an extensible pipeline in the future.


def extract_and_parse_json(text: str) -> dict[str, Any]:
    """Extract and parse JSON from text, with multiple fallback strategies"""
    original_text = text
    text = text.strip()

    markdown_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    markdown_match = re.search(markdown_pattern, text, re.DOTALL)
    if markdown_match:
        _record_llm_metric("json_fence_removed")
        text = markdown_match.group(1).strip()

    # Strip leading/trailing markdown bold/italic markers (* or **)
    text_before = text
    text = re.sub(r"^\*{1,3}\s*", "", text)
    text = re.sub(r"\s*\*{1,3}$", "", text)
    if text != text_before:
        _record_llm_metric("json_markdown_bold_stripped")

    if not text:
        raise json.JSONDecodeError(
            f"Empty text after processing. Original: `{original_text[:200]}` ...", original_text, 0
        )

    try:
        result = json.loads(text)
        # Handle double-encoded JSON in nested values
        return _recursively_parse_json_strings(result)
    except json.JSONDecodeError as first_error:
        if "[...]" in text or '"..."' in text or ": ..." in text:
            raise json.JSONDecodeError(
                "JSON contains abbreviations/ellipsis ([...] or \"...\" or ': ...'). "
                "You MUST provide the complete, unabbreviated JSON. Do not truncate or use placeholders. "
                "Write out ALL values in full.",
                text,
                first_error.pos,
            ) from first_error

    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            _record_llm_metric("json_nested_extraction")
            return _recursively_parse_json_strings(result)
        except json.JSONDecodeError:
            pass

    text_before = text
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    if text != text_before:
        _record_llm_metric("json_control_chars_removed")
    text_before = text
    text = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", text)
    if text != text_before:
        _record_llm_metric("json_escape_fixed")

    try:
        result = json.loads(text)
        return _recursively_parse_json_strings(result)
    except json.JSONDecodeError as e:
        preview = text[:500] if len(text) > 500 else text
        raise json.JSONDecodeError(
            f"Failed to parse JSON after multiple cleanup attempts. Text preview: {preview}",
            text,
            e.pos,
        ) from e


def _resolve_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $ref references by inlining $defs definitions.

    Pydantic generates $ref/$defs for nested models. LLM APIs need
    flat schemas with type on every property. This inlines all
    references and removes $defs.
    """
    defs = schema.get("$defs", {})
    if not defs:
        return schema

    def _inline(node, _resolving=frozenset()):
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref_path = node["$ref"]
            if ref_path.startswith("#/$defs/"):
                def_name = ref_path[len("#/$defs/") :]
                if def_name in _resolving:
                    return {"type": "object"}  # break cycle for recursive models
                if def_name in defs:
                    return _inline(dict(defs[def_name]), _resolving | {def_name})
            # Unresolvable $ref (not in $defs or non-local) — replace with generic object
            # since LLM providers don't support $ref in tool schemas
            return {"type": "object"}
        result = {}
        for k, v in node.items():
            if k == "$defs":
                continue
            elif k == "properties" and isinstance(v, dict):
                result[k] = {pk: _inline(pv, _resolving) for pk, pv in v.items()}
            elif k in ("items", "additionalProperties") and isinstance(v, dict):
                result[k] = _inline(v, _resolving)
            elif k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
                result[k] = [_inline(i, _resolving) if isinstance(i, dict) else i for i in v]
            else:
                result[k] = v
        # Unwrap single-item allOf (Pydantic wraps $ref in allOf when Field has description)
        if "allOf" in result and len(result["allOf"]) == 1:
            merged = dict(result["allOf"][0])
            del result["allOf"]
            # Preserve sibling keys (e.g., description) alongside the unwrapped schema
            for k, v in result.items():
                if k not in merged:
                    merged[k] = v
            result = merged
        return result

    return _inline(schema)


def _strip_schema_noise(schema: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """Strip Pydantic noise (title, default) from a JSON schema recursively.

    These fields are auto-generated by Pydantic but add no value for LLM APIs
    and cause strict-mode rejections on some providers.

    When ``strict=True`` (OpenAI Responses API / Chat Completions strict mode):
    - Forces ``additionalProperties: false`` on every object type
    - Ensures every object has ``properties`` and ``required`` keys
    - Strips extra keys not in the strict-mode allowlist
    """
    STRIP_KEYS = frozenset({"title", "default"})
    # Strict mode only allows these keys (OpenAI spec)
    STRICT_ALLOWED = frozenset(
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
            cleaned = {k: v for k, v in node.items() if k in STRICT_ALLOWED}
        else:
            cleaned = {k: v for k, v in node.items() if k not in STRIP_KEYS}
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
    """Clean a Pydantic-generated JSON schema for LLM tool use.

    1. Resolves $ref/$defs inline (providers need flat schemas)
    2. Strips title/default noise (Pydantic artifacts, no LLM value)
    3. Ensures top-level structure has type/properties/required
    4. When ``strict=True``: enforces OpenAI strict-mode constraints
       (additionalProperties: false everywhere, properties+required on all objects)
    """
    resolved = _resolve_schema_refs(schema)
    cleaned = _strip_schema_noise(resolved, strict=strict)
    # Ensure standard top-level structure
    result = {
        "type": cleaned.get("type", "object"),
        "properties": cleaned.get("properties", {}),
        "required": cleaned.get("required", []),
    }
    if strict:
        result["additionalProperties"] = False
    return result


def _strict_schema_valid(schema: dict[str, Any]) -> bool:
    """Check whether a schema satisfies OpenAI strict-mode requirements.

    Every property node must have a ``type`` key (or ``anyOf``/``oneOf``),
    and every object must have ``required`` listing ALL property keys.
    Returns False if any node violates these constraints.
    """

    def _check(node: dict[str, Any]) -> bool:
        if not isinstance(node, dict):
            return True
        # A property node needs type or a union discriminator
        if (
            "type" not in node
            and "anyOf" not in node
            and "oneOf" not in node
            and "allOf" not in node
        ):
            return False
        # Object nodes: required must list every property key
        if (
            node.get("type") == "object"
            and "properties" in node
            and set(node.get("required", [])) != set(node["properties"].keys())
        ):
            return False
        # Array nodes: strict mode requires typed `items`. Strict cleaning drops
        # prefixItems (tuples) and other non-allowlisted array keywords, which can
        # leave an array with no usable `items` — the Responses API then rejects
        # "array schema missing items". Treat such arrays as strict-invalid so the
        # caller falls back to a Responses-safe non-strict schema. See issue 232.
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
            strict: When True, enforce OpenAI strict-mode constraints
                (additionalProperties: false everywhere, all objects need
                properties+required).
        """
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


@dataclass
class ToolCall:
    """Standardized tool call representation across all LLM APIs"""

    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    """Provider-neutral response returned by every UnifiedLLM implementation.

    ``usage`` remains the one-release compatibility storage field. It contains
    only normalized primitive values; ``reported_usage`` is the canonical typed
    read view used by new code.
    """

    content: str | BaseModel
    tool_calls: list[ToolCall]
    finish_reason: Literal["stop", "tool_calls", "length", "error"]
    assistant_message: dict[str, Any]
    reasoning: str | None = None
    usage: dict[str, int | float] | LLMUsage | None = None

    @property
    def reported_usage(self) -> LLMUsage | None:
        """Canonical provider-neutral usage view; never used for runtime control."""
        return _normalize_usage(self.usage)

    def to_wire(self) -> dict[str, Any]:
        """Return a provider-neutral, JSON-friendly response representation."""
        return {
            "message": copy.deepcopy(self.assistant_message),
            "content": self.content.model_dump()
            if isinstance(self.content, BaseModel)
            else self.content,
            "tool_calls": [vars(call) for call in self.tool_calls],
            "finish_reason": self.finish_reason,
            "reasoning": self.reasoning,
            "usage": _usage_dict(self.reported_usage),
        }

    @property
    def message(self) -> str | BaseModel | None:
        """Backward-compatible alias for content."""
        return self.content


# --- Bedrock JSON schema sanitization (gl-134) ---
# Bedrock Claude rejects schemas with certain JSON schema keywords.
# We strip/fix these for Bedrock models and rely on Pydantic's client-side
# validation instead.
#
# Source of truth: AWS ML Blog "Structured outputs on Amazon Bedrock"
# https://aws.amazon.com/blogs/machine-learning/structured-outputs-on-amazon-bedrock-schema-compliant-ai-responses/
# The blog lists numerical constraints, string constraints, and
# additionalProperties != false as "Not supported". As of 2025-04-21,
# numerical + maxItems actively 400; string constraints are silently
# accepted today but could be tightened at any time (like the numerical
# constraints were on Apr 20), so we strip them defensively.

_BEDROCK_STRIP_KEYWORDS = frozenset(
    {
        # Numerical — actively rejected (HTTP 400)
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        # Array — actively rejected (HTTP 400). maxItems, plus prefixItems
        # (heterogeneous tuples) and uniqueItems (sets): Bedrock's
        # output_config.format.schema reports these "not supported". Stripping
        # them degrades the schema to a plain array; PredictStrategy still
        # validates the exact tuple/set type client-side. See issue 232.
        "maxItems",
        "prefixItems",
        "uniqueItems",
        # String — blog says unsupported; currently accepted but stripped
        # defensively to avoid the next silent enforcement tightening.
        "minLength",
        "maxLength",
        "pattern",
    }
)


def _is_bedrock_model(model: str) -> bool:
    """Return True if the model string routes to AWS Bedrock."""
    m = model.lower()
    return "bedrock" in m or "/aws/" in m or m.startswith("aws/")


def _is_anthropic_model(model: str) -> bool:
    """Return True if the model is served by Anthropic (direct or via Bedrock).

    Used to gate features that are only meaningful for Anthropic's API,
    such as the explicit ``cache_control`` markers that don't affect
    OpenAI's automatic byte-prefix cache.

    Bedrock routes are only counted as Anthropic when the model id
    actually mentions Anthropic or Claude — otherwise we'd over-match
    Bedrock-hosted Titan/Cohere/Llama, which don't use the marker.
    """
    m = model.lower()
    if "anthropic" in m or m.startswith(("claude/", "claude-")):
        return True
    return _is_bedrock_model(model) and "claude" in m


def _sanitize_schema_for_bedrock(schema: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy *schema* and strip/fix keywords unsupported by Bedrock.

    Bedrock Claude rejects:
    - Numerical: minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf
    - String: minLength, maxLength, pattern (docs say unsupported; stripped defensively)
    - Array: maxItems (rejected), minItems > 1 (only 0 and 1 allowed)
    - Object: additionalProperties set to anything other than false

    Assumes Pydantic v2 schema output shapes. Does not recurse into
    prefixItems, patternProperties, or dependentSchemas (Pydantic v2
    does not emit these for typical models).
    """
    schema = copy.deepcopy(schema)
    _strip_unsupported_keys(schema)
    return schema


def _strip_unsupported_keys(node: Any) -> None:
    """Recursively fix unsupported Bedrock keywords in a schema node in-place."""
    if not isinstance(node, dict):
        return

    # Strip keywords that Bedrock rejects outright
    for key in list(node.keys()):
        if key in _BEDROCK_STRIP_KEYWORDS:
            del node[key]

    # minItems: Bedrock only accepts 0 or 1; clamp higher values to 1
    if "minItems" in node and isinstance(node["minItems"], int) and node["minItems"] > 1:
        node["minItems"] = 1

    # additionalProperties: Bedrock only accepts false
    if "additionalProperties" in node and node["additionalProperties"] is not False:
        node["additionalProperties"] = False

    # Recurse into sub-schemas
    for key in ("properties", "$defs", "definitions"):
        if key in node and isinstance(node[key], dict):
            for v in node[key].values():
                _strip_unsupported_keys(v)
    if "items" in node and isinstance(node["items"], dict):
        _strip_unsupported_keys(node["items"])
    for key in ("allOf", "anyOf", "oneOf"):
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                _strip_unsupported_keys(item)
    if "not" in node and isinstance(node["not"], dict):
        _strip_unsupported_keys(node["not"])


# OpenAI structured-output (json_schema) supports only this keyword subset. Anything
# else (uniqueItems, prefixItems, minItems/maxItems, pattern, format, numeric bounds, …)
# is rejected outright — even in non-strict mode. We strip to this set when sending a
# non-strict schema for return types that cannot satisfy strict mode.
_RESPONSE_SCHEMA_ALLOWED_KEYS = frozenset(
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


def _schema_strict_compatible(schema: dict[str, Any]) -> bool:
    """Return True if *schema* can be expressed under OpenAI strict structured outputs.

    Strict mode cannot represent several JSON Schema shapes that Pydantic emits for
    perfectly valid Python return types:

    - **free-form objects** (``dict[str, T]`` / bare ``dict``) — strict requires
      ``additionalProperties: false`` and every key declared in ``properties``;
    - **untyped arrays** (bare ``list``) — strict requires ``items`` with a type;
    - **heterogeneous tuples** (``tuple[int, str]``) — emitted as ``prefixItems``;
    - **unique arrays** (``set[T]``) — emitted with ``uniqueItems``.

    When this returns False the caller falls back to a non-strict json_schema so the
    request is accepted; PredictStrategy still validates the parsed output against the
    real Pydantic model (with retries) client-side.

    Note: callers pass the ``_resolve_schema_refs`` output, whose cycle-breaking turns
    recursive models into a property-less ``{"type": "object"}``. Such models therefore
    classify as incompatible and take the (safe) non-strict path — the request still
    succeeds and the value is validated client-side; only the schema hint is loosened.
    """

    def _has_type(node: Any) -> bool:
        # A schema node is "typed" (expressible in strict mode) if it declares a type
        # or a union/enum discriminator. Pydantic emits an empty ``{}`` for untyped
        # members (bare ``list`` items, ``Any``), which strict mode rejects.
        return isinstance(node, dict) and any(
            k in node for k in ("type", "anyOf", "oneOf", "allOf", "enum", "const")
        )

    def _check(node: Any) -> bool:
        if not isinstance(node, dict):
            return True
        # Untyped node (Pydantic emits ``{}`` for ``Any`` / untyped members). Strict
        # mode requires a type on every node, so route these to the non-strict path.
        if not _has_type(node):
            return False
        node_type = node.get("type")
        if node_type == "object":
            extra = node.get("additionalProperties")
            if isinstance(extra, dict) or extra is True:
                return False  # free-form dict
            if "properties" not in node:
                return False  # free-form object with no declared keys
        if node_type == "array":
            if "prefixItems" in node:
                return False  # heterogeneous tuple
            if node.get("uniqueItems"):
                return False  # set
            if not _has_type(node.get("items")):
                return False  # untyped / bare list
        for value in node.get("properties", {}).values():
            if not _check(value):
                return False
        if isinstance(node.get("items"), dict) and not _check(node["items"]):
            return False
        if isinstance(node.get("additionalProperties"), dict) and not _check(
            node["additionalProperties"]
        ):
            return False
        for key in ("anyOf", "oneOf", "allOf"):
            for item in node.get(key, []):
                if isinstance(item, dict) and not _check(item):
                    return False
        return True

    return _check(schema)


def _loose_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Reduce *schema* to the OpenAI-supported keyword subset for a non-strict request.

    Resolves ``$ref``/``$defs`` and recursively drops keywords OpenAI rejects
    (``uniqueItems``, ``prefixItems``, ``minItems``/``maxItems``, ``pattern``, numeric
    bounds, Pydantic ``title``/``default`` noise, …). The result is intentionally loose:
    it guides the model, while PredictStrategy enforces the exact type client-side.
    """
    resolved = _resolve_schema_refs(schema)

    def _strip(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        out = {k: v for k, v in node.items() if k in _RESPONSE_SCHEMA_ALLOWED_KEYS}
        if "properties" in out and isinstance(out["properties"], dict):
            out["properties"] = {k: _strip(v) for k, v in out["properties"].items()}
        if isinstance(out.get("items"), dict):
            out["items"] = _strip(out["items"])
        if isinstance(out.get("additionalProperties"), dict):
            out["additionalProperties"] = _strip(out["additionalProperties"])
        for key in ("anyOf", "oneOf", "allOf"):
            if isinstance(out.get(key), list):
                out[key] = [_strip(i) if isinstance(i, dict) else i for i in out[key]]
        # The Azure Responses endpoint rejects "object schema missing properties" and
        # "array schema missing items" even in non-strict mode. Supply empty defaults so
        # free-form dicts and tuples (which legitimately omit these) are accepted; an
        # empty schema means "any", matching the loose intent.
        if out.get("type") == "object" and "properties" not in out:
            out["properties"] = {}
        if out.get("type") == "array" and "items" not in out:
            out["items"] = {}
        return out

    return _strip(resolved)


def _maybe_sanitize_response_format(
    model: str, output_model: type[BaseModel]
) -> type[BaseModel] | dict[str, Any]:
    """Choose the response_format payload for *output_model* per provider.

    - **Bedrock**: always a sanitized strict json_schema dict (unchanged).
    - **Other providers** (OpenAI/Azure/NIM chat completions): return the Pydantic model
      as-is so AnyLLM builds a strict json_schema — UNLESS the schema cannot satisfy
      strict mode (free-form dict, bare/untyped list, tuple, set), in which case we send
      a non-strict json_schema so the request is accepted. See issue 232.
    """
    if _is_bedrock_model(model):
        schema = _sanitize_schema_for_bedrock(output_model.model_json_schema())
        return {
            "type": "json_schema",
            "json_schema": {
                "name": output_model.__name__,
                "strict": True,
                "schema": schema,
            },
        }

    raw_schema = output_model.model_json_schema()
    if _schema_strict_compatible(_resolve_schema_refs(raw_schema)):
        return output_model

    return {
        "type": "json_schema",
        "json_schema": {
            "name": output_model.__name__,
            "strict": False,
            "schema": _loose_response_schema(raw_schema),
        },
    }


def _responses_output_params(output_model: type[BaseModel]) -> dict[str, Any]:
    """Structured-output params for the Responses API (``the provider Responses API``).

    The Responses API is the strict-mode counterpart of chat completions'
    ``_maybe_sanitize_response_format``. the transport's ``text_format`` convenience builds a
    *strict* ``text.format`` json_schema from the Pydantic model, which the API rejects
    for free-form dicts, bare/untyped lists, tuples, and sets (see issue 232).

    - strict-compatible schema → ``{"text_format": output_model}`` (the transport builds strict);
    - otherwise → an explicit non-strict ``text.format`` so the request is accepted.
      the transport passes a provided ``text`` through verbatim (``text_format`` is then ignored).
      PredictStrategy still validates the parsed output against the real model client-side.
    """
    raw_schema = output_model.model_json_schema()
    if _schema_strict_compatible(_resolve_schema_refs(raw_schema)):
        return {"text_format": output_model}
    return {
        "text": {
            "format": {
                "type": "json_schema",
                "name": output_model.__name__,
                "strict": False,
                "schema": _loose_response_schema(raw_schema),
            }
        }
    }


# Bedrock/Anthropic reject messages containing tool_call blocks when no tools= param
# is set. Some providers require a placeholder tool in this case, so handle it locally.
_DUMMY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "_placeholder",
        "description": "Placeholder tool (not callable).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def _needs_dummy_tool(model: str) -> bool:
    """Return True if the model's provider requires tools= when tool_calls are present."""
    model_lower = model.lower()
    # Bedrock models come in many prefix forms: "bedrock/", "aws/anthropic/bedrock-...", etc.
    if _is_bedrock_model(model):
        return True
    # Direct Anthropic API calls
    return model_lower.startswith(("anthropic/", "anthropic."))


def _messages_have_tool_calls(messages: list[dict[str, Any]]) -> bool:
    """Return True if any message contains tool_call blocks."""
    for msg in messages:
        if msg.get("role") == "assistant":
            if msg.get("tool_calls"):
                return True
            # Anthropic-style: content list with tool_use blocks
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        return True
    return False


def _instantiate_output_model(output_model: type[BaseModel], json_data: Any) -> BaseModel:
    """Instantiate a Pydantic model from parsed JSON data.

    Handles both regular BaseModel (kwargs) and RootModel (positional arg).
    RootModel is used for dict/list return types where the value is returned directly.
    """
    if issubclass(output_model, RootModel):
        # RootModel takes the value directly as a positional argument
        return output_model(json_data)
    else:
        # Regular BaseModel takes kwargs
        return output_model(**json_data)


class UnifiedLLM(ABC):
    _registry_config: dict[str, Any] | None

    def __init__(self, model: str, **config):
        self.model = model
        self.config = config
        self._registry_config = None
        # Cache control injection — shared by CompletionClient and ResponsesClient
        self.cache_control_injection_points: list[dict[str, Any]] = (
            DEFAULT_CACHE_CONTROL_INJECTION_POINTS
        )
        # Per-client HTTP transport (managed httpx clients). Set by
        # concrete subclasses; guarded here so base helpers stay safe.
        self._http: _ClientHttp | None = None

    def close(self) -> None:
        """Release this client's sync HTTP resources (its own httpx clients)."""
        if self._http is not None:
            self._http.close()

    async def aclose(self) -> None:
        """Release this client's sync + async HTTP resources."""
        if self._http is not None:
            await self._http.aclose()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        await self.aclose()

    @staticmethod
    def _inject_cache_control_on_content(msg: dict) -> None:
        """Add cache_control to the last content block of a message.

        Anthropic's API requires cache_control on content blocks (not message level)
        for non-system messages.  When content is a plain string, converts it to
        the array-of-blocks format so cache_control can be attached.

        Mutates ``msg`` in place.
        """
        content = msg.get("content")
        if content is None:
            msg["cache_control"] = {"type": "ephemeral"}
        elif isinstance(content, str):
            msg["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
        elif isinstance(content, list) and len(content) > 0:
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = {"type": "ephemeral"}
        else:
            msg["cache_control"] = {"type": "ephemeral"}

    def _inject_cache_control(
        self, messages: list[dict[str, Any]], injection_points: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Add cache_control to designated messages for prompt caching.

        Adds cache_control at the message level (sibling of role/content), which
        is the format expected by NVIDIA's OpenAI-compatible gateway endpoints.
        This format also survives OpenAI SDK validation since the SDK only strips
        extra fields from content blocks, not from messages themselves.

        Supports two injection modes:

        1. **Role-based** (existing): marks ALL messages of a given role.
           ``{"role": "system"}``

        2. **Position-based** (new): marks only the last message of a given role.
           ``{"role": "assistant", "position": "last"}``

        Anthropic supports up to 4 cache breakpoints. Using both modes together
        caches both the stable system prompt and the conversation history prefix::

            [
                {"role": "system"},                        # breakpoint 1
                {"role": "tool", "position": "last"},      # breakpoint 2
            ]

        Args:
            messages: The message list to process.
            injection_points: List of dicts specifying where to add cache_control.
                Each dict must have a "role" key. Optional "position" key with
                value "last" restricts marking to only the last message of that role.

        Returns:
            A deep copy of messages with cache_control injected at breakpoints.
        """
        if not injection_points:
            return messages

        roles_to_cache_all: set[str] = set()
        roles_to_cache_last: set[str] = set()
        for p in injection_points:
            role = p.get("role")
            if not role:
                continue
            if p.get("position") == "last":
                roles_to_cache_last.add(role)
            else:
                roles_to_cache_all.add(role)

        if not roles_to_cache_all and not roles_to_cache_last:
            return messages

        messages = [copy.deepcopy(msg) for msg in messages]

        # Map role names to native Responses API type equivalents
        _ROLE_TO_TYPE = {"tool": "function_call_output"}

        for msg in messages:
            role = msg.get("role")
            if role and role in roles_to_cache_all:
                msg["cache_control"] = {"type": "ephemeral"}
            elif not role:
                # Native Responses format: match by type equivalent
                msg_type = msg.get("type")
                for r, t in _ROLE_TO_TYPE.items():
                    if t == msg_type and r in roles_to_cache_all:
                        msg["cache_control"] = {"type": "ephemeral"}
                        break

        # Anthropic needs cache_control on a content block (parts form); other providers
        # reject a content list on non-user roles, so mark at the message level instead.
        anthropic = _is_anthropic_model(self.model)
        for role in roles_to_cache_last:
            # Search for matching messages by role OR by equivalent native type
            native_type = _ROLE_TO_TYPE.get(role)
            for msg in reversed(messages):
                if msg.get("role") == role or (native_type and msg.get("type") == native_type):
                    if anthropic:
                        self._inject_cache_control_on_content(msg)
                    else:
                        msg["cache_control"] = {"type": "ephemeral"}
                    break

        return messages

    def get_model_info(self) -> "Any":
        """Get model metadata from the configured provider registry.

        Returns:
            Dict with model info (max_input_tokens, max_output_tokens, etc.)
            or None if model is not in the configured registry.
        """
        return None

    @property
    def context_window(self) -> int | None:
        """Get context window size (max input tokens).

        Resolution order:
        1. Explicit ``context_window`` config passed to the client constructor
        2. Registry config (if created via get_llm_client())
        3. Registry lookup by model name or model_name field
        4. the transport model info (for known models)
        5. None (unknown model)

        Returns:
            Maximum input tokens for this model, or None if unknown.
        """
        # First, honor explicit direct-client config.
        cw = self.config.get("context_window")
        if cw is not None:
            return cw

        # Then check registry config (set by get_llm_client()).
        if self._registry_config is not None:
            cw = self._registry_config.get("context_window")
            if cw is not None:
                return cw
            # Registry entry exists but lacks context_window — fall through

        # Try registry lookup by model string. The property is reachable
        # from any UnifiedLLM instance — including ones constructed
        # directly via CompletionClient(...) — so trigger the lazy
        # auto-load to match what users got from the pre-refactor
        # import-time side effect.
        from nooa.unifiedllm.registry import (
            MODELS,
            _registry_lock,
            ensure_loaded,
        )

        ensure_loaded()

        model_str = self.model

        # Snapshot under the lock so a concurrent reload_registry()
        # can't make us observe a half-cleared MODELS dict mid-lookup.
        with _registry_lock:
            models_snapshot = dict(MODELS)

        # Direct key match
        if model_str in models_snapshot:
            cw = models_snapshot[model_str].get("context_window")
            if cw is not None:
                return cw

        # Reverse lookup: check if any registry entry's model_name matches
        for _key, cfg in models_snapshot.items():
            if cfg.get("model_name") == model_str:
                cw = cfg.get("context_window")
                if cw is not None:
                    return cw

        # Fallback to provider metadata
        info = self.get_model_info()
        return info.get("max_input_tokens") if info else None

    @abstractmethod
    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Single method that:
        1. Transforms messages to API-specific format (if needed)
        2. Calls the LLM API
        3. Extracts tool calls (if any) and returns early
        4. If no tool calls, parses structured output (if requested)
        5. Returns everything in standardized LLMResponse

        Raises:
        - ValidationError: if output_model validation fails
        - json.JSONDecodeError: if JSON parsing fails
        - Other exceptions for API errors
        """
        pass

    @abstractmethod
    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Async version of call"""
        pass


def _usage_dict(value: LLMUsage | None) -> dict[str, int | float] | None:
    """Build the one-release compatibility mapping from canonical usage."""
    if value is None:
        return None
    result: dict[str, int | float] = {}
    if value.input_tokens is not None:
        result["prompt_tokens"] = value.input_tokens
    if value.output_tokens is not None:
        result["completion_tokens"] = value.output_tokens
    if value.input_tokens is not None and value.output_tokens is not None:
        result["total_tokens"] = value.input_tokens + value.output_tokens
    if value.cached_input_tokens is not None:
        result["cached_input_tokens"] = value.cached_input_tokens
    if value.reasoning_tokens is not None:
        result["reasoning_tokens"] = value.reasoning_tokens
    if value.cost_usd is not None:
        result["cost_usd"] = value.cost_usd
    return result


def _normalize_usage(value: Any) -> LLMUsage | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        value = {
            name: getattr(value, name, None) for name in ("prompt_tokens", "completion_tokens")
        }
    input_details = value.get("input_tokens_details") or value.get("prompt_tokens_details") or {}
    output_details = (
        value.get("output_tokens_details") or value.get("completion_tokens_details") or {}
    )
    if hasattr(input_details, "model_dump"):
        input_details = input_details.model_dump()
    if hasattr(output_details, "model_dump"):
        output_details = output_details.model_dump()
    return LLMUsage(
        input_tokens=value.get("input_tokens", value.get("prompt_tokens")),
        output_tokens=value.get("output_tokens", value.get("completion_tokens")),
        cached_input_tokens=value.get("cached_input_tokens", input_details.get("cached_tokens")),
        reasoning_tokens=value.get("reasoning_tokens", output_details.get("reasoning_tokens")),
        cost_usd=value.get("cost_usd"),
    )


def _chat_chunk_parts(chunk: Any) -> tuple[str, str, list[Any], str | None, Any]:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return "", "", [], None, getattr(chunk, "usage", None)
    choice = choices[0]
    delta = getattr(choice, "delta", None) or getattr(choice, "message", None)
    tool_calls = tuple(
        LLMToolCallChunk(
            index=getattr(call, "index", position),
            id=getattr(call, "id", None) or "",
            name=getattr(getattr(call, "function", None), "name", None) or "",
            arguments=getattr(getattr(call, "function", None), "arguments", None) or "",
        )
        for position, call in enumerate(getattr(delta, "tool_calls", None) or ())
    )
    return (
        getattr(delta, "content", None) or "",
        getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) or "",
        tool_calls,
        getattr(choice, "finish_reason", None),
        getattr(chunk, "usage", None),
    )


def _event_value(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _responses_chunk(event: Any) -> LLMChunk:
    """Normalize one Responses API stream event without retaining SDK objects."""
    event_type = _event_value(event, "type", "")
    content = ""
    calls: tuple[LLMToolCallChunk, ...] = ()
    finish: str | None = None
    usage = None
    if event_type == "response.output_text.delta":
        content = str(_event_value(event, "delta", "") or "")
    elif event_type == "response.function_call_arguments.delta":
        calls = (
            LLMToolCallChunk(
                index=int(_event_value(event, "output_index", 0) or 0),
                id=str(_event_value(event, "item_id", "") or ""),
                arguments=str(_event_value(event, "delta", "") or ""),
            ),
        )
    elif event_type == "response.output_item.added":
        item = _event_value(event, "item")
        if _event_value(item, "type") == "function_call":
            calls = (
                LLMToolCallChunk(
                    index=int(_event_value(event, "output_index", 0) or 0),
                    id=str(_event_value(item, "call_id", "") or _event_value(item, "id", "") or ""),
                    name=str(_event_value(item, "name", "") or ""),
                    arguments=str(_event_value(item, "arguments", "") or ""),
                ),
            )
    if event_type in {"response.completed", "response.incomplete", "response.failed"}:
        response = _event_value(event, "response", event)
        usage = _event_value(response, "usage")
        status = _event_value(response, "status")
        if event_type == "response.incomplete" or status == "incomplete":
            details = _event_value(response, "incomplete_details")
            finish = "length" if _event_value(details, "reason") == "max_output_tokens" else "error"
        elif event_type == "response.failed" or status == "failed":
            finish = "error"
        else:
            finish = "stop"
    return LLMChunk(content, None, calls, finish, _normalize_usage(usage))


def _aggregate_chat_chunks(chunks: list[Any]) -> Any:
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    finish = None
    usage = None
    for chunk in chunks:
        text, thought, tool_parts, chunk_finish, chunk_usage = _chat_chunk_parts(chunk)
        content.append(text)
        reasoning.append(thought)
        finish = chunk_finish or finish
        usage = chunk_usage or usage
        for pos, tc in enumerate(tool_parts):
            idx = getattr(tc, "index", pos)
            item = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            item["id"] += getattr(tc, "id", None) or ""
            item["name"] += tc.name
            item["arguments"] += tc.arguments
    tool_calls = [
        SimpleNamespace(
            id=v["id"], function=SimpleNamespace(name=v["name"], arguments=v["arguments"])
        )
        for _, v in sorted(calls.items())
    ]
    message = SimpleNamespace(
        content="".join(content),
        reasoning_content="".join(reasoning) or None,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=usage
    )


def _collect_sync(raw: Any) -> Any:
    """Consume a provider-neutral sync stream, leaving complete responses alone."""
    if hasattr(raw, "choices") or hasattr(raw, "output"):
        return raw
    if hasattr(raw, "__iter__") and not isinstance(raw, (dict, str, bytes)):
        return _aggregate_chat_chunks(list(raw))
    return raw


async def _collect_async(raw: Any) -> Any:
    """Consume a provider-neutral async stream, leaving complete responses alone."""
    if hasattr(raw, "choices") or hasattr(raw, "output"):
        return raw
    if hasattr(raw, "__aiter__"):
        chunks = [chunk async for chunk in raw]
        # Responses streams are mappings/events rather than chat choices.
        if chunks and all(isinstance(c, dict) and "type" in c for c in chunks):
            text = "".join(
                str(c.get("delta", ""))
                for c in chunks
                if c.get("type") == "response.output_text.delta"
            )
            return SimpleNamespace(output=[], output_text=text, status="completed", usage=None)
        return _aggregate_chat_chunks(chunks)
    return raw


def _map_completion_finish_reason(
    provider_response: Any,
) -> Literal["stop", "tool_calls", "length", "error"]:
    """Map a Chat-Completions provider finish_reason onto LLMResponse.finish_reason.

    Provider APIs report the provider's stop condition on
    ``provider_response.choices[0].finish_reason``. We surface ``"length"`` (output
    tokens exhausted) and ``"error"`` (e.g. ``content_filter``) so downstream
    logic (e.g. CodeAct's max-tokens abort) can react. Callers that have already
    detected tool calls should keep ``finish_reason="tool_calls"`` rather than
    calling this.
    """
    raw = None
    try:
        raw = provider_response.choices[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        raw = None

    if raw == "length":
        return "length"
    if raw == "tool_calls":
        return "tool_calls"
    if raw in ("content_filter", "error"):
        return "error"
    return "stop"


def _map_responses_finish_reason(
    provider_response: Any,
) -> Literal["stop", "tool_calls", "length", "error"]:
    """Map a Responses-API response onto LLMResponse.finish_reason.

    The Responses API reports truncation via ``status == "incomplete"`` with
    ``incomplete_details.reason == "max_output_tokens"`` (rather than a
    per-choice finish_reason). A ``status == "failed"`` response is surfaced as
    ``"error"``. Callers that have already detected tool calls should keep
    ``finish_reason="tool_calls"`` rather than calling this.
    """
    status = getattr(provider_response, "status", None)

    if status == "incomplete":
        details = getattr(provider_response, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        if reason is None and isinstance(details, dict):
            reason = details.get("reason")
        if reason == "max_output_tokens":
            return "length"
        return "error"
    if status == "failed":
        return "error"
    return "stop"


def _extract_reasoning_and_usage(
    provider_response: Any,
) -> tuple[str | None, dict[str, int] | None]:
    """Extract reasoning and usage from raw LLM response."""
    reasoning = None
    usage = None

    # Extract reasoning (o1-style or DeepSeek/QwQ)
    if hasattr(provider_response, "choices") and provider_response.choices:
        msg = provider_response.choices[0].message
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)

    # Extract usage
    if hasattr(provider_response, "usage") and provider_response.usage:
        usage_obj = provider_response.usage
        if hasattr(usage_obj, "_asdict"):
            usage = usage_obj._asdict()
        elif hasattr(usage_obj, "model_dump"):
            usage = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            usage = usage_obj
        else:
            # Try to extract common fields
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                "total_tokens": getattr(usage_obj, "total_tokens", 0),
            }

    return reasoning, usage


def _extract_xml_tool_calls(content: str) -> list["ToolCall"]:
    """Extract tool calls from XML format used by Nemotron/NIM models.

    vLLM's hermes parser expects JSON inside <tool_call> but these models output:
        <tool_call><function=name><parameter=p>v</parameter>...</function></tool_call>

    This is called as a fallback when raw_tool_calls is empty but content has <tool_call>.
    """
    import uuid as _uuid

    tool_calls = []
    tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    for match in tool_call_pattern.finditer(content):
        block = match.group(1).strip()

        # Try JSON format first (standard hermes: {"name": ..., "arguments": ...})
        try:
            import json as _json

            data = _json.loads(block)
            name = data.get("name", "")
            args = _json.dumps(data.get("arguments", data.get("parameters", {})))
            if name:
                tool_calls.append(
                    ToolCall(id=f"call_{_uuid.uuid4().hex[:8]}", name=name, arguments=args)
                )
            continue
        except (ValueError, TypeError):
            pass

        # XML format: <function=name><parameter=p1>v1</parameter>...</function>
        func_match = re.match(r"<function=([^>]+)>(.*?)</function>", block, re.DOTALL)
        if not func_match:
            continue

        func_name = func_match.group(1).strip()
        params_block = func_match.group(2)
        params: dict[str, str] = {}
        for param_match in re.finditer(
            r"<parameter=([^>]+)>(.*?)</parameter>", params_block, re.DOTALL
        ):
            params[param_match.group(1).strip()] = param_match.group(2).strip()

        import json as _json

        tool_calls.append(
            ToolCall(
                id=f"call_{_uuid.uuid4().hex[:8]}", name=func_name, arguments=_json.dumps(params)
            )
        )

    return tool_calls


def _extract_think_tags(content: str) -> tuple[str, str | None]:
    """Response cleanup: extract <think>...</think> tags from content.

    Returns (cleaned_content, reasoning) where:
    - cleaned_content is the content with think tags removed
    - reasoning is the extracted thinking content (or None if no tags found)

    Handles both complete tags and malformed tags (missing opening tag due to the transport bug).
    """
    # Pattern for complete <think>...</think> tags
    think_pattern = r"<think>(.*?)</think>"
    match = re.search(think_pattern, content, re.DOTALL)

    if match:
        reasoning = match.group(1).strip()
        cleaned = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()
        _record_llm_metric("think_tag_extracted")
        return cleaned, reasoning

    # Handle malformed case: content starts with thinking and ends with </think>
    # (the transport bug strips opening <think> but leaves closing </think>)
    if "</think>" in content:
        parts = content.split("</think>", 1)
        if len(parts) == 2:
            reasoning = parts[0].strip()
            cleaned = parts[1].strip()
            _record_llm_metric("malformed_think_tag_fixed")
            return cleaned, reasoning

    return content, None


DEFAULT_CACHE_CONTROL_INJECTION_POINTS = [
    {"role": "system"},
    {"role": "tool", "position": "last"},
]


class CompletionClient(UnifiedLLM):
    def __init__(
        self,
        model: str,
        retry_config: RetryConfig | None = None,
        http_config: HttpConfig | None = None,
        # use system as default for cache_control_injection_points
        cache_control_injection_points: list[dict[str, Any]] | None = None,
        **config,
    ):
        """
        Initialize CompletionClient.

        Args:
            model: The model identifier (e.g., "gpt-4o-mini", "nvidia_nim/...").
            retry_config: Optional retry configuration for API-level retries.
                         Defaults to RetryConfig(), which retries transient endpoint
                         failures such as rate limits, server errors, timeouts,
                         disconnects, and unreachable endpoints. Pass
                         RetryConfig(max_retries=0, rate_limit_extra_retries=0) to disable endpoint retries. Set
                         retry_on_empty_content=True to also retry when reasoning
                         models return empty content.
            http_config: Optional per-client HTTP connection-pool and timeout
                         settings. Applied only to THIS client's requests: the
                         client builds its own httpx client from these values and
                         passes it to the transport per call. No global state and no
                         monkey-patching of httpx — two clients with different
                         http_configs are fully independent.
            cache_control_injection_points: Optional list of role/position rules to
                enable prompt caching (for example: {"role": "system"} or
                {"role": "tool", "position": "last"}). Applied to all calls.
                Note: Do NOT manually add cache_control to message content when using this.
            **config: Additional configuration passed to the transport (api_key, api_base, etc.)
        """
        super().__init__(model, **config)
        self.retry_config = retry_config or RetryConfig()
        self._http_config = http_config or HttpConfig()
        self._http = _ClientHttp.for_completion(self.model, self.config, self._http_config)
        self._transport = AnyLLMTransport(self.model, self.config, self._http.httpx_async)
        transport_keys = {
            "provider",
            "api_key",
            "api_base",
            "base_url",
            "endpoint",
            "context_window",
            "provider_options",
            "capabilities",
            "client_type",
        }
        self._request_config = {
            key: value for key, value in self.config.items() if key not in transport_keys
        }
        # Only set default if explicitly None (not if empty list is passed)
        if cache_control_injection_points is not None:
            self.cache_control_injection_points = cache_control_injection_points
        else:
            self.cache_control_injection_points = DEFAULT_CACHE_CONTROL_INJECTION_POINTS

    def _convert_tool_to_schema(self, tool: Tool) -> dict[str, Any]:
        """Convert Tool object to Completion API schema format"""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.get_parameter_schema(),
            },
        }

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        cache_control_injection_points: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Sync version: Completion API uses standard message format, so no transformation needed.
        Messages are passed directly to the API.

        If retry_config.retry_on_empty_content is True, will retry when the model
        returns empty content but has reasoning_content (common with some reasoning models).
        """
        # Inject cache_control at the message level for prompt caching
        cache_points = (
            self.cache_control_injection_points
            if cache_control_injection_points is None
            else cache_control_injection_points
        )
        prepared_messages = self._inject_cache_control(messages, cache_points)

        api_params = {
            "messages": prepared_messages,
            **self._request_config,
            **kwargs,
        }

        if tools:
            api_params["tools"] = [self._convert_tool_to_schema(tool) for tool in tools]
            api_params["parallel_tool_calls"] = False

        if output_model is not None:
            api_params["response_format"] = _maybe_sanitize_response_format(
                self.model, output_model
            )

        # Bedrock/Anthropic reject messages with tool_call blocks when tools= is absent.
        if (
            "tools" not in api_params
            and _needs_dummy_tool(self.model)
            and _messages_have_tool_calls(prepared_messages)
        ):
            api_params["tools"] = [_DUMMY_TOOL_SCHEMA]

        # tool_choice/parallel_tool_calls are meaningless without tools — strip to avoid
        # provider rejections (e.g. Bedrock) when kwargs leak from CodeAct to PredictStrategy.
        if "tools" not in api_params:
            api_params.pop("tool_choice", None)
            api_params.pop("parallel_tool_calls", None)

        retry_on_empty = self.retry_config.retry_on_empty_content if self.retry_config else False

        http_client = self._http
        assert http_client is not None

        def _make_call():
            provider_response = _collect_sync(self._transport.completion(**api_params))
            reasoning, _ = _extract_reasoning_and_usage(provider_response)
            text_content = provider_response.choices[0].message.content or ""  # type: ignore[union-attr]

            # Raise EmptyContentError to trigger retry if configured
            if not text_content and reasoning and retry_on_empty:
                raise EmptyContentError(reasoning)

            return provider_response

        # Track LLM call for debugging (visible via SIGUSR2 if nooa debug handler installed)
        with _track_llm_call(model=self.model, endpoint=self.config.get("api_base")):
            provider_response = (
                sync_retry(_make_call, config=self.retry_config)
                if self.retry_config
                else _make_call()
            )

        reasoning, usage = _extract_reasoning_and_usage(provider_response)
        if usage:
            _record_llm_metric("token_usage", usage)
        raw_tool_calls = provider_response.choices[0].message.tool_calls  # type: ignore[union-attr]

        if raw_tool_calls:
            tool_calls = [
                ToolCall(id=tc.id, name=tc.function.name or "", arguments=tc.function.arguments)
                for tc in raw_tool_calls
            ]

            return LLMResponse(
                content="",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                assistant_message={
                    "role": "assistant",
                    "content": provider_response.choices[0].message.content or "",  # type: ignore[union-attr]
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in raw_tool_calls
                    ],
                },
                reasoning=reasoning,
                usage=_usage_dict(_normalize_usage(usage)),
            )

        text_content = provider_response.choices[0].message.content or ""  # type: ignore[union-attr]

        # Fallback: vLLM's hermes parser fails on Nemotron's XML tool call format.
        # Extract <tool_call><function=name><parameter=...> from content.
        if not raw_tool_calls and tools and "<tool_call>" in text_content:
            xml_tool_calls = _extract_xml_tool_calls(text_content)
            if xml_tool_calls:
                return LLMResponse(
                    content="",
                    tool_calls=xml_tool_calls,
                    finish_reason="tool_calls",
                    assistant_message={
                        "role": "assistant",
                        "content": text_content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": tc.arguments},
                            }
                            for tc in xml_tool_calls
                        ],
                    },
                    reasoning=reasoning,
                    usage=_usage_dict(_normalize_usage(usage)),
                )

        if output_model:
            # ── Response cleanup: reasoning-as-content fallback ───────
            # Intercept point: some reasoning models (e.g. Nemotron) put
            # structured output JSON in reasoning_content instead of content.
            # Consider making this an extensible transform in the future.
            parseable_content = text_content if text_content else (reasoning or "")
            if not text_content and reasoning:
                _record_llm_metric("reasoning_as_structured_output")
            json_data = extract_and_parse_json(parseable_content)
            parsed_content = _instantiate_output_model(output_model, json_data)

            return LLMResponse(
                content=parsed_content,
                tool_calls=[],
                finish_reason=_map_completion_finish_reason(provider_response),
                assistant_message={"role": "assistant", "content": text_content},
                reasoning=reasoning if text_content else None,
                usage=_usage_dict(_normalize_usage(usage)),
            )

        return LLMResponse(
            content=text_content,
            tool_calls=[],
            finish_reason=_map_completion_finish_reason(provider_response),
            assistant_message={"role": "assistant", "content": text_content},
            reasoning=reasoning,
            usage=_usage_dict(_normalize_usage(usage)),
        )

    def stream(
        self, messages: list[dict[str, Any]], tools: list[Tool] | None = None, **kwargs: Any
    ):
        """Yield provider-neutral chunks and always close the provider stream."""
        prepared = self._inject_cache_control(messages, self.cache_control_injection_points)
        params: dict[str, Any] = {
            "messages": prepared,
            **self._request_config,
            **kwargs,
            "stream": True,
        }
        if tools:
            params["tools"] = [self._convert_tool_to_schema(tool) for tool in tools]
        raw = self._transport.completion(**params)
        try:
            for item in raw:
                content, reasoning, calls, finish, usage = _chat_chunk_parts(item)
                yield LLMChunk(
                    content, reasoning or None, tuple(calls), finish, _normalize_usage(usage)
                )
        finally:
            close = getattr(raw, "close", None)
            if close:
                close()

    async def astream(
        self, messages: list[dict[str, Any]], tools: list[Tool] | None = None, **kwargs: Any
    ):
        """Yield provider-neutral async chunks and always close the provider stream."""
        prepared = self._inject_cache_control(messages, self.cache_control_injection_points)
        params: dict[str, Any] = {
            "messages": prepared,
            **self._request_config,
            **kwargs,
            "stream": True,
        }
        if tools:
            params["tools"] = [self._convert_tool_to_schema(tool) for tool in tools]
        raw = await self._transport.acompletion(**params)
        try:
            async for item in raw:
                content, reasoning, calls, finish, usage = _chat_chunk_parts(item)
                yield LLMChunk(
                    content, reasoning or None, tuple(calls), finish, _normalize_usage(usage)
                )
        finally:
            close = getattr(raw, "aclose", None)
            if close:
                await close()

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        cache_control_injection_points: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Async version: Completion API uses standard message format, so no transformation needed.
        Messages are passed directly to the API.

        If retry_config.retry_on_empty_content is True, will retry when the model
        returns empty content but has reasoning_content (common with some reasoning models).
        """
        # Inject cache_control at the message level for prompt caching
        cache_points = (
            self.cache_control_injection_points
            if cache_control_injection_points is None
            else cache_control_injection_points
        )
        prepared_messages = self._inject_cache_control(messages, cache_points)

        api_params = {
            "messages": prepared_messages,
            **self._request_config,
            **kwargs,
        }

        if tools:
            api_params["tools"] = [self._convert_tool_to_schema(tool) for tool in tools]
            api_params["parallel_tool_calls"] = False

        if output_model is not None:
            api_params["response_format"] = _maybe_sanitize_response_format(
                self.model, output_model
            )

        # Bedrock/Anthropic reject messages with tool_call blocks when tools= is absent.
        if (
            "tools" not in api_params
            and _needs_dummy_tool(self.model)
            and _messages_have_tool_calls(prepared_messages)
        ):
            api_params["tools"] = [_DUMMY_TOOL_SCHEMA]

        # tool_choice/parallel_tool_calls are meaningless without tools — strip to avoid
        # provider rejections (e.g. Bedrock) when kwargs leak from CodeAct to PredictStrategy.
        if "tools" not in api_params:
            api_params.pop("tool_choice", None)
            api_params.pop("parallel_tool_calls", None)

        retry_on_empty = self.retry_config.retry_on_empty_content if self.retry_config else False

        http_client = self._http
        assert http_client is not None

        async def _make_call():
            provider_response = await _collect_async(
                await self._transport.acompletion(**api_params)
            )
            reasoning, _ = _extract_reasoning_and_usage(provider_response)
            text_content = provider_response.choices[0].message.content or ""  # type: ignore[union-attr]

            # Raise EmptyContentError to trigger retry if configured
            if not text_content and reasoning and retry_on_empty:
                raise EmptyContentError(reasoning)

            return provider_response

        # Track LLM call for debugging (visible via SIGUSR2 if nooa debug handler installed)
        with _track_llm_call(model=self.model, endpoint=self.config.get("api_base")):
            provider_response = (
                await with_retry(_make_call, config=self.retry_config)
                if self.retry_config
                else await _make_call()
            )

        reasoning, usage = _extract_reasoning_and_usage(provider_response)
        if usage:
            _record_llm_metric("token_usage", usage)
        raw_tool_calls = provider_response.choices[0].message.tool_calls  # type: ignore[union-attr]

        if raw_tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id, name=tc.function.name or "", arguments=tc.function.arguments or ""
                )  # type: ignore[union-attr]
                for tc in raw_tool_calls
            ]

            return LLMResponse(
                content="",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                assistant_message={
                    "role": "assistant",
                    "content": provider_response.choices[0].message.content or "",  # type: ignore[union-attr]
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in raw_tool_calls
                    ],
                },
                reasoning=reasoning,
                usage=_usage_dict(_normalize_usage(usage)),
            )

        text_content = provider_response.choices[0].message.content or ""  # type: ignore[union-attr]

        # Fallback: vLLM's hermes parser fails on Nemotron's XML tool call format.
        # Extract <tool_call><function=name><parameter=...> from content.
        if not raw_tool_calls and tools and "<tool_call>" in text_content:
            xml_tool_calls = _extract_xml_tool_calls(text_content)
            if xml_tool_calls:
                return LLMResponse(
                    content="",
                    tool_calls=xml_tool_calls,
                    finish_reason="tool_calls",
                    assistant_message={
                        "role": "assistant",
                        "content": text_content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": tc.arguments},
                            }
                            for tc in xml_tool_calls
                        ],
                    },
                    reasoning=reasoning,
                    usage=_usage_dict(_normalize_usage(usage)),
                )

        if output_model:
            # ── Response cleanup: reasoning-as-content fallback ───────
            # Intercept point: some reasoning models (e.g. Nemotron) put
            # structured output JSON in reasoning_content instead of content.
            # Consider making this an extensible transform in the future.
            parseable_content = text_content if text_content else (reasoning or "")
            if not text_content and reasoning:
                _record_llm_metric("reasoning_as_structured_output")
            json_data = extract_and_parse_json(parseable_content)
            parsed_content = _instantiate_output_model(output_model, json_data)

            return LLMResponse(
                content=parsed_content,
                tool_calls=[],
                finish_reason=_map_completion_finish_reason(provider_response),
                assistant_message={"role": "assistant", "content": text_content},
                reasoning=reasoning if text_content else None,
                usage=_usage_dict(_normalize_usage(usage)),
            )

        return LLMResponse(
            content=text_content,
            tool_calls=[],
            finish_reason=_map_completion_finish_reason(provider_response),
            assistant_message={"role": "assistant", "content": text_content},
            reasoning=reasoning,
            usage=_usage_dict(_normalize_usage(usage)),
        )


class ReasoningCompletionClient(CompletionClient):
    """
    CompletionClient for reasoning models that output <think>...</think> tags.

    This client:
    1. Extracts reasoning from <think>...</think> tags in the content
    2. Handles provider behavior where opening <think> tag is stripped
    3. Returns clean content with reasoning in the `reasoning` field

    Use this for models like:
    - nvidia/Nemotron-3-Nano-30B-A3B
    - nvidia/llama-3.3-nemotron-super-49b-v1.5
    - Any other model that outputs <think> tags

    Example:
        client = ReasoningCompletionClient(
            model="nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5",
            api_base="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.6,
            top_p=0.95,
        )
        response = await client.acall(messages)
        print(response.content)    # Clean content without think tags
        print(response.reasoning)  # Extracted reasoning
    """

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        cache_control_injection_points: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Call with <think> tag extraction."""
        response = super().call(
            messages, tools, output_model, cache_control_injection_points, **kwargs
        )

        # Extract think tags from content
        if isinstance(response.content, str) and response.content:
            cleaned_content, think_reasoning = _extract_think_tags(response.content)

            # Combine extracted reasoning with any existing reasoning
            if think_reasoning:
                existing_reasoning = response.reasoning or ""
                combined_reasoning = (
                    f"{existing_reasoning}\n\n{think_reasoning}".strip()
                    if existing_reasoning
                    else think_reasoning
                )

                return LLMResponse(
                    content=cleaned_content,
                    tool_calls=response.tool_calls,
                    finish_reason=response.finish_reason,
                    assistant_message={
                        "role": "assistant",
                        "content": cleaned_content,
                    },
                    reasoning=combined_reasoning,
                    usage=response.usage,
                )

        return response

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        cache_control_injection_points: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Async call with <think> tag extraction."""
        response = await super().acall(
            messages, tools, output_model, cache_control_injection_points, **kwargs
        )

        # Extract think tags from content
        if isinstance(response.content, str) and response.content:
            cleaned_content, think_reasoning = _extract_think_tags(response.content)

            # Combine extracted reasoning with any existing reasoning
            if think_reasoning:
                existing_reasoning = response.reasoning or ""
                combined_reasoning = (
                    f"{existing_reasoning}\n\n{think_reasoning}".strip()
                    if existing_reasoning
                    else think_reasoning
                )

                return LLMResponse(
                    content=cleaned_content,
                    tool_calls=response.tool_calls,
                    finish_reason=response.finish_reason,
                    assistant_message={
                        "role": "assistant",
                        "content": cleaned_content,
                    },
                    reasoning=combined_reasoning,
                    usage=response.usage,
                )

        return response


def _provider_neutral(value: Any) -> Any:
    """Recursively detach response data from provider SDK model instances."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseModel):
        return _provider_neutral(value.model_dump())
    if hasattr(value, "model_dump"):
        return _provider_neutral(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _provider_neutral(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_provider_neutral(item) for item in value]
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            key: _provider_neutral(item)
            for key, item in attributes.items()
            if not key.startswith("_")
        }
    return str(value)


class ResponsesClient(UnifiedLLM):
    def __init__(
        self,
        model: str,
        retry_config: RetryConfig | None = None,
        http_config: HttpConfig | None = None,
        cache_control_injection_points: list[dict[str, Any]] | None = None,
        **config,
    ):
        """
        Initialize ResponsesClient.

        Mirrors CompletionClient so the Responses API path gets the same retry,
        HTTP, and cache-control behaviour. Accepting these as named parameters
        also keeps them out of ``self.config`` — otherwise they would leak into
        ``the provider Responses API`` as bogus API params.

        Args:
            model: The model identifier (e.g., "openai/gpt-5.3-codex").
            retry_config: Optional retry configuration for API-level retries.
                          Defaults to RetryConfig(), which retries transient endpoint
                          failures such as rate limits, server errors, timeouts,
                          disconnects, and unreachable endpoints. Pass
                          RetryConfig(max_retries=0, rate_limit_extra_retries=0) to
                          disable endpoint retries.
            http_config: Optional per-client HTTP connection-pool and timeout
                         settings. Applied only to THIS client's requests (its
                         own httpx client is passed to the transport per call). No
                         global state and no monkey-patching of httpx.
            cache_control_injection_points: Optional list of role/position rules to
                enable prompt caching (for example: {"role": "system"} or
                {"role": "tool", "position": "last"}). Applied to all calls.
            **config: Additional configuration passed to the transport (api_key, api_base, etc.)
        """
        capabilities = config.get("capabilities") or {}
        responses_enabled = (
            capabilities.get("responses", False)
            if isinstance(capabilities, dict)
            else bool(getattr(capabilities, "responses", False))
        )
        if not responses_enabled:
            raise ValueError(
                "ResponsesClient requires capabilities.responses=true; "
                "declare endpoint support before using the Responses API"
            )
        super().__init__(model, **config)
        self.retry_config = retry_config or RetryConfig()
        self._http_config = http_config or HttpConfig()
        self._http = _ClientHttp.for_responses(self.model, self.config, self._http_config)
        transport_config = {**self.config, "_responses_api": True}
        self._transport = AnyLLMTransport(self.model, transport_config, self._http.httpx_async)
        transport_keys = {
            "provider",
            "api_key",
            "api_base",
            "base_url",
            "endpoint",
            "context_window",
            "provider_options",
            "capabilities",
            "client_type",
        }
        self._request_config = {
            key: value for key, value in self.config.items() if key not in transport_keys
        }
        # Only set default if explicitly None (not if empty list is passed)
        if cache_control_injection_points is not None:
            self.cache_control_injection_points = cache_control_injection_points
        else:
            self.cache_control_injection_points = DEFAULT_CACHE_CONTROL_INJECTION_POINTS

    def _convert_tool_to_schema(self, tool: Tool) -> dict[str, Any]:
        """Convert Tool object to Responses API schema format."""
        schema_loose = tool.get_parameter_schema()
        required = schema_loose.get("required", [])
        properties = schema_loose.get("properties", {})
        use_strict = len(required) >= len(properties)

        if use_strict:
            schema = tool.get_parameter_schema(strict=True)
            if not _strict_schema_valid(schema):
                logger.warning(
                    "[ResponsesClient] Tool '%s' has parameters that cannot satisfy "
                    "strict-mode schema requirements (e.g. Any type, untyped properties). "
                    "Falling back to non-strict mode.",
                    tool.name,
                )
                schema = _loose_response_schema(schema_loose)
                use_strict = False
        else:
            schema = _loose_response_schema(schema_loose)

        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
            "strict": use_strict,
        }

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        cache_control_injection_points: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Sync version: Call LLM and parse response.

        Handles both native Responses format (from ResponsesProviderFormatter) and
        legacy OpenAI Chat format. System messages are extracted to the `instructions` param.
        """
        if kwargs.pop("stream", False):
            response = _stream_response(list(self.stream(messages, tools, **kwargs)))
            if output_model is not None:
                response.content = _instantiate_output_model(
                    output_model, extract_and_parse_json(str(response.content))
                )
            return response

        # Inject cache_control only for Anthropic-served models. the provider Responses API
        # passes input[] through verbatim — no equivalent of the Chat Completions
        # OpenAIGPTConfig.remove_cache_control_flag strip — so leaving the marker
        # on OpenAI/Azure/NIM Responses calls triggers a 400 "Unknown parameter:
        # input[N].cache_control" at the gateway.
        if _is_anthropic_model(self.model):
            cache_points = (
                self.cache_control_injection_points
                if cache_control_injection_points is None
                else cache_control_injection_points
            )
            prepared_messages = self._inject_cache_control(messages, cache_points)
        else:
            prepared_messages = messages
        input_messages, instructions = self._transform_messages(prepared_messages)

        api_params = {
            "input_data": input_messages,
            "truncation": "disabled",
            **self._request_config,
            **kwargs,
        }

        if instructions:
            api_params["instructions"] = instructions

        if "base_url" in api_params:
            api_params["api_base"] = api_params.pop("base_url")

        if tools:
            api_params["tools"] = [self._convert_tool_to_schema(tool) for tool in tools]
            api_params["tool_choice"] = "auto"
            api_params["parallel_tool_calls"] = False

        if output_model is not None:
            api_params.update(_responses_output_params(output_model))

        if reasoning := self.config.get("reasoning"):
            api_params["reasoning"] = reasoning

        http_client = self._http
        assert http_client is not None

        def _make_call():
            return self._transport.responses(**api_params)

        # Track LLM call for debugging (visible via SIGUSR2 if nooa debug handler installed)
        with _track_llm_call(model=self.model, endpoint=self.config.get("api_base")):
            provider_response = (
                sync_retry(_make_call, config=self.retry_config)
                if self.retry_config
                else _make_call()
            )

        # Extract usage if available (Responses API may have different structure)
        usage = None
        if hasattr(provider_response, "usage") and provider_response.usage:
            usage_obj = provider_response.usage
            if hasattr(usage_obj, "model_dump"):
                usage = usage_obj.model_dump()
            elif isinstance(usage_obj, dict):
                usage = usage_obj
        output: list[Any] = provider_response.output  # type: ignore[assignment]
        raw_tool_calls = [item for item in output if item.type == "function_call"]

        if raw_tool_calls:
            tool_calls = [
                ToolCall(id=tc.call_id or "", name=tc.name or "", arguments=tc.arguments or "")
                for tc in raw_tool_calls
            ]

            assistant_messages = [_provider_neutral(item) for item in output]

            return LLMResponse(
                content="",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                assistant_message={"_batch": assistant_messages},
                reasoning=None,  # Responses API doesn't have reasoning
                usage=_usage_dict(_normalize_usage(usage)),
            )

        text_content = self._extract_text_from_output(provider_response)

        if output_model:
            json_data = extract_and_parse_json(text_content)
            parsed_content = _instantiate_output_model(output_model, json_data)

            return LLMResponse(
                content=parsed_content,
                tool_calls=[],
                finish_reason=_map_responses_finish_reason(provider_response),
                assistant_message={"role": "assistant", "content": text_content},
                reasoning=None,
                usage=_usage_dict(_normalize_usage(usage)),
            )

        return LLMResponse(
            content=text_content,
            tool_calls=[],
            finish_reason=_map_responses_finish_reason(provider_response),
            assistant_message={"role": "assistant", "content": text_content},
            reasoning=None,
            usage=_usage_dict(_normalize_usage(usage)),
        )

    def stream(
        self, messages: list[dict[str, Any]], tools: list[Tool] | None = None, **kwargs: Any
    ):
        """Yield provider-neutral Responses API chunks and close the provider stream."""
        prepared = (
            self._inject_cache_control(messages, self.cache_control_injection_points)
            if _is_anthropic_model(self.model)
            else messages
        )
        input_messages, instructions = self._transform_messages(prepared)
        params: dict[str, Any] = {
            "input_data": input_messages,
            "truncation": "disabled",
            **self._request_config,
            **kwargs,
            "stream": True,
        }
        if instructions:
            params["instructions"] = instructions
        if "base_url" in params:
            params["api_base"] = params.pop("base_url")
        if tools:
            params.update(
                tools=[self._convert_tool_to_schema(tool) for tool in tools],
                tool_choice="auto",
                parallel_tool_calls=False,
            )
        if reasoning := self.config.get("reasoning"):
            params["reasoning"] = reasoning
        raw = self._transport.responses(**params)
        try:
            for event in raw:
                yield _responses_chunk(event)
        finally:
            close = getattr(raw, "close", None)
            if close:
                close()

    async def astream(
        self, messages: list[dict[str, Any]], tools: list[Tool] | None = None, **kwargs: Any
    ):
        """Yield provider-neutral async Responses API chunks and close the provider stream."""
        prepared = (
            self._inject_cache_control(messages, self.cache_control_injection_points)
            if _is_anthropic_model(self.model)
            else messages
        )
        input_messages, instructions = self._transform_messages(prepared)
        params: dict[str, Any] = {
            "input_data": input_messages,
            "truncation": "disabled",
            **self._request_config,
            **kwargs,
            "stream": True,
        }
        if instructions:
            params["instructions"] = instructions
        if "base_url" in params:
            params["api_base"] = params.pop("base_url")
        if tools:
            params.update(
                tools=[self._convert_tool_to_schema(tool) for tool in tools],
                tool_choice="auto",
                parallel_tool_calls=False,
            )
        if reasoning := self.config.get("reasoning"):
            params["reasoning"] = reasoning
        raw = await self._transport.aresponses(**params)
        try:
            async for event in raw:
                yield _responses_chunk(event)
        finally:
            close = getattr(raw, "aclose", None)
            if close:
                await close()

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        cache_control_injection_points: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Async version: Call LLM and parse response.

        Handles both native Responses format (from ResponsesProviderFormatter) and
        legacy OpenAI Chat format. System messages are extracted to the `instructions` param.
        """
        if kwargs.pop("stream", False):
            chunks = [chunk async for chunk in self.astream(messages, tools, **kwargs)]
            response = _stream_response(chunks)
            if output_model is not None:
                response.content = _instantiate_output_model(
                    output_model, extract_and_parse_json(str(response.content))
                )
            return response

        # See ResponsesClient.call for why cache_control injection is gated on
        # Anthropic models only.
        if _is_anthropic_model(self.model):
            cache_points = (
                self.cache_control_injection_points
                if cache_control_injection_points is None
                else cache_control_injection_points
            )
            prepared_messages = self._inject_cache_control(messages, cache_points)
        else:
            prepared_messages = messages
        input_messages, instructions = self._transform_messages(prepared_messages)

        api_params = {
            "input_data": input_messages,
            "truncation": "disabled",
            **self._request_config,
            **kwargs,
        }

        if instructions:
            api_params["instructions"] = instructions

        if "base_url" in api_params:
            api_params["api_base"] = api_params.pop("base_url")

        if tools:
            api_params["tools"] = [self._convert_tool_to_schema(tool) for tool in tools]
            api_params["tool_choice"] = "auto"
            api_params["parallel_tool_calls"] = False

        if output_model is not None:
            api_params.update(_responses_output_params(output_model))

        if reasoning := self.config.get("reasoning"):
            api_params["reasoning"] = reasoning

        http_client = self._http
        assert http_client is not None

        async def _make_call():
            return await _collect_async(await self._transport.aresponses(**api_params))

        # Track LLM call for debugging (visible via SIGUSR2 if nooa debug handler installed)
        with _track_llm_call(model=self.model, endpoint=self.config.get("api_base")):
            provider_response = (
                await with_retry(_make_call, config=self.retry_config)
                if self.retry_config
                else await _make_call()
            )

        # Extract usage if available (Responses API may have different structure)
        usage = None
        if hasattr(provider_response, "usage") and provider_response.usage:
            usage_obj = provider_response.usage
            if hasattr(usage_obj, "model_dump"):
                usage = usage_obj.model_dump()
            elif isinstance(usage_obj, dict):
                usage = usage_obj
        output: list[Any] = provider_response.output  # type: ignore[assignment]
        raw_tool_calls = [item for item in output if item.type == "function_call"]

        if raw_tool_calls:
            tool_calls = [
                ToolCall(id=tc.call_id or "", name=tc.name or "", arguments=tc.arguments or "")
                for tc in raw_tool_calls
            ]

            assistant_messages = [_provider_neutral(item) for item in output]

            return LLMResponse(
                content="",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                assistant_message={"_batch": assistant_messages},
                reasoning=None,
                usage=_usage_dict(_normalize_usage(usage)),
            )

        text_content = self._extract_text_from_output(provider_response)

        if output_model:
            json_data = extract_and_parse_json(text_content)
            parsed_content = _instantiate_output_model(output_model, json_data)

            return LLMResponse(
                content=parsed_content,
                tool_calls=[],
                finish_reason=_map_responses_finish_reason(provider_response),
                assistant_message={"role": "assistant", "content": text_content},
                reasoning=None,
                usage=_usage_dict(_normalize_usage(usage)),
            )

        return LLMResponse(
            content=text_content,
            tool_calls=[],
            finish_reason=_map_responses_finish_reason(provider_response),
            assistant_message={"role": "assistant", "content": text_content},
            reasoning=None,
            usage=_usage_dict(_normalize_usage(usage)),
        )

    def _transform_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Transform messages to Responses API format and extract instructions.

        Handles two input formats:
        1. Native Responses format (from ResponsesProviderFormatter): messages contain
           "type": "function_call" / "function_call_output" items alongside role-based messages.
           System messages have {"role": "system", ...} and are extracted to instructions.
        2. Legacy OpenAI Chat format: messages use {"role": "tool", "tool_call_id": ...} and
           {"role": "assistant", "tool_calls": [...]}. These are converted to native format.

        Returns (input_messages, instructions) where instructions is the concatenated
        system message content (or None if no system messages).
        """
        instructions_parts: list[str] = []
        transformed: list[dict[str, Any]] = []

        for msg in messages:
            # System messages → extract to instructions
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content:
                    instructions_parts.append(content)
                continue

            # Already in native Responses format (from ResponsesProviderFormatter)
            if "type" in msg:
                transformed.append(msg)
                continue

            # Legacy OpenAI format: tool result messages
            if msg.get("role") == "tool":
                # Extract output text: content may be a string OR a list of blocks
                # (position-based cache injection converts strings to block arrays)
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Extract text from content blocks and cache_control from the last block
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("text"):
                                text_parts.append(block["text"])
                    output_str = "".join(text_parts)
                else:
                    output_str = content

                item: dict[str, Any] = {
                    "type": "function_call_output",
                    "call_id": msg["tool_call_id"],
                    "output": output_str,
                }
                # Preserve cache_control for prompt caching
                if "cache_control" in msg:
                    item["cache_control"] = msg["cache_control"]
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "cache_control" in block:
                            item["cache_control"] = block["cache_control"]
                            break
                transformed.append(item)
                continue

            # Legacy OpenAI format: assistant messages with tool_calls
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Preserve assistant text that precedes tool calls (matches native formatter)
                if msg.get("content"):
                    transformed.append({"role": "assistant", "content": msg["content"]})
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    transformed.append(
                        {
                            "type": "function_call",
                            "call_id": tc["id"],
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", ""),
                        }
                    )
                continue

            # User/Assistant text messages → passthrough with cache_control preservation
            if msg.get("role") in ["user", "assistant"]:
                content = msg.get("content", "")
                if content is None:
                    content = ""
                item = {"role": msg["role"], "content": content}
                if "cache_control" in msg:
                    item["cache_control"] = msg["cache_control"]
                transformed.append(item)
                continue

            # Unknown format → passthrough
            transformed.append(msg)

        instructions = "\n\n".join(instructions_parts) if instructions_parts else None
        return transformed, instructions

    def _extract_text_from_output(self, response: Any) -> str:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        if hasattr(response, "output"):
            for item in response.output:
                if item.type == "message":
                    texts = []
                    for content_item in item.content:
                        if hasattr(content_item, "text"):
                            texts.append(content_item.text)  # type: ignore
                    return "\n".join(texts)
        return ""


# Lifecycle instrumentation is applied at the normalized client boundary, outside
# retry loops. Re-entrant wrappers let subclasses extend normalization without
# emitting a second terminal event.
_lifecycle_depth: ContextVar[int] = ContextVar("unifiedllm_lifecycle_depth", default=0)


def _instrument_sync_call(fn: Callable[..., LLMResponse]) -> Callable[..., LLMResponse]:
    def wrapped(
        self: UnifiedLLM, messages: list[dict[str, Any]], tools=None, *args: Any, **kwargs: Any
    ) -> LLMResponse:
        if _lifecycle_depth.get():
            return fn(self, messages, tools, *args, **kwargs)
        from nooa.runtime.llm_lifecycle import begin_llm_call, end_llm_call

        schemas = [self._convert_tool_to_schema(t) for t in tools] if tools else None
        call = begin_llm_call(self.model, messages, schemas, **kwargs)
        token = _lifecycle_depth.set(1)
        try:
            response = fn(self, messages, tools, *args, **kwargs)
        except BaseException as exc:
            end_llm_call(call, exception=exc)
            raise
        else:
            end_llm_call(call, response=response)
            return response
        finally:
            _lifecycle_depth.reset(token)

    return wrapped


def _instrument_async_call(fn: Callable[..., Any]) -> Callable[..., Any]:
    async def wrapped(
        self: UnifiedLLM, messages: list[dict[str, Any]], tools=None, *args: Any, **kwargs: Any
    ) -> LLMResponse:
        if _lifecycle_depth.get():
            return await fn(self, messages, tools, *args, **kwargs)
        from nooa.runtime.llm_lifecycle import begin_llm_call, end_llm_call

        schemas = [self._convert_tool_to_schema(t) for t in tools] if tools else None
        call = begin_llm_call(self.model, messages, schemas, **kwargs)
        token = _lifecycle_depth.set(1)
        try:
            response = await fn(self, messages, tools, *args, **kwargs)
        except BaseException as exc:
            end_llm_call(call, exception=exc)
            raise
        else:
            end_llm_call(call, response=response)
            return response
        finally:
            _lifecycle_depth.reset(token)

    return wrapped


def _stream_response(chunks: list[LLMChunk]) -> LLMResponse:
    """Build the normalized terminal response exposed to lifecycle consumers."""
    content = "".join(chunk.content for chunk in chunks)
    reasoning = "".join(chunk.reasoning or "" for chunk in chunks) or None
    calls: dict[int, dict[str, str]] = {}
    for chunk in chunks:
        for position, chunk_call in enumerate(chunk.tool_calls):
            index = chunk_call.index if chunk_call.index is not None else position
            item = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            item["id"] += chunk_call.id
            item["name"] += chunk_call.name
            item["arguments"] += chunk_call.arguments
    tool_calls = [
        ToolCall(id=value["id"], name=value["name"], arguments=value["arguments"])
        for _, value in sorted(calls.items())
    ]
    usage = next((chunk.usage for chunk in reversed(chunks) if chunk.usage is not None), None)
    finish_reason = next(
        (chunk.finish_reason for chunk in reversed(chunks) if chunk.finish_reason is not None),
        "tool_calls" if tool_calls else "stop",
    )
    assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {"name": tool_call.name, "arguments": tool_call.arguments},
            }
            for tool_call in tool_calls
        ]
    return LLMResponse(
        content="" if tool_calls else content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        assistant_message=assistant_message,
        reasoning=reasoning,
        usage=_usage_dict(_normalize_usage(usage)),
    )


def _instrument_sync_stream(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(
        self: UnifiedLLM, messages: list[dict[str, Any]], tools=None, *args: Any, **kwargs: Any
    ):
        if _lifecycle_depth.get():
            yield from fn(self, messages, tools, *args, **kwargs)
            return
        from nooa.runtime.llm_lifecycle import begin_llm_call, end_llm_call

        schemas = [self._convert_tool_to_schema(t) for t in tools] if tools else None
        call = begin_llm_call(self.model, messages, schemas, **kwargs)
        chunks: list[LLMChunk] = []
        stream = None
        try:
            stream = fn(self, messages, tools, *args, **kwargs)
            for chunk in stream:
                chunks.append(chunk)
                yield chunk
        except BaseException as exc:
            try:
                close = getattr(stream, "close", None)
                if close:
                    close()
            finally:
                end_llm_call(call, exception=exc)
            raise
        else:
            try:
                close = getattr(stream, "close", None)
                if close:
                    close()
            except BaseException as exc:
                end_llm_call(call, exception=exc)
                raise
            else:
                end_llm_call(call, response=_stream_response(chunks))

    return wrapped


def _instrument_async_stream(fn: Callable[..., Any]) -> Callable[..., Any]:
    async def wrapped(
        self: UnifiedLLM, messages: list[dict[str, Any]], tools=None, *args: Any, **kwargs: Any
    ):
        if _lifecycle_depth.get():
            async for chunk in fn(self, messages, tools, *args, **kwargs):
                yield chunk
            return
        from nooa.runtime.llm_lifecycle import begin_llm_call, end_llm_call

        schemas = [self._convert_tool_to_schema(t) for t in tools] if tools else None
        call = begin_llm_call(self.model, messages, schemas, **kwargs)
        chunks: list[LLMChunk] = []
        stream = None
        try:
            stream = fn(self, messages, tools, *args, **kwargs)
            async for chunk in stream:
                chunks.append(chunk)
                yield chunk
        except BaseException as exc:
            try:
                close = getattr(stream, "aclose", None)
                if close:
                    await close()
            finally:
                end_llm_call(call, exception=exc)
            raise
        else:
            try:
                close = getattr(stream, "aclose", None)
                if close:
                    await close()
            except BaseException as exc:
                end_llm_call(call, exception=exc)
                raise
            else:
                end_llm_call(call, response=_stream_response(chunks))

    return wrapped


CompletionClient.call = _instrument_sync_call(CompletionClient.call)
CompletionClient.acall = _instrument_async_call(CompletionClient.acall)
CompletionClient.stream = _instrument_sync_stream(CompletionClient.stream)
CompletionClient.astream = _instrument_async_stream(CompletionClient.astream)
ReasoningCompletionClient.call = _instrument_sync_call(ReasoningCompletionClient.call)
ReasoningCompletionClient.acall = _instrument_async_call(ReasoningCompletionClient.acall)
ResponsesClient.call = _instrument_sync_call(ResponsesClient.call)
ResponsesClient.acall = _instrument_async_call(ResponsesClient.acall)
ResponsesClient.stream = _instrument_sync_stream(ResponsesClient.stream)
ResponsesClient.astream = _instrument_async_stream(ResponsesClient.astream)
