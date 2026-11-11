# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import inspect
import json
import logging
import math
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, cast

import litellm
from pydantic import BaseModel, RootModel

from nooa.llm_types import (
    AssistantReasoning,
    AssistantText,
    CacheBoundary,
    LLMResponse,
    LLMUsage,
    ToolCall,
)
from nooa.unifiedllm.cache_policy import (
    apply_cache_policy,
    reject_legacy_cache_config,
)
from nooa.unifiedllm.media_inputs import (
    MediaInput,
    explicitly_rejected,
    filter_media_messages,
    media_occurrences,
    validate_media_inputs,
)

from .admission import _current_admission_controller, _run_admitted_attempt
from .errors import EmptyContentError
from .http_config import HttpConfig
from .limits import REPLY_CAP_KEYS, ContextLimits
from .reasoning import ReasoningConfig, apply_reasoning_level
from .retry import sync_retry, with_retry
from .retry_config import RetryConfig
from .structured_output import (
    StructuredOutputMode,
    validate_structured_output_mode,
)

logger = logging.getLogger(__name__)

# Bedrock/Anthropic reject requests where messages contain tool_call blocks but
# no tools= param is passed (e.g. PredictStrategy after a CodeAct turn).
# This flag tells litellm to auto-insert a dummy tool instead of raising.
litellm.modify_params = True

# litellm defaults to aiohttp for async HTTP (faster than httpx at high RPS).
# But it never closes sessions on shutdown, producing noisy ResourceWarnings:
#   "Unclosed client session" / "Unclosed connector"
# For a TUI agent making sequential calls the perf difference is irrelevant,
# and we already patch httpx for connection management. Disable aiohttp.
litellm.disable_aiohttp_transport = True

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


def _record_admission_observation(detail: dict[str, Any]) -> None:
    """Record one admission outcome on the generation span and harness metrics."""
    _record_llm_metric("llm_queue", detail)
    try:
        from opentelemetry import trace as otel_trace

        span = otel_trace.get_current_span()
        if span and span.is_recording():
            span.add_event("llm.queue", attributes=detail)
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not record LLM admission event: %s", e)


@contextmanager
def _track_llm_call(model: str, endpoint: str | None = None, prompt_tokens: int | None = None):
    """Track LLM call for debug purposes (if nooa debug handler is available)."""
    if _HAS_DEBUG_HANDLER and _llm_call_context:
        with _llm_call_context(model=model, endpoint=endpoint, prompt_tokens=prompt_tokens):
            yield
    else:
        yield


# Suppress harmless warning about litellm's async callback not being awaited
# (occurs during shutdown when async logging callbacks aren't fully cleaned up)
warnings.filterwarnings(
    "ignore",
    message="coroutine 'Logging.async_success_handler' was never awaited",
    category=RuntimeWarning,
)


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
# its HttpConfig, and passes them to litellm per call via litellm's
# caller-provided-client support. No global state, no monkey-patch, and two
# clients with different HttpConfigs stay fully independent.


class _ClientHttp:
    """Per-client HTTP transport: owns httpx clients + litellm wrappers.

    Builds one ``httpx.AsyncClient`` and one ``httpx.Client`` from the given
    ``HttpConfig`` (so the configured connection-pool limits — notably
    ``max_keepalive_connections`` — and timeouts apply to exactly this client's
    requests) and wraps them in the object litellm expects for the target
    provider:

    * The Responses API always routes through litellm's ``base_llm_http_handler``,
      which accepts an ``AsyncHTTPHandler`` / ``HTTPHandler`` for any provider, so
      responses clients always use those wrappers.
    * Chat Completions is provider-specific: OpenAI and OpenAI-compatible
      providers go through the OpenAI SDK path (``client=`` must be an
      ``AsyncOpenAI`` / ``OpenAI``), while anthropic/bedrock/etc. accept the
      ``AsyncHTTPHandler`` / ``HTTPHandler`` wrappers.

    If the correct wrapper can't be built (e.g. provider detection fails or the
    OpenAI SDK client can't be constructed) the corresponding wrapper is left as
    ``None`` and litellm falls back to building its own default client — the call
    still succeeds, it just doesn't get this client's custom pool/timeout.
    """

    def __init__(self, model: str, config: dict[str, Any], http_config: HttpConfig):
        import httpx

        self.http_config = http_config
        self._sync_closed = False
        self._async_closed = False
        self._timeout = http_config.to_httpx_timeout()
        self.limits = http_config.to_httpx_limits()

        # Mirror the SSL / redirect / default-header hardening litellm applies to
        # its own httpx clients, so handing litellm our client only changes the
        # connection-pool limits + timeout — not TLS verification, client certs,
        # or redirect handling (see GitLab #329 review). Falls back to plain
        # limits+timeout if litellm's internals move.
        hardening = self._httpx_hardening()

        # The per-client httpx clients. These are what carry this client's
        # connection-pool limits (incl. max_keepalive_connections) + timeouts.
        # transport is left as httpx's default so ``limits`` actually applies.
        self.httpx_async: httpx.AsyncClient = httpx.AsyncClient(
            limits=self.limits, timeout=self._timeout, **hardening
        )
        self.httpx_sync: httpx.Client = httpx.Client(
            limits=self.limits, timeout=self._timeout, **hardening
        )

        # litellm wrappers, filled in by _build_* below.
        self.async_client: Any = None
        self.sync_client: Any = None
        self._openai_clients: list[Any] = []

    @staticmethod
    def _httpx_hardening() -> dict[str, Any]:
        """Transport kwargs mirroring litellm's own httpx client construction.

        litellm builds its clients with SSL verification (``litellm.ssl_verify`` /
        ``SSL_VERIFY``), an optional client cert (``SSL_CERTIFICATE`` /
        ``litellm.ssl_certificate``), ``follow_redirects=True``, and a default
        User-Agent. We replicate that here so a client that supplies its own
        HttpConfig doesn't silently lose TLS/redirect behaviour.
        """
        try:
            import os

            from litellm.llms.custom_httpx.http_handler import (
                get_default_headers,
                get_ssl_configuration,
            )

            return {
                "verify": get_ssl_configuration(),
                "cert": os.getenv("SSL_CERTIFICATE", getattr(litellm, "ssl_certificate", None)),
                "follow_redirects": True,
                "headers": get_default_headers(),
            }
        except Exception as e:  # noqa: BLE001
            logger.debug("Falling back to minimal httpx transport config: %s", e)
            return {"follow_redirects": True}

    def _build_handler_wrappers(self) -> None:
        """Wrap the httpx clients in litellm's AsyncHTTPHandler / HTTPHandler."""
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

        # AsyncHTTPHandler eagerly creates an httpx.AsyncClient in __init__.
        # Build the wrapper object directly so there is no throwaway async client
        # to leak before replacing it with this _ClientHttp's managed client.
        async_handler = AsyncHTTPHandler.__new__(AsyncHTTPHandler)
        async_handler.timeout = self._timeout
        async_handler.event_hooks = None
        async_handler.client = self.httpx_async
        async_handler.client_alias = None
        self.async_client = async_handler

        sync_handler = HTTPHandler(timeout=self._timeout)
        try:
            sync_handler.client.close()  # close the throwaway sync client
        except Exception:  # noqa: BLE001
            pass
        sync_handler.client = self.httpx_sync
        self.sync_client = sync_handler

    def _build_completion_wrappers(self, model: str, config: dict[str, Any]) -> None:
        """Pick the right litellm client type for a Chat Completions provider."""
        try:
            _, provider, dynamic_api_key, dynamic_api_base = litellm.get_llm_provider(
                model,
                api_key=config.get("api_key"),
                api_base=config.get("api_base") or config.get("base_url"),
                custom_llm_provider=config.get("custom_llm_provider"),
            )
        except Exception as e:  # noqa: BLE001
            # Unknown/ambiguous model — don't risk handing an incompatible client
            # to a handler we can't identify. Let litellm build its own.
            logger.debug(
                "Could not detect provider for %r (%s); using litellm's default HTTP client.",
                model,
                e,
            )
            return

        openai_family = provider == "openai" or provider in getattr(
            litellm, "openai_compatible_providers", []
        )
        if not openai_family:
            # anthropic / bedrock / vertex / ... accept AsyncHTTPHandler|HTTPHandler
            # (guarded by isinstance in their handlers).
            self._build_handler_wrappers()
            return

        # OpenAI SDK path: client= must be an AsyncOpenAI / OpenAI wrapping httpx.
        api_key = config.get("api_key") or dynamic_api_key
        api_base = config.get("api_base") or config.get("base_url") or dynamic_api_base
        # CompletionClient's RetryConfig owns retries; calls set their own value.
        common: dict[str, Any] = {"timeout": self._timeout, "max_retries": 0}
        if api_key:
            common["api_key"] = api_key
        if api_base:
            common["base_url"] = api_base
        try:
            from openai import AsyncOpenAI, OpenAI

            self.async_client = AsyncOpenAI(http_client=self.httpx_async, **common)
            self.sync_client = OpenAI(http_client=self.httpx_sync, **common)
            self._openai_clients = [self.async_client, self.sync_client]
        except Exception as e:  # noqa: BLE001
            # e.g. no API key resolvable — fall back to litellm's own client so
            # auth/behaviour is preserved (this client just loses its custom pool).
            logger.debug(
                "Could not build OpenAI client for %r (%s); using litellm's default HTTP client.",
                model,
                e,
            )
            self.async_client = None
            self.sync_client = None

    @classmethod
    def for_completion(cls, model: str, config: dict[str, Any], http_config: HttpConfig):
        inst = cls(model, config, http_config)
        inst._build_completion_wrappers(model, config)
        return inst

    @classmethod
    def for_responses(cls, model: str, config: dict[str, Any], http_config: HttpConfig):
        inst = cls(model, config, http_config)
        # The Responses API always accepts the handler wrappers, regardless of
        # provider.
        inst._build_handler_wrappers()
        return inst

    def close(self) -> None:
        """Close the sync HTTP resources owned by this client."""
        if self._sync_closed:
            return
        self._sync_closed = True
        for oc in self._openai_clients:
            close = getattr(oc, "close", None)
            if close is not None and not inspect.iscoroutinefunction(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
        try:
            self.httpx_sync.close()
        except Exception:  # noqa: BLE001
            pass

    async def aclose(self) -> None:
        """Close both the sync and async HTTP resources owned by this client."""
        if not self._async_closed:
            self._async_closed = True
            for oc in self._openai_clients:
                close = getattr(oc, "close", None)
                if close is None or not inspect.iscoroutinefunction(close):
                    continue
                try:
                    await close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                await self.httpx_async.aclose()
            except Exception:  # noqa: BLE001
                pass
        self.close()


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

    markdown_pattern = r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n?```"
    markdown_match = re.fullmatch(markdown_pattern, text, re.DOTALL)
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
      as-is so litellm builds a strict json_schema — UNLESS the schema cannot satisfy
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
    """Structured-output params for the Responses API (``litellm.responses``).

    The Responses API is the strict-mode counterpart of chat completions'
    ``_maybe_sanitize_response_format``. litellm's ``text_format`` convenience builds a
    *strict* ``text.format`` json_schema from the Pydantic model, which the API rejects
    for free-form dicts, bare/untyped lists, tuples, and sets (see issue 232).

    - strict-compatible schema → ``{"text_format": output_model}`` (litellm builds strict);
    - otherwise → an explicit non-strict ``text.format`` so the request is accepted.
      litellm passes a provided ``text`` through verbatim (``text_format`` is then ignored).
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
# is set. litellm.modify_params=True should add a dummy tool, but doesn't work in all
# code paths (e.g. litellm router). We detect and handle this ourselves.
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


def _messages_have_tool_calls(
    messages: Sequence[dict[str, Any] | LLMResponse | CacheBoundary],
) -> bool:
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


class TokenCalibration:
    """Per-model EMA calibration of token estimates against API-reported usage.

    litellm's token_counter uses OpenAI's cl100k_base tokenizer for all models
    and ignores chat-template overhead, under-counting by 1.4–2.4× depending on
    the model family.  After each LLM call we observe the *actual* prompt token
    count from ``response.usage`` and maintain a running ratio so that future
    estimates are corrected.

    The ratio is an exponential moving average (EMA) that adapts quickly —
    after ~10 observations the initial value's influence drops below 3%.
    Before any observation arrives, ``default_ratio`` (1.0) is used — callers
    who want a conservative first estimate can raise this.
    """

    __slots__ = ("_ratios", "_alpha", "_default_ratio")

    def __init__(self, *, alpha: float = 0.3, default_ratio: float = 1.0):
        self._ratios: dict[str, float] = {}
        self._alpha = alpha
        self._default_ratio = default_ratio

    def update(self, model: str, estimated: int, actual: int) -> None:
        """Record one observation after an LLM call."""
        if estimated <= 0 or actual <= 0:
            return
        observed = actual / estimated
        prev = self._ratios.get(model)
        if prev is None:
            self._ratios[model] = observed
        else:
            self._ratios[model] = self._alpha * observed + (1 - self._alpha) * prev

    def ratio(self, model: str) -> float:
        """Current calibration ratio for *model* (default if unseen)."""
        return self._ratios.get(model, self._default_ratio)

    def calibrate(self, model: str, estimated: int) -> int:
        """Apply the calibration ratio to a raw estimate."""
        return int(estimated * self.ratio(model))

    def __repr__(self) -> str:
        entries = ", ".join(f"{m}: {r:.3f}" for m, r in self._ratios.items())
        return f"TokenCalibration({{{entries}}})"


# Module-level singleton so all UnifiedLLM instances share calibration data.
_token_calibration = TokenCalibration()


def _wrapped_block_text(value: Any) -> str | None:
    """Join text from a list of input_text/output_text/text blocks, if any."""
    if not isinstance(value, list):
        return None
    texts = [
        block["text"]
        for block in value
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]
    return "\n".join(texts) if texts else None


def _token_counter_block(block: Any) -> Any:
    """Retag one Responses content block into a shape litellm's token_counter
    recognizes (text/image_url/tool_use/tool_result/thinking/tool_reference).
    Anything else (e.g. an already-portable block) passes through unchanged."""
    if not isinstance(block, dict):
        return block
    if block.get("type") in {"input_text", "output_text"}:
        return {**block, "type": "text"}
    if block.get("type") == "input_image":
        return {"type": "image_url", "image_url": block.get("image_url")}
    return block


def _token_counter_messages(
    messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
) -> list[Any]:
    """Build a calibration-only view litellm.token_counter can actually count.

    litellm's token_counter only recognizes text/image_url/tool_use/
    tool_result/thinking/tool_reference block types and raises on
    ResponsesClient's input_text/output_text/input_image blocks (see
    _transform_messages), which would otherwise always fall through to the
    per-message fallback below and silently drop that content from the
    estimate -- collapsing it toward zero and inflating the calibration
    ratio by whatever multiple was missed. The fallback itself only sums
    "text" blocks, so an image billed real tokens by the API still counted
    as zero even after falling back.

    A native function_call_output item's wrapped `output` (see
    _transform_messages) is invisible to token_counter entirely -- confirmed
    against the real counter, a short and a very long tool result count
    identically -- so it never raises and never falls back to the
    per-message loop either; the text is just silently worth zero tokens.
    Representing it as an ordinary role/content message gives it the same
    real chat-template accounting every other message gets.
    """
    view = []
    for msg in messages:
        if not isinstance(msg, dict):
            view.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, list):
            msg = {
                **msg,
                "content": [_token_counter_block(block) for block in content],
            }
        elif msg.get("type") == "function_call_output":
            text = _wrapped_block_text(msg.get("output"))
            if text is None and isinstance(msg.get("output"), str):
                text = msg["output"]
            if text:
                msg = {"role": "tool", "content": text}
        view.append(msg)
    return view


def _update_token_calibration(
    model: str,
    messages: Sequence[dict[str, Any] | LLMResponse | CacheBoundary],
    usage: LLMUsage,
    tools: list[dict[str, Any]] | None = None,
    *,
    instructions: str | None = None,
) -> None:
    """Update token calibration from an API response's usage data.

    The recorded ratio is ``actual / estimated`` where ``actual`` is the API's
    reported ``prompt_tokens``. For the ratio to reflect the model tokenizer's
    real skew (and not a *coverage* gap), ``estimated`` must count the SAME
    request the API billed:

    * **messages-mode** ``token_counter`` (not a per-message text sum) so the
      chat-template / role framing the API charges is included, and
    * the **tool/function schemas** that were sent (``tools``) — for an agent
      with a large tool surface these are a big, fixed per-call cost that the
      API bills in ``prompt_tokens``. Omitting them (the old behavior, which
      summed only message text) made ``estimated`` far smaller than ``actual``
      and inflated the ratio (observed ~2.7x), which then scaled every
      displayed/triggering token count up by that bogus factor.
    """
    actual = usage.input_tokens
    if actual <= 0:
        return
    # Responses lifts the leading system prompt out of input. It is still
    # billed input, so include it in the estimate without copying history.
    message_list = list(messages)
    if instructions:
        message_list = [{"role": "system", "content": instructions}, *message_list]
    # Calibration is best-effort: it must NEVER raise out of the (already paid)
    # response path. The whole estimate — primary AND fallback — is guarded.
    try:
        counted = _token_counter_messages(message_list)
        try:
            estimated = litellm.token_counter(model=model, messages=counted)
            if tools:
                # Count the full messages+tools payload the way the API bills it,
                # then take the larger of the bare and with-tools counts
                # (with_tools is normally >= bare; max only guards a tokenizer
                # that returns less with tools attached).
                with_tools = litellm.token_counter(
                    model=model, messages=counted, tools=cast(Any, tools)
                )
                estimated = max(estimated, with_tools)
        except Exception:
            # token_counter can reject some message/tool shapes; fall back to the
            # per-message text sum rather than skip calibration entirely.
            estimated = 0
            for msg in counted:
                content = msg.get("content")
                if isinstance(content, str):
                    estimated += litellm.token_counter(model=model, text=content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            estimated += litellm.token_counter(
                                model=model, text=part.get("text", "")
                            )
        _token_calibration.update(model, estimated, actual)
    except Exception:
        logger.debug("token calibration skipped (estimate failed)", exc_info=True)


class UnifiedLLM(ABC):
    _registry_config: dict[str, Any] | None
    cache_breakpoint: Literal["auto", "openai", "anthropic"] | None
    retry_config: RetryConfig | None = None  # endpoint retries; also bounds media recovery

    def __init__(
        self,
        model: str,
        *,
        structured_output_mode: StructuredOutputMode = None,
        reasoning_levels: dict[str, dict[str, Any]] | None = None,
        reasoning_default: str | None = None,
        reasoning_level: str | None = None,
        media_inputs: list[MediaInput] | tuple[MediaInput, ...] | None = None,
        context_window: int | None = None,
        **config,
    ):
        self.structured_output_mode = validate_structured_output_mode(structured_output_mode)
        if self.structured_output_mode == "anthropic" and not isinstance(self, CompletionClient):
            raise ValueError(
                "structured_output_mode='anthropic' requires CompletionClient, not Responses"
            )
        reject_legacy_cache_config(config)
        # Freeze prevents field assignment, not mutations inside nested Any
        # settings. Detach this small configuration once, never the history.
        self._reasoning_config = ReasoningConfig(
            levels=reasoning_levels, default=reasoning_default
        ).model_copy(deep=True)
        if reasoning_level is not None:
            self._reasoning_config.settings(reasoning_level)
        self.reasoning_level = reasoning_level
        self.media_inputs = validate_media_inputs(media_inputs)
        # Attachment occurrences the current connection rejected; reset when
        # a request goes to another model or endpoint.
        self._media_exclusions: tuple[tuple[Any, ...] | None, set[Any]] = (None, set())
        self.model = model
        self.config = config
        self._context_window = context_window
        self._registry_config = None
        self.cache_breakpoint = None
        # Per-client HTTP transport (httpx clients + litellm wrappers). Set by
        # concrete subclasses; guarded here so base helpers stay safe.
        self._http: _ClientHttp | None = None

    @property
    def reasoning_levels(self) -> tuple[str, ...] | None:
        """Selectable levels; None means unknown, () means unsupported."""
        levels = self._reasoning_config.levels
        return None if levels is None else tuple(levels)

    @property
    def reasoning_default(self) -> str | None:
        """Documented route default; not a request override."""
        return self._reasoning_config.default

    def _prepare_call_config(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge the client configuration with per-call ``overrides`` for one request.

        ``api_base`` and ``base_url`` are two spellings of the same endpoint:

        - An endpoint override of ``None``, ``""`` or only whitespace counts as no
          override. Clearing the
          configured endpoint would send the request, with its credentials, to
          the provider's default endpoint instead of the configured gateway.
        - A non-None override of one spelling also replaces the other spelling if
          the client configuration sets it. Leaving the old value would give
          LiteLLM two different endpoints, and Responses requests copy
          ``base_url`` over ``api_base``. Both spellings are kept (rather than
          dropping one) because LiteLLM's async provider detection reads
          ``base_url``, and because an override equal to the configured URL must
          leave the route, and therefore the media permissions, unchanged.
        """
        from nooa.unifiedllm.route_identity import no_endpoint_override

        overrides = {
            key: value
            for key, value in overrides.items()
            if not (key in ("api_base", "base_url") and no_endpoint_override(value))
        }
        params = apply_reasoning_level(
            self._reasoning_config, self.model, self.config, overrides, self.reasoning_level
        )
        for alias, other in (("api_base", "base_url"), ("base_url", "api_base")):
            if alias in overrides and other not in overrides and other in params:
                params[other] = overrides[alias]
        return params

    def _request_media_inputs(self, call_config: dict[str, Any]) -> tuple[MediaInput, ...]:
        """A replacement route never inherits permissions from this connection.

        ``api_base`` and ``base_url`` are compared as one endpoint, so the
        configured URL under the other spelling keeps the permissions. URLs are
        normalized by route_identity.endpoint_changed, like context windows.
        """
        from nooa.unifiedllm.route_identity import endpoint_changed

        if (
            self._effective_model(call_config) != self.model
            or endpoint_changed(self.config, call_config)
            # A request with no endpoint goes to the provider default.
            or (
                any(self.config.get(key) for key in ("api_base", "base_url"))
                and not any(call_config.get(key) for key in ("api_base", "base_url"))
            )
            or any(
                call_config.get(key) != self.config.get(key)
                for key in ("custom_llm_provider", "client")
            )
        ):
            return ()
        return validate_media_inputs(self.media_inputs)

    def _prepare_media_messages(self, messages, call_config):
        enabled = self._request_media_inputs(call_config)
        # Media the interface cannot send is rejected after projection, by
        # _validate_implemented_media: only the projected request also shows the
        # native content of replayed assistant turns.
        return filter_media_messages(messages, enabled, self._rejected_media(call_config))

    def _rejected_media(self, call_config: dict[str, Any]) -> set[Any]:
        """Rejected attachment occurrences of this request's connection only."""
        client = call_config.get("client")
        connection = (
            self._effective_model(call_config),
            *(call_config.get(key) for key in ("api_base", "base_url", "custom_llm_provider")),
            None if client is None else id(client),
        )
        exclusions = self._media_exclusions
        if exclusions[0] != connection:
            # Another model or endpoint may read what this one could not.
            exclusions = (connection, set())
            self._media_exclusions = exclusions
        return exclusions[1]

    def _exclude_rejected_media(
        self, messages, kwargs: dict[str, Any], exc: Exception, staged: list[tuple[set, Any]]
    ) -> bool:
        """Exclude sent attachments the provider explicitly rejected, for a retry.

        The first exclusion of a request takes the newest sent attachment: the
        only one when the request carries one, and usually the one just added.
        A provider rarely says which one it could not read, so if the retry is
        rejected too, the next one excludes every remaining attachment of that
        kind at once. Each exclusion is appended to ``staged``; the caller
        drops them all again unless the request finally succeeds, so a request
        that still fails never leaves good attachments replaced by a note.

        True means the request changed and may be retried; a generic error, or
        one with nothing left to exclude, returns False.
        """
        try:
            call_config = self._prepare_call_config(kwargs)
        except Exception:
            return False  # The request itself is invalid; re-raise the original error.
        rejected = self._rejected_media(call_config)
        sent = media_occurrences(messages, self._request_media_inputs(call_config))
        candidates = [
            (kind, key)
            for kind, key in reversed(sent)
            if key not in rejected and explicitly_rejected(exc, kind)
        ]
        if not candidates:
            return False
        if staged:  # the newest was not it: the provider did not say which one
            kind = candidates[0][0]
            chosen = [(k, key) for k, key in candidates if k == kind]
        else:
            chosen = candidates[:1]
        for kind, key in chosen:
            rejected.add(key)
            staged.append((rejected, key))
            _record_llm_metric("media_rejected_retry", kind)
        logger.warning(
            "Provider rejected an attached %s; retrying with a note in place of %d: %s",
            chosen[0][0],
            len(chosen),
            str(exc)[:300],
        )
        return True

    def _media_recovery_attempts(self) -> int:
        # Shares the endpoint retry budget: max_retries=0 (e.g. a connect
        # probe, which must see the provider's own answer) disables it.
        return max(0, self.retry_config.max_retries) if self.retry_config else 0

    @staticmethod
    def _unstage_media(staged: list[tuple[set, Any]]) -> None:
        """The request still failed: its exclusions were not the fix, so undo them."""
        for rejected, key in staged:
            rejected.discard(key)

    def _validate_implemented_media(self, messages, call_config):
        from nooa.media_policy import reject_unsupported_media
        from nooa.request_interface import resolved_input_interface

        interface = resolved_input_interface(
            self._effective_model(call_config),
            call_config,
            client_type="responses" if isinstance(self, ResponsesClient) else "completion",
        )
        reject_unsupported_media(messages, interface)

    def get_context_limits(
        self, overrides: dict[str, Any] | None = None, *, fallback_reserve: int = 0
    ) -> ContextLimits:
        """Resolve context limits using the same settings as the next request.

        Includes selected reasoning levels, per-call caps and extra_body. Does
        not infer a reply cap from model metadata. ``fallback_reserve`` is only
        a planning allowance when no cap is configured. For a per-call route
        switch, never reuse the original endpoint's window; pass an explicit
        context_window or use a separately configured client for that model.
        """
        params = self._prepare_call_config(overrides or {})
        body = {**params, **(params.get("extra_body") or {})}
        caps = [body[k] for k in REPLY_CAP_KEYS if body.get(k) is not None]
        if len(caps) > 1:
            raise ValueError("Use only one reply token limit field")
        cap = caps[0] if caps else None
        if cap is not None and (type(cap) is not int or cap <= 0):
            raise ValueError("Reply token limit must be a positive integer")
        from nooa.unifiedllm.route_identity import same_context_route

        client_type = "responses" if isinstance(self, ResponsesClient) else "completion"
        same_route = same_context_route(
            self.model,
            self.config,
            client_type,
            self._effective_model(params),
            params,
            client_type,
        )
        window = (overrides or {}).get("context_window")
        if window is None and same_route:
            window = self.context_window
        if not isinstance(window, int) or window <= 0:
            window = None
        return ContextLimits(window, cap if cap is not None else fallback_reserve, cap is None)

    def _with_reduced_reply_limit(self, overrides: dict[str, Any], limit: int) -> dict[str, Any]:
        """Recovery keeps resolved effort settings, but lowers its reply cap."""
        params = self._prepare_call_config(overrides)
        extra = params.get("extra_body") or {}
        key = next((k for k in REPLY_CAP_KEYS if k in extra or k in params), "max_tokens")
        for alias in REPLY_CAP_KEYS:
            params.pop(alias, None)
        if extra:
            params["extra_body"] = {k: v for k, v in extra.items() if k not in REPLY_CAP_KEYS}
        params[key] = limit
        # Do not re-apply a selected level's original cap on the retry.
        params["reasoning_level"] = None
        return params

    def _effective_model(self, call_config: dict[str, Any]) -> str:
        """Return the model this individual request will actually dispatch."""
        model = call_config.get("model", self.model)
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        return model

    def _llm_request(self, messages, tools, output_model, kwargs):
        """The LLMRequest for one call. The wire values still come from kwargs.

        The reply cap is the one ``kwargs`` set (top level, then ``extra_body``),
        so the adapter's fill-in of request fields never adds a second cap.
        """
        from nooa.unifiedllm.request import LLMRequest

        tool_choice = kwargs.get("tool_choice", "auto")
        if not isinstance(tool_choice, (str, dict)):
            tool_choice = "auto"
        level = kwargs.get("reasoning_level", self.reasoning_level)
        extra = kwargs.get("extra_body")
        layers = [kwargs, extra if isinstance(extra, Mapping) else {}]
        cap = next(
            (
                layer[k]
                for layer in layers
                for k in sorted(REPLY_CAP_KEYS)
                if type(layer.get(k)) is int
            ),
            None,
        )
        return LLMRequest(
            messages=messages,
            tools=tools or [],
            tool_choice=tool_choice,
            output_type=output_model,
            reasoning_level=level if isinstance(level, str) else None,
            max_output_tokens=cap,
        )

    def _dispatch_request(self, messages, tools, output_model, kwargs, managed):
        """Check one call's settings and return its LLMRequest and effective model.

        The adapter resolves the settings again per attempt; doing it once here
        keeps invalid calls failing before tracking and retries start.
        """
        call_config = self._prepare_call_config(kwargs)
        self._validate_request_config(managed, call_config)
        model = self._effective_model(call_config)
        return self._llm_request(messages, tools, output_model, kwargs), model

    def _validate_cache_breakpoint_model(self, effective_model: str) -> None:
        """Reject applying a model-declared cache mapping to a call override."""
        if self.cache_breakpoint not in {None, "auto"} and effective_model != self.model:
            raise ValueError(
                "cache_breakpoint is declared for the client model and cannot be used "
                "with a per-call model override"
            )

    @staticmethod
    def _validate_request_config(name: str, call_config: dict[str, Any]) -> None:
        """Keep provider payloads and routing on their validated top-level paths."""
        reject_legacy_cache_config(call_config)
        if "structured_output_mode" in call_config or (
            isinstance(call_config.get("extra_body"), Mapping)
            and "structured_output_mode" in call_config["extra_body"]
        ):
            raise ValueError("structured_output_mode is a framework-only profile setting")
        if "media_inputs" in call_config or (
            isinstance(call_config.get("extra_body"), Mapping)
            and "media_inputs" in call_config["extra_body"]
        ):
            raise ValueError("media_inputs is a client setting, not a provider request parameter")
        if name in call_config:
            raise ValueError(
                f"{name!r} is managed by UnifiedLLM; pass conversation data through "
                "the messages argument"
            )
        extra_body = call_config.get("extra_body")
        if extra_body is not None and not isinstance(extra_body, Mapping):
            raise ValueError("extra_body must be a mapping")
        if "cache_breakpoint" in call_config or (
            isinstance(extra_body, Mapping) and "cache_breakpoint" in extra_body
        ):
            raise ValueError(
                "cache_breakpoint is a client setting; pass it to the client constructor, "
                "not call/acall or extra_body"
            )
        if isinstance(extra_body, Mapping) and (reserved := {name, "model"} & set(extra_body)):
            fields = ", ".join(repr(field) for field in sorted(reserved))
            raise ValueError(
                f"extra_body may not override reserved field(s) {fields}; pass model at "
                "the top level and conversation data through the messages argument"
            )

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

    def _resolve_cache_mapping(self, model: str | None = None, *, responses: bool) -> str | None:
        """The single source of truth for which cache_control mapping applies.

        Both the marking decision here and the rendering decision in
        call()/acall() (whether content is pre-wrapped so marking doesn't
        change a message's wire shape) must agree, or the exact class of bug
        this stability fix exists to prevent reappears -- silently desynced
        between two copies of the same logic.
        """
        mapping = self.cache_breakpoint
        if mapping == "auto" and not responses:
            mapping = "anthropic" if _is_anthropic_model(model or self.model) else None
        return mapping

    def _prepare_cache_boundary(self, messages, *, responses, model=None, instructions=None):
        mapping = self._resolve_cache_mapping(model, responses=responses)
        return apply_cache_policy(messages, mapping, responses=responses, instructions=instructions)

    def count_tokens(self, text: str) -> int:
        """Count tokens using model-appropriate tokenizer.

        Uses litellm's token_counter with a calibration correction derived
        from API-reported usage.  Before the first LLM call completes the
        raw litellm estimate is returned unchanged (ratio = 1.0).

        Args:
            text: The text to count tokens for.

        Returns:
            Calibrated number of tokens in the text.
        """
        raw = litellm.token_counter(model=self.model, text=text)
        return _token_calibration.calibrate(self.model, raw)

    def get_model_info(self) -> "Any":
        """Get model metadata from litellm registry.

        Returns:
            Dict with model info (max_input_tokens, max_output_tokens, etc.)
            or None if model is not in litellm's registry.
        """
        try:
            return litellm.get_model_info(self.model)
        except Exception:
            return None

    @property
    def context_window(self) -> int | None:
        """Get context window size (max input tokens).

        Resolution order:
        1. Explicit ``context_window`` passed to the client constructor
        2. Registry config (if created via get_llm_client())
        3. Unambiguous registry lookup by compatible route (direct clients only)
        4. litellm model info (for known models)
        5. None (unknown model)

        Returns:
            Maximum input tokens for this model, or None if unknown.
        """
        # Keep local context-budget metadata out of provider API requests.
        if self._context_window is not None:
            return self._context_window

        from nooa.unifiedllm.route_identity import same_context_route

        client_type = "responses" if isinstance(self, ResponsesClient) else "completion"
        # Clients built by the registry carry their entry's settings. Use only
        # that entry; if it declares no context_window, do not borrow one from
        # another alias that happens to use the same model.
        if self._registry_config is not None:
            route = getattr(self, "_registry_context_route", None)
            if route is None or same_context_route(
                self.model, self.config, client_type, route[0], route[1], route[2]
            ):
                cw = self._registry_config.get("context_window")
                if cw is not None:
                    return cw
            info = self.get_model_info()
            return info.get("max_input_tokens") if info else None

        # Try registry lookup by model string. The property is reachable
        # from any UnifiedLLM instance — including ones constructed
        # directly via CompletionClient(...) — so trigger the lazy
        # auto-load to match what users got from the pre-refactor
        # import-time side effect.
        from nooa.unifiedllm.registry import (
            MODELS,
            _lookup_api_key_env,
            _registry_lock,
            ensure_loaded,
        )

        ensure_loaded()

        model_str = self.model

        # Snapshot under the lock so a concurrent reload_registry()
        # can't make us observe a half-cleared MODELS dict mid-lookup.
        with _registry_lock:
            models_snapshot = dict(MODELS)

        # A directly built client has no entry of its own. Use a registry
        # context_window only if every alias with the same route declares the
        # same positive value. If they disagree, or one of them declares none,
        # the result would depend on YAML order, so fall back instead.
        #
        # A direct client never has ``api_key_env``, so credentials are compared
        # here rather than in same_context_route: a client without an api_key
        # matches regardless of credentials; a client with one matches only an
        # entry whose resolved key (api_key, else the value of api_key_env) is
        # equal. Keys are compared in memory only and never logged.
        client_key = self.config.get("api_key")
        own_route = {
            name: value
            for name, value in self.config.items()
            if name not in ("api_key", "api_key_env")
        }
        windows = set()
        undeclared = False
        for key, cfg in models_snapshot.items():
            entry_client_type = cfg.get("client_type", "completion")
            if client_key is not None:
                # Resolve the entry key as the registry does (validated name,
                # NVIDIA synonyms, empty means unset), but without its warnings:
                # this loop visits every entry. A malformed api_key_env gives None.
                entry_key = cfg.get("api_key")
                if entry_key is None:
                    entry_key, _ = _lookup_api_key_env(cfg)
                if entry_key != client_key:
                    continue
            # Completion and Responses clients are never sent the registry-only
            # ``endpoint`` key (the decision-route spelling), so leave it out of
            # the comparison, as client_from_config does. Decision entries keep it.
            skipped = ("api_key", "api_key_env") + (
                () if entry_client_type == "decision" else ("endpoint",)
            )
            route = {name: value for name, value in cfg.items() if name not in skipped}
            if same_context_route(
                model_str,
                own_route,
                client_type,
                cfg.get("model_name", key),
                route,
                entry_client_type,
            ):
                cw = cfg.get("context_window")
                if type(cw) is int and cw > 0:
                    windows.add(cw)
                else:
                    undeclared = True
        if len(windows) == 1 and not undeclared:
            return windows.pop()

        # Fallback to litellm
        info = self.get_model_info()
        return info.get("max_input_tokens") if info else None

    def call(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
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

        Pass prior LLMResponse objects directly in messages to retain compatible state.
        An attachment the provider explicitly rejects is retried as a note
        (``_exclude_rejected_media``); each request is one ``_call_once``.

        Raises:
        - ValidationError: if output_model validation fails
        - json.JSONDecodeError: if JSON parsing fails
        - Other exceptions for API errors
        """
        staged: list[tuple[set, Any]] = []
        try:
            for _ in range(self._media_recovery_attempts()):
                try:
                    return self._call_once(messages, tools, output_model, **kwargs)
                except Exception as exc:
                    if not self._exclude_rejected_media(messages, kwargs, exc, staged):
                        raise
            return self._call_once(messages, tools, output_model, **kwargs)
        except BaseException:
            self._unstage_media(staged)
            raise

    async def acall(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Async version of call"""
        staged: list[tuple[set, Any]] = []
        try:
            for _ in range(self._media_recovery_attempts()):
                try:
                    return await self._acall_once(messages, tools, output_model, **kwargs)
                except Exception as exc:
                    if not self._exclude_rejected_media(messages, kwargs, exc, staged):
                        raise
            return await self._acall_once(messages, tools, output_model, **kwargs)
        except BaseException:
            self._unstage_media(staged)
            raise

    @abstractmethod
    def _call_once(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """One request; ``call`` adds media recovery around it. Subclasses implement it."""

    @abstractmethod
    async def _acall_once(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Async ``_call_once``; ``acall`` adds media recovery around it. Subclasses implement it."""


def _collect_sync(raw: Any) -> "litellm.ModelResponse":
    """Consume a sync streaming or non-streaming litellm response, returning ModelResponse."""
    if isinstance(raw, litellm.CustomStreamWrapper):
        chunks = list(raw)
        result = litellm.stream_chunk_builder(chunks)
        if result is None:
            raise ValueError("stream_chunk_builder returned None for empty stream")
        if not isinstance(result, litellm.ModelResponse):
            raise TypeError(f"Expected ModelResponse, got {type(result)}")
        return result
    if not isinstance(raw, litellm.ModelResponse):
        raise TypeError(f"Expected ModelResponse, got {type(raw)}")
    return raw


def _with_sdk_retries(client: Any, max_retries: int) -> Any:
    """Return a per-call copy of an OpenAI SDK client with its own retry setting.

    LiteLLM writes ``max_retries`` onto the SDK client it is given, so a shared
    client would let concurrent calls change each other's retry budget. The
    copy keeps the same HTTP pool. Other client types are returned unchanged.
    """
    with_options = getattr(client, "with_options", None)
    return with_options(max_retries=max_retries) if callable(with_options) else client


async def _collect_async(raw: Any) -> "litellm.ModelResponse":
    """Consume an async streaming or non-streaming litellm response, returning ModelResponse."""
    if isinstance(raw, litellm.CustomStreamWrapper):
        chunks = [chunk async for chunk in raw]  # type: ignore[attr-defined]
        result = litellm.stream_chunk_builder(chunks)
        if result is None:
            raise ValueError("stream_chunk_builder returned None for empty stream")
        if not isinstance(result, litellm.ModelResponse):
            raise TypeError(f"Expected ModelResponse, got {type(result)}")
        return result
    if not isinstance(raw, litellm.ModelResponse):
        raise TypeError(f"Expected ModelResponse, got {type(raw)}")
    return raw


async def _run_async_provider_call[T](
    call: Callable[[], Coroutine[Any, Any, T]],
    *,
    request_timeout: float | None,
) -> T:
    """Run one provider attempt, holding admission through its actual exit.

    Acquisition happens before the provider task is created, so cancelling a
    queued caller cannot dispatch abandoned work.  Once dispatched, the
    provider task owns the permit and releases it in ``finally``.  Shielding
    keeps that accounting correct when the caller is cancelled while remote
    work or a stream is still active.

    ``request_timeout`` bounds the whole attempt, stream included. On expiry it
    cancels the task and raises TimeoutError. None disables it.
    """
    admission_controller = _current_admission_controller()
    if admission_controller is None:
        return await _await_provider_task(asyncio.create_task(call()), request_timeout)

    permit = await admission_controller.acquire(_record_admission_observation)
    if permit is None:
        return await _await_provider_task(asyncio.create_task(call()), request_timeout)

    try:
        task = asyncio.create_task(_run_admitted_attempt(call, permit))
    except BaseException:
        permit.release()
        raise

    return await _await_provider_task(task, request_timeout)


async def _await_provider_task[T](task: asyncio.Task[T], request_timeout: float | None) -> T:
    """Await a shielded provider task, bounded by ``request_timeout`` if set.

    The deadline is armed on the task, not on this await, so it still fires
    after the caller cancels. A caller cancel raises CancelledError, even when
    it lands with the deadline.
    """
    task.add_done_callback(_consume_async_provider_result)
    if request_timeout is None:
        return await asyncio.shield(task)

    timed_out = False

    def _on_deadline() -> None:
        nonlocal timed_out
        if not task.done():
            timed_out = True
            task.cancel()

    handle = asyncio.get_running_loop().call_later(request_timeout, _on_deadline)
    task.add_done_callback(lambda _done: handle.cancel())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if timed_out and (current is None or not current.cancelling()):
            raise TimeoutError from None
        raise


def _consume_async_provider_result(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except BaseException:
        pass


def _map_completion_finish_reason(
    raw_response: Any,
) -> Literal["stop", "tool_calls", "length", "error"]:
    """Map a Chat-Completions provider finish_reason onto LLMResponse.finish_reason.

    litellm/OpenAI report the provider's stop condition on
    ``raw_response.choices[0].finish_reason``. We surface ``"length"`` (output
    tokens exhausted) and ``"error"`` (e.g. ``content_filter``) so downstream
    logic (e.g. CodeAct's max-tokens abort) can react.
    """
    raw = None
    try:
        raw = raw_response.choices[0].finish_reason
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
    raw_response: Any,
) -> Literal["stop", "tool_calls", "length", "error"]:
    """Map a Responses-API response onto LLMResponse.finish_reason.

    The Responses API reports truncation via ``status == "incomplete"`` with
    ``incomplete_details.reason == "max_output_tokens"`` (rather than a
    per-choice finish_reason). A ``status == "failed"`` response is surfaced as
    ``"error"``.
    """
    status = getattr(raw_response, "status", None)

    if status == "incomplete":
        details = getattr(raw_response, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        if reason is None and isinstance(details, dict):
            reason = details.get("reason")
        if reason == "max_output_tokens":
            return "length"
        return "error"
    if status == "failed":
        return "error"
    return "stop"


def _finish_reason_for_tool_calls(
    provider_finish_reason: Literal["stop", "tool_calls", "length", "error"],
) -> Literal["tool_calls", "length", "error"]:
    """Keep provider failure/truncation authoritative over parsed tool calls."""
    if provider_finish_reason in ("length", "error"):
        return provider_finish_reason
    return "tool_calls"


def _extract_usage(raw_response: Any) -> LLMUsage | None:
    """Normalize provider usage plus LiteLLM's per-response cost metadata."""
    usage = LLMUsage.from_provider(getattr(raw_response, "usage", None))
    if usage is None:
        return None

    hidden = getattr(raw_response, "_hidden_params", None)
    response_cost = hidden.get("response_cost") if isinstance(hidden, Mapping) else None
    if response_cost is None:
        return usage
    if (
        isinstance(response_cost, bool)
        or not isinstance(response_cost, (int, float))
        or not math.isfinite(response_cost)
        or response_cost < 0
    ):
        logger.warning("Ignoring malformed LiteLLM response_cost metadata: %r", response_cost)
        return usage
    return usage.model_copy(update={"cost_usd": float(response_cost)})


def _extract_reasoning_and_usage(raw_response: Any) -> tuple[str | None, LLMUsage | None]:
    """Extract reasoning and normalized usage from a raw LLM response."""
    reasoning = None

    # Extract reasoning (o1-style or DeepSeek/QwQ)
    if hasattr(raw_response, "choices") and raw_response.choices:
        msg = raw_response.choices[0].message
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)

    return reasoning, _extract_usage(raw_response)


def _raise_if_empty_content(response: LLMResponse, enabled: bool) -> None:
    """Raise EmptyContentError for a reply with reasoning but no text, if enabled."""
    if enabled and not response.content and response.reasoning:
        raise EmptyContentError(response.reasoning)


def _item_field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


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

    Handles both complete tags and malformed tags (missing opening tag due to litellm bug).
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
    # (litellm bug strips opening <think> but leaves closing </think>)
    if "</think>" in content:
        parts = content.split("</think>", 1)
        if len(parts) == 2:
            reasoning = parts[0].strip()
            cleaned = parts[1].strip()
            _record_llm_metric("malformed_think_tag_fixed")
            return cleaned, reasoning

    return content, None


# ============================================================================
# PATCH: Prevent litellm from stripping cache_control for Anthropic models
# ============================================================================
# litellm's OpenAIGPTConfig.remove_cache_control_flag_from_messages_and_tools()
# unconditionally strips cache_control from all messages and tools before sending.
# For Anthropic models behind an OpenAI-compatible endpoint (e.g., NVIDIA gateway),
# we need cache_control to survive so the API can enable prompt caching.
_cache_control_patch_applied = False


def _apply_cache_control_preserve_patch():
    """Patch litellm to preserve cache_control for Anthropic models."""
    global _cache_control_patch_applied
    if _cache_control_patch_applied:
        return

    try:
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        _original_remove = OpenAIGPTConfig.remove_cache_control_flag_from_messages_and_tools

        def _patched_remove(self, model, messages, tools=None):
            if _is_anthropic_model(model):
                return messages, tools
            return _original_remove(self, model, messages, tools)

        OpenAIGPTConfig.remove_cache_control_flag_from_messages_and_tools = _patched_remove
        _cache_control_patch_applied = True
        logger.debug("Applied cache_control preserve patch for Anthropic models")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not apply cache_control preserve patch: {e}")


_apply_cache_control_preserve_patch()


# The output schema of the native-output call running in this context, read by
# the request hooks below. A ContextVar keeps concurrent calls separate.
_NATIVE_OUTPUT_CONTRACT: ContextVar[dict[str, Any] | None] = ContextVar(
    "nooa_native_output_contract", default=None
)


def _check_native_output_request(request):
    """httpx request hook: verify the outgoing body still has the expected schema.

    Raises ``ValueError`` (which stops the request) if
    ``output_config.format`` is missing or differs from the schema built for
    this call, or if another output format or tools were added. Requests
    outside a native-output call are not checked. This is one module-level
    function, so a shared httpx client gets a single hook that keeps no NOOA
    client alive; the expected schema comes from the ContextVar set by the
    active call's ``_native_output_request_scope``.
    """
    expected = _NATIVE_OUTPUT_CONTRACT.get()
    if expected is None:
        return
    wire = json.loads(request.content)
    output = wire.get("output_config")
    if (
        not isinstance(output, dict)
        or output.get("format") != expected
        or any(
            key in wire
            for key in (
                "response_format",
                "output_format",
                "text_format",
                "text.format",
                "tools",
                "tool_choice",
            )
        )
        or isinstance(wire.get("text"), dict)
        and "format" in wire["text"]
    ):
        raise ValueError(
            "Native output_config.format schema was changed or dropped before send; "
            "generic structured-output carriers are not permitted"
        )


async def _acheck_native_output_request(request):
    """Async version of ``_check_native_output_request`` for ``httpx.AsyncClient``."""
    _check_native_output_request(request)


class CompletionClient(UnifiedLLM):
    def _prepare_call_config(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Validate and remove a per-call ``structured_output_mode`` before the base logic.

        The mode is NOOA's own setting, so it must not reach the provider
        parameters; ``_native_output_for_call`` reads it from the overrides.
        """
        overrides = dict(overrides)
        if "structured_output_mode" in overrides:
            validate_structured_output_mode(overrides.pop("structured_output_mode"))
        return super()._prepare_call_config(overrides)

    def _with_reduced_reply_limit(self, overrides: dict[str, Any], limit: int) -> dict[str, Any]:
        params = super()._with_reduced_reply_limit(overrides, limit)
        # _prepare_call_config removed structured_output_mode because it is a
        # NOOA setting, not a provider parameter. Put an explicit per-call value
        # (including None) back so the retry with a smaller reply limit uses it.
        if "structured_output_mode" in overrides:
            params["structured_output_mode"] = validate_structured_output_mode(
                overrides["structured_output_mode"]
            )
        return params

    def _call_endpoint_changed(self, call_config) -> bool:
        """Whether this call sends to a different endpoint than the constructor's.

        ``api_base`` and ``base_url`` are two spellings of one endpoint (the
        same comparison as ``_request_media_inputs``); a call with no endpoint
        while one is configured goes to the provider default, a change.
        """
        from nooa.unifiedllm.route_identity import endpoint_changed

        return endpoint_changed(self.config, call_config) or (
            any(self.config.get(key) for key in ("api_base", "base_url"))
            and not any(call_config.get(key) for key in ("api_base", "base_url"))
        )

    def _native_output_for_call(self, overrides, call_config) -> bool:
        """Return whether this call sends its output schema as ``output_config.format``.

        The per-call ``structured_output_mode`` wins over the client's. The
        client's ``"anthropic"`` mode is not applied when the call overrides the
        model, base URL, provider or client: the endpoint is then different,
        and the call must set ``structured_output_mode`` itself (else
        ``ValueError``).
        """
        mode = validate_structured_output_mode(
            overrides.get("structured_output_mode", self.structured_output_mode)
        )
        route_changed = (
            self._effective_model(call_config) != self.model
            or self._call_endpoint_changed(call_config)
            or any(
                call_config.get(key) != self.config.get(key)
                for key in ("custom_llm_provider", "client")
            )
        )
        if mode == "anthropic" and route_changed and "structured_output_mode" not in overrides:
            raise ValueError(
                "Route override cannot inherit structured_output_mode='anthropic'; "
                "declare the replacement route's profile explicitly"
            )
        return mode == "anthropic"

    # The wire code lives in adapters/litellm_chat.py. These methods stay as the
    # hooks the adapter calls, so subclasses and tests can still replace them.

    @staticmethod
    def _normalize_chat_extensions(api_params, effective_model):
        """Move ``thinking`` and ``output_config`` into ``extra_body`` for OpenAI-SDK requests."""
        from .adapters import litellm_chat

        litellm_chat.normalize_chat_extensions(api_params, effective_model)

    @staticmethod
    def _set_output_format(api_params, output_model, effective_model, native_output):
        """Add the output schema of ``output_model`` to the request parameters."""
        from .adapters import litellm_chat

        litellm_chat.set_output_format(api_params, output_model, effective_model, native_output)

    def _native_output_request_scope(self, api_params, output_model, native_output, *, is_async):
        """Check the HTTP request LiteLLM sends during a native-output call."""
        from .adapters import litellm_chat

        return litellm_chat.native_output_request_scope(
            api_params, output_model, native_output, is_async=is_async
        )

    @staticmethod
    def _response_from_chat(raw_response, scope, tools, output_model, *, native_output=False):
        """Turn a Chat Completions reply into an LLMResponse (adapter hook)."""
        from .adapters import litellm_chat

        return litellm_chat.response_from_chat(
            raw_response, scope, tools, output_model, native_output=native_output
        )

    def __init__(
        self,
        model: str,
        retry_config: RetryConfig | None = None,
        http_config: HttpConfig | None = None,
        cache_breakpoint: Literal["auto", "anthropic"] | None = "auto",
        *,
        structured_output_mode: StructuredOutputMode = None,
        media_inputs: list[MediaInput] | tuple[MediaInput, ...] | None = None,
        context_window: int | None = None,
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
                         models return empty content. The OpenAI SDK's own
                         retries are disabled unless ``max_retries`` is passed
                         in ``**config``.
            http_config: Optional per-client HTTP connection-pool and timeout
                         settings. Applied only to THIS client's requests: the
                         client builds its own httpx client from these values and
                         passes it to litellm per call. No global state and no
                         monkey-patching of httpx — two clients with different
                         http_configs are fully independent.
            cache_breakpoint: Set to ``"anthropic"`` to map the cached
                renderer's stable-prefix boundary to native ``cache_control``.
                Default ``"auto"`` enables this for recognized Anthropic routes;
                other routes use provider-default caching. ``None`` disables
                NOOA markers. Without a boundary, only leading instructions
                are marked.
            structured_output_mode: Request field that carries the output
                schema of ``output_model`` calls. ``'anthropic'`` sends
                ``output_config.format``; ``None`` and ``'auto'`` send the
                default ``response_format``. Applies only to this client's own
                model and endpoint and is never chosen from the model name.
            media_inputs: Explicit allowed inputs: image, audio, video, pdf.
                Absent/empty is text-only. PDF means application/pdf only.
            context_window: Optional model context-window metadata used for local
                context budgeting. It is never sent to the inference API.
            **config: Additional configuration passed to litellm (api_key, api_base, etc.)
        """
        if cache_breakpoint not in {None, "auto", "anthropic"}:
            raise ValueError(
                "CompletionClient cache_breakpoint must be 'auto', 'anthropic', or None"
            )
        super().__init__(
            model,
            structured_output_mode=structured_output_mode,
            media_inputs=media_inputs,
            context_window=context_window,
            **config,
        )
        self.retry_config = retry_config or RetryConfig()
        self.cache_breakpoint = cache_breakpoint
        self._http_config = http_config or HttpConfig()
        self._http = _ClientHttp.for_completion(self.model, self.config, self._http_config)

        from .adapters.litellm_chat import LiteLLMChatAdapter

        self._adapter = LiteLLMChatAdapter(self)

    def _convert_tool_to_schema(self, tool: Tool) -> dict[str, Any]:
        """Convert Tool object to Completion API schema format"""
        from .adapters import litellm_chat

        return litellm_chat.tool_schema(tool)

    def _completion_http_client(
        self, call_config: dict[str, Any], *, is_async: bool, max_retries: int
    ) -> Any:
        """Reuse the owned transport only while its constructor routing still applies."""
        return self._adapter.http_client(call_config, is_async=is_async, max_retries=max_retries)

    def _call_once(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Sync version: Completion API uses standard message format, so no transformation needed.
        Messages are passed directly to the API.

        Each attempt is one ``adapter.send_sync``; retries, the empty-content
        check, debug tracking and the usage metric stay here.

        If retry_config.retry_on_empty_content is True, will retry when the model
        returns empty content but has reasoning_content (common with some reasoning models).
        """
        request, model = self._dispatch_request(messages, tools, output_model, kwargs, "messages")
        retry_on_empty = self.retry_config.retry_on_empty_content if self.retry_config else False

        def _make_call():
            """One attempt; an empty reply with reasoning may raise for a retry."""
            response = self._adapter.send_sync(request, **kwargs)
            _raise_if_empty_content(response, retry_on_empty)
            return response

        # Track LLM call for debugging (visible via SIGUSR2 if nooa debug handler installed)
        with _track_llm_call(model=model, endpoint=self.config.get("api_base")):
            response = (
                sync_retry(_make_call, config=self.retry_config)
                if self.retry_config
                else _make_call()
            )
        if response.usage:
            _record_llm_metric("token_usage", response.usage)
        return response

    async def _acall_once(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Async version: Completion API uses standard message format, so no transformation needed.
        Messages are passed directly to the API.

        Each attempt is one ``await adapter.send``; retries, the empty-content
        check, debug tracking and the usage metric stay here.

        If retry_config.retry_on_empty_content is True, will retry when the model
        returns empty content but has reasoning_content (common with some reasoning models).
        """
        request, model = self._dispatch_request(messages, tools, output_model, kwargs, "messages")
        retry_on_empty = self.retry_config.retry_on_empty_content if self.retry_config else False

        async def _make_call():
            """One attempt; an empty reply with reasoning may raise for a retry."""
            response = await self._adapter.send(request, **kwargs)
            _raise_if_empty_content(response, retry_on_empty)
            return response

        # Track LLM call for debugging (visible via SIGUSR2 if nooa debug handler installed)
        with _track_llm_call(model=model, endpoint=self.config.get("api_base")):
            response = (
                await with_retry(_make_call, config=self.retry_config)
                if self.retry_config
                else await _make_call()
            )
        if response.usage:
            _record_llm_metric("token_usage", response.usage)
        return response


class ReasoningCompletionClient(CompletionClient):
    """
    CompletionClient for reasoning models that output <think>...</think> tags.

    This client:
    1. Extracts reasoning from <think>...</think> tags in the content
    2. Handles litellm's bug where opening <think> tag is stripped
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
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Call with <think> tag extraction."""
        response = super().call(messages, tools, output_model, **kwargs)

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

                parsed = response.parsed
                response = response.replace_parts(
                    (
                        AssistantReasoning(text=combined_reasoning),
                        AssistantText(text=cleaned_content),
                        *response.tool_calls,
                    )
                )
                # Separating think tags does not change the validated answer.
                response.parsed = parsed

        return response

    async def acall(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Async call with <think> tag extraction."""
        response = await super().acall(messages, tools, output_model, **kwargs)

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

                parsed = response.parsed
                response = response.replace_parts(
                    (
                        AssistantReasoning(text=combined_reasoning),
                        AssistantText(text=cleaned_content),
                        *response.tool_calls,
                    )
                )
                # Separating think tags does not change the validated answer.
                response.parsed = parsed

        return response


class ResponsesClient(UnifiedLLM):
    def __init__(
        self,
        model: str,
        retry_config: RetryConfig | None = None,
        http_config: HttpConfig | None = None,
        cache_breakpoint: Literal["auto", "openai"] | None = "auto",
        *,
        media_inputs: list[MediaInput] | tuple[MediaInput, ...] | None = None,
        context_window: int | None = None,
        **config,
    ):
        """
        Initialize ResponsesClient.

        Mirrors CompletionClient so the Responses API path gets the same retry,
        HTTP, and cache-control behaviour. Accepting these as named parameters
        also keeps them out of ``self.config`` — otherwise they would leak into
        ``litellm.responses()`` as bogus API params.

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
                         own httpx client is passed to litellm per call). No
                         global state and no monkey-patching of httpx.
            cache_breakpoint: Default ``"auto"`` maps a rendered boundary with
                eligible stable input to a Responses explicit breakpoint.
                Without a usable boundary, leaves provider-default caching unchanged.
                ``None`` disables NOOA markers, not the provider's implicit cache.
                Anthropic cache mapping is supported by CompletionClient only.
                With ``"openai"`` and no eligible stable block, warns and keeps
                explicit mode without a breakpoint, avoiding all cache writes.
            media_inputs: Explicit allowed inputs: image, audio, video, pdf.
                Absent/empty is text-only. PDF means application/pdf only.
            context_window: Optional model context-window metadata used for local
                context budgeting. It is never sent to the inference API.
            **config: Additional configuration passed to litellm (api_key, api_base, etc.)
        """
        if cache_breakpoint not in {None, "auto", "openai"}:
            raise ValueError("ResponsesClient cache_breakpoint must be 'auto', 'openai', or None")
        super().__init__(model, media_inputs=media_inputs, context_window=context_window, **config)
        self.retry_config = retry_config or RetryConfig()
        self.cache_breakpoint = cache_breakpoint
        self._http_config = http_config or HttpConfig()
        self._http = _ClientHttp.for_responses(self.model, self.config, self._http_config)

        from .adapters.litellm_responses import LiteLLMResponsesAdapter

        self._adapter = LiteLLMResponsesAdapter(self)

    def _prepare_call_config(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge the call config as the base class does, spelling the cap max_output_tokens."""
        params = super()._prepare_call_config(overrides)
        caps = REPLY_CAP_KEYS & params.keys()
        if caps:
            params["max_output_tokens"] = params.pop(next(iter(caps)))
        return params

    # The wire code lives in adapters/litellm_responses.py. These methods stay as
    # the hooks the adapter calls, so subclasses and tests can still replace them.

    def _convert_tool_to_schema(self, tool: Tool) -> dict[str, Any]:
        """Convert Tool object to Responses API schema format."""
        from .adapters import litellm_responses

        return litellm_responses.tool_schema(tool)

    def _call_once(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Sync version: Call LLM and parse response.

        Accepts public message dictionaries and LLMResponse objects. Stored turns
        are projected here; only leading system messages become `instructions`.
        Each attempt is one ``adapter.send_sync``; retries and tracking stay here.
        """
        request, model = self._dispatch_request(messages, tools, output_model, kwargs, "input")

        def _make_call():
            """One attempt."""
            return self._adapter.send_sync(request, **kwargs)

        # Track LLM call for debugging (visible via SIGUSR2 if nooa debug handler installed)
        with _track_llm_call(model=model, endpoint=self.config.get("api_base")):
            return (
                sync_retry(_make_call, config=self.retry_config)
                if self.retry_config
                else _make_call()
            )

    async def _acall_once(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Async version: Call LLM and parse response.

        Accepts public message dictionaries and LLMResponse objects. Stored turns
        are projected here; only leading system messages become `instructions`.
        Each attempt is one ``await adapter.send``; retries and tracking stay here.
        """
        request, model = self._dispatch_request(messages, tools, output_model, kwargs, "input")

        async def _make_call():
            """One attempt."""
            return await self._adapter.send(request, **kwargs)

        # Track LLM call for debugging (visible via SIGUSR2 if nooa debug handler installed)
        with _track_llm_call(model=model, endpoint=self.config.get("api_base")):
            return (
                await with_retry(_make_call, config=self.retry_config)
                if self.retry_config
                else await _make_call()
            )

    def _prepare_input(
        self, messages, state_scope, model, native_encrypted_reasoning=False, *, media_inputs=None
    ):
        """Choose cache markers only after canonical turn projection."""
        input_messages, instructions = self._transform_messages(
            messages, state_scope, native_encrypted_reasoning, media_inputs=media_inputs
        )
        return self._prepare_cache_boundary(
            input_messages, responses=True, model=model, instructions=instructions
        )

    def _response_from_output(
        self, raw_response, scope, usage, output_model, native_encrypted_reasoning=False
    ):
        """Turn a Responses reply into an LLMResponse (adapter hook)."""
        from .adapters import litellm_responses

        return litellm_responses.response_from_output(
            raw_response, scope, usage, output_model, native_encrypted_reasoning
        )

    @staticmethod
    def _transform_content_blocks(blocks, text_kind="input_text"):
        """Project documented public slots, preserving request-owned metadata."""
        from .adapters import litellm_responses

        return litellm_responses.transform_content_blocks(blocks, text_kind)

    def _transform_messages(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        state_scope: str | None = None,
        native_encrypted_reasoning: bool = False,
        *,
        media_inputs=None,
    ) -> tuple[list[dict[str, Any] | CacheBoundary], str | None]:
        """Expand turns at dispatch; only leading system messages become instructions."""
        from .adapters import litellm_responses

        return litellm_responses.transform_messages(
            messages, state_scope, native_encrypted_reasoning, media_inputs=media_inputs
        )

    def _extract_text_from_output(self, response: Any) -> str:
        """The reply text of a Responses reply."""
        from .adapters import litellm_responses

        return litellm_responses.extract_text_from_output(response)
