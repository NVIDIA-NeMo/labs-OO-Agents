# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import inspect
import json
import logging
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal, cast

import litellm
from pydantic import BaseModel, RootModel

from .http_config import HttpConfig
from .retry import EmptyContentError, sync_retry, with_retry
from .retry_config import RetryConfig

# Private helpers and factory are imported with explicit aliases so Ruff treats
# them as intentional compatibility re-exports from this historical module.
from .types import (
    LLMResponse,
    Tool,
    ToolCall,
    _resolve_schema_refs,
    _strict_schema_valid,
)
from .types import (
    _clean_schema as _clean_schema,
)
from .types import (
    _strip_schema_noise as _strip_schema_noise,
)
from .types import (
    create_tool_from_callable as create_tool_from_callable,
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
                api_base=config.get("api_base"),
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
        api_base = config.get("api_base") or dynamic_api_base
        common: dict[str, Any] = {"timeout": self._timeout}
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


def _update_token_calibration(
    model: str,
    messages: list[dict[str, Any]],
    usage: dict[str, int],
    tools: list[dict[str, Any]] | None = None,
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
    actual = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    if actual <= 0:
        return
    # Calibration is best-effort: it must NEVER raise out of the (already paid)
    # response path. The whole estimate — primary AND fallback — is guarded.
    try:
        try:
            estimated = litellm.token_counter(model=model, messages=messages)
            if tools:
                # Count the full messages+tools payload the way the API bills it,
                # then take the larger of the bare and with-tools counts
                # (with_tools is normally >= bare; max only guards a tokenizer
                # that returns less with tools attached).
                with_tools = litellm.token_counter(
                    model=model, messages=messages, tools=cast(Any, tools)
                )
                estimated = max(estimated, with_tools)
        except Exception:
            # token_counter can reject some message/tool shapes; fall back to the
            # per-message text sum rather than skip calibration entirely.
            estimated = 0
            for msg in messages:
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

    def __init__(self, model: str, **config):
        self.model = model
        self.config = config
        self._registry_config = None
        # Cache control injection — shared by CompletionClient and ResponsesClient
        self.cache_control_injection_points: list[dict[str, Any]] = (
            DEFAULT_CACHE_CONTROL_INJECTION_POINTS
        )
        # Per-client HTTP transport (httpx clients + litellm wrappers). Set by
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
        1. Explicit ``context_window`` config passed to the client constructor
        2. Registry config (if created via get_llm_client())
        3. Registry lookup by model name or model_name field
        4. litellm model info (for known models)
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

        # Fallback to litellm
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


async def _litellm_acompletion(api_params: dict[str, Any]) -> Any:
    """Await LiteLLM without cancelling its nested provider coroutine.

    LiteLLM runs sync ``completion()`` in an executor for async chat calls.
    OpenAI-compatible providers return ``OpenAIChatCompletion.acompletion``
    from that sync frame, then LiteLLM awaits it on the event loop. If a TUI
    soft-cancel lands in that handoff window, Python can garbage-collect the
    provider coroutine before it is awaited and print::

        RuntimeWarning: coroutine 'OpenAIChatCompletion.acompletion' was never awaited

    Shielding lets LiteLLM finish consuming that provider coroutine while the
    caller still receives ``CancelledError`` immediately.
    """
    task = asyncio.create_task(litellm.acompletion(**api_params))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.add_done_callback(_consume_litellm_acompletion_result)
        raise


def _consume_litellm_acompletion_result(task: asyncio.Task[Any]) -> None:
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
    logic (e.g. CodeAct's max-tokens abort) can react. Callers that have already
    detected tool calls should keep ``finish_reason="tool_calls"`` rather than
    calling this.
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
    ``"error"``. Callers that have already detected tool calls should keep
    ``finish_reason="tool_calls"`` rather than calling this.
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


def _extract_reasoning_and_usage(raw_response: Any) -> tuple[str | None, dict[str, int] | None]:
    """Extract reasoning and usage from raw LLM response."""
    reasoning = None
    usage = None

    # Extract reasoning (o1-style or DeepSeek/QwQ)
    if hasattr(raw_response, "choices") and raw_response.choices:
        msg = raw_response.choices[0].message
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)

    # Extract usage
    if hasattr(raw_response, "usage") and raw_response.usage:
        usage_obj = raw_response.usage
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


DEFAULT_CACHE_CONTROL_INJECTION_POINTS = [
    {"role": "system"},
    {"role": "tool", "position": "last"},
]


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
                         passes it to litellm per call. No global state and no
                         monkey-patching of httpx — two clients with different
                         http_configs are fully independent.
            cache_control_injection_points: Optional list of role/position rules to
                enable prompt caching (for example: {"role": "system"} or
                {"role": "tool", "position": "last"}). Applied to all calls.
                Note: Do NOT manually add cache_control to message content when using this.
            **config: Additional configuration passed to litellm (api_key, api_base, etc.)
        """
        super().__init__(model, **config)
        self.retry_config = retry_config or RetryConfig()
        self._http_config = http_config or HttpConfig()
        self._http = _ClientHttp.for_completion(self.model, self.config, self._http_config)
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
            "model": self.model,
            "messages": prepared_messages,
            **self.config,
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
        if http_client.sync_client is not None:
            api_params.setdefault("client", http_client.sync_client)

        def _make_call():
            raw_response = _collect_sync(litellm.completion(**api_params))
            reasoning, _ = _extract_reasoning_and_usage(raw_response)
            text_content = raw_response.choices[0].message.content or ""  # type: ignore[union-attr]

            # Raise EmptyContentError to trigger retry if configured
            if not text_content and reasoning and retry_on_empty:
                raise EmptyContentError(reasoning)

            return raw_response

        # Track LLM call for debugging (visible via SIGUSR2 if nooa debug handler installed)
        with _track_llm_call(model=self.model, endpoint=self.config.get("api_base")):
            raw_response = (
                sync_retry(_make_call, config=self.retry_config)
                if self.retry_config
                else _make_call()
            )

        reasoning, usage = _extract_reasoning_and_usage(raw_response)
        if usage:
            _record_llm_metric("token_usage", usage)
            _update_token_calibration(
                self.model, prepared_messages, usage, tools=api_params.get("tools")
            )
        raw_tool_calls = raw_response.choices[0].message.tool_calls  # type: ignore[union-attr]

        if raw_tool_calls:
            tool_calls = [
                ToolCall(id=tc.id, name=tc.function.name or "", arguments=tc.function.arguments)
                for tc in raw_tool_calls
            ]

            return LLMResponse(
                raw_response=raw_response,
                content="",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                assistant_message={
                    "role": "assistant",
                    "content": raw_response.choices[0].message.content or "",  # type: ignore[union-attr]
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
                usage=usage,
            )

        text_content = raw_response.choices[0].message.content or ""  # type: ignore[union-attr]

        # Fallback: vLLM's hermes parser fails on Nemotron's XML tool call format.
        # Extract <tool_call><function=name><parameter=...> from content.
        if not raw_tool_calls and tools and "<tool_call>" in text_content:
            xml_tool_calls = _extract_xml_tool_calls(text_content)
            if xml_tool_calls:
                return LLMResponse(
                    raw_response=raw_response,
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
                    usage=usage,
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
                raw_response=raw_response,
                content=parsed_content,
                tool_calls=[],
                finish_reason=_map_completion_finish_reason(raw_response),
                assistant_message={"role": "assistant", "content": text_content},
                reasoning=reasoning if text_content else None,
                usage=usage,
            )

        return LLMResponse(
            raw_response=raw_response,
            content=text_content,
            tool_calls=[],
            finish_reason=_map_completion_finish_reason(raw_response),
            assistant_message={"role": "assistant", "content": text_content},
            reasoning=reasoning,
            usage=usage,
        )

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
            "model": self.model,
            "messages": prepared_messages,
            **self.config,
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
        if http_client.async_client is not None:
            api_params.setdefault("client", http_client.async_client)

        async def _make_call():
            raw_response = await _collect_async(await _litellm_acompletion(api_params))
            reasoning, _ = _extract_reasoning_and_usage(raw_response)
            text_content = raw_response.choices[0].message.content or ""  # type: ignore[union-attr]

            # Raise EmptyContentError to trigger retry if configured
            if not text_content and reasoning and retry_on_empty:
                raise EmptyContentError(reasoning)

            return raw_response

        # Track LLM call for debugging (visible via SIGUSR2 if nooa debug handler installed)
        with _track_llm_call(model=self.model, endpoint=self.config.get("api_base")):
            raw_response = (
                await with_retry(_make_call, config=self.retry_config)
                if self.retry_config
                else await _make_call()
            )

        reasoning, usage = _extract_reasoning_and_usage(raw_response)
        if usage:
            _record_llm_metric("token_usage", usage)
            _update_token_calibration(
                self.model, prepared_messages, usage, tools=api_params.get("tools")
            )
        raw_tool_calls = raw_response.choices[0].message.tool_calls  # type: ignore[union-attr]

        if raw_tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id, name=tc.function.name or "", arguments=tc.function.arguments or ""
                )  # type: ignore[union-attr]
                for tc in raw_tool_calls
            ]

            return LLMResponse(
                raw_response=raw_response,
                content="",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                assistant_message={
                    "role": "assistant",
                    "content": raw_response.choices[0].message.content or "",  # type: ignore[union-attr]
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
                usage=usage,
            )

        text_content = raw_response.choices[0].message.content or ""  # type: ignore[union-attr]

        # Fallback: vLLM's hermes parser fails on Nemotron's XML tool call format.
        # Extract <tool_call><function=name><parameter=...> from content.
        if not raw_tool_calls and tools and "<tool_call>" in text_content:
            xml_tool_calls = _extract_xml_tool_calls(text_content)
            if xml_tool_calls:
                return LLMResponse(
                    raw_response=raw_response,
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
                    usage=usage,
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
                raw_response=raw_response,
                content=parsed_content,
                tool_calls=[],
                finish_reason=_map_completion_finish_reason(raw_response),
                assistant_message={"role": "assistant", "content": text_content},
                reasoning=reasoning if text_content else None,
                usage=usage,
            )

        return LLMResponse(
            raw_response=raw_response,
            content=text_content,
            tool_calls=[],
            finish_reason=_map_completion_finish_reason(raw_response),
            assistant_message={"role": "assistant", "content": text_content},
            reasoning=reasoning,
            usage=usage,
        )


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
                    raw_response=response.raw_response,
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
                    raw_response=response.raw_response,
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
            cache_control_injection_points: Optional list of role/position rules to
                enable prompt caching (for example: {"role": "system"} or
                {"role": "tool", "position": "last"}). Applied to all calls.
            **config: Additional configuration passed to litellm (api_key, api_base, etc.)
        """
        super().__init__(model, **config)
        self.retry_config = retry_config or RetryConfig()
        self._http_config = http_config or HttpConfig()
        self._http = _ClientHttp.for_responses(self.model, self.config, self._http_config)
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
        # Inject cache_control only for Anthropic-served models. litellm.responses
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
            "model": self.model,
            "input": input_messages,
            "truncation": "disabled",
            **self.config,
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
        if http_client.sync_client is not None:
            api_params.setdefault("client", http_client.sync_client)

        def _make_call():
            return cast("litellm.ResponsesAPIResponse", litellm.responses(**api_params))

        # Track LLM call for debugging (visible via SIGUSR2 if nooa debug handler installed)
        with _track_llm_call(model=self.model, endpoint=self.config.get("api_base")):
            raw_response = (
                sync_retry(_make_call, config=self.retry_config)
                if self.retry_config
                else _make_call()
            )

        # Extract usage if available (Responses API may have different structure)
        usage = None
        if hasattr(raw_response, "usage") and raw_response.usage:
            usage_obj = raw_response.usage
            if hasattr(usage_obj, "model_dump"):
                usage = usage_obj.model_dump()
            elif isinstance(usage_obj, dict):
                usage = usage_obj
        if usage:
            _update_token_calibration(self.model, messages, usage, tools=api_params.get("tools"))

        output: list[Any] = raw_response.output  # type: ignore[assignment]
        raw_tool_calls = [item for item in output if item.type == "function_call"]

        if raw_tool_calls:
            tool_calls = [
                ToolCall(id=tc.call_id or "", name=tc.name or "", arguments=tc.arguments or "")
                for tc in raw_tool_calls
            ]

            assistant_messages = []
            for item in output:
                if hasattr(item, "model_dump"):
                    assistant_messages.append(item.model_dump())
                else:
                    assistant_messages.append(item)

            return LLMResponse(
                raw_response=raw_response,
                content="",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                assistant_message={"_batch": assistant_messages},
                reasoning=None,  # Responses API doesn't have reasoning
                usage=usage,
            )

        text_content = self._extract_text_from_output(raw_response)

        if output_model:
            json_data = extract_and_parse_json(text_content)
            parsed_content = _instantiate_output_model(output_model, json_data)

            return LLMResponse(
                raw_response=raw_response,
                content=parsed_content,
                tool_calls=[],
                finish_reason=_map_responses_finish_reason(raw_response),
                assistant_message={"role": "assistant", "content": text_content},
                reasoning=None,
                usage=usage,
            )

        return LLMResponse(
            raw_response=raw_response,
            content=text_content,
            tool_calls=[],
            finish_reason=_map_responses_finish_reason(raw_response),
            assistant_message={"role": "assistant", "content": text_content},
            reasoning=None,
            usage=usage,
        )

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
            "model": self.model,
            "input": input_messages,
            "truncation": "disabled",
            **self.config,
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
        if http_client.async_client is not None:
            api_params.setdefault("client", http_client.async_client)

        async def _make_call():
            return cast("litellm.ResponsesAPIResponse", await litellm.aresponses(**api_params))

        # Track LLM call for debugging (visible via SIGUSR2 if nooa debug handler installed)
        with _track_llm_call(model=self.model, endpoint=self.config.get("api_base")):
            raw_response = (
                await with_retry(_make_call, config=self.retry_config)
                if self.retry_config
                else await _make_call()
            )

        # Extract usage if available (Responses API may have different structure)
        usage = None
        if hasattr(raw_response, "usage") and raw_response.usage:
            usage_obj = raw_response.usage
            if hasattr(usage_obj, "model_dump"):
                usage = usage_obj.model_dump()
            elif isinstance(usage_obj, dict):
                usage = usage_obj
        if usage:
            _update_token_calibration(self.model, messages, usage, tools=api_params.get("tools"))

        output: list[Any] = raw_response.output  # type: ignore[assignment]
        raw_tool_calls = [item for item in output if item.type == "function_call"]

        if raw_tool_calls:
            tool_calls = [
                ToolCall(id=tc.call_id or "", name=tc.name or "", arguments=tc.arguments or "")
                for tc in raw_tool_calls
            ]

            assistant_messages = []
            for item in output:
                if hasattr(item, "model_dump"):
                    assistant_messages.append(item.model_dump())
                else:
                    assistant_messages.append(item)

            return LLMResponse(
                raw_response=raw_response,
                content="",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                assistant_message={"_batch": assistant_messages},
                reasoning=None,
                usage=usage,
            )

        text_content = self._extract_text_from_output(raw_response)

        if output_model:
            json_data = extract_and_parse_json(text_content)
            parsed_content = _instantiate_output_model(output_model, json_data)

            return LLMResponse(
                raw_response=raw_response,
                content=parsed_content,
                tool_calls=[],
                finish_reason=_map_responses_finish_reason(raw_response),
                assistant_message={"role": "assistant", "content": text_content},
                reasoning=None,
                usage=usage,
            )

        return LLMResponse(
            raw_response=raw_response,
            content=text_content,
            tool_calls=[],
            finish_reason=_map_responses_finish_reason(raw_response),
            assistant_message={"role": "assistant", "content": text_content},
            reasoning=None,
            usage=usage,
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
