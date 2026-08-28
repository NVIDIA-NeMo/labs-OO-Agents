# SPDX-License-Identifier: Apache-2.0
"""Private AnyLLM transport adapter. Provider objects never cross this module."""

from __future__ import annotations

from typing import Any

from any_llm import (
    AnyLLM,
    AuthenticationError,
    ContextLengthExceededError,
    InvalidRequestError,
    LLMProvider,
    ModelNotFoundError,
    RateLimitError,
    UnsupportedParameterError,
)


def close_async_client_sync(client: Any) -> None:
    """Close an owned async client on AnyLLM's sync runner loop."""
    from any_llm.utils.aio import run_async_in_sync

    run_async_in_sync(client.aclose(), allow_running_loop=False)


class LLMError(Exception):
    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.provider, self.status_code, self.code = provider, status_code, code


class LLMAuthenticationError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    retry_after: str | None = None


class LLMInvalidRequestError(LLMError):
    pass


class LLMModelNotFoundError(LLMError):
    pass


class LLMContextLengthError(LLMInvalidRequestError):
    pass


class LLMTransportError(LLMError):
    pass


class LLMProviderError(LLMError):
    pass


class LLMUnsupportedError(LLMInvalidRequestError):
    pass


class LLMContentFilterError(LLMInvalidRequestError):
    pass


class LLMInsufficientFundsError(LLMError):
    pass


class LLMUpstreamError(LLMProviderError):
    pass


class LLMGatewayTimeoutError(LLMTransportError):
    pass


def _error_code(exc: Exception) -> str | None:
    """Extract SDK error codes without depending on a particular SDK class."""
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            code = code or error.get("code") or error.get("type")
    response = getattr(exc, "response", None)
    try:
        data = response.json() if response is not None else None
    except Exception:
        data = None
    if isinstance(data, dict):
        error = data.get("error", data)
        if isinstance(error, dict):
            code = code or error.get("code") or error.get("type")
    return str(code) if code is not None else None


def _normalize_error(exc: Exception, provider: str) -> LLMError:
    """Normalize AnyLLM and raw SDK/httpx failures locally."""
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    status = status or getattr(response, "status_code", None)
    code = _error_code(exc)
    normalized = (code or "").lower()
    name = type(exc).__name__.lower()
    kw = {"provider": provider, "status_code": status, "code": code}
    context_codes = {
        "context_length_exceeded",
        "context_window_exceeded",
        "prompt_too_long",
        "max_tokens_exceeded",
    }
    content_codes = {"content_filter", "content_policy_violation", "safety_violation"}
    funds_codes = {"insufficient_quota", "insufficient_funds", "billing_hard_limit_reached"}
    unsupported_codes = {"unsupported_parameter", "unsupported_value", "not_supported"}
    if isinstance(exc, ContextLengthExceededError) or normalized in context_codes:
        cls = LLMContextLengthError
    elif normalized in content_codes:
        cls = LLMContentFilterError
    elif normalized in funds_codes or status == 402:
        cls = LLMInsufficientFundsError
    elif isinstance(exc, UnsupportedParameterError) or normalized in unsupported_codes:
        cls = LLMUnsupportedError
    elif isinstance(exc, AuthenticationError) or status in (401, 403):
        cls = LLMAuthenticationError
    elif isinstance(exc, RateLimitError) or status == 429:
        cls = LLMRateLimitError
    elif isinstance(exc, ModelNotFoundError) or status == 404:
        cls = LLMModelNotFoundError
    elif status in (408, 504) or "timeout" in name:
        cls = LLMGatewayTimeoutError
    elif status in (502, 503):
        cls = LLMUpstreamError
    elif isinstance(exc, (InvalidRequestError,)) or status == 400:
        cls = LLMInvalidRequestError
    elif (
        isinstance(exc, (ConnectionError, TimeoutError))
        or "connection" in name
        or "connect" in name
    ):
        cls = LLMTransportError
    else:
        cls = LLMProviderError
    error = cls(str(exc), **kw)
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is None and response is not None:
        retry_after = getattr(response, "headers", {}).get("retry-after")
    if isinstance(error, LLMRateLimitError):
        error.retry_after = str(retry_after) if retry_after is not None else None
    return error


def _provider_from_model(model: str, config: dict[str, Any]) -> tuple[str, str]:
    """Resolve explicit and one-release legacy model routing.

    New code should use ``provider:model`` (or the explicit ``provider=``
    argument).  Slash-prefixed LiteLLM names remain accepted where their
    prefix is unambiguous.  A small compatibility inference table preserves
    the historically documented bare model names without sending them to the
    wrong provider.
    """
    explicit = config.pop("provider", None)
    if explicit:
        return str(explicit), model
    aliases = {"nvidia_nim": "openai-compatible", "openai-compatible": "openai-compatible"}
    known = {p.value for p in LLMProvider}
    for separator in (":", "/"):
        if separator in model:
            prefix, remainder = model.split(separator, 1)
            if prefix in aliases or prefix in known:
                return aliases.get(prefix, prefix), remainder
    lowered = model.lower()
    inferred = (
        ("claude", "anthropic"),
        ("gemini", "gemini"),
        ("grok", "xai"),
        ("command", "cohere"),
        ("mistral", "mistral"),
    )
    for prefix, provider in inferred:
        if lowered.startswith(prefix):
            return provider, model
    return "openai", model


class AnyLLMTransport:
    """Lazily constructed provider transport with normalized exceptions."""

    def __init__(self, model: str, config: dict[str, Any], http_client: Any = None):
        cfg = dict(config)
        self.provider_name, self.model = _provider_from_model(model, cfg)
        self.api_key = cfg.get("api_key")
        self.api_base = cfg.get("endpoint") or cfg.get("api_base") or cfg.get("base_url")
        # Provider options are request-scoped by contract.  The owned HTTP
        # client is the sole extra provider-constructor option.
        self.provider_options = dict(cfg.get("provider_options") or {})
        reserved_request_options = {
            "model",
            "messages",
            "input",
            "instructions",
            "tools",
            "response_format",
            "stream",
            "temperature",
            "top_p",
            "max_tokens",
            "max_output_tokens",
            "stop",
            "reasoning_effort",
            "prompt_cache_key",
            "timeout",
            "tool_choice",
            "parallel_tool_calls",
        }
        collisions = reserved_request_options.intersection(self.provider_options)
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"provider_options collides with normalized request key(s): {names}")
        self.http_client = http_client
        self.responses_api = bool(cfg.get("_responses_api", False))
        self._provider: Any = None

    @property
    def provider(self) -> Any:
        if self._provider is None:
            constructor_options = {}
            if self.http_client is not None:
                constructor_options["http_client"] = self.http_client
            if self.provider_name == "openai-compatible":
                if not self.api_base:
                    raise ValueError("openai-compatible provider requires api_base/endpoint")
                if self.responses_api:
                    # AnyLLM's generic compatible provider only declares chat
                    # completions.  Its OpenAI provider supports /responses and
                    # accepts a caller-supplied base URL.
                    self._provider = AnyLLM.create(
                        "openai",
                        api_key=self.api_key,
                        api_base=self.api_base,
                        **constructor_options,
                    )
                else:
                    self._provider = AnyLLM.create_openai_compatible(
                        "nooa", self.api_base, self.api_key, **constructor_options
                    )
            else:
                self._provider = AnyLLM.create(
                    self.provider_name,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    **constructor_options,
                )
        return self._provider

    def _request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        collisions = self.provider_options.keys() & kwargs.keys()
        if collisions:
            names = ", ".join(sorted(collisions))
            raise TypeError(f"provider_options collides with request option(s): {names}")
        return {**kwargs, **self.provider_options}

    def embedding(self, inputs: str | list[str], **kwargs: Any) -> Any:
        """Create embeddings through the configured AnyLLM provider."""
        try:
            return self.provider._embedding(
                model=self.model, inputs=inputs, **self._request_kwargs(kwargs)
            )
        except Exception as exc:
            raise _normalize_error(exc, self.provider_name) from exc

    def completion(self, **kwargs: Any) -> Any:
        try:
            return self.provider.completion(model=self.model, **self._request_kwargs(kwargs))
        except Exception as exc:
            raise _normalize_error(exc, self.provider_name) from exc

    async def acompletion(self, **kwargs: Any) -> Any:
        try:
            return await self.provider.acompletion(model=self.model, **self._request_kwargs(kwargs))
        except Exception as exc:
            raise _normalize_error(exc, self.provider_name) from exc

    def responses(self, **kwargs: Any) -> Any:
        try:
            return self.provider.responses(model=self.model, **self._request_kwargs(kwargs))
        except Exception as exc:
            raise _normalize_error(exc, self.provider_name) from exc

    async def aresponses(self, **kwargs: Any) -> Any:
        try:
            return await self.provider.aresponses(model=self.model, **self._request_kwargs(kwargs))
        except Exception as exc:
            raise _normalize_error(exc, self.provider_name) from exc


def embedding(model: str, inputs: str | list[str], **config: Any) -> list[list[float]]:
    """Return provider-neutral embedding vectors using the AnyLLM adapter."""
    request_keys = {"dimensions", "timeout", "num_retries"}
    request = {key: config.pop(key) for key in tuple(config) if key in request_keys}
    response = AnyLLMTransport(model, config).embedding(inputs, **request)
    data = response.get("data", []) if isinstance(response, dict) else getattr(response, "data", [])
    return [
        list(item.get("embedding", [])) if isinstance(item, dict) else list(item.embedding)
        for item in data
    ]
