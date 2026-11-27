# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LiteLLM Chat Completions adapter: request building, the call and reply parsing.

The code here was moved out of ``CompletionClient`` without changing the wire
body. The client owns retries, metrics and debug tracking and calls only
``send`` / ``send_sync``, one attempt each. ``prepare``, ``transmit`` and
``parse`` are this adapter's own steps; no client calls them.
"""

import copy
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import litellm

from nooa._provider_ownership import _owned_provider_region
from nooa.llm_types import AssistantReasoning, LLMResponse
from nooa.unifiedllm import replay_state
from nooa.unifiedllm._observations import _ObservationOwner, _observe_attempt
from nooa.unifiedllm.adapters._numeric_usage import _numeric_usage
from nooa.unifiedllm.adapters._provider_logging import _prepare_owned_litellm_call
from nooa.unifiedllm.adapters.base import Capabilities
from nooa.unifiedllm.cache_policy import add_session_affinity_header
from nooa.unifiedllm.capability_settings import json_object_instruction, map_instruction_roles
from nooa.unifiedllm.errors import _parse_reply
from nooa.unifiedllm.media_inputs import filter_media_messages
from nooa.unifiedllm.request import LLMRequest
from nooa.unifiedllm.structured_output import anthropic_output_format
from nooa.unifiedllm.unifiedllm import (
    _DUMMY_TOOL_SCHEMA,
    _NATIVE_OUTPUT_CONTRACT,
    Tool,
    _acheck_native_output_request,
    _check_native_output_request,
    _collect_async,
    _collect_sync,
    _estimate_projected_input_tokens,
    _estimate_raw_input_tokens,
    _extract_reasoning_and_usage,
    _extract_xml_tool_calls,
    _finish_reason_for_tool_calls,
    _instantiate_output_model,
    _map_completion_finish_reason,
    _maybe_sanitize_response_format,
    _messages_have_tool_calls,
    _needs_dummy_tool,
    _record_llm_metric,
    _run_async_provider_call,
    _update_token_calibration,
    _with_sdk_retries,
    extract_and_parse_json,
)

from ._failures import failure_boundary

if TYPE_CHECKING:
    from nooa.unifiedllm.unifiedllm import CompletionClient, UnifiedLLM


def client_capabilities(client: "UnifiedLLM") -> Capabilities:
    """Capabilities as the client declares them today."""
    return Capabilities(
        media_kinds=frozenset(client.media_inputs),
        reasoning_levels=client.reasoning_levels,
        tools=True,
        context_window=client.context_window,
        max_output_tokens=None,
        developer_role=getattr(client, "developer_role", None),
        media_capabilities=_client_media_capabilities(client),
    )


def _client_media_capabilities(client: Any):
    """The client's declared media capabilities; None for foreign clients."""
    declared = getattr(client, "_request_media_capabilities", None)
    return declared() if callable(declared) and client.media_inputs else None


def _call_output_format(client: Any, call_config: dict[str, Any]) -> str:
    """The client's typed-output format for this call; json_schema for foreign clients."""
    chooser = getattr(client, "_call_output_format", None)
    return chooser(call_config) if callable(chooser) else "json_schema"


def _call_instruction_roles(client: Any, call_config: dict[str, Any]) -> tuple[str, bool | None]:
    """The client's instruction role for this call; unchanged roles for foreign clients."""
    chooser = getattr(client, "_call_instruction_roles", None)
    return chooser(call_config) if callable(chooser) else ("system", None)


def request_options(request: LLMRequest, options: Mapping[str, Any]) -> dict[str, Any]:
    """Translate content only; legacy views retain their full private options."""
    return {**options, "tool_choice": request.tool_choice}


# Tool choices that are not tool names; any other str names a tool.
RESERVED_TOOL_CHOICES = frozenset({"auto", "required", "none"})


def named_tool_choice(choice: Any, tools: list[Tool] | None) -> str | None:
    """The tool a str ``tool_choice`` names, or None for reserved values and mappings.

    Raises:
        ValueError: If the name is not one of ``tools``.
    """
    if not isinstance(choice, str) or choice in RESERVED_TOOL_CHOICES:
        return None
    names = [tool.name for tool in tools or ()]
    if choice not in names:
        raise ValueError(
            f"tool_choice {choice!r} names no tool of this request; "
            f"tools: {', '.join(names) or 'none'}"
        )
    return choice


@dataclass
class PreparedChatCall:
    """Everything one Chat Completions attempt needs, built once per call."""

    api_params: dict[str, Any]
    effective_model: str
    state_scope: str | None
    prepared_messages: list[Any]
    tools: list[Tool] | None
    output_model: Any
    native_output: bool


def tool_schema(tool: Tool) -> dict[str, Any]:
    """Convert a Tool to the Chat Completions schema."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.get_parameter_schema(),
        },
    }


_KEEP_DEVELOPER_ROLE: ContextVar[bool] = ContextVar("keep_developer_role", default=False)
_install_lock = threading.Lock()


def _install_developer_role_guard() -> None:
    """Let ``keep_developer_role()`` skip LiteLLM's developer-to-system rewrite.

    Wraps LiteLLM's base ``translate_developer_role_to_system_role`` once per
    process. The wrapper calls the original unless the current context is
    inside ``keep_developer_role()``, so requests outside the check behave
    exactly as before. LiteLLM runs sync completion work in a copy of the
    caller's context, so the flag reaches it.
    """
    from litellm.llms.base_llm.chat.transformation import BaseConfig

    with _install_lock:
        original = BaseConfig.translate_developer_role_to_system_role
        if getattr(original, "_connect_keeps_developer_role", False):
            return

        def translate_developer_role_to_system_role(self, messages):
            if _KEEP_DEVELOPER_ROLE.get():
                return messages
            return original(self, messages)

        translate_developer_role_to_system_role._connect_keeps_developer_role = True  # type: ignore[attr-defined]
        BaseConfig.translate_developer_role_to_system_role = (  # type: ignore[method-assign]
            translate_developer_role_to_system_role
        )


@contextmanager
def keep_developer_role():
    """Send ``developer`` messages unchanged in this context.

    LiteLLM Chat Completions rewrites ``developer`` to ``system`` for
    OpenAI-compatible models other than o-series
    (``translate_developer_role_to_system_role``). Connect's developer-role
    check uses this so its request really carries the role; nothing else
    does, so normal requests keep LiteLLM's behavior.
    """
    _install_developer_role_guard()
    token = _KEEP_DEVELOPER_ROLE.set(True)
    try:
        yield
    finally:
        _KEEP_DEVELOPER_ROLE.reset(token)


def normalize_chat_extensions(api_params, effective_model):
    """Move ``thinking`` and ``output_config`` into ``extra_body`` for OpenAI-SDK requests.

    LiteLLM sends OpenAI and OpenAI-compatible providers through the OpenAI
    Python SDK, which does not accept these fields as keyword arguments but
    copies ``extra_body`` into the request JSON. Providers that LiteLLM
    handles with its own HTTP code (such as Anthropic's Messages API) read
    these fields as top-level parameters, so they are left unchanged.

    This applies to any model and does not depend on ``output_model``. The
    provider is resolved the same way LiteLLM resolves it, because a call
    that overrides the endpoint may not use the SDK client NOOA created.

    Raises:
        ValueError: If a field is set both top-level and in ``extra_body``
            with different values.
    """
    fields = ("thinking", "output_config")
    if not any(key in api_params for key in fields):
        return
    _, provider, _, _ = litellm.get_llm_provider(
        model=effective_model,
        custom_llm_provider=api_params.get("custom_llm_provider"),
        api_base=api_params.get("api_base") or api_params.get("base_url"),
    )
    if provider != "openai" and provider not in getattr(litellm, "openai_compatible_providers", []):
        return
    # These OpenAI-compatible providers have their own LiteLLM HTTP
    # handlers and do not go through the OpenAI SDK, so leave them alone.
    if (
        provider
        in {
            "deepseek",
            "azure_ai",
            "fireworks_ai",
            "ragflow",
            "xai",
            "groq",
            "cometapi",
            "hosted_vllm",
        }
        or api_params.get("text_completion") is True
    ):
        return
    from litellm.secret_managers.main import get_secret_bool

    # With this LiteLLM setting, OpenAI requests use LiteLLM's own HTTP
    # handler instead of the SDK; check it the same way LiteLLM does.
    if get_secret_bool("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER"):
        return
    extra = copy.deepcopy(dict(api_params.get("extra_body") or {}))
    # Check every field for conflicts before moving any of them. Each
    # field is moved as a whole; a conflicting extra_body value is an
    # error rather than being merged or overwritten.
    for key in fields:
        if key in api_params and key in extra and api_params[key] != extra[key]:
            raise ValueError(f"Conflicting {key} settings in extra_body")
    for key in fields:
        if key in api_params:
            extra[key] = copy.deepcopy(api_params.pop(key))
    api_params["extra_body"] = extra


def set_output_format(api_params, output_model, effective_model, native_output):
    """Add the output schema of ``output_model`` to the request parameters.

    Without native output this sets ``response_format``. With native
    output (``structured_output_mode="anthropic"``) it sets
    ``output_config.format``, merged with any other ``output_config``
    settings, either as a top-level parameter (Anthropic provider) or in
    ``extra_body`` (OpenAI-compatible gateways).

    Raises:
        ValueError: If native output is combined with another output
            format, tools or a conflicting ``output_config``, or the
            provider is neither Anthropic nor OpenAI-compatible.
    """
    if output_model is None:
        return
    if not native_output:
        api_params["response_format"] = _maybe_sanitize_response_format(
            effective_model, output_model
        )
        return
    extra = dict(api_params.get("extra_body") or {})
    for settings in (api_params, extra):
        if any(
            key in settings
            for key in (
                "response_format",
                "text_format",
                "text",
                "output_format",
                "tools",
                "tool_choice",
            )
        ):
            raise ValueError(
                "Native output_model conflicts with explicit structured format settings"
            )
        output_config = settings.get("output_config")
        if output_config is not None and not isinstance(output_config, Mapping):
            raise ValueError("output_config must be a mapping")
        if isinstance(output_config, Mapping) and "format" in output_config:
            raise ValueError("output_model conflicts with explicit output_config.format")
    output_config = dict(api_params.pop("output_config", {}) or {})
    for key, value in (extra.get("output_config") or {}).items():
        if key in output_config and output_config[key] != value:
            raise ValueError("Conflicting output_config settings")
        output_config[key] = value
    output_config["format"] = anthropic_output_format(output_model)
    # The mode decides that output_config.format is sent; the provider
    # only decides where it goes. The OpenAI SDK copies extra_body into
    # the request JSON, while LiteLLM's Anthropic Messages handler expects
    # a top-level parameter (it would nest an extra_body value).
    _, provider, _, _ = litellm.get_llm_provider(
        model=effective_model,
        custom_llm_provider=api_params.get("custom_llm_provider"),
        api_base=api_params.get("api_base") or api_params.get("base_url"),
    )
    if provider == "anthropic":
        extra.pop("output_config", None)
        api_params["output_config"] = output_config
        if extra:
            api_params["extra_body"] = extra
        else:
            api_params.pop("extra_body", None)
    elif provider in {"openai", "azure"} or provider in getattr(
        litellm, "openai_compatible_providers", []
    ):
        extra["output_config"] = output_config
        api_params["extra_body"] = extra
    else:
        raise ValueError(
            "Native Anthropic carrier requires Anthropic Messages or a Chat-compatible gateway"
        )


@contextmanager
def native_output_request_scope(api_params, output_model, native_output, *, is_async):
    """Check the HTTP request LiteLLM sends during a native-output call.

    Installs the request hook on the httpx client of this call's SDK client
    and records the expected schema for the duration of the ``with``
    block. Does nothing when the call does not use native output.

    Raises:
        ValueError: If the call has no per-client httpx client to hook.
    """
    if not native_output or output_model is None:
        yield
        return
    import httpx

    # Check the request body that is actually sent, not only the
    # parameters given to LiteLLM, which could still drop or rewrite the
    # schema. The hook is added to this call's httpx client, never to a
    # global one. It is a module-level function, added once per httpx
    # client, so a caller-owned client shared by several CompletionClients
    # holds no reference to them; concurrent calls each see their own
    # expected schema through the ContextVar.
    client = api_params.get("client")
    http = getattr(client, "_client", None)
    if http is None:
        http = getattr(client, "client", None)
    expected_type = httpx.AsyncClient if is_async else httpx.Client
    if not isinstance(http, expected_type):
        raise ValueError(
            "Native structured output requires a verifiable per-client HTTP transport; "
            "construct a separate CompletionClient for a replacement route"
        )
    hook = _acheck_native_output_request if is_async else _check_native_output_request
    hooks = http.event_hooks.setdefault("request", [])
    if hook not in hooks:
        hooks.append(hook)
    # Build the expected schema again from output_model instead of reusing
    # the dict in api_params, which LiteLLM could modify in place.
    token = _NATIVE_OUTPUT_CONTRACT.set(anthropic_output_format(output_model))
    try:
        yield
    finally:
        _NATIVE_OUTPUT_CONTRACT.reset(token)


def response_from_chat(raw_response, scope, tools, output_model, *, native_output=False):
    """Turn a Chat Completions reply into an LLMResponse."""
    from nooa.unifiedllm.chat_parts import capture_chat_parts

    reasoning, usage = _extract_reasoning_and_usage(raw_response)
    turn = LLMResponse(
        raw_response=raw_response,
        parts=capture_chat_parts(raw_response.choices[0].message, scope),
        replay_scope=scope,
        finish_reason=_map_completion_finish_reason(raw_response),
        usage=usage,
    )
    if native_output and output_model is not None:
        if turn.finish_reason != "stop" or turn.tool_calls or not turn.content.strip():
            raise ValueError(
                "Native structured output requires a complete final stop with JSON text, not tools or reasoning alone"
            )
        # Parse the whole reply as JSON for output_model; unlike the default
        # path, there is no fallback that extracts JSON from fenced or
        # surrounding text.
        turn.parsed = output_model.model_validate_json(turn.content)
        return turn
    if turn.tool_calls:
        turn.finish_reason = _finish_reason_for_tool_calls(turn.finish_reason)
        return turn

    # A parsed XML fallback is an edit of the model turn, so native state
    # is deliberately discarded through the same replacement API.
    if tools and "<tool_call>" in turn.content:
        if calls := _extract_xml_tool_calls(turn.content):
            turn = turn.replace_parts(
                (
                    *(part for part in turn.parts if isinstance(part, AssistantReasoning)),
                    *calls,
                )
            )
            turn.finish_reason = _finish_reason_for_tool_calls(turn.finish_reason)
            return turn
    if output_model:
        parseable_content = turn.content or reasoning or ""
        if not turn.content and reasoning:
            _record_llm_metric("reasoning_as_structured_output")
        turn.parsed = _instantiate_output_model(
            output_model, extract_and_parse_json(parseable_content)
        )
    return turn


class LiteLLMChatAdapter:
    """Chat Completions through ``litellm.completion`` / ``litellm.acompletion``.

    Built by a CompletionClient, which it reads for the model, configuration,
    HTTP transport, media rules and cache mapping. The client's private
    methods (``_set_output_format`` and the others) stay the hooks this
    adapter calls, so subclasses and tests that replace them keep working.
    """

    def __init__(self, client: "CompletionClient"):
        """Wrap ``client``, whose settings and hooks every call reads."""
        self._client = client
        self._legacy_options: dict[str, Any] | None = None
        # Private lazy numeric bridge; legacy success hooks remain unchanged.
        self._route_calibration_record = None
        self._route_calibration_identity = object()

    def _numeric_participation(self):
        """Exact built-ins only: inherited participation never bypasses send overrides."""
        if (
            type(self) is not LiteLLMChatAdapter
            or "send" in self.__dict__
            or "send_sync" in self.__dict__
        ):
            return None
        from nooa.unifiedllm._route_calibration import _CalibrationAdapter

        from ._calibration import _route_calibration

        return _CalibrationAdapter(
            self, _route_calibration(self) if self._legacy_options is None else None
        )

    @property
    def capabilities(self) -> Capabilities:
        """The capabilities the client declares."""
        return getattr(self, "_bound_capabilities", None) or client_capabilities(self._client)

    def http_client(self, call_config: dict[str, Any], *, is_async: bool, max_retries: int) -> Any:
        """Reuse the owned transport only while its constructor routing still applies.

        The endpoint is compared with route_identity.endpoint_changed, like media
        permissions, reasoning levels and context windows: the configured URL
        under the other spelling, with a trailing slash or a different host case
        is the same route and keeps the owned transport.
        """
        from nooa.unifiedllm.route_identity import endpoint_changed

        client = self._client
        if (
            client._effective_model(call_config) != client.model
            or endpoint_changed(client.config, call_config)
            or any(
                call_config.get(key) != client.config.get(key)
                for key in ("api_key", "custom_llm_provider")
            )
        ):
            # LiteLLM uses a supplied OpenAI SDK client's bound URL/key, ignoring
            # the corresponding call parameters. Let it build the correct client
            # for overrides; these calls use LiteLLM's default HTTP pool settings.
            return None
        assert client._http is not None
        sdk_client = client._http.async_client if is_async else client._http.sync_client
        return _with_sdk_retries(sdk_client, max_retries)

    def prepare(
        self,
        request: LLMRequest,
        call_config: dict[str, Any],
        *,
        native_output: bool,
        is_async: bool,
    ) -> PreparedChatCall:
        """Build the LiteLLM keyword arguments for one call."""
        client = self._client
        messages = request.messages
        tools = request.tools or None
        output_model = request.output_type
        effective_model = client._effective_model(call_config)
        client._validate_cache_breakpoint_model(effective_model)
        state_scope = replay_state.replay_scope(effective_model, "chat", call_config)
        # Scope's resolved provider (from litellm) and _resolve_cache_mapping
        # can disagree for gateway-routed models; the latter -- not scope --
        # decides whether Anthropic-style cache_control marking is applied.
        cache_mapping = client._resolve_cache_mapping(effective_model, responses=False)
        # Legacy hooks keep sticky connection exclusions; strict/public sends
        # use only a request-local permission view and never touch that state.
        media_messages = (
            client._prepare_media_messages(messages, call_config)
            if self._legacy_options is not None
            else filter_media_messages(messages, client._request_media_inputs(call_config))
        )
        prepared_messages = replay_state.prepare_chat_messages(
            media_messages,
            state_scope,
            anthropic_cache_marking=cache_mapping == "anthropic",
            media_inputs=client._request_media_inputs(call_config),
        )

        client._validate_implemented_media(prepared_messages, call_config)

        # Choose the stable-prefix breakpoint on projected provider messages.
        prepared_messages, _, _ = client._prepare_cache_boundary(
            prepared_messages, responses=False, model=effective_model
        )
        role, developer_role = _call_instruction_roles(client, call_config)
        prepared_messages = map_instruction_roles(prepared_messages, role, developer_role)
        json_object = (
            output_model is not None
            and not native_output
            and _call_output_format(client, call_config) == "json_object"
        )
        if json_object:
            # JSON mode: the schema travels as a final instruction, after the
            # cacheable prefix, in the role this route uses for instructions.
            instruction = json_object_instruction(output_model)
            prepared_messages = [*prepared_messages, {**instruction, "role": role}]

        api_params = {
            "model": client.model,
            **call_config,
            "messages": prepared_messages,
        }

        if tools:
            api_params["tools"] = [client._convert_tool_to_schema(tool) for tool in tools]
            api_params["parallel_tool_calls"] = False
        if (name := named_tool_choice(api_params.get("tool_choice"), tools)) is not None:
            api_params["tool_choice"] = {"type": "function", "function": {"name": name}}

        client._normalize_chat_extensions(api_params, effective_model)
        if json_object:
            api_params["response_format"] = {"type": "json_object"}
        else:
            client._set_output_format(api_params, output_model, effective_model, native_output)

        # Bedrock/Anthropic reject messages with tool_call blocks when tools= is absent.
        if (
            "tools" not in api_params
            and not (native_output and output_model is not None)
            and _needs_dummy_tool(effective_model)
            and _messages_have_tool_calls(prepared_messages)
        ):
            api_params["tools"] = [_DUMMY_TOOL_SCHEMA]

        # tool_choice/parallel_tool_calls are meaningless without tools — strip to avoid
        # provider rejections (e.g. Bedrock) when kwargs leak from CodeAct to PredictStrategy.
        if "tools" not in api_params:
            api_params.pop("tool_choice", None)
            api_params.pop("parallel_tool_calls", None)
        elif api_params.get("tool_choice") == "auto":
            # Auto is the default with tools; sending it alongside parallel=False
            # can create conflicting tool-choice settings in compatible servers.
            api_params.pop("tool_choice")

        add_session_affinity_header(api_params)
        # RetryConfig owns endpoint retries. LiteLLM otherwise has the OpenAI SDK
        # retry each attempt twice more, tripling the configured budget. An
        # explicit max_retries passthrough still opts into SDK retries.
        api_params.setdefault("max_retries", 0)

        if "client" in api_params:
            api_params["client"] = _with_sdk_retries(
                api_params["client"], api_params["max_retries"]
            )
        else:
            http_client = client._completion_http_client(
                call_config, is_async=is_async, max_retries=api_params["max_retries"]
            )
            if http_client is not None:
                api_params["client"] = http_client

        return PreparedChatCall(
            api_params=api_params,
            effective_model=effective_model,
            state_scope=state_scope,
            prepared_messages=prepared_messages,
            tools=tools,
            output_model=output_model,
            native_output=native_output,
        )

    def transmit_sync(self, prepared: PreparedChatCall, *, _attempt_owner=None) -> Any:
        """One sync provider attempt; no retries."""
        if _attempt_owner is not None:
            _attempt_owner.writer.entering_provider(
                hidden_retries=bool(
                    prepared.api_params.get("num_retries", 0)
                    or prepared.api_params.get("max_retries", 0)
                )
            )
        with self._client._native_output_request_scope(
            prepared.api_params, prepared.output_model, prepared.native_output, is_async=False
        ):
            with _owned_provider_region(
                prepared.api_params,
                owner=_attempt_owner,
                call_type="completion",
                prepare=_prepare_owned_litellm_call,
            ) as params:
                return _collect_sync(litellm.completion(**params))

    async def transmit(
        self, prepared: PreparedChatCall, *, _wire_owner=None, _attempt_owner=None
    ) -> Any:
        """One async provider attempt under admission and the request deadline."""

        async def provider_call():
            """The LiteLLM call, under the native-output request check."""
            with self._client._native_output_request_scope(
                prepared.api_params, prepared.output_model, prepared.native_output, is_async=True
            ):
                with _owned_provider_region(
                    prepared.api_params,
                    owner=_attempt_owner,
                    call_type="acompletion",
                    prepare=_prepare_owned_litellm_call,
                ) as params:
                    return await _collect_async(await litellm.acompletion(**params))

        return await _run_async_provider_call(
            provider_call,
            request_timeout=self._client._http_config.request_timeout,
            _wire_owner=_wire_owner,
            _attempt_owner=_attempt_owner,
            _hidden_retries=bool(
                prepared.api_params.get("num_retries", 0)
                or prepared.api_params.get("max_retries", 0)
            ),
        )

    def parse(self, prepared: PreparedChatCall, raw_response: Any) -> LLMResponse:
        """Turn the provider reply into an LLMResponse."""
        return self._client._response_from_chat(
            raw_response,
            prepared.state_scope,
            prepared.tools,
            prepared.output_model,
            native_output=prepared.native_output,
        )

    def _settings(self, request: LLMRequest):
        """Legacy views keep exact option precedence, including explicit None.

        Do not fill from the recorded request in a legacy view: an inherited
        reasoning level may define its own cap, or recovery may disable it.
        """
        options = self._legacy_options
        if options is None:
            options = request_options(request, {})
            # Strict/public sends own one wire attempt, even if the backing
            # legacy client's constructor opted into SDK retries. Legacy views
            # retain those overrides. This does not create a new HTTP pool.
            options["max_retries"] = 0
            options["num_retries"] = 0
            if "tool_choice" in self._client.config:
                options["tool_choice"] = request.tool_choice
        if hasattr(self, "_bound_settings"):
            from ._binding import _bound_call_config, _settings_client

            call_config = _bound_call_config(self, request, options)
            settings_client = _settings_client(self)
        else:
            settings_client = self._client
            call_config = settings_client._prepare_call_config(options)
        settings_client._validate_request_config("messages", call_config)
        native_output = settings_client._native_output_for_call(options, call_config)
        return call_config, native_output

    @staticmethod
    def _calibrate(prepared: PreparedChatCall, raw_response: Any) -> None:
        """Feed the reply's input-token count to the token-estimate calibration.

        Done here because the estimate must count the projected wire messages
        and tool schemas, which only this adapter has.
        """
        _, usage = _extract_reasoning_and_usage(raw_response)
        if usage:
            _update_token_calibration(
                prepared.effective_model,
                prepared.prepared_messages,
                usage,
                tools=prepared.api_params.get("tools"),
            )

    def _finish(
        self, prepared: PreparedChatCall, raw_response: Any, *, legacy_calibration: bool = True
    ) -> LLMResponse:
        """Keep normal parsing; an explicitly owned send skips the legacy hook."""
        if legacy_calibration:
            self._calibrate(prepared, raw_response)
        return self.parse(prepared, raw_response)

    def send_sync(self, request: LLMRequest) -> LLMResponse:
        """One sync attempt for a strict request; no retries."""
        return self._send_sync(request)

    def _send_sync(
        self,
        request: LLMRequest,
        *,
        _owner: _ObservationOwner | None = None,
        _decision=None,
        _legacy_calibration: bool = True,
    ) -> LLMResponse:
        """Explicit shell participation applies only to this exact send."""
        strict = type(self) is LiteLLMChatAdapter and self._legacy_options is None
        with _observe_attempt(_owner) as observation:
            if observation is not None:
                observation._replace(dispatch_known=True)
            with failure_boundary(strict, "config", observation=observation):
                call_config, native_output = self._settings(request)
            with failure_boundary(strict, "prepare", observation=observation):
                prepared = self.prepare(
                    request, call_config, native_output=native_output, is_async=False
                )
                if observation is not None:
                    if _decision is not None:
                        estimated, quality = _estimate_projected_input_tokens(
                            prepared.effective_model,
                            prepared.prepared_messages,
                            tools=prepared.api_params.get("tools"),
                        )
                        observation.projected(estimated)
                        from ._prepared_limits import _apply_decision

                        _apply_decision(
                            prepared.api_params, estimated, quality, _decision, responses=False
                        )
                    elif not (
                        _legacy_calibration
                        and _owner is not None
                        and _owner.writer.record.logical_id is not None
                        and _owner.writer._collector is None
                    ):
                        observation.projected(
                            _estimate_raw_input_tokens(
                                prepared.effective_model,
                                prepared.prepared_messages,
                                tools=prepared.api_params.get("tools"),
                            )
                        )
                    observation.transmitting()
            with failure_boundary(strict, "wire", observation=observation):
                if (
                    _owner is not None
                    and type(self) is LiteLLMChatAdapter
                    and "transmit_sync" not in self.__dict__
                ):
                    raw_response = self.transmit_sync(prepared, _attempt_owner=_owner)
                else:
                    if observation is not None:
                        observation._replace(dispatch_known=False)
                    raw_response = self.transmit_sync(prepared)
            if observation is not None:
                observation.delivered(_numeric_usage(raw_response))
            with failure_boundary(strict, "parse", reply=raw_response, observation=observation):
                return _parse_reply(
                    lambda: self._finish(
                        prepared, raw_response, legacy_calibration=_legacy_calibration
                    )
                )

    async def send(self, request: LLMRequest) -> LLMResponse:
        """One async attempt for a strict request; no retries."""
        return await self._send(request)

    async def _send(
        self,
        request: LLMRequest,
        *,
        _owner: _ObservationOwner | None = None,
        _decision=None,
        _wire_owner=None,
        _legacy_calibration: bool = True,
    ) -> LLMResponse:
        """Explicit shell participation applies only to this exact send."""
        strict = type(self) is LiteLLMChatAdapter and self._legacy_options is None
        with _observe_attempt(_owner) as observation:
            if observation is not None:
                observation._replace(dispatch_known=True)
            with failure_boundary(strict, "config", observation=observation):
                call_config, native_output = self._settings(request)
            with failure_boundary(strict, "prepare", observation=observation):
                prepared = self.prepare(
                    request, call_config, native_output=native_output, is_async=True
                )
                if observation is not None:
                    if _decision is not None:
                        estimated, quality = _estimate_projected_input_tokens(
                            prepared.effective_model,
                            prepared.prepared_messages,
                            tools=prepared.api_params.get("tools"),
                        )
                        observation.projected(estimated)
                        from ._prepared_limits import _apply_decision

                        _apply_decision(
                            prepared.api_params, estimated, quality, _decision, responses=False
                        )
                    elif not (
                        _legacy_calibration
                        and _owner is not None
                        and _owner.writer.record.logical_id is not None
                        and _owner.writer._collector is None
                    ):
                        observation.projected(
                            _estimate_raw_input_tokens(
                                prepared.effective_model,
                                prepared.prepared_messages,
                                tools=prepared.api_params.get("tools"),
                            )
                        )
                    observation.transmitting()
            with failure_boundary(strict, "wire", observation=observation):
                if (
                    _owner is not None
                    and type(self) is LiteLLMChatAdapter
                    and "transmit" not in self.__dict__
                ):
                    raw_response = await self.transmit(
                        prepared, _wire_owner=_wire_owner, _attempt_owner=_owner
                    )
                else:
                    if observation is not None:
                        observation._replace(dispatch_known=False)
                    raw_response = await (
                        self.transmit(prepared)
                        if _wire_owner is None
                        else self.transmit(prepared, _wire_owner=_wire_owner)
                    )
            if observation is not None:
                observation.delivered(_numeric_usage(raw_response))
            with failure_boundary(strict, "parse", reply=raw_response, observation=observation):
                return _parse_reply(
                    lambda: self._finish(
                        prepared, raw_response, legacy_calibration=_legacy_calibration
                    )
                )
