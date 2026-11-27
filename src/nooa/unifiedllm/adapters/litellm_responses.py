# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LiteLLM Responses adapter: request building, the call and reply parsing.

The code here was moved out of ``ResponsesClient`` without changing the wire
body. The client owns retries, metrics and debug tracking and calls only
``send`` / ``send_sync``, one attempt each. ``prepare``, ``transmit`` and
``parse`` are this adapter's own steps; no client calls them.
"""

import copy
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import litellm

from nooa._provider_ownership import _owned_provider_region
from nooa.llm_types import CacheBoundary, LLMResponse
from nooa.unifiedllm import replay_state, response_parts
from nooa.unifiedllm import unifiedllm as _core
from nooa.unifiedllm._observations import _ObservationOwner, _observe_attempt
from nooa.unifiedllm.adapters._numeric_usage import _numeric_usage
from nooa.unifiedllm.adapters._provider_logging import _prepare_owned_litellm_call
from nooa.unifiedllm.adapters.base import Capabilities
from nooa.unifiedllm.adapters.litellm_chat import (
    _call_instruction_roles,
    _call_output_format,
    client_capabilities,
    named_tool_choice,
    request_options,
)
from nooa.unifiedllm.cache_policy import (
    add_session_affinity_header,
    enable_openai_explicit_cache,
    reject_boundary_dict,
    wrap_responses_text,
)
from nooa.unifiedllm.capability_settings import json_object_instruction, map_instruction_roles
from nooa.unifiedllm.errors import _parse_reply
from nooa.unifiedllm.media_inputs import attachment_kind, filter_media_messages
from nooa.unifiedllm.request import LLMRequest
from nooa.unifiedllm.unifiedllm import (
    Tool,
    _finish_reason_for_tool_calls,
    _instantiate_output_model,
    _loose_response_schema,
    _map_responses_finish_reason,
    _responses_output_params,
    _run_async_provider_call,
    _strict_schema_valid,
    extract_and_parse_json,
)

from ._failures import failure_boundary

if TYPE_CHECKING:
    from nooa.unifiedllm.unifiedllm import ResponsesClient

# Keep the logger name of the code's old home so log filters keep matching.
logger = logging.getLogger("nooa.unifiedllm.unifiedllm")


@dataclass
class PreparedResponsesCall:
    """Everything one Responses attempt needs, built once per call."""

    api_params: dict[str, Any]
    effective_model: str
    state_scope: str | None
    input_messages: list[Any]
    output_model: Any
    native_encrypted_reasoning: bool


def tool_schema(tool: Tool) -> dict[str, Any]:
    """Convert a Tool to the Responses API schema."""
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


def response_from_output(
    raw_response, scope, usage, output_model, native_encrypted_reasoning=False
):
    """Turn a Responses reply into an LLMResponse."""
    parts = response_parts.capture_parts(
        raw_response.output, scope, native_encrypted_reasoning=native_encrypted_reasoning
    )
    response = LLMResponse(
        raw_response=raw_response,
        parts=parts,
        replay_scope=scope if any(part.native for part in parts) else None,
        usage=usage,
        finish_reason=_map_responses_finish_reason(raw_response),
    )
    if response.tool_calls:
        response.finish_reason = _finish_reason_for_tool_calls(response.finish_reason)
    elif output_model:
        response.parsed = _instantiate_output_model(
            output_model, extract_and_parse_json(response.content)
        )
    return response


def transform_content_blocks(blocks, text_kind="input_text"):
    """Project documented public slots, preserving request-owned metadata."""
    result = copy.deepcopy(blocks)
    for block in result:
        if block.get("type") == "text":
            block["type"] = text_kind
        elif block.get("type") == "image_url":
            image = block["image_url"]
            if isinstance(image, dict):
                if not image.get("url"):
                    raise ValueError("image_url dict has no 'url'")
                block["image_url"] = image["url"]
                if image.get("detail"):
                    block["detail"] = image["detail"]
            block["type"] = "input_image"
        elif block.get("type") == "file":
            file = block.get("file")
            if not isinstance(file, Mapping):
                raise ValueError(
                    "Responses file content requires a nested 'file' mapping; "
                    "use {'type': 'file', 'file': {...}} or a native "
                    "{'type': 'input_file', ...} block."
                )
            if not any(file.get(key) for key in ("file_data", "file_id", "file_url")):
                raise ValueError("Responses file mapping requires file_data, file_id or file_url.")
            block.pop("file")
            block.update(
                {
                    key: value
                    for key, value in file.items()
                    if key in {"detail", "file_data", "file_id", "file_url", "filename"}
                }
            )
            block["type"] = "input_file"
        elif block.get("type") == "input_file":
            # A MIME hint only tells the media check that a file_id is a PDF;
            # input_file has no such field, so strict endpoints reject it.
            for key in ("mime_type", "media_type", "format"):
                block.pop(key, None)
    return result


def transform_messages(
    messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
    state_scope: str | None = None,
    native_encrypted_reasoning: bool = False,
    *,
    media_inputs=None,
) -> tuple[list[dict[str, Any] | CacheBoundary], str | None]:
    """Expand turns at dispatch; only leading system messages become instructions."""
    instructions: list[str] = []
    transformed: list[dict[str, Any] | CacheBoundary] = []
    leading_system = True
    for original in messages:
        if not isinstance(original, Mapping):
            raise TypeError("Each message must be a mapping or LLMResponse.")
        # Moving a later system message to instructions would reorder history.
        leading_system = leading_system and original.get("role") == "system"
        if isinstance(original, LLMResponse):
            transformed.extend(
                response_parts.project_turn(
                    original,
                    state_scope,
                    native_encrypted_reasoning=native_encrypted_reasoning,
                    media_inputs=media_inputs,
                )
            )
            continue
        if isinstance(original, CacheBoundary):
            transformed.append(original)
            continue
        msg = dict(original)
        reject_boundary_dict(msg)
        replay_state.reject_native_message(msg, state_scope)
        if isinstance(msg.get("content"), list) and any(
            not isinstance(block, dict) for block in msg["content"]
        ):
            raise TypeError("Message content blocks must be dictionaries.")
        if leading_system:
            content = msg.get("content")
            if isinstance(content, list):
                if any(
                    block.get("type") not in {"text", "input_text"}
                    or not isinstance(block.get("text"), str)
                    for block in content
                ):
                    raise ValueError(
                        "Leading system content requires text blocks with string text."
                    )
                content = "".join(block["text"] for block in content)
            if content:
                instructions.append(content)
        elif msg.get("role") == "tool":
            if not isinstance(msg.get("tool_call_id"), str):
                raise ValueError("Tool result requires a string 'tool_call_id'.")
            content = msg.get("content", "")
            cache_control = msg.get("cache_control")
            if isinstance(content, list):
                cache_control = next(
                    (block["cache_control"] for block in content if "cache_control" in block),
                    cache_control,
                )
                if any(attachment_kind(block) for block in content):
                    output = transform_content_blocks(content)
                else:
                    output = [
                        wrap_responses_text("".join(block.get("text", "") for block in content))
                    ]
            else:
                output = [wrap_responses_text(content)]
            # Same stability rationale as the input_text wrapping below --
            # a plain string here (including "") would render differently
            # than its cache-marked list form once this tool result
            # becomes history; apply_cache_policy's marker wraps a string
            # unconditionally, so this must too.
            item = {
                "type": "function_call_output",
                "call_id": msg["tool_call_id"],
                "output": output,
            }
            if cache_control:
                # This is caller-owned mutable metadata, not retained native
                # state. Keep request mutations isolated from future sends.
                item["cache_control"] = copy.deepcopy(cache_control)
            transformed.append(item)
        elif msg.get("role") == "assistant" and (
            msg.get("tool_calls") or msg.get("reasoning_content")
        ):
            # Public input accepts content blocks, unlike normalized Chat
            # output. Preserve those blocks directly rather than passing
            # them through the provider-response capture contract.
            reasoning = msg.get("reasoning_content")
            if reasoning is not None and not isinstance(reasoning, str):
                raise ValueError("Assistant reasoning_content must be a string.")
            if reasoning:
                transformed.append({"role": "assistant", "content": reasoning})
            content = msg.get("content")
            if content is not None and not isinstance(content, (str, list)):
                raise ValueError("Assistant content must be a string or list of blocks.")
            if content:
                content = copy.deepcopy(content)
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            block["type"] = "output_text"
                transformed.append({"role": "assistant", "content": content})
            for call in msg.get("tool_calls") or []:
                function = call["function"]  # Shape checked by reject_native_message.
                if not all(
                    isinstance(value, str)
                    for value in (
                        call.get("id"),
                        function.get("name"),
                        function.get("arguments"),
                    )
                ):
                    raise ValueError("Tool call id, name and arguments must be strings.")
                transformed.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": function["name"],
                        "arguments": function["arguments"],
                    }
                )
        else:
            # Raw dictionaries are mutable caller input. We rewrite nested
            # block types here and the SDK may mutate them again; detach the
            # containers once. deepcopy shares immutable strings (including
            # encoded images). LLMResponse took the projection path above,
            # so this does not copy its retained reasoning blobs.
            item = copy.deepcopy(msg)
            if isinstance(item.get("content"), str):
                # Always emit input_text/output_text blocks in list form so a
                # message's wire shape is stable whether or not it happens to
                # be the one apply_cache_policy marks with a cache breakpoint
                # this turn. A plain string (including "") would flip to a
                # marked block on the turn it's cached and back to a bare
                # string the next turn, breaking the provider's
                # stable-prefix cache match -- apply_cache_policy's marker
                # wraps a string unconditionally, so this must too.
                kind = "output_text" if item.get("role") == "assistant" else "input_text"
                item["content"] = [wrap_responses_text(item["content"], kind)]
            for slot in ("content", "output"):
                if isinstance(item.get(slot), list):
                    item[slot] = transform_content_blocks(
                        item[slot],
                        "output_text" if item.get("role") == "assistant" else "input_text",
                    )
            if item.get("type") == "function_call_output" and isinstance(item.get("output"), str):
                # Same stability rationale as the tool-message conversion
                # branch above -- an already-native function_call_output
                # (not converted from a role="tool" message) took this
                # generic path unwrapped, so it still flipped shape
                # (including "", per apply_cache_policy's unconditional
                # string wrap) whenever apply_cache_policy marked it.
                item["output"] = [wrap_responses_text(item["output"])]
            transformed.append(item)
    return transformed, "\n\n".join(instructions) or None


def extract_text_from_output(response: Any) -> str:
    """The reply text of a Responses reply."""
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    if hasattr(response, "output"):
        return replay_state.responses_output_text(response.output)
    return ""


class LiteLLMResponsesAdapter:
    """The Responses API through ``litellm.responses`` / ``litellm.aresponses``.

    Built by a ResponsesClient, which it reads for the model, configuration,
    HTTP transport, media rules and cache mapping. The client's private
    methods (``_prepare_input`` and the others) stay the hooks this adapter
    calls, so subclasses and tests that replace them keep working.
    """

    def __init__(self, client: "ResponsesClient"):
        """Wrap ``client``, whose settings and hooks every call reads."""
        self._client = client
        self._legacy_options: dict[str, Any] | None = None
        # Private lazy numeric bridge; legacy success hooks remain unchanged.
        self._route_calibration_record = None
        self._route_calibration_identity = object()

    def _numeric_participation(self):
        """Exact built-ins only: inherited participation never bypasses send overrides."""
        if (
            type(self) is not LiteLLMResponsesAdapter
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

    def prepare(
        self, request: LLMRequest, call_config: dict[str, Any], *, is_async: bool
    ) -> PreparedResponsesCall:
        """Build the LiteLLM keyword arguments for one call."""
        client = self._client
        tools = request.tools or None
        output_model = request.output_type
        effective_model = client._effective_model(call_config)
        client._validate_cache_breakpoint_model(effective_model)
        state_scope = replay_state.replay_scope(effective_model, "responses", call_config)
        native_encrypted_reasoning = replay_state.native_encrypted_reasoning_expected(
            call_config, state_scope
        )
        # Do not import a legacy client's sticky media exclusions into strict
        # sends. Opaque native replay still goes through the adapter's checks.
        media_messages = (
            client._prepare_media_messages(request.messages, call_config)
            if self._legacy_options is not None
            else filter_media_messages(request.messages, client._request_media_inputs(call_config))
        )
        input_messages, instructions, openai_explicit = client._prepare_input(
            media_messages,
            state_scope,
            effective_model,
            native_encrypted_reasoning,
            media_inputs=client._request_media_inputs(call_config),
        )

        client._validate_implemented_media(input_messages, call_config)
        # Leading system messages already became ``instructions``; the role
        # choice applies to the instruction items that stay in ``input``.
        role, developer_role = _call_instruction_roles(client, call_config)
        input_messages = map_instruction_roles(input_messages, role, developer_role)
        json_object = (
            output_model is not None and _call_output_format(client, call_config) == "json_object"
        )
        if json_object:
            # JSON mode: the schema travels as a final instruction item.
            instruction = json_object_instruction(output_model)
            input_messages = [
                *input_messages,
                {"role": role, "content": [{"type": "input_text", "text": instruction["content"]}]},
            ]

        api_params = {
            "model": client.model,
            "truncation": "disabled",
            **call_config,
            "input": input_messages,
        }
        if openai_explicit:
            enable_openai_explicit_cache(api_params)

        if instructions:
            api_params["instructions"] = instructions

        if "base_url" in api_params:
            api_params["api_base"] = api_params.pop("base_url")

        if (name := named_tool_choice(api_params.get("tool_choice"), tools)) is not None:
            api_params["tool_choice"] = {"type": "function", "name": name}
        if tools:
            api_params["tools"] = [client._convert_tool_to_schema(tool) for tool in tools]
            # Honour the requested choice; "auto" when the call names none.
            api_params["tool_choice"] = api_params.get("tool_choice") or "auto"
            api_params["parallel_tool_calls"] = False

        if json_object:
            api_params["text"] = {
                **(api_params.get("text") or {}),
                "format": {"type": "json_object"},
            }
        elif output_model is not None:
            api_params.update(_responses_output_params(output_model))

        replay_state.add_encrypted_reasoning_include(
            api_params, state_scope, native_encrypted_reasoning=native_encrypted_reasoning
        )
        add_session_affinity_header(api_params)

        http_client = client._http
        assert http_client is not None
        sdk_client = http_client.async_client if is_async else http_client.sync_client
        if sdk_client is not None:
            api_params.setdefault("client", sdk_client)
        if self._legacy_options is None:
            # Bind the SDK copy to the strict attempt budget, sharing its pool.
            api_params["max_retries"] = 0
            api_params["num_retries"] = 0
            if api_params.get("client") is not None:
                api_params["client"] = _core._with_sdk_retries(api_params["client"], 0)

        return PreparedResponsesCall(
            api_params=api_params,
            effective_model=effective_model,
            state_scope=state_scope,
            input_messages=input_messages,
            output_model=output_model,
            native_encrypted_reasoning=native_encrypted_reasoning,
        )

    def transmit_sync(self, prepared: PreparedResponsesCall, *, _attempt_owner=None) -> Any:
        """One sync provider attempt; no retries."""
        if _attempt_owner is not None:
            _attempt_owner.writer.entering_provider(
                hidden_retries=bool(
                    prepared.api_params.get("num_retries", 0)
                    or prepared.api_params.get("max_retries", 0)
                )
            )
        with _owned_provider_region(
            prepared.api_params,
            owner=_attempt_owner,
            call_type="responses",
            prepare=_prepare_owned_litellm_call,
        ) as params:
            return cast("litellm.ResponsesAPIResponse", litellm.responses(**params))

    async def transmit(
        self, prepared: PreparedResponsesCall, *, _wire_owner=None, _attempt_owner=None
    ) -> Any:
        """One async provider attempt under admission and the request deadline."""

        async def call_provider():
            """The LiteLLM call."""
            with _owned_provider_region(
                prepared.api_params,
                owner=_attempt_owner,
                call_type="aresponses",
                prepare=_prepare_owned_litellm_call,
            ) as params:
                return cast("litellm.ResponsesAPIResponse", await litellm.aresponses(**params))

        return await _run_async_provider_call(
            call_provider,
            request_timeout=self._client._http_config.request_timeout,
            _wire_owner=_wire_owner,
            _attempt_owner=_attempt_owner,
            _hidden_retries=bool(
                prepared.api_params.get("num_retries", 0)
                or prepared.api_params.get("max_retries", 0)
            ),
        )

    def parse(self, prepared: PreparedResponsesCall, raw_response: Any, usage: Any) -> LLMResponse:
        """Turn the provider reply into an LLMResponse; ``usage`` is read by the client."""
        return self._client._response_from_output(
            raw_response,
            prepared.state_scope,
            usage,
            prepared.output_model,
            prepared.native_encrypted_reasoning,
        )

    def _call_config(self, request: LLMRequest) -> dict[str, Any]:
        """Resolve exact legacy options or the fields of a strict request."""
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
            from ._binding import _bound_call_config

            call_config = _bound_call_config(self, request, options)
        else:
            call_config = self._client._prepare_call_config(options)
        self._client._validate_request_config("input", call_config)
        return call_config

    @staticmethod
    def _finish(
        prepared: PreparedResponsesCall, raw_response: Any, *, legacy_calibration: bool = True
    ):
        """Return normal usage and retain the direct legacy calibration hook.

        Only an explicitly shell-owned send skips that hook. Its numeric wire
        observation is consumed by the shell even if reply parsing fails.
        """
        usage = _core._extract_usage(raw_response)
        if usage and legacy_calibration:
            _core._update_token_calibration(
                prepared.effective_model,
                prepared.input_messages,
                usage,
                tools=prepared.api_params.get("tools"),
                instructions=prepared.api_params.get("instructions"),
            )
        return usage

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
        strict = type(self) is LiteLLMResponsesAdapter and self._legacy_options is None
        with _observe_attempt(_owner) as observation:
            if observation is not None:
                observation._replace(dispatch_known=True)
            with failure_boundary(strict, "config", observation=observation):
                call_config = self._call_config(request)
            with failure_boundary(strict, "prepare", observation=observation):
                prepared = self.prepare(request, call_config, is_async=False)
                if observation is not None:
                    if _decision is not None:
                        estimated, quality = _core._estimate_projected_input_tokens(
                            prepared.effective_model,
                            prepared.input_messages,
                            tools=prepared.api_params.get("tools"),
                            instructions=prepared.api_params.get("instructions"),
                        )
                        observation.projected(estimated)
                        from ._prepared_limits import _apply_decision

                        _apply_decision(
                            prepared.api_params, estimated, quality, _decision, responses=True
                        )
                    elif not (
                        _legacy_calibration
                        and _owner is not None
                        and _owner.writer.record.logical_id is not None
                        and _owner.writer._collector is None
                    ):
                        observation.projected(
                            _core._estimate_raw_input_tokens(
                                prepared.effective_model,
                                prepared.input_messages,
                                tools=prepared.api_params.get("tools"),
                                instructions=prepared.api_params.get("instructions"),
                            )
                        )
                    observation.transmitting()
            with failure_boundary(strict, "wire", observation=observation):
                if (
                    _owner is not None
                    and type(self) is LiteLLMResponsesAdapter
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
                    lambda: self.parse(
                        prepared,
                        raw_response,
                        self._finish(
                            prepared, raw_response, legacy_calibration=_legacy_calibration
                        ),
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
        strict = type(self) is LiteLLMResponsesAdapter and self._legacy_options is None
        with _observe_attempt(_owner) as observation:
            if observation is not None:
                observation._replace(dispatch_known=True)
            with failure_boundary(strict, "config", observation=observation):
                call_config = self._call_config(request)
            with failure_boundary(strict, "prepare", observation=observation):
                prepared = self.prepare(request, call_config, is_async=True)
                if observation is not None:
                    if _decision is not None:
                        estimated, quality = _core._estimate_projected_input_tokens(
                            prepared.effective_model,
                            prepared.input_messages,
                            tools=prepared.api_params.get("tools"),
                            instructions=prepared.api_params.get("instructions"),
                        )
                        observation.projected(estimated)
                        from ._prepared_limits import _apply_decision

                        _apply_decision(
                            prepared.api_params, estimated, quality, _decision, responses=True
                        )
                    elif not (
                        _legacy_calibration
                        and _owner is not None
                        and _owner.writer.record.logical_id is not None
                        and _owner.writer._collector is None
                    ):
                        observation.projected(
                            _core._estimate_raw_input_tokens(
                                prepared.effective_model,
                                prepared.input_messages,
                                tools=prepared.api_params.get("tools"),
                                instructions=prepared.api_params.get("instructions"),
                            )
                        )
                    observation.transmitting()
            with failure_boundary(strict, "wire", observation=observation):
                if (
                    _owner is not None
                    and type(self) is LiteLLMResponsesAdapter
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
                    lambda: self.parse(
                        prepared,
                        raw_response,
                        self._finish(
                            prepared, raw_response, legacy_calibration=_legacy_calibration
                        ),
                    )
                )
