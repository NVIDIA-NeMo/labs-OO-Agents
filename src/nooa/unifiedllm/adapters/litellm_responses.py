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

from nooa.llm_types import CacheBoundary, LLMResponse
from nooa.unifiedllm import replay_state, response_parts
from nooa.unifiedllm import unifiedllm as _core
from nooa.unifiedllm.adapters.base import Capabilities
from nooa.unifiedllm.adapters.litellm_chat import (
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
from nooa.unifiedllm.media_inputs import attachment_kind
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

    @property
    def capabilities(self) -> Capabilities:
        """The capabilities the client declares."""
        return client_capabilities(self._client)

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
        input_messages, instructions, openai_explicit = client._prepare_input(
            client._prepare_media_messages(request.messages, call_config),
            state_scope,
            effective_model,
            native_encrypted_reasoning,
            media_inputs=client._request_media_inputs(call_config),
        )

        client._validate_implemented_media(input_messages, call_config)

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

        if output_model is not None:
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

        return PreparedResponsesCall(
            api_params=api_params,
            effective_model=effective_model,
            state_scope=state_scope,
            input_messages=input_messages,
            output_model=output_model,
            native_encrypted_reasoning=native_encrypted_reasoning,
        )

    def transmit_sync(self, prepared: PreparedResponsesCall) -> Any:
        """One sync provider attempt; no retries."""
        return cast("litellm.ResponsesAPIResponse", litellm.responses(**prepared.api_params))

    async def transmit(self, prepared: PreparedResponsesCall) -> Any:
        """One async provider attempt under admission and the request deadline."""

        async def call_provider():
            """The LiteLLM call."""
            return cast(
                "litellm.ResponsesAPIResponse", await litellm.aresponses(**prepared.api_params)
            )

        return await _run_async_provider_call(
            call_provider,
            request_timeout=self._client._http_config.request_timeout,
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

    def _call_config(self, request: LLMRequest, options: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve and check the call config for one attempt."""
        call_config = self._client._prepare_call_config(request_options(request, options))
        self._client._validate_request_config("input", call_config)
        return call_config

    @staticmethod
    def _finish(prepared: PreparedResponsesCall, raw_response: Any):
        """Read usage, feed the token-estimate calibration and return the usage.

        Calibration happens here because the estimate must count the projected
        input, tool schemas and instructions, which only this adapter has.
        """
        usage = _core._extract_usage(raw_response)
        if usage:
            _core._update_token_calibration(
                prepared.effective_model,
                prepared.input_messages,
                usage,
                tools=prepared.api_params.get("tools"),
                instructions=prepared.api_params.get("instructions"),
            )
        return usage

    def send_sync(self, request: LLMRequest, **options: Any) -> LLMResponse:
        """One attempt, sync. ``options`` are LiteLLM call settings (temporary)."""
        prepared = self.prepare(request, self._call_config(request, options), is_async=False)
        raw_response = self.transmit_sync(prepared)
        return self.parse(prepared, raw_response, self._finish(prepared, raw_response))

    async def send(self, request: LLMRequest, **options: Any) -> LLMResponse:
        """One attempt, async. ``options`` are LiteLLM call settings (temporary)."""
        prepared = self.prepare(request, self._call_config(request, options), is_async=True)
        raw_response = await self.transmit(prepared)
        return self.parse(prepared, raw_response, self._finish(prepared, raw_response))
