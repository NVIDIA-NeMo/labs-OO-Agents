# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native wire projection; finite families, no provider/model-name discovery."""

import hashlib
import json
import re
from collections.abc import Mapping

from nooa._immutable_json import json_containers
from nooa.llm_types import AssistantReasoning, AssistantText, CacheBoundary, LLMResponse, ToolCall
from nooa.media_inputs import attachment_kind
from nooa.unifiedllm.cache_policy import (
    apply_cache_policy,
    enable_openai_explicit_cache,
    reject_boundary_dict,
)
from nooa.unifiedllm.capability_settings import json_object_instruction, map_instruction_roles
from nooa.unifiedllm.errors import FailureKind, LLMFailure
from nooa.unifiedllm.structured_output import anthropic_output_format

from ._binding import _thaw
from ._native_transport import _FAMILIES
from .schema import _clean_schema, _strict_schema_valid


def _bad(message):
    return LLMFailure(FailureKind.BAD_REQUEST, message)


def _envelope(family, wire):
    return {"family": family, "version": 1, "wire": wire}


def _replay(response, family, scope):
    items = []
    covered = set()
    for index, part in enumerate(response.parts):
        if part.native is None:
            continue
        native = json_containers(part.native)
        # Foreign legacy state is not native state. Supported malformed native
        # state is terminal, even when a portable cross-family fallback exists.
        if not isinstance(native, dict) or native.get("family") not in _FAMILIES:
            continue
        if native.get("version") != 1 or not isinstance(native.get("wire"), dict):
            raise _bad("Malformed supported native replay state")
        source = native["family"]
        wire = native["wire"]
        if not _valid_replay_item(source, wire):
            raise _bad("Malformed supported native replay item")
        if source == family and response.replay_scope == scope:
            items.append(wire)
            covered.add(index)
    if items:
        if family == "openai-chat":
            # Chat carries one COMPLETE message, not one wire item per part.
            # Its covered portable parts must match exactly; appended or edited
            # parts cannot be silently discarded by restoring the old carrier.
            if len(items) != 1 or _chat_parts(items[0]) != _part_values(response.parts):
                raise _bad("Incomplete native Chat replay cannot drop portable parts")
        elif any(
            index not in covered
            and (
                part.native is not None
                or isinstance(part, ToolCall)
                or isinstance(part, (AssistantText, AssistantReasoning))
                and part.text.strip()
            )
            for index, part in enumerate(response.parts)
        ):
            raise _bad("Incomplete native replay cannot silently drop portable parts")
    return items


def _valid_replay_item(family, wire):
    """Validate supported opaque structures without repairing signed bytes."""

    def string(key):
        return isinstance(wire.get(key), str)

    def calls(value):
        return isinstance(value, list) and all(
            isinstance(call, dict)
            and isinstance(call.get("id"), str)
            and call.get("type") == "function"
            and isinstance(call.get("function"), dict)
            and isinstance(call["function"].get("name"), str)
            and isinstance(call["function"].get("arguments"), str)
            for call in value
        )

    kind = wire.get("type")
    if family == "openai-chat":
        return (
            wire.get("role") == "assistant"
            and isinstance(wire.get("content"), (str, type(None)))
            and calls(wire.get("tool_calls", []))
        )
    if family == "openai-responses":
        if kind == "function_call":
            return all(string(key) for key in ("call_id", "name", "arguments"))
        if kind == "reasoning":
            return (
                string("id")
                and isinstance(wire.get("summary"), list)
                and all(
                    isinstance(block, dict)
                    and block.get("type") == "summary_text"
                    and isinstance(block.get("text"), str)
                    for block in wire["summary"]
                )
                and (
                    "encrypted_content" not in wire
                    or wire["encrypted_content"] is None
                    or string("encrypted_content")
                )
            )
        if kind == "message":
            return (
                wire.get("role") == "assistant"
                and isinstance(wire.get("content"), list)
                and all(
                    isinstance(block, dict)
                    and (
                        block.get("type") == "output_text"
                        and isinstance(block.get("text"), str)
                        or block.get("type") == "refusal"
                        and isinstance(block.get("refusal"), str)
                    )
                    for block in wire["content"]
                )
            )
        return False
    if kind == "text":
        return string("text")
    if kind == "tool_use":
        return string("id") and string("name") and isinstance(wire.get("input"), dict)
    if kind == "thinking":
        return string("thinking") and string("signature")
    return kind == "redacted_thinking" and string("data")


def _part_values(parts):
    values = []
    for part in parts:
        if isinstance(part, ToolCall):
            values.append(("tool", part.id, part.name, part.arguments))
        elif isinstance(part, AssistantReasoning):
            values.append(("reasoning", part.text))
        elif isinstance(part, AssistantText):
            values.append(("text", part.text))
    return values


def _chat_parts(wire):
    values = []
    if wire.get("reasoning_content"):
        values.append(("reasoning", wire["reasoning_content"]))
    values.append(("text", wire.get("content") or ""))
    for call in wire.get("tool_calls", []):
        function = call.get("function", {})
        values.append(("tool", call.get("id"), function.get("name"), function.get("arguments")))
    return values


def _check_media(value, enabled):
    if isinstance(value, Mapping):
        kind = attachment_kind(value)
        if kind is not None and kind not in enabled:
            raise LLMFailure(FailureKind.MEDIA_REJECTED, "Opaque replay contains unsupported media")
        for child in value.values():
            _check_media(child, enabled)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_media(child, enabled)


def _media(block, family, enabled):
    kind = attachment_kind(block)
    if kind not in enabled:
        raise LLMFailure(FailureKind.MEDIA_REJECTED, "Native request contains unsupported media")
    if kind not in {"image", "pdf"}:
        raise _bad("Unsupported native media representation")
    source = block.get("source", {})
    if block.get("type") == "image_url":
        node = block.get("image_url")
        url = node.get("url") if isinstance(node, Mapping) else node
    else:
        url = block.get("image_url") or block.get("file_url") or source.get("url")
    data = block.get("file_data") or source.get("data")
    mime = (
        source.get("media_type")
        or block.get("mime_type")
        or ("application/pdf" if kind == "pdf" else None)
    )
    pdf_uri = (
        (
            data
            if isinstance(data, str) and data.startswith("data:")
            else url
            if isinstance(url, str) and url.startswith("data:")
            else None
        )
        if kind == "pdf"
        else None
    )
    if pdf_uri is not None:
        head, separator, payload = pdf_uri.partition(",")
        if (
            head != "data:application/pdf;base64"
            or not separator
            or not payload
            or mime != "application/pdf"
        ):
            raise _bad("Invalid or incompatible explicit PDF data URI")
        # Opaque base64 bytes are forwarded, not decoded or reencoded. Responses
        # requires the URI; Anthropic requires its payload and explicit MIME.
        data = payload
    elif isinstance(url, str) and url.startswith("data:"):
        try:
            head, data = url.split(",", 1)
            mime = head[5:].split(";", 1)[0]
        except ValueError as exc:
            raise _bad("Invalid native data URL") from exc
    if family == "anthropic-messages":
        if data is not None and isinstance(data, str) and mime:
            output_source = {"type": "base64", "media_type": mime, "data": data}
        elif isinstance(url, str):
            output_source = {"type": "url", "url": url}
        else:
            raise _bad("Native media requires explicit URL or base64 data/MIME")
        return {"type": "image" if kind == "image" else "document", "source": output_source}
    if data is not None and not isinstance(url, str):
        if not mime:
            raise _bad("Native base64 media requires MIME")
        url = "data:" + mime + ";base64," + data
    if family == "openai-chat":
        if kind != "image" or not isinstance(url, str):
            raise _bad("Native Chat supports URL/base64 images only")
        node = {"url": url}
        detail = block.get("detail") or (
            block.get("image_url", {}).get("detail")
            if isinstance(block.get("image_url"), Mapping)
            else None
        )
        if detail:
            node["detail"] = detail
        return {"type": "image_url", "image_url": node}
    if kind == "image" and isinstance(url, str):
        return {"type": "input_image", "image_url": url, "detail": block.get("detail", "auto")}
    if kind == "pdf" and isinstance(url, str):
        return {
            "type": "input_file",
            **(
                {"file_data": url, "filename": block.get("filename", "document.pdf")}
                if url.startswith("data:")
                else {"file_url": url}
            ),
        }
    raise _bad("Native media requires explicit URL or data")


def _content(content, role, family, enabled):
    if content is None:
        content = ""
    if isinstance(content, str):
        if family == "openai-chat":
            return content
        return [
            {
                "type": "text"
                if family == "anthropic-messages"
                else "output_text"
                if role == "assistant"
                else "input_text",
                "text": content,
            }
        ]
    if not isinstance(content, (list, tuple)):
        raise _bad("Native message content must be text or content blocks")
    result = []
    for block in content:
        if not isinstance(block, Mapping):
            raise _bad("Native content blocks must be mappings")
        if block.get("type") in {"text", "input_text", "output_text"}:
            if not isinstance(block.get("text"), str):
                raise _bad("Native text block requires text")
            result.append(
                {
                    "type": "text"
                    if family != "openai-responses"
                    else "output_text"
                    if role == "assistant"
                    else "input_text",
                    "text": block["text"],
                }
            )
        elif attachment_kind(block) is not None:
            result.append(_media(block, family, enabled))
        else:
            raise _bad("Unknown portable content block")
    return result


def _tool_id(identifier, settings, ids):
    if not isinstance(identifier, str) or not identifier:
        raise _bad("Tool IDs must be nonempty strings")
    if settings.host_profile.get("tool_id_pattern") == "alphanumeric" and not re.fullmatch(
        r"[A-Za-z0-9]+", identifier
    ):
        return ids.setdefault(identifier, "call" + hashlib.sha256(identifier.encode()).hexdigest())
    return identifier


def _remap_native(wire, settings, ids):
    wire = json_containers(wire)
    if wire.get("type") == "function_call":
        wire["call_id"] = _tool_id(wire.get("call_id"), settings, ids)
    elif wire.get("type") == "tool_use":
        wire["id"] = _tool_id(wire.get("id"), settings, ids)
    for call in wire.get("tool_calls", []):
        call["id"] = _tool_id(call.get("id"), settings, ids)
    return wire


def _calls(message):
    result = []
    for call in message.get("tool_calls", []):
        function = call.get("function", {})
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise _bad("Portable tool arguments must be a JSON string")
        try:
            if not isinstance(json.loads(arguments), dict):
                raise ValueError("not an object")
        except (ValueError, TypeError) as exc:
            raise _bad("Malformed portable tool arguments") from exc
        result.append((call.get("id"), function.get("name"), arguments))
    return result


def _history(request, settings):
    family = settings.family
    output, ids = [], {}
    for item in request.messages:
        if isinstance(item, CacheBoundary):
            output.append(item)
            continue
        if isinstance(item, LLMResponse):
            replay = _replay(item, family, settings.replay_scope)
            if replay:
                for wire in replay:
                    _check_media(wire, settings.capabilities.media_kinds)
                    wire = _remap_native(wire, settings, ids)
                    output.append(
                        {"role": "assistant", "content": [wire]}
                        if family == "anthropic-messages"
                        else wire
                    )
                # Chat native message is the complete turn. Other native items
                # are independently captured in response order by normalization.
                continue
            item = item.public_message()
        if not isinstance(item, Mapping):
            raise _bad("Native history must contain public messages")
        reject_boundary_dict(item)
        role = item.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise _bad("Unsupported native message role")
        content = _content(item.get("content"), role, family, settings.capabilities.media_kinds)
        if role == "tool":
            identifier = _tool_id(item.get("tool_call_id"), settings, ids)
            if family == "openai-chat":
                output.append({"role": "tool", "tool_call_id": identifier, "content": content})
            elif family == "openai-responses":
                output.append(
                    {
                        "type": "function_call_output",
                        "call_id": identifier,
                        "output": item.get("content") or ""
                        if isinstance(item.get("content"), (str, type(None)))
                        else content,
                    }
                )
            else:
                output.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": identifier, "content": content}
                        ],
                    }
                )
            continue
        calls = _calls(item)
        if family == "openai-chat":
            message = {"role": role, "content": content}
            if calls:
                message["tool_calls"] = [
                    {
                        "id": _tool_id(identifier, settings, ids),
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                    for identifier, name, arguments in calls
                ]
            output.append(message)
        elif family == "openai-responses":
            if content:
                output.append({"role": role, "content": content})
            for identifier, name, arguments in calls:
                output.append(
                    {
                        "type": "function_call",
                        "call_id": _tool_id(identifier, settings, ids),
                        "name": name,
                        "arguments": arguments,
                    }
                )
        else:
            for identifier, name, arguments in calls:
                # Anthropic accepts an object, so only this family deliberately
                # loses JSON whitespace. Portable captured strings remain intact.
                content.append(
                    {
                        "type": "tool_use",
                        "id": _tool_id(identifier, settings, ids),
                        "name": name,
                        "input": json.loads(arguments),
                    }
                )
            output.append({"role": role, "content": content})
    return output


def _tools(request, family):
    tools = []
    for tool in request.tools:
        strict_schema = tool.get_parameter_schema(strict=True)
        strict = _strict_schema_valid(strict_schema)
        schema = strict_schema if strict else tool.get_parameter_schema(strict=False)
        if family == "anthropic-messages":
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": schema,
                    "strict": strict,
                }
            )
        else:
            function = {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
                "strict": strict,
            }
            tools.append(
                {"type": "function", "function": function}
                if family == "openai-chat"
                else {"type": "function", **function}
            )
    choice = request.tool_choice
    if choice not in {"auto", "required", "none"} and choice not in {
        tool.name for tool in request.tools
    }:
        raise _bad("Named tool choice must name a declared tool")
    if not tools:
        if choice not in {"auto", "none"}:
            raise _bad("Tool choice requires tools")
        return {}
    if family == "anthropic-messages":
        wire_choice = (
            {"type": {"required": "any"}.get(choice, choice)}
            if choice in {"auto", "required", "none"}
            else {"type": "tool", "name": choice}
        )
    else:
        wire_choice = (
            choice
            if choice in {"auto", "required", "none"}
            else {"type": "function", "function": {"name": choice}}
            if family == "openai-chat"
            else {"type": "function", "name": choice}
        )
    return {"tools": tools, "tool_choice": wire_choice}


def _json_object_message(request, settings, family):
    """JSON mode's final schema instruction, in this route's instruction role."""
    message = {**json_object_instruction(request.output_type), "role": settings.instruction_role}
    if family == "openai-responses":
        message["content"] = _content(message["content"], message["role"], family, frozenset())
    return message


def _project_request(request, settings):
    family = settings.family
    params = _thaw(settings.selected)
    # Runtime identity is not a request/setting/route carrier. Only these SDK
    # families declare a representable top-level cache key; configured wins.
    from nooa.unifiedllm._call_identity import current_call_identity

    identity = current_call_identity()
    if identity is not None and family in {"openai-chat", "openai-responses"}:
        params.setdefault("prompt_cache_key", identity)
    params["model"] = settings.model
    params.update(_tools(request, family))
    if (
        family == "anthropic-messages"
        and params.get("thinking", {}).get("type") in {"enabled", "adaptive"}
        and params.get("tool_choice", {}).get("type") in {"any", "tool"}
    ):
        raise LLMFailure(FailureKind.CONFIG, "Native thinking cannot force a tool choice")
    history = _history(request, settings)
    json_object = request.output_type is not None and settings.output_format == "json_object"
    if family in {"openai-chat", "openai-responses"}:
        history = map_instruction_roles(
            history, settings.instruction_role, settings.capabilities.developer_role
        )
    if family == "openai-chat":
        history, _, _ = apply_cache_policy(history, None, responses=False)
        if json_object:
            history = [*history, _json_object_message(request, settings, family)]
        if settings.cache_breakpoint is not None:
            raise LLMFailure(
                FailureKind.CONFIG, "Native Chat has no explicit cache-boundary extension"
            )
        params["messages"] = history
    elif family == "openai-responses":
        # Ordered system/developer items remain in input; never hoist a mid-turn
        # instruction. This is the SDK's representable chronology-preserving path.
        history, _, explicit = apply_cache_policy(
            history, settings.cache_breakpoint, responses=True
        )
        if explicit:
            enable_openai_explicit_cache(params)
        if json_object:
            history = [*history, _json_object_message(request, settings, family)]
        params["input"] = history
    else:
        history, _, _ = apply_cache_policy(history, settings.cache_breakpoint, responses=False)
        system, messages = [], []
        for item in history:
            if item.get("role") in {"system", "developer"}:
                if messages:
                    raise _bad(
                        "Anthropic cannot represent mid-history instructions without changing chronology"
                    )
                system.extend(item["content"])
            elif messages and messages[-1]["role"] == item["role"]:
                messages[-1] = {
                    "role": item["role"],
                    "content": messages[-1]["content"] + item["content"],
                }
            else:
                messages.append(item)
        params["messages"] = messages
        if system:
            params["system"] = system
    if request.output_type is not None:
        if family == "anthropic-messages":
            params["output_config"] = {
                **params.get("output_config", {}),
                "format": anthropic_output_format(request.output_type),
            }
        elif json_object:
            if family == "openai-chat":
                params["response_format"] = {"type": "json_object"}
            else:
                params["text"] = {**params.get("text", {}), "format": {"type": "json_object"}}
        else:
            strict_schema = _clean_schema(request.output_type.model_json_schema(), strict=True)
            strict = _strict_schema_valid(strict_schema)
            schema = strict_schema if strict else request.output_type.model_json_schema()
            format = {
                "type": "json_schema",
                "name": request.output_type.__name__,
                "schema": schema,
                "strict": strict,
            }
            if family == "openai-chat":
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {key: value for key, value in format.items() if key != "type"},
                }
            else:
                params["text"] = {**params.get("text", {}), "format": format}
    return params
