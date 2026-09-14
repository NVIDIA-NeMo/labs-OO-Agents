# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Official SDK dispatch behind the existing UnifiedLLM message contract.

The OpenAI SDK sends Chat and Responses requests. The Anthropic SDK sends
Messages requests; its adapter translates public Chat fields and normalizes the
reply for the same capture code used by legacy calls. Neither path guesses
reasoning settings from model names. Unknown request fields go in extra_body,
so registry-declared settings reach compatible servers unchanged.
"""

import inspect
import json
import re
from typing import Any

import httpx
from pydantic import BaseModel

from nooa.tracing._llm_hooks import capture_async_request, capture_request

from .http_config import HttpConfig

# These are routing prefixes, not a model catalogue. Other model ids, including
# ids containing slashes, are sent verbatim. Nonstandard routes require a URL.
_PREFIXES = {
    "openai",
    "anthropic",
    "azure",
    "deepseek",
    "nvidia_nim",
    "openrouter",
    "hosted_vllm",
    "together_ai",
    "xai",
    "gemini",
}
_LEGACY_OPTIONS = {"drop_params", "allowed_openai_params", "additional_drop_params"}


def _blocks(content: Any) -> list[dict]:
    if content is None or content == "":
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list) or not all(isinstance(b, dict) for b in content):
        raise ValueError("Message content must be text or a list of blocks")
    result = []
    for block in content:
        block = dict(block)
        kind = block.get("type")
        if kind in {"input_text", "output_text"}:
            block["type"] = "text"
        elif kind == "image_url":
            image = block.pop("image_url")
            url = image["url"] if isinstance(image, dict) else image
            if url.startswith("data:"):
                media, data = url[5:].split(";base64,", 1)
                source = {"type": "base64", "media_type": media, "data": data}
            else:
                source = {"type": "url", "url": url}
            block.update(type="image", source=source)
        elif kind == "file":
            file = block.pop("file")
            if "file_data" not in file:
                raise ValueError(
                    "Anthropic file blocks require file_data; use a native document for file ids"
                )
            media, data = file["file_data"][5:].split(";base64,", 1)
            block.update(
                type="document", source={"type": "base64", "media_type": media, "data": data}
            )
        result.append(block)
    return result


def anthropic_request(params: dict) -> dict:
    """Translate projected Chat messages, keeping signed blocks and cache markers."""
    result = dict(params)
    messages, system = [], []
    for message in result.pop("messages"):
        role = message["role"]
        content = _blocks(message.get("content"))
        if role in {"system", "developer"}:
            if messages:
                raise ValueError("Anthropic accepts only leading system messages")
            system.extend(content)
            continue
        if role == "tool":
            content = [
                {"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": content}
            ]
            role = "user"
        elif role == "assistant":
            content = [*message.get("thinking_blocks", []), *content]
            if message.get("reasoning_content"):
                content.insert(0, {"type": "text", "text": message["reasoning_content"]})
            for call in message.get("tool_calls") or []:
                content.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "input": json.loads(call["function"]["arguments"]),
                    }
                )
        elif role != "user":
            raise ValueError(f"Unsupported Anthropic message role: {role!r}")
        if content:
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].extend(content)
            else:
                messages.append({"role": role, "content": content})
    result["messages"] = messages
    if system:
        result["system"] = system
    if "tools" in result:
        result["tools"] = [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"]["parameters"],
            }
            for t in result["tools"]
        ]
    parallel = result.pop("parallel_tool_calls", None)
    choice = result.pop("tool_choice", None)
    if choice is not None:
        if isinstance(choice, str):
            choice = {"type": "any" if choice == "required" else choice}
        elif choice.get("type") == "function":
            choice = {"type": "tool", "name": choice["function"]["name"]}
        else:
            choice = dict(choice)
        if parallel is False and choice.get("type") != "none":
            choice["disable_parallel_tool_use"] = True
        result["tool_choice"] = choice
    elif parallel is False and result.get("tools"):
        result["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
    if "response_format" in result:
        fmt = result.pop("response_format")["json_schema"]
        result["output_config"] = {
            **result.get("output_config", {}),
            "format": {"type": "json_schema", "schema": fmt["schema"]},
        }
    # A cache shard key has meaning only on the OpenAI APIs.
    result.pop("prompt_cache_key", None)
    if "max_tokens" not in result:
        raise ValueError("Anthropic direct calls require an explicit max_tokens reply limit")
    return result


def anthropic_response(response):
    """Normalize SDK Messages blocks without copying native text into metadata."""
    from openai.types.chat import ChatCompletion

    text, thinking, calls = [], [], []
    for part in response.content:
        block = part.model_dump(exclude_none=True)
        kind = block["type"]
        if kind == "text":
            text.append(block["text"])
        elif kind in {"thinking", "redacted_thinking"}:
            thinking.append(block)
        elif kind == "tool_use":
            calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"], ensure_ascii=False),
                    },
                }
            )
        else:
            raise ValueError(f"Unsupported Anthropic response block: {kind!r}")
    reason = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
    }.get(response.stop_reason, "content_filter")
    usage = response.usage.model_dump(exclude_none=True)
    usage["prompt_tokens"] = sum(
        usage.get(k, 0)
        for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    )
    usage["input_tokens"] = usage["prompt_tokens"]
    usage["completion_tokens"] = usage["output_tokens"]
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return ChatCompletion.model_validate(
        {
            "id": response.id,
            "model": response.model,
            "created": 0,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": reason,
                    "message": {
                        "role": "assistant",
                        "content": "".join(text),
                        "tool_calls": calls,
                        "thinking_blocks": thinking,
                    },
                }
            ],
            "usage": usage,
        }
    )


class DirectTransport:
    """Own one HTTP pool per client; SDK retries are off, NOOA owns retries.

    SDK wrappers are per request so a per-call URL or key cannot accidentally
    reuse an earlier client's bound credentials. They borrow our HTTP pools;
    only close/aclose below closes those pools. Errors never invoke LiteLLM.
    """

    sync_client = None
    async_client = None

    def __init__(self, model, api_style, replay_vendor, config, http_config: HttpConfig):
        if api_style not in {"chat", "responses", "anthropic"}:
            raise ValueError("api_style must be chat, responses, or anthropic")
        self.api_style = api_style
        if replay_vendor is not None and (
            not isinstance(replay_vendor, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", replay_vendor)
        ):
            raise ValueError("replay_vendor must be a nonempty lowercase provider name")
        self.replay_vendor = replay_vendor
        self.route(model, config)
        settings = self._http_settings(http_config)
        self.httpx_sync = httpx.Client(**settings)
        self.httpx_async = httpx.AsyncClient(**settings)
        self.httpx_sync.event_hooks["request"].append(capture_request)
        self.httpx_async.event_hooks["request"].append(capture_async_request)

    @staticmethod
    def _http_settings(config):
        return {"timeout": config.to_httpx_timeout(), "limits": config.to_httpx_limits()}

    def route(self, model, params):
        prefix, separator, rest = model.partition("/")
        recognized = bool(separator and prefix in _PREFIXES)
        wire_model = rest if recognized else model
        vendor = self.replay_vendor or (
            prefix if recognized else ("anthropic" if self.api_style == "anthropic" else "openai")
        )
        if params.get("custom_llm_provider") or params.get("client"):
            raise ValueError(
                "direct transport uses api_style, replay_vendor and api_base, not client/custom_llm_provider"
            )
        if (
            recognized
            and prefix not in {"openai", "anthropic"}
            and not (params.get("api_base") or params.get("base_url"))
        ):
            raise ValueError(
                "This direct route requires api_base; SDKs do not resolve provider aliases"
            )
        return wire_model, vendor

    def _request(self, params, *, asynchronous):
        from openai import AsyncOpenAI, OpenAI
        from openai.lib._parsing._completions import type_to_response_format_param

        body = dict(params)
        if body.get("additional_drop_params"):
            raise ValueError(
                "direct transport does not silently drop fields; remove additional_drop_params"
            )
        if body.get("num_retries", 0) != 0:
            raise ValueError("Use retry_config for direct transport retries, not num_retries")
        body.pop("num_retries", None)
        model, _ = self.route(body.pop("model"), body)
        body["model"] = model
        if body.pop("stream", False):
            raise ValueError("direct transport currently requires stream=False")
        for key in _LEGACY_OPTIONS | {"context_window"}:
            body.pop(key, None)
        api_base = body.pop("base_url", None) or body.pop("api_base", None)
        body.pop("api_base", None)
        api_key = body.pop("api_key", None)
        kwargs = {
            "max_retries": 0,
            "http_client": self.httpx_async if asynchronous else self.httpx_sync,
        }
        if api_base:
            if self.api_style == "anthropic":
                api_base = api_base.rstrip("/").removesuffix("/v1")
            kwargs["base_url"] = api_base
        if api_key is not None:
            kwargs["api_key"] = api_key
        fmt = body.get("response_format")
        if isinstance(fmt, type) and issubclass(fmt, BaseModel):
            body["response_format"] = type_to_response_format_param(fmt)
        if self.api_style == "anthropic":
            from anthropic import Anthropic, AsyncAnthropic

            body = anthropic_request(body)
            client = (AsyncAnthropic if asynchronous else Anthropic)(**kwargs)
            method = client.messages.create
        else:
            client = (AsyncOpenAI if asynchronous else OpenAI)(**kwargs)
            if self.api_style == "responses":
                if "max_tokens" in body:
                    if "max_output_tokens" in body:
                        raise ValueError("Use either max_tokens or max_output_tokens, not both")
                    body["max_output_tokens"] = body.pop("max_tokens")
                if "text_format" in body:
                    fmt = type_to_response_format_param(body.pop("text_format"))["json_schema"]
                    body["text"] = {"format": {"type": "json_schema", **fmt}}
                method = client.responses.create
            else:
                method = client.chat.completions.create
        # The SDK signature identifies transport kwargs. Provider extensions
        # still reach the server verbatim via the SDK's explicit escape hatch.
        accepted = inspect.signature(method).parameters
        extra = dict(body.pop("extra_body", {}) or {})
        for key in list(body):
            if key not in accepted:
                extra[key] = body.pop(key)
        if extra:
            body["extra_body"] = extra
        return method, body

    def call(self, params):
        method, body = self._request(params, asynchronous=False)
        response = method(**body)
        return anthropic_response(response) if self.api_style == "anthropic" else response

    async def acall(self, params):
        method, body = self._request(params, asynchronous=True)
        response = await method(**body)
        return anthropic_response(response) if self.api_style == "anthropic" else response

    def close(self):
        self.httpx_sync.close()

    async def aclose(self):
        self.close()
        await self.httpx_async.aclose()
