# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One tracing boundary for both transports; HTTP hooks observe the actual body.

Request hooks record sanitized wire JSON, never headers. Completion records the
public LLMResponse projection, never the SDK object. Journals and spans share
this boundary, so enabling telemetry neither imports a provider nor changes
which transport sends a request. Telemetry errors cannot retry a paid call.
"""

import functools
import inspect
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from uuid import uuid4

from opentelemetry import trace

from ._secret_scrubber import scrub_value

logger = logging.getLogger(__name__)
callbacks: list = []
_active: ContextVar = ContextVar("nooa_llm_trace", default=None)


def _tracer():
    return trace.get_tracer("nooa.unifiedllm")


def _notify(method, *args):
    for callback in tuple(callbacks):
        try:
            getattr(callback, method)(*args)
        except Exception as exc:
            logger.warning(
                "LLM journal callback failed (%s); the model call is unaffected", type(exc).__name__
            )


def _messages(span, direction, messages):
    for index, message in enumerate(messages):
        prefix = f"llm.{direction}_messages.{index}.message"
        if role := message.get("role"):
            span.set_attribute(prefix + ".role", role)
        if content := message.get("content"):
            span.set_attribute(
                prefix + ".content",
                content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
            )
        for i, call in enumerate(message.get("tool_calls") or []):
            base = f"{prefix}.tool_calls.{i}.tool_call"
            span.set_attribute(base + ".id", call["id"])
            span.set_attribute(base + ".function.name", call["function"]["name"])
            span.set_attribute(base + ".function.arguments", call["function"]["arguments"])


def capture_request(request):
    """httpx sync hook, also used by the async hook after the body is prepared."""
    state = _active.get()
    if state is None:
        return
    try:
        body, _ = scrub_value(json.loads(request.content))
        if not isinstance(body, dict):
            return
        span, metadata = state
        metadata["model"] = body.get("model", metadata["model"])
        messages = body.get("messages", body.get("input", []))
        if "contents" in body:
            messages = [
                {
                    "role": "assistant"
                    if item.get("role") == "model"
                    else item.get("role", "user"),
                    "content": item.get("parts", []),
                }
                for item in body["contents"]
            ]
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        leading = body.get(
            "system",
            body.get("instructions", body.get("system_instruction", body.get("systemInstruction"))),
        )
        if leading:
            messages = [{"role": "system", "content": leading}, *messages]
        span.set_attribute("llm.model_name", metadata["model"])
        # SDKs can order JSON keys differently for the same request. Stable
        # serialization lets traces compare the fields without that noise.
        span.set_attribute("input.value", json.dumps(body, ensure_ascii=False, sort_keys=True))
        span.set_attribute("input.mime_type", "application/json")
        _messages(span, "input", messages)
        invocation = {
            k: v
            for k, v in body.items()
            if k not in {"messages", "input", "system", "instructions", "tools"}
        }
        span.set_attribute(
            "llm.invocation_parameters", json.dumps(invocation, ensure_ascii=False, sort_keys=True)
        )
        for i, tool in enumerate(body.get("tools") or []):
            span.set_attribute(
                f"llm.tools.{i}.tool.json_schema",
                json.dumps(tool, ensure_ascii=False, sort_keys=True),
            )
        # A NOOA retry belongs to the same journal call. Re-notifying consumes
        # the one-shot block-reference sideband and replaces it with raw input.
        if not metadata.get("input_recorded"):
            _notify("log_pre_api_call", metadata["model"], messages, metadata)
            metadata["input_recorded"] = True
    except Exception as exc:
        logger.warning("Could not record LLM request (%s)", type(exc).__name__)


async def capture_async_request(request):
    capture_request(request)


@contextmanager
def _call(model):
    start = time.time()
    metadata = {"model": model, "litellm_call_id": str(uuid4())}
    with _tracer().start_as_current_span(
        "llm.call",
        attributes={"openinference.span.kind": "LLM", "nooa.viewer.plugin": "llm_call"},
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        token = _active.set((span, metadata))
        try:

            def finish(response):
                try:
                    public, _ = scrub_value(response.public_message())
                    span.set_attribute("output.value", json.dumps(public, ensure_ascii=False))
                    span.set_attribute("output.mime_type", "application/json")
                    _messages(span, "output", [public])
                    if response.reasoning:
                        span.set_attribute(
                            "llm.reasoning_content", scrub_value(response.reasoning)[0]
                        )
                    if usage := response.usage:
                        for name, value in {
                            "prompt": usage.input_tokens,
                            "completion": usage.output_tokens,
                            "total": usage.total_tokens,
                            "prompt_details.cache_read": usage.cached_input_tokens,
                            "prompt_details.cache_write": usage.cache_write_input_tokens,
                            "completion_details.reasoning": usage.reasoning_tokens,
                        }.items():
                            span.set_attribute("llm.token_count." + name, value)
                        span.set_attribute("llm.cost.total", usage.cost_usd)
                    _notify(
                        "log_success_event",
                        metadata,
                        SimpleNamespace(
                            choices=[SimpleNamespace(message=public)], usage=response.usage
                        ),
                        start,
                        time.time(),
                    )
                except Exception as exc:
                    logger.warning("Could not record LLM outcome (%s)", type(exc).__name__)

            yield finish
        except BaseException as exc:
            # Provider errors may echo requests or credentials. Record the
            # exception type, not its repr or a traceback containing the body.
            span.set_status(trace.Status(trace.StatusCode.ERROR, type(exc).__name__))
            _notify("log_failure_event", metadata, None, start, time.time())
            raise
        finally:
            _active.reset(token)


def trace_llm_call(function):
    """Wrap sync and async UnifiedLLM calls without inspecting their transport."""
    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def async_call(self, *args, **kwargs):
            if _active.get() is not None:
                return await function(self, *args, **kwargs)
            with _call(kwargs.get("model", self.model)) as finish:
                response = await function(self, *args, **kwargs)
                finish(response)
                return response

        return async_call

    @functools.wraps(function)
    def sync_call(self, *args, **kwargs):
        if _active.get() is not None:
            return function(self, *args, **kwargs)
        with _call(kwargs.get("model", self.model)) as finish:
            response = function(self, *args, **kwargs)
            finish(response)
            return response

    return sync_call
