# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Three exact native built-ins; the public shell owns retry and settlement."""

import json
from contextlib import contextmanager

from nooa.unifiedllm._accounting import _current_account
from nooa.unifiedllm._observations import _observe_attempt
from nooa.unifiedllm._route_calibration import _CalibrationAdapter
from nooa.unifiedllm._transport_lifetime import _run_async_provider_call
from nooa.unifiedllm.errors import FailureKind, LLMFailure

from ._native_failures import _failure
from ._native_projection import _project_request
from ._native_response import _normalize_response, _parse_typed
from ._native_transport import _resource
from ._native_usage import _numeric_usage
from ._prepared_limits import _apply_decision


@contextmanager
def _native_entry(owner, model):
    # SDK-independent suppression, scoped to actual entry, never global callbacks.
    if owner is None or owner.writer.record.logical_id is None:
        yield
        return
    account = _current_account.get()
    if account is not None:
        # The shell has already retained the portable effective history. Do not
        # expose signed/encrypted native wire carriers to the journal renderer.
        account.describe_request(model=model)
    from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, attach, detach, set_value

    token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
    try:
        yield
    finally:
        detach(token)


class _NativeAdapter:
    def __init__(self, settings, transport, record):
        self._settings = settings
        self._transport = transport
        self._route_calibration_record = record

    @property
    def capabilities(self):
        return self._settings.capabilities

    def _numeric_participation(self):
        if type(self) not in _BUILTINS or "send" in self.__dict__ or "send_sync" in self.__dict__:
            return None
        return _CalibrationAdapter(self, self._route_calibration_record)

    def _prepare(self, request, writer, decision):
        if writer is not None:
            writer.entering_adapter()
        body = _project_request(request, self._settings)
        # All projected text, system, tools and schema are included. Native SDK
        # framing/tokenization and media are not known: PARTIAL, never full clamp.
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        estimate = max(1, (len(encoded) + 3) // 4)
        if writer is not None:
            writer.projected(estimate)
        if decision is not None:
            _apply_decision(
                body,
                estimate,
                "partial",
                decision,
                responses=self._settings.family == "openai-responses",
            )
        timeout = body.pop("timeout", None)
        if writer is not None:
            writer.transmitting()
        return body, timeout

    def _received(self, raw, request, writer):
        header = self._settings.host_profile.get("cost_header")
        # HTTP acknowledgement is evidence even if the SDK cannot decode the
        # body. Enrich this SAME slot after parsing; the shell folds it once.
        if writer is not None:
            writer.delivered(
                _numeric_usage(self._settings.family, None, raw.headers, cost_header=header)
            )
        try:
            reply = raw.parse()
            usage = _numeric_usage(self._settings.family, reply, raw.headers, cost_header=header)
            if writer is not None:
                writer.delivered(usage)
            response = _normalize_response(
                self._settings.family,
                self._settings.model,
                reply,
                usage,
                replay_scope=self._settings.replay_scope,
            )
            return _parse_typed(response, request.output_type)
        except LLMFailure:
            raise
        except Exception as exc:
            # This is a received-body failure, not a configuration or retryable
            # transport failure. Keep the original cause private and unchanged.
            raise LLMFailure(
                FailureKind.BAD_REQUEST,
                "Native response could not be decoded or normalized",
                cause=exc,
            ) from exc

    async def send(self, request):
        return await self._send(request)

    def send_sync(self, request):
        return self._send_sync(request)

    async def _send(
        self, request, *, _owner=None, _decision=None, _wire_owner=None, _legacy_calibration=False
    ):
        with _observe_attempt(_owner) as writer:
            try:
                body, timeout = self._prepare(request, writer, _decision)
                sdk = self._transport._sdk("async", timeout)

                async def call():
                    with _native_entry(_owner, self._settings.model):
                        raw = await _resource(sdk, self._settings.family).with_raw_response.create(
                            **body
                        )
                        return self._received(raw, request, writer)

                return await _run_async_provider_call(
                    call,
                    request_timeout=self._transport.http_config.request_timeout,
                    _wire_owner=_wire_owner,
                    _attempt_owner=_owner,
                )
            except Exception as exc:
                failure = _failure(exc)
                if writer is not None:
                    writer.failed(failure)
                if failure is exc:
                    raise
                raise failure from exc

    def _send_sync(self, request, *, _owner=None, _decision=None, _legacy_calibration=False):
        with _observe_attempt(_owner) as writer:
            try:
                body, timeout = self._prepare(request, writer, _decision)
                sdk = self._transport._sdk("sync", timeout)
                if writer is not None:
                    writer.admission()
                    writer.entering_provider()
                with _native_entry(_owner, self._settings.model):
                    raw = _resource(sdk, self._settings.family).with_raw_response.create(**body)
                    return self._received(raw, request, writer)
            except Exception as exc:
                failure = _failure(exc)
                if writer is not None:
                    writer.failed(failure)
                if failure is exc:
                    raise
                raise failure from exc


class _OpenAIChatAdapter(_NativeAdapter):
    pass


class _OpenAIResponsesAdapter(_NativeAdapter):
    pass


class _AnthropicMessagesAdapter(_NativeAdapter):
    pass


_BUILTINS = {_OpenAIChatAdapter, _OpenAIResponsesAdapter, _AnthropicMessagesAdapter}
_NATIVE_TYPES = {
    "openai-chat": _OpenAIChatAdapter,
    "openai-responses": _OpenAIResponsesAdapter,
    "anthropic-messages": _AnthropicMessagesAdapter,
}
