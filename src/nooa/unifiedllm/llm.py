# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public, provider-neutral request facade.

This incremental shell owns portable request preflight and calls the shared
logical/attempt executors. Built-ins normalize provider failures; the shell types unknown terminal errors.
Logical accounting and tracing remain separate migration steps. Wire validation and
opaque native replay belong to adapters. Third-party adapters must make one
wire attempt per send; the shell cannot discover hidden SDK retries.
"""

from collections.abc import Mapping
from dataclasses import replace
from threading import RLock
from typing import Literal

from pydantic import BaseModel

from nooa.llm_types import CacheBoundary, LLMResponse
from nooa.media_capabilities import MediaCapabilities
from nooa.media_inputs import attachment_kind

from ._accounting import _logical_account_scope
from ._route_calibration import _calibration_adapter
from .adapters.base import Adapter, Capabilities
from .errors import FailureKind, LLMFailure
from .execution import _public_terminal, check_empty_response, execute_async, execute_sync
from .logical import logical_async, logical_sync
from .media_inputs import filter_media_messages
from .request import LLMRequest
from .retry_config import RetryConfig
from .tools import Tool

_MEDIA_KINDS = frozenset({"image", "pdf", "audio", "video"})


def _snapshot_capabilities(facts: Capabilities) -> Capabilities:
    """Freeze known route facts without discovery or borrowing another route."""
    if not isinstance(facts, Capabilities):
        raise LLMFailure(FailureKind.CONFIG, "adapter.capabilities must be Capabilities")
    try:
        media = frozenset(facts.media_kinds)
        levels = None if facts.reasoning_levels is None else tuple(facts.reasoning_levels)
    except TypeError as exc:
        raise LLMFailure(FailureKind.CONFIG, "Invalid capability collections", cause=exc) from exc
    if not media <= _MEDIA_KINDS:
        raise LLMFailure(FailureKind.CONFIG, "Unknown capability media kind")
    if levels is not None and (
        isinstance(facts.reasoning_levels, str)
        or any(not isinstance(level, str) or not level for level in levels)
        or len(set(levels)) != len(levels)
    ):
        raise LLMFailure(FailureKind.CONFIG, "Invalid capability reasoning levels")
    if facts.tools is not None and type(facts.tools) is not bool:
        raise LLMFailure(FailureKind.CONFIG, "Capability tools must be bool or unknown (None)")
    for name in ("context_window", "max_output_tokens"):
        value = getattr(facts, name)
        if value is not None and (type(value) is not int or value <= 0):
            raise LLMFailure(FailureKind.CONFIG, f"Capability {name} must be positive or unknown")
    if facts.developer_role is not None and type(facts.developer_role) is not bool:
        raise LLMFailure(
            FailureKind.CONFIG, "Capability developer_role must be bool or unknown (None)"
        )
    if facts.media_capabilities is not None and not isinstance(
        facts.media_capabilities, MediaCapabilities
    ):
        raise LLMFailure(
            FailureKind.CONFIG, "Capability media_capabilities must be MediaCapabilities or None"
        )
    # Independently advertised ceilings need not be ordered. Validate the
    # selected request budget below, rather than rejecting usable route facts.
    if (
        type(facts) is Capabilities
        and type(facts.media_kinds) is frozenset
        and (facts.reasoning_levels is None or type(facts.reasoning_levels) is tuple)
    ):
        return facts  # Already immutable, still defensively validated above.
    return replace(facts, media_kinds=media, reasoning_levels=levels)


def _portable_media(messages):
    """Only the mapping attachment slots supported by filter_media_messages.

    Yields (kind, block); ``block`` is None for an assistant ``audio`` field.
    Never traverse Tool arguments, vendor extensions or opaque LLMResponse data.
    This does not implement the design's future Message/MediaPart model.
    """
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        if (kind := attachment_kind(message)) is not None:
            yield kind, message
        if message.get("role") == "assistant" and message.get("audio") is not None:
            yield "audio", None
        slot = "output" if message.get("type") == "function_call_output" else "content"
        content = message.get(slot)
        if isinstance(content, (list, tuple)):
            for block in content:
                if isinstance(block, Mapping) and (kind := attachment_kind(block)) is not None:
                    yield kind, block


class LLM:
    """One loaded route wrapping only capabilities/send/send_sync.

    Unknown capability facts are not refusals. To switch routes, construct a new
    LLM with the new adapter; there are no per-call route overrides. Requests
    retain stored history and native response identities. Loader views own leases;
    directly supplied adapters remain borrowed.
    """

    def __init__(
        self,
        adapter: Adapter,
        *,
        on_unsupported_media: Literal["note", "error"] = "note",
        retry_config: RetryConfig | None = None,
    ) -> None:
        if on_unsupported_media not in ("note", "error"):
            raise LLMFailure(FailureKind.CONFIG, "on_unsupported_media must be note or error")
        if retry_config is not None and not isinstance(retry_config, RetryConfig):
            raise LLMFailure(FailureKind.CONFIG, "retry_config must be RetryConfig or None")
        self._view_lock = RLock()
        self._lease = None
        self._closed = False
        self._adapter = adapter
        try:
            facts = adapter.capabilities
        except LLMFailure:
            raise
        except Exception as exc:
            raise LLMFailure(
                FailureKind.CONFIG, "Cannot read adapter capabilities", cause=exc
            ) from exc
        self._capabilities = _snapshot_capabilities(facts)
        self._media_policy = on_unsupported_media
        self._retry_config = retry_config

    @property
    def adapter(self) -> Adapter:
        return self._adapter

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    def _prepare(self, request: LLMRequest) -> LLMRequest:
        if not isinstance(request, LLMRequest):
            raise LLMFailure(FailureKind.BAD_REQUEST, "call requires an LLMRequest")
        if not isinstance(request.messages, list) or any(
            not isinstance(message, (Mapping, LLMResponse, CacheBoundary))
            for message in request.messages
        ):
            raise LLMFailure(FailureKind.BAD_REQUEST, "messages must be mapping history entries")
        if not isinstance(request.tools, list) or any(
            not isinstance(tool, Tool) for tool in request.tools
        ):
            raise LLMFailure(FailureKind.BAD_REQUEST, "tools must be a list of Tool")
        if request.output_type is not None and (
            not isinstance(request.output_type, type)
            or not issubclass(request.output_type, BaseModel)
        ):
            raise LLMFailure(FailureKind.BAD_REQUEST, "output_type must be a BaseModel type")
        choice = request.tool_choice
        if not isinstance(choice, str):
            raise LLMFailure(FailureKind.BAD_REQUEST, "tool_choice must be a neutral string")
        if choice == "required" and not request.tools:
            raise LLMFailure(FailureKind.BAD_REQUEST, "required tool_choice needs tools")
        if choice not in ("auto", "required", "none") and choice not in {
            tool.name for tool in request.tools
        }:
            raise LLMFailure(FailureKind.BAD_REQUEST, "tool_choice names no tool of this request")
        facts = self.capabilities
        if facts.tools is False and (request.tools or choice not in ("auto", "none")):
            raise LLMFailure(FailureKind.BAD_REQUEST, "This route does not support tools")
        media_caps = facts.media_capabilities
        if self._media_policy == "error":
            for kind, block in _portable_media(request.messages):
                reason = None
                if kind in facts.media_kinds and media_caps is not None and block is not None:
                    reason = media_caps.refusal(block, kind)
                if kind not in facts.media_kinds or reason:
                    raise LLMFailure(
                        FailureKind.MEDIA_REJECTED,
                        f"Unsupported {kind} media"
                        + (f": this model does not accept {reason}" if reason else ""),
                        media_kind=kind if kind in _MEDIA_KINDS else None,
                    )
        messages = filter_media_messages(
            request.messages, tuple(facts.media_kinds), capabilities=media_caps
        )
        return request.model_copy(update={"messages": messages})

    def _acquire(self):
        with self._view_lock:
            if self._closed:
                raise LLMFailure(FailureKind.CONFIG, "LLM view is closed")
            lease = self._lease
            if lease is not None:
                lease.acquire()
            return lease

    def close(self):
        """Release this view, not other live views or in-flight calls.

        Inside a running event loop use aclose to await final async disposal.
        Directly supplied adapters and caller-supplied SDKs remain borrowed.
        """
        with self._view_lock:
            if not self._closed:
                self._closed = True
                if self._lease is not None:
                    self._lease.release()

    async def aclose(self):
        """Idempotently release this view, awaiting final owned pool disposal."""
        with self._view_lock:
            if not self._closed:
                self._closed = True
                if self._lease is not None:
                    self._lease.release()
            lease = self._lease
        if lease is not None:
            await lease.join_disposal()

    def __enter__(self):
        if self._closed:
            raise LLMFailure(FailureKind.CONFIG, "LLM view is closed")
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *args):
        await self.aclose()

    def call(self, request: LLMRequest) -> LLMResponse:
        """Validate a request-local view and run the common logical executor."""
        with _logical_account_scope(
            messages=request.messages if isinstance(request, LLMRequest) else ()
        ):
            lease = self._acquire()
            try:
                return logical_sync(
                    lambda: self._prepare(request),
                    lambda prepared: execute_sync(
                        _calibration_adapter(self.adapter),
                        prepared,
                        self._retry_config,
                        check_empty=check_empty_response,
                        _public_failures=True,
                    ),
                )
            except Exception as error:
                normalized = _public_terminal(error)
                if normalized is error:
                    raise
                raise normalized from normalized.__cause__
            finally:
                if lease is not None:
                    lease.release()

    async def acall(self, request: LLMRequest) -> LLMResponse:
        """Async call, sharing preflight, retry ownership and cancellation semantics."""
        with _logical_account_scope(
            messages=request.messages if isinstance(request, LLMRequest) else ()
        ):
            lease = self._acquire()
            try:
                return await logical_async(
                    lambda: self._prepare(request),
                    lambda prepared: execute_async(
                        _calibration_adapter(self.adapter, lease),
                        prepared,
                        self._retry_config,
                        check_empty=check_empty_response,
                        _public_failures=True,
                    ),
                )
            except Exception as error:
                normalized = _public_terminal(error)
                if normalized is error:
                    raise
                raise normalized from normalized.__cause__
            finally:
                if lease is not None:
                    await lease.release_async()
