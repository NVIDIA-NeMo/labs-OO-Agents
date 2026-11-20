# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public runtime dispatch and finite, temporary settings views.

The legacy branch is intentional for Gym and existing duck-typed clients. A
public client always receives the strict four-field request, never legacy kwargs.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from nooa.runtime.model_binding import derive_client, public_base
from nooa.unifiedllm._call_identity import call_identity
from nooa.unifiedllm.request import LLMRequest

SETTINGS = frozenset(
    {
        "temperature",
        "top_p",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "reasoning_level",
        "max_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "timeout",
        "reasoning",
        "thinking",
        "output_config",
        "reasoning_effort",
    }
)
REQUEST_FIELDS = frozenset(
    {"tools", "tool_choice", "output_model", "output_type", "prompt_cache_key"}
)


def model_label(client: Any) -> str:
    base = public_base(client)
    if base is not None:
        if base._lease is None:
            return "custom"
        from nooa.unifiedllm.adapters._view_limits import _loaded_runtime_metadata

        return _loaded_runtime_metadata(base)["label"]
    return getattr(client, "model", "") or ""


def context_limits(client: Any, params: dict | None = None, *, fallback_reserve=0):
    base = public_base(client)
    if base is not None:
        if base._lease is None:
            from nooa.unifiedllm.limits import ContextLimits

            facts = base.capabilities
            return ContextLimits(facts.context_window, fallback_reserve, True)
        from nooa.unifiedllm.limits import context_limits_for

        return context_limits_for(base, fallback_reserve=fallback_reserve)
    from nooa.unifiedllm.limits import context_limits_for

    return context_limits_for(client, params, fallback_reserve=fallback_reserve)


_accepted_generation = ContextVar("accepted_generation", default=None)


@dataclass(frozen=True)
class AcceptedCall:
    """Private dispatched provenance, never stored on mutable middleware context."""

    client: Any
    messages: Any
    params: dict


class CallViews:
    """Retain derived views until the whole generation (including recovery) exits."""

    def __init__(self, owner=None):
        self._owner = owner
        self._owned: list[Any] = []
        self._bindings: list[Any] = []
        self._cleanup: asyncio.Task | None = None
        self._strict = False
        self.accepted: AcceptedCall | None = None

    def _retain(self, binding):
        agent = getattr(self._owner, "agent", None)
        if agent is None:
            return
        retirement = getattr(agent, "_retired_bindings", None)
        if retirement is None:
            # A standalone runtime carries obligations back to its invocation;
            # the temporary stub is not an independent lifecycle owner.
            retirement = getattr(self._owner, "_dispatch_bindings", None)
            if retirement is None:
                retirement = self._owner._dispatch_bindings = []
        retirement.append(binding)

        def settled():
            if binding in retirement:
                retirement.remove(binding)
            callback = getattr(agent, "_cleanup_settled", None)
            if callback is not None:
                callback()

        binding._on_settled = settled

    def _binding_for(self, client):
        for binding in self._bindings:
            if binding._current is client:
                return binding
        agent = getattr(self._owner, "agent", None)
        for binding in getattr(agent, "_model_bindings", {}).values():
            matched = binding._binding_for(client)
            if matched is not None:
                return matched
        return None

    def prepare(self, client: Any, params: dict[str, Any]):
        base = public_base(client)
        if self._strict:
            from nooa.unifiedllm.adapters._loader import _check_derivable

            if base is None:
                raise TypeError("A loaded public generation cannot use a legacy replacement")
            try:
                _check_derivable(base)
            except Exception as exc:
                raise TypeError(
                    "A loaded public generation requires an unchanged loaded view"
                ) from exc
        elif base is not None and base._lease is not None:
            self._strict = True
            # Validate even when there are no settings to derive.
            from nooa.unifiedllm.adapters._loader import _check_derivable

            try:
                _check_derivable(base)
            except Exception as exc:
                raise TypeError("Public generation requires an unchanged loaded view") from exc
        if base is None:
            return client
        unknown = params.keys() - SETTINGS - REQUEST_FIELDS
        if unknown:
            raise TypeError("Unsupported public runtime request option")
        selected = {key: value for key, value in params.items() if key in SETTINGS}
        if not selected:
            return client
        view = derive_client(client, **selected)
        from nooa.runtime.model_binding import ModelBinding

        binding = ModelBinding(view, owned=True)
        self._bindings.append(binding)
        self._retain(binding)
        self._owned.append(view)
        if self._owner is not None and not getattr(self._owner, "_runtime_settings_warned", False):
            import warnings

            warnings.warn(
                "Generation settings are deprecated; configure a selected model view instead",
                DeprecationWarning,
                stacklevel=3,
            )
            self._owner._runtime_settings_warned = True
        return view

    async def aclose(self):
        if not self._bindings and self._cleanup is None:
            return
        if self._cleanup is None:
            from nooa.runtime.model_binding import _cleanup_result

            async def close_all():
                failure = None
                for binding in reversed(self._bindings):
                    try:
                        binding._drop_reference(owner=True)
                    except BaseException as error:
                        if failure is None:
                            failure = error
                    # A held native receipt belongs to explicit owner cleanup,
                    # never to the prompt logical cancellation boundary.
                    if not binding._records:
                        try:
                            await binding._join_cleanup()
                            binding._notify_settled()
                        except BaseException as error:
                            if failure is None:
                                failure = error
                self._bindings.clear()
                self._owned.clear()
                if failure is not None:
                    raise failure

            self._cleanup = asyncio.create_task(_cleanup_result(close_all))
        from nooa.runtime.model_binding import _join_cleanup_tasks

        await _join_cleanup_tasks((self._cleanup,))


def freeze_request(client, messages, params):
    """Validate and freeze public call input before tracing/user hooks."""
    if public_base(client) is None:
        return None
    import copy

    unknown = params.keys() - SETTINGS - REQUEST_FIELDS
    if unknown:
        raise TypeError("Unsupported public runtime request option")
    return LLMRequest(
        messages=copy.deepcopy(messages),
        tools=copy.deepcopy(params.get("tools") or []),
        tool_choice=params.get("tool_choice", "auto"),
        output_type=params.get("output_type", params.get("output_model")),
    )


async def dispatch(client, messages, params, *, request=None):
    if public_base(client) is None:
        return await client.acall(messages, **params)
    request = request if request is not None else freeze_request(client, messages, params)
    # The context is established before LLM creates any provider task. Native
    # projection and LiteLLM's existing affinity helper consume it separately.
    from nooa.runtime.model_binding import ModelBinding, _CaptureRecord
    from nooa.unifiedllm._transport_lifetime import _dispatch_registrar

    registrar = _dispatch_registrar.get()
    views = _accepted_generation.get()
    agent = getattr(getattr(views, "_owner", None), "agent", None)
    with call_identity(params.get("prompt_cache_key")):
        if (isinstance(registrar, _CaptureRecord) and registrar.view is client) or agent is None:
            # Keep the matching lexical owner, notably an isolated summary fork.
            return await client.acall(request)
        binding = views._binding_for(client)
        temporary = binding is None
        if temporary:
            binding = ModelBinding(client)
            views._retain(binding)
        try:
            async with binding._capture_exact(client, owner=agent):
                return await client.acall(request)
        finally:
            if temporary:
                binding._drop_reference(owner=True)
