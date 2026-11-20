# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-use admission wrapper for UnifiedLLM-compatible clients."""

from __future__ import annotations

from typing import Any

from nooa.unifiedllm.admission import (
    AdmissionControlConfig,
    AdmissionController,
    _admission_controller_scope,
)
from nooa.unifiedllm.unifiedllm import UnifiedLLM


class AdmissionControl(UnifiedLLM):
    """Apply admission to one use of an existing LLM client.

    The wrapper selects a call-scoped controller consumed by ``UnifiedLLM``
    immediately before each provider attempt. It does not acquire around the
    whole logical ``acall``, so retries reacquire capacity and never retain a
    permit during backoff. Synchronous calls are delegated unchanged.
    """

    def __init__(self, llm: Any, config: AdmissionControlConfig):
        if not callable(getattr(llm, "acall", None)):
            raise TypeError("AdmissionControl requires an LLM object with an async acall method")
        if not isinstance(config, AdmissionControlConfig):
            raise TypeError("config must be an AdmissionControlConfig")
        self.base_llm = llm
        self.admission_config = config
        from nooa.unifiedllm.llm import LLM

        api_base = None
        try:
            facade_base = vars(llm).get("base_llm")
        except TypeError:
            facade_base = None
        public = isinstance(llm, LLM) or isinstance(facade_base, LLM)
        if public:
            _validate_public_bridge(self, installed=False)
            if config.controller is None:
                from nooa.unifiedllm.adapters._view_limits import _loaded_endpoint

                api_base = _loaded_endpoint(llm)
        self.controller = config._controller_for(llm, api_base=api_base)
        if public:
            _validate_public_bridge(self)
        self.model = getattr(llm, "model", "")
        # Public injected roots are borrowed; private bridges own their views.
        # Preserve legacy caller-explicit close delegation, not runtime ownership.
        self._owns_base = not public

    @property
    def context_window(self) -> int | None:
        """Return the wrapped client's context window when available."""
        return getattr(self.base_llm, "context_window", None)

    @property
    def retry_config(self) -> Any:
        """Return the wrapped client's retry configuration (the base declares None)."""
        return getattr(self.base_llm, "retry_config", None)

    @property
    def reasoning_levels(self) -> tuple[str, ...] | None:
        """Return the wrapped client's selectable reasoning levels."""
        return getattr(self.base_llm, "reasoning_levels", None)

    @property
    def reasoning_default(self) -> str | None:
        """Return the wrapped client's documented reasoning default."""
        return getattr(self.base_llm, "reasoning_default", None)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate a synchronous call without applying async admission."""
        return self.base_llm.call(*args, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate one logical async call under this wrapper's controller."""
        from nooa.unifiedllm.llm import LLM

        base = vars(self).get("base_llm")
        if isinstance(base, LLM):
            controller = _validate_public_bridge(self)
        else:
            controller = self.controller  # Preserve genuine legacy delegation.
        with _admission_controller_scope(controller):
            return await self.base_llm.acall(*args, **kwargs)

    # A delegating wrapper: call/acall forward whole calls (the wrapped
    # client does its own media recovery), so the one-request hooks do too.
    _call_once = call
    _acall_once = acall

    def _prepare_call_config(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Delegate per-call configuration to the wrapped client."""
        return self.base_llm._prepare_call_config(overrides)

    def get_context_limits(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate context-limit resolution to the wrapped client."""
        return self.base_llm.get_context_limits(*args, **kwargs)

    def _with_reduced_reply_limit(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate reply-limit recovery to the wrapped client."""
        return self.base_llm._with_reduced_reply_limit(*args, **kwargs)

    def count_tokens(self, text: str) -> int:
        """Delegate token counting to the wrapped client."""
        return self.base_llm.count_tokens(text)

    def get_model_info(self) -> Any:
        """Delegate model metadata lookup to the wrapped client."""
        return self.base_llm.get_model_info()

    def _with_public_view(self, view):
        """Exact private bridge: preserve the SAME attempt controller."""
        from nooa.unifiedllm.llm import LLM

        _validate_public_bridge(self)
        if type(view) is not LLM:
            raise TypeError("Unsupported public admission bridge")
        wrapped = AdmissionControl(view, AdmissionControlConfig(controller=self.controller))
        wrapped._owns_base = True
        return wrapped

    def close(self) -> None:
        """Close a legacy base or bridge-created view, never a public injected root."""
        if self._owns_base:
            self.base_llm.close()

    async def aclose(self) -> None:
        """Join legacy/owned-view disposal; public injected roots remain borrowed."""
        if self._owns_base:
            await self.base_llm.aclose()

    def __getattr__(self, name: str) -> Any:
        """Delegate remaining attributes and methods to the wrapped client."""
        return getattr(self.base_llm, name)


# Capture static descriptors at module definition time. Even a replacement
# before runtime import must fail the same-file/slot proof, not become canonical.
def _bridge_behavior():
    result = {}
    for owner in reversed(AdmissionControl.__mro__[:-1]):
        for name, descriptor in vars(owner).items():
            if callable(descriptor) or isinstance(
                descriptor, (property, staticmethod, classmethod)
            ):
                result[name] = descriptor
    return result


_BRIDGE_MRO = AdmissionControl.__mro__
_BRIDGE_BEHAVIOR = _bridge_behavior()


def _validate_public_bridge(client, *, installed=True):
    import inspect

    from nooa.unifiedllm.adapters._loader import _check_derivable
    from nooa.unifiedllm.llm import LLM

    if type(client) is not AdmissionControl or AdmissionControl.__mro__ != _BRIDGE_MRO:
        raise TypeError("Unsupported public admission bridge")
    current = _bridge_behavior()
    if current.keys() != _BRIDGE_BEHAVIOR.keys() or any(
        inspect.getattr_static(AdmissionControl, name) is not descriptor or name in vars(client)
        for name, descriptor in _BRIDGE_BEHAVIOR.items()
    ):
        raise TypeError("Admission bridge requires unchanged behavior")
    for name, descriptor in current.items():
        method = descriptor.fget if isinstance(descriptor, property) else descriptor
        if isinstance(method, (staticmethod, classmethod)):
            method = method.__func__
        method = inspect.unwrap(method)
        owner = next(owner for owner in AdmissionControl.__mro__ if name in vars(owner))
        slot = {"_acall_once": "acall", "_call_once": "call"}.get(name, name)
        if (
            getattr(method, "__module__", None) != owner.__module__
            or getattr(method, "__qualname__", None) != f"{owner.__qualname__}.{slot}"
        ):
            raise TypeError("Admission bridge requires canonical behavior")
    base = vars(client).get("base_llm")
    if type(base) is not LLM:
        raise TypeError("Admission bridge requires an exact loaded base")
    try:
        _check_derivable(base)
    except Exception as exc:
        raise TypeError("Admission bridge requires unchanged loaded base") from exc
    if installed:
        controller = vars(client).get("controller")
        if (
            controller is None
            or not isinstance(controller, AdmissionController)
            or not callable(getattr(controller, "acquire", None))
        ):
            raise TypeError("Admission bridge requires a compatible installed controller")
        return controller
    return None


__all__ = ["AdmissionControl"]
