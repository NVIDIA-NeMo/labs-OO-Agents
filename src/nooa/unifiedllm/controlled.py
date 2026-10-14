# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-use admission wrapper for UnifiedLLM-compatible clients."""

from __future__ import annotations

from typing import Any

from nooa.unifiedllm.admission import (
    AdmissionControlConfig,
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
        self.controller = config._controller_for(llm)
        self.model = getattr(llm, "model", "")

    @property
    def context_window(self) -> int | None:
        """Return the wrapped client's context window when available."""
        return getattr(self.base_llm, "context_window", None)

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
        with _admission_controller_scope(self.controller):
            return await self.base_llm.acall(*args, **kwargs)

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

    def close(self) -> None:
        """Close resources owned by the wrapped client."""
        self.base_llm.close()

    async def aclose(self) -> None:
        """Asynchronously close resources owned by the wrapped client."""
        await self.base_llm.aclose()

    def __getattr__(self, name: str) -> Any:
        """Delegate remaining attributes and methods to the wrapped client."""
        return getattr(self.base_llm, name)


__all__ = ["AdmissionControl"]
