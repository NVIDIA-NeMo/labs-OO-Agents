# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime components: actor loop, prompts, PLAN, REPL, context.

Core runtime contains NO OpenTelemetry code - only the hooks protocol.
For tracing, use: from openinference_instrumentation_nooa import enable_tracing

Middleware types are imported directly from ``nooa.runtime.middleware``::

    from nooa.runtime.middleware import LLMCallContext, LLMCallMiddleware
"""

from __future__ import annotations

from typing import Any

_MODULE_BY_NAME = {
    "ActorRuntime": "nooa.runtime.actor",
    "EventManager": "nooa.runtime.event_manager",
    "EventQuery": "nooa.runtime.event_query",
    "EventsApi": "nooa.runtime.events",
    "InstrumentationHooks": "nooa.runtime.hooks",
    "set_hooks": "nooa.runtime.hooks",
    "get_hooks": "nooa.runtime.hooks",
    "TruncationConfig": "nooa.config.truncation_config",
    "TruncatingStringIO": "nooa.agentdoc",
    "FileBackedTruncatingStringIO": "nooa.agentdoc",
    "pprint": "nooa.runtime.pprint",
    "show": "nooa.runtime.media_capture",
}

__all__ = [
    "ActorRuntime",
    # Event system
    "EventManager",
    "EventQuery",
    "EventsApi",
    # Hook-based instrumentation protocol
    "InstrumentationHooks",
    "set_hooks",
    "get_hooks",
    # Truncation system
    "TruncationConfig",
    "TruncatingStringIO",
    "FileBackedTruncatingStringIO",
    "pprint",
    "show",
]


def __getattr__(name: str) -> Any:
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
