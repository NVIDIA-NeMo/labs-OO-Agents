# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime components: actor loop, prompts, PLAN, REPL, context.

Core runtime contains NO OpenTelemetry code - only the hooks protocol.
For tracing, use: from openinference_instrumentation_nooa import enable_tracing

Middleware types are imported directly from ``nooa.runtime.middleware``::

    from nooa.runtime.middleware import LLMCallContext, LLMCallMiddleware
"""

from nooa.agentdoc import FileBackedTruncatingStringIO, TruncatingStringIO
from nooa.config.truncation_config import TruncationConfig
from nooa.runtime.actor import ActorRuntime
from nooa.runtime.event_manager import EventManager
from nooa.runtime.event_query import EventQuery
from nooa.runtime.events import EventsApi
from nooa.runtime.hooks import InstrumentationHooks, get_hooks, set_hooks
from nooa.runtime.media_capture import show
from nooa.runtime.pprint import pprint

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
