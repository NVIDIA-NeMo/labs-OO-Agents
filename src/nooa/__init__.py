# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
nooa - Code-Generating Agent Orchestration System

A minimal viable runtime for agent orchestration with event sourcing,
serialized execution, and complete transparency.
"""

import logging as _logging

from nooa._version import __version__

# ---------------------------------------------------------------------------
# Library logging: add NullHandler so applications that don't configure
# logging never see "No handlers could be found for logger 'nooa'".
# This is the only handler a library should ever add (see Python docs).
# ---------------------------------------------------------------------------
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# Export core types
# Export agent and decorators

from nooa._logging import enable_logging  # noqa: E402
from nooa._visible import visible  # noqa: E402
from nooa.agent import Agent  # noqa: E402
from nooa.agentdoc import hidden, spec  # noqa: E402
from nooa.context_blocks import Context, ContextWindowStats, DynamicContext  # noqa: E402
from nooa.decorators import strategy  # noqa: E402

# Export errors
from nooa.errors import (  # noqa: E402
    DynamicMethodAdditionError,
    GenerationError,
    NemoOOAgentsError,
    NemoOOAgentsRuntimeError,
    RestrictedCodeError,
    SerializationError,
    SnapshotNotFoundError,
    StorageNotConfiguredError,
    ValidationError,
)
from nooa.library_manager import LibraryManager  # noqa: E402
from nooa.media import Audio, File, Image, Media, Video  # noqa: E402
from nooa.metaclass import AgentMeta, no_trace  # noqa: E402

# Export prompt inspection utilities
from nooa.prompts import PromptData, build_prompt_data, print_prompt  # noqa: E402
from nooa.runtime.channels import Channel, QueueManager, QueueOutput  # noqa: E402

# Export runtime API classes
from nooa.runtime.context import ContextApi  # noqa: E402
from nooa.runtime.context_manager import ContextManager  # noqa: E402

# Export event filtering
from nooa.runtime.event_query import EventQuery  # noqa: E402
from nooa.runtime.events import EventsApi  # noqa: E402
from nooa.skill import Skill, TextSkill, get_slash_commands, slash_command  # noqa: E402
from nooa.skill_registry import skill_from_module  # noqa: E402

# Export storage
from nooa.storage import StorageManager  # noqa: E402

# Export strategy base class and implementations.
# NOTE: CodeActLiteStrategy and ReflexionStrategy are experimental. They are NOT
# imported raw here — that would bypass the FutureWarning gate in
# nooa.experimental. Instead they are exposed lazily via __getattr__
# below, which returns the warning-emitting factories.
from nooa.strategies import (  # noqa: E402
    CodeActStrategy,
    GenerationStrategy,
    InspectInputsPrefill,
    PredictStrategy,
    get_default_strategy,
    set_default_strategy,
)
from nooa.strategy_validation import (  # noqa: E402
    InvariantError,
    MethodPostcondition,
    MethodPrecondition,
)
from nooa.token_counter import char_approximate_token_counter  # noqa: E402
from nooa.unifiedllm import LLMResponse  # noqa: E402


# Lazy re-export of llm_config_chain — defer importing the llm_config /
# paths machinery (and the registry it touches) until the helper is
# actually called, keeping ``import nooa`` cheap.
def __getattr__(name):
    if name == "llm_config_chain":
        from nooa.llm_config import llm_config_chain

        return llm_config_chain
    # Experimental strategies: route through the warning factories so that
    # `from nooa import CodeActLiteStrategy; CodeActLiteStrategy()`
    # emits the same FutureWarning as importing from nooa.experimental.
    if name in ("CodeActLiteStrategy", "ReflexionStrategy"):
        from nooa import experimental

        return getattr(experimental, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "llm_config_chain",  # Lazy re-export (see __getattr__)
    # Types
    "ContextWindowStats",  # Re-exported from context_blocks
    "Context",  # Re-exported from context_blocks
    "DynamicContext",  # Re-exported from context_blocks (deprecated, use Context)
    "EventQuery",  # Event filtering configuration
    "ContextApi",  # LLM-facing context API wrapper (Skill)
    "ContextManager",  # Context block state backend
    "EventsApi",  # Runtime events query API
    "Channel",  # Named queue/event channel for agent input
    "QueueManager",  # Channel registry + race + status
    "QueueOutput",  # Event emitted by event-mode channels
    "LLMResponse",  # Re-exported from unifiedllm
    # Strategies
    "GenerationStrategy",
    "CodeActStrategy",
    "CodeActLiteStrategy",
    "ReflexionStrategy",
    "PredictStrategy",
    "get_default_strategy",
    "set_default_strategy",
    # Method validators
    "InvariantError",
    "MethodPrecondition",
    "MethodPostcondition",
    # Prefill plugins
    "InspectInputsPrefill",
    # Prompt inspection
    "print_prompt",
    "build_prompt_data",
    "PromptData",
    # Media types
    "Media",
    "Image",
    "Audio",
    "Video",
    "File",
    # Agent and decorators
    "Agent",
    "Skill",
    "TextSkill",
    "slash_command",
    "get_slash_commands",
    "skill_from_module",
    "LibraryManager",
    "strategy",
    "no_trace",
    "AgentMeta",
    # Logging
    "enable_logging",
    # Visibility
    "hidden",
    "spec",
    "visible",
    # Errors
    "NemoOOAgentsError",
    "GenerationError",
    "ValidationError",
    "RestrictedCodeError",
    "NemoOOAgentsRuntimeError",
    "DynamicMethodAdditionError",
    "SerializationError",
    "SnapshotNotFoundError",
    "StorageNotConfiguredError",
    # Storage
    "StorageManager",
    # Token counting
    "char_approximate_token_counter",
]

# Install debug handler by default (zero overhead until SIGUSR2 received)
# Usage: kill -USR2 <pid> → dumps traceback + cell code to debug_dump_<pid>.txt in cwd
from nooa.runtime.debug_handler import install_debug_handler  # noqa: E402

install_debug_handler()
