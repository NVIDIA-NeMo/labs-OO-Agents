# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
nooa - Code-Generating Agent Orchestration System

A minimal viable runtime for agent orchestration with event sourcing,
serialized execution, and complete transparency.
"""

import logging as _logging
import typing as _typing

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
from nooa.decisions import (  # noqa: E402
    BooleanDecision,
    ChoiceDecision,
    Criteria,
    Decision,
    DecisionClient,
    DecisionModel,
    DecisionModelRequiredError,
    DecisionRequest,
    DecisionResponse,
    Instructions,
    ScoreDecision,
    Threshold,
    UnifiedDecisionModel,
)
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
from nooa.strategy_validation import (  # noqa: E402
    InvariantError,
    MethodPostcondition,
    MethodPrecondition,
)
from nooa.token_counter import char_approximate_token_counter  # noqa: E402

# Names loaded on first use. The strategies import CodeAct, which imports
# the LLM client and LiteLLM (about two seconds); llm_config_chain touches
# the llm_config / paths machinery. ``import nooa`` stays cheap, and
# ``from nooa import CodeActStrategy`` still works.
# NOTE: ReflexionStrategy is experimental; it resolves to the
# warning-emitting factory in nooa.experimental (below).
if _typing.TYPE_CHECKING:  # the same names, for type checkers and IDEs
    from nooa.llm_config import llm_config_chain
    from nooa.llm_types import LLMResponse
    from nooa.strategies import (
        CodeActStrategy,
        CodeActV2,
        DecideStrategy,
        GenerationStrategy,
        InspectInputsPrefill,
        PredictStrategy,
        TextOnlyResponseAction,
        TextOnlyResponseContext,
        TextOnlyResponseHandler,
        get_default_strategy,
        retry_text_only_response,
        return_text_as_result,
        set_default_strategy,
    )

_LAZY = {
    "CodeActStrategy": "nooa.strategies",
    "CodeActV2": "nooa.strategies",
    "DecideStrategy": "nooa.strategies",
    "GenerationStrategy": "nooa.strategies",
    "InspectInputsPrefill": "nooa.strategies",
    "PredictStrategy": "nooa.strategies",
    "TextOnlyResponseAction": "nooa.strategies",
    "TextOnlyResponseContext": "nooa.strategies",
    "TextOnlyResponseHandler": "nooa.strategies",
    "get_default_strategy": "nooa.strategies",
    "retry_text_only_response": "nooa.strategies",
    "return_text_as_result": "nooa.strategies",
    "set_default_strategy": "nooa.strategies",
    "LLMResponse": "nooa.llm_types",  # the class nooa.unifiedllm re-exports
    "llm_config_chain": "nooa.llm_config",
}


def __getattr__(name):
    module_name = _LAZY.get(name)
    if module_name is not None:
        import importlib

        value = getattr(importlib.import_module(module_name), name)
        globals()[name] = value
        return value
    # Experimental strategies: route through the warning factories so that
    # `from nooa import ReflexionStrategy; ReflexionStrategy()`
    # emits the same FutureWarning as importing from nooa.experimental.
    if name == "ReflexionStrategy":
        from nooa import experimental

        return getattr(experimental, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "llm_config_chain",  # Lazy (see _LAZY)
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
    "CodeActV2",
    "TextOnlyResponseAction",
    "TextOnlyResponseContext",
    "TextOnlyResponseHandler",
    "retry_text_only_response",
    "return_text_as_result",
    "ReflexionStrategy",
    "PredictStrategy",
    "DecideStrategy",
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
    # Decisions
    "BooleanDecision",
    "ChoiceDecision",
    "Criteria",
    "Decision",
    "DecisionClient",
    "DecisionModel",
    "DecisionModelRequiredError",
    "DecisionRequest",
    "DecisionResponse",
    "Instructions",
    "ScoreDecision",
    "Threshold",
    "UnifiedDecisionModel",
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
