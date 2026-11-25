# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
nooa - Code-Generating Agent Orchestration System

A minimal viable runtime for agent orchestration with event sourcing,
serialized execution, and complete transparency.
"""

import logging as _logging
import typing as _typing

from nooa._platform_policy import check_platform as _check_platform
from nooa._version import __version__

# Native Windows is unsupported: stop here, before any other nooa module loads,
# unless NOOA_SKIP_PLATFORM_CHECK=1 (see nooa._platform_policy).
_check_platform()

# ---------------------------------------------------------------------------
# Library logging: add NullHandler so applications that don't configure
# logging never see "No handlers could be found for logger 'nooa'".
# This is the only handler a library should ever add (see Python docs).
# ---------------------------------------------------------------------------
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# Install debug handler by default (zero overhead until SIGUSR2 received)
# Usage: kill -USR2 <pid> → dumps traceback + cell code to debug_dump_<pid>.txt in cwd
# nooa._debug_handler is stdlib-only, so this does not load the runtime.
from nooa._debug_handler import install_debug_handler  # noqa: E402

install_debug_handler()

# Every public name loads on first use (PEP 562), so ``import nooa`` and
# ``import nooa.<subpackage>`` (e.g. nooa.overflow) stay cheap. The strategies
# import CodeAct, which imports the LLM client and LiteLLM (about two
# seconds); Agent and the runtime pull in about a hundred nooa modules.
# ``from nooa import Agent`` still works.
# NOTE: ReflexionStrategy is experimental; it resolves to the
# warning-emitting factory in nooa.experimental (below).
if _typing.TYPE_CHECKING:  # the same names, for type checkers and IDEs
    from nooa._logging import enable_logging
    from nooa._visible import visible
    from nooa.agent import Agent
    from nooa.agentdoc import hidden, spec
    from nooa.context_blocks import Context, ContextWindowStats, DynamicContext
    from nooa.decisions import (
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
    from nooa.decorators import strategy
    from nooa.errors import (
        CodeSyntaxError,
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
    from nooa.library_manager import LibraryManager
    from nooa.llm_config import llm_config_chain
    from nooa.llm_types import LLMResponse
    from nooa.media import Audio, File, Image, Media, Video
    from nooa.metaclass import AgentMeta, no_trace
    from nooa.prompts import PromptData, build_prompt_data, print_prompt
    from nooa.runtime.channels import Channel, QueueManager, QueueOutput
    from nooa.runtime.context import ContextApi
    from nooa.runtime.context_manager import ContextManager
    from nooa.runtime.event_query import EventQuery
    from nooa.runtime.events import EventsApi
    from nooa.skill import Skill, TextSkill, get_slash_commands, slash_command
    from nooa.skill_registry import skill_from_module
    from nooa.storage import StorageManager
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
    from nooa.strategy_validation import (
        InvariantError,
        MethodPostcondition,
        MethodPrecondition,
    )
    from nooa.token_counter import char_approximate_token_counter

_LAZY = {
    "enable_logging": "nooa._logging",
    "visible": "nooa._visible",
    "Agent": "nooa.agent",
    "hidden": "nooa.agentdoc",
    "spec": "nooa.agentdoc",
    "Context": "nooa.context_blocks",
    "ContextWindowStats": "nooa.context_blocks",
    "DynamicContext": "nooa.context_blocks",
    "BooleanDecision": "nooa.decisions",
    "ChoiceDecision": "nooa.decisions",
    "Criteria": "nooa.decisions",
    "Decision": "nooa.decisions",
    "DecisionClient": "nooa.decisions",
    "DecisionModel": "nooa.decisions",
    "DecisionModelRequiredError": "nooa.decisions",
    "DecisionRequest": "nooa.decisions",
    "DecisionResponse": "nooa.decisions",
    "Instructions": "nooa.decisions",
    "ScoreDecision": "nooa.decisions",
    "Threshold": "nooa.decisions",
    "UnifiedDecisionModel": "nooa.decisions",
    "strategy": "nooa.decorators",
    "CodeSyntaxError": "nooa.errors",
    "DynamicMethodAdditionError": "nooa.errors",
    "GenerationError": "nooa.errors",
    "NemoOOAgentsError": "nooa.errors",
    "NemoOOAgentsRuntimeError": "nooa.errors",
    "RestrictedCodeError": "nooa.errors",
    "SerializationError": "nooa.errors",
    "SnapshotNotFoundError": "nooa.errors",
    "StorageNotConfiguredError": "nooa.errors",
    "ValidationError": "nooa.errors",
    "LibraryManager": "nooa.library_manager",
    "llm_config_chain": "nooa.llm_config",
    "LLMResponse": "nooa.llm_types",  # the class nooa.unifiedllm re-exports
    "Audio": "nooa.media",
    "File": "nooa.media",
    "Image": "nooa.media",
    "Media": "nooa.media",
    "Video": "nooa.media",
    "AgentMeta": "nooa.metaclass",
    "no_trace": "nooa.metaclass",
    "PromptData": "nooa.prompts",
    "build_prompt_data": "nooa.prompts",
    "print_prompt": "nooa.prompts",
    "Channel": "nooa.runtime.channels",
    "QueueManager": "nooa.runtime.channels",
    "QueueOutput": "nooa.runtime.channels",
    "ContextApi": "nooa.runtime.context",
    "ContextManager": "nooa.runtime.context_manager",
    "EventQuery": "nooa.runtime.event_query",
    "EventsApi": "nooa.runtime.events",
    "Skill": "nooa.skill",
    "TextSkill": "nooa.skill",
    "get_slash_commands": "nooa.skill",
    "slash_command": "nooa.skill",
    "skill_from_module": "nooa.skill_registry",
    "StorageManager": "nooa.storage",
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
    "InvariantError": "nooa.strategy_validation",
    "MethodPostcondition": "nooa.strategy_validation",
    "MethodPrecondition": "nooa.strategy_validation",
    "char_approximate_token_counter": "nooa.token_counter",
}


def __getattr__(name):
    import importlib

    module_name = _LAZY.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(module_name), name)
        globals()[name] = value
        return value
    # Experimental strategies: route through the warning factories so that
    # `from nooa import ReflexionStrategy; ReflexionStrategy()`
    # emits the same FutureWarning as importing from nooa.experimental.
    if name == "ReflexionStrategy":
        from nooa import experimental

        return getattr(experimental, name)
    # ``import nooa`` used to load most submodules eagerly, so code may rely
    # on ``nooa.<submodule>`` without importing it. Import it on access.
    # Only plain names: ``getattr(nooa, "a.b", default)`` must get AttributeError.
    if name.isidentifier() and not name.startswith("__"):
        import importlib.util

        if importlib.util.find_spec(f"{__name__}.{name}") is not None:
            return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_LAZY, "ReflexionStrategy"})


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
    "CodeSyntaxError",
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
