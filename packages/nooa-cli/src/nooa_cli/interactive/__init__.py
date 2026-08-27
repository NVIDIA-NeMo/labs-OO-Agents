"""Python-native interactive agent API."""

from .state import (
    AgentJobState,
    AgentJobSummary,
    AgentLifecycle,
    AgentState,
    AgentWorkspaceState,
    CancellationState,
    InteractiveAgent,
    Observation,
    UIScheduler,
)

_LAZY_EXPORTS = {
    "AgentRuntime": (".runtime", "AgentRuntime"),
    "InteractiveSessionDispatcher": (".dispatcher", "InteractiveSessionDispatcher"),
    "JobSnapshot": (".runtime", "JobSnapshot"),
    "LocalAgentRunner": (".local_agent", "LocalAgentRunner"),
}

__all__ = [
    "AgentJobState",
    "AgentJobSummary",
    "AgentLifecycle",
    "AgentRuntime",
    "AgentState",
    "AgentWorkspaceState",
    "CancellationState",
    "InteractiveAgent",
    "InteractiveSessionDispatcher",
    "JobSnapshot",
    "LocalAgentRunner",
    "Observation",
    "UIScheduler",
]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(importlib.import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
