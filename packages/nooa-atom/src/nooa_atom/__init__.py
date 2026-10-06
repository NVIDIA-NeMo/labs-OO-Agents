# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session layer, session tree and in-process host for NOOA interactive agents."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nooa_coder.hosts.headless import TaskRun, Tree, open_tree, run_task
    from nooa_coder.session.items import (
        ChildFailed,
        ChildFailedError,
        ChildQuestion,
        ChildRef,
        ChildResult,
        CommandInfo,
        CommandResult,
        Receipt,
        SessionEvent,
        SessionInfo,
        TaskResult,
        TranscriptEntry,
        TurnCancelled,
        TurnCancelledOutcome,
        Usage,
    )
    from nooa_coder.session.loader import AgentFactory, AgentSpecError, load_agent_class
    from nooa_coder.session.options import SessionOptions
    from nooa_coder.session.port import SessionPort
    from nooa_coder.session.registry import (
        ChildActiveElsewhereError,
        DepthLimitError,
        SessionRegistry,
    )
    from nooa_coder.session.session import (
        ItemWithdrawnError,
        Outcome,
        Session,
        SessionClosedError,
        TurnFailedError,
    )
    from nooa_coder.session.store import SessionNotFoundError, SessionStore

# The public names load on first use: importing a light submodule (the
# `nooa coder` command, which `nooa` loads at startup with every plugin)
# must not import the framework.
_EXPORT_MODULES = {
    **dict.fromkeys(("TaskRun", "Tree", "open_tree", "run_task"), "hosts.headless"),
    **dict.fromkeys(
        (
            "ChildFailed ChildFailedError ChildQuestion ChildRef ChildResult CommandInfo "
            "CommandResult Receipt SessionEvent SessionInfo TaskResult TranscriptEntry "
            "TurnCancelled TurnCancelledOutcome Usage"
        ).split(),
        "session.items",
    ),
    **dict.fromkeys(("AgentFactory", "AgentSpecError", "load_agent_class"), "session.loader"),
    "SessionOptions": "session.options",
    "SessionPort": "session.port",
    **dict.fromkeys(
        ("ChildActiveElsewhereError", "DepthLimitError", "SessionRegistry"), "session.registry"
    ),
    **dict.fromkeys(
        ("ItemWithdrawnError", "Outcome", "Session", "SessionClosedError", "TurnFailedError"),
        "session.session",
    ),
    **dict.fromkeys(("SessionNotFoundError", "SessionStore"), "session.store"),
}


def __getattr__(name: str) -> Any:
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


try:
    __version__ = version("nooa-coder")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = [
    "AgentFactory",
    "AgentSpecError",
    "ChildActiveElsewhereError",
    "ChildFailed",
    "ChildFailedError",
    "ChildQuestion",
    "ChildRef",
    "ChildResult",
    "CommandInfo",
    "CommandResult",
    "DepthLimitError",
    "ItemWithdrawnError",
    "Outcome",
    "Receipt",
    "Session",
    "SessionClosedError",
    "SessionEvent",
    "SessionInfo",
    "SessionNotFoundError",
    "SessionOptions",
    "SessionPort",
    "SessionRegistry",
    "SessionStore",
    "TaskResult",
    "TaskRun",
    "TranscriptEntry",
    "Tree",
    "TurnCancelled",
    "TurnCancelledOutcome",
    "TurnFailedError",
    "Usage",
    "__version__",
    "load_agent_class",
    "open_tree",
    "run_task",
]
