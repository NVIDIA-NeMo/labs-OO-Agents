# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session layer, session tree and in-process host for NOOA interactive agents."""

from importlib.metadata import PackageNotFoundError, version

from nooa_coder.hosts.headless import Tree, open_tree
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
    "TranscriptEntry",
    "Tree",
    "TurnCancelled",
    "TurnCancelledOutcome",
    "TurnFailedError",
    "Usage",
    "__version__",
    "load_agent_class",
    "open_tree",
]
