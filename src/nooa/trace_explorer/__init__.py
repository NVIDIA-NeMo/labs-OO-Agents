# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trace_explorer - Programmatic interface for exploring .jsonl files.

Designed for agent-driven trace analysis.

Usage:
    from nooa.trace_explorer import TraceExplorer

    trace = await TraceExplorer.from_file("path/to/trace.jsonl")
    print(await trace.help())
    print(await trace.get_overview())
"""

from nooa.trace_explorer.client import TraceExplorerClient
from nooa.trace_explorer.explorer import (
    AgentSession,
    EvalContextData,
    ExecutionTurn,
    LLMMessage,
    LLMTurn,
    OverviewData,
    OverviewStats,
    RootSessionInfo,
    ScoreDetail,
    SearchMatches,
    SearchResult,
    SessionData,
    SessionSummary,
    TimelineData,
    TimelineEvent,
    ToolCall,
    ToolDefinition,
    TraceExplorer,
    TurnInfo,
    get_quiet_mode,
    main,
    set_quiet_mode,
)

__all__ = [
    "TraceExplorerClient",
    "TraceExplorer",
    "AgentSession",
    "LLMTurn",
    "ExecutionTurn",
    "LLMMessage",
    "ToolCall",
    "ToolDefinition",
    "TurnInfo",
    "SessionData",
    "SessionSummary",
    "EvalContextData",
    "ScoreDetail",
    "SearchMatches",
    "SearchResult",
    "TimelineData",
    "TimelineEvent",
    "OverviewData",
    "OverviewStats",
    "RootSessionInfo",
    "main",
    "set_quiet_mode",
    "get_quiet_mode",
]
