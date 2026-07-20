# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for trace viewer."""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    """A single trace event in nooa format (OTel-compatible)."""

    model_config = {"extra": "allow"}

    # Core fields
    schema_version: str = "1.0"
    timestamp: str
    type: str

    # Correlation IDs (OTel-compatible)
    ids: dict[str, Any]

    # Optional fields
    session_start: str | None = None
    body: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class OTelSpan(BaseModel):
    """An OpenTelemetry span exported to JSONL format."""

    model_config = {"extra": "allow"}

    # OTel span identifiers
    span_id: str
    trace_id: str
    parent_span_id: str | None = None

    # Span metadata
    name: str
    start_time: int  # nanoseconds since epoch
    end_time: int | None = None  # nanoseconds since epoch
    duration_ns: int | None = None

    # Span data
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)  # Span events
    status: dict[str, Any] = Field(default_factory=dict)  # status_code, description


class TraceFile(BaseModel):
    """Information about a trace file (legacy, use TraceGroup for new code)."""

    path: str
    name: str
    size: int
    modified: str
    event_count: int | None = None


class TraceGroup(BaseModel):
    """Provider-agnostic trace group representation.

    Replaces TraceFile as the public API model. A group is identified by
    session_id and may contain one or more underlying traces/files.

    For HybridProvider, id is prefixed with source (e.g., "local::abc123").
    """

    id: str  # Group handle (typically session.id or "ungrouped")
    name: str  # Display name
    modified: str  # Last modification timestamp (ISO or Unix)
    size: int = 0  # Optional size indicator
    event_count: int | None = None  # Total spans/events in group
    source: str | None = None  # Provider source: "local" | "langfuse" (set by HybridProvider)
    counterpart_id: str | None = None  # ID of same trace in another source (for hybrid provider)
    batch_id: str | None = None  # Import batch identifier


class FilterConfig(BaseModel):
    """Filter configuration."""

    event_types: list[str] = Field(default_factory=list)
    text_search: str = ""
    field_filters: dict[str, str] = Field(default_factory=dict)
    time_range: list[float] = Field(default_factory=lambda: [0, 9999999999.0])


class TimelineConfig(BaseModel):
    """Timeline configuration."""

    expanded: bool = False
    zoom: float = 1.0
    position: float = 0.0


class ViewerConfig(BaseModel):
    """Complete viewer configuration."""

    id: str
    name: str
    filters: FilterConfig
    timeline: TimelineConfig


class ConfigFile(BaseModel):
    """Configuration file structure."""

    configurations: dict[str, ViewerConfig] = Field(default_factory=dict)
    default_config: str | None = None

    # Provider selection: "local" (default) or "langfuse"
    provider: str = Field(default="local")

    # Upload target: provider to upload sessions to (e.g., "langfuse")
    # Upload is available only if upload_target is set and differs from provider
    upload_target: str | None = None

    # Langfuse provider configuration (only used if provider="langfuse")
    langfuse: dict[str, str] | None = None

    # Legacy: explicit trace directories (still supported for local provider)
    trace_directories: list[str] = Field(default_factory=list)

    # Auto-discovery: scan roots for .jsonl trace files (local provider only)
    scan_roots: list[str] = Field(default_factory=lambda: ["../../"])
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["node_modules", ".git", "__pycache__", ".venv", "venv"]
    )


class ChainSpan(BaseModel):
    """A span representing a call chain."""

    chain_type: str  # 'plan', 'delegation', 'llm_call'
    chain_id: str
    start_event_index: int
    end_event_index: int
    start_timestamp: str
    end_timestamp: str
    events: list[int]  # Indices of all events in chain


class SpanHierarchyNode(BaseModel):
    """A node in the span hierarchy tree (OpenTelemetry-compatible)."""

    model_config = {"ser_json_inf_nan": "constants"}  # Allow deep nesting serialization

    span_id: str
    parent_span_id: str | None = None
    start_event_index: int
    end_event_index: int
    start_timestamp: str
    end_timestamp: str
    events: list[int]  # Indices of all events with this span_id
    children: list["SpanHierarchyNode"] = Field(default_factory=list)
    depth: int = 0  # Nesting level (0 = root)


class ChainRelationship(BaseModel):
    """Relationships between events via correlation IDs."""

    plan_chains: list[ChainSpan] = Field(default_factory=list)
    delegation_chains: list[ChainSpan] = Field(default_factory=list)
    llm_call_chains: list[ChainSpan] = Field(default_factory=list)
    span_hierarchy: list[SpanHierarchyNode] = Field(default_factory=list)  # Root spans only


class Annotation(BaseModel):
    """User annotation on a trace span.

    Compatible with Langfuse scores API:
    - span_id maps to Langfuse observationId
    - score/label maps to Langfuse value/stringValue
    - source maps to Langfuse source field
    - target is stored in metadata for Langfuse
    """

    # Identity
    id: str = Field(default_factory=lambda: str(uuid4()))

    # Link to span
    session_id: str  # Required: routing scope (which trace group/session)
    span_id: str | None = None  # Optional: specific span (None = whole session)

    # NEW: Annotation target (part of span)
    target: str | None = None  # Optional: which part (e.g., "prompt", "output", "code", "result")

    # Annotation content
    name: str  # Category name, e.g., "quality", "correctness"
    score: float | None = None  # Numeric value (e.g., 0.0-1.0, 1-5)
    label: str | None = None  # Categorical value (e.g., "good", "bad", "neutral")
    comment: str | None = None  # Free-form text feedback

    # Tags for taxonomy evolution
    tags: list[str] = Field(
        default_factory=list
    )  # Freeform tags, e.g., ["tool-error", "hallucination"]

    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    author_id: str | None = None  # Who created it (user ID or name)
    source: Literal["human", "llm", "mechanical", "eval"] = "human"  # How it was created
    metadata: dict[str, Any] | None = None  # Additional structured metadata
