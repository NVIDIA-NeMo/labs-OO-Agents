# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured output types for NOOA frontends.

Commands and agent events produce ``Output`` instances; the
``TerminalFrontend`` renders them as Rich panels in the terminal.

Every concrete type is a plain dataclass with a ``to_json()`` method that
returns a dict suitable for JSON serialisation.  This is the **single source
of truth** for the JSON form of rich content pushed to the browser via the
PTY web terminal's rich side-channel (``nooa-term``).  Adding a new Output
type keeps working as long as ``to_json()`` is defined.
"""

import re
from dataclasses import dataclass
from typing import Literal

from nooa.agentdoc import hidden

_STOP_REASON_ESCAPE_SEQUENCE_RE = re.compile(
    "\x1b(?:[@-Z\\-_]|\\[[0-?]*[ -/]*[@-~]|\\][^\x07\x1b]*(?:\x07|\x1b\\\\)?)"
)
_STOP_REASON_CONTROL_RE = re.compile("[\x00-\x1f\x7f-\x9f]+")


def _sanitize_stop_reason_text(value: str) -> str:
    value = _STOP_REASON_ESCAPE_SEQUENCE_RE.sub(" ", str(value))
    value = _STOP_REASON_CONTROL_RE.sub(" ", value)
    return " ".join(value.split())


# ---------------------------------------------------------------------------
# Concrete output types
# ---------------------------------------------------------------------------


@dataclass
class TextOutput:
    """A plain text message with a severity level."""

    content: str
    level: Literal["info", "error", "warning", "success", "status"] = "info"

    def to_json(self) -> dict:
        return {"type": "text", "content": self.content, "level": self.level}


@dataclass
class TableOutput:
    """Tabular data."""

    columns: list[str]
    rows: list[list[str]]
    title: str = ""
    footer: str = ""
    show_header: bool = True

    def to_json(self) -> dict:
        return {
            "type": "table",
            "title": self.title,
            "columns": self.columns,
            "rows": self.rows,
            "footer": self.footer,
            "show_header": self.show_header,
        }


@dataclass
class HelpOutput:
    """Slash-command help listing."""

    commands: dict[str, str]  # {"/cmd subcmd": "description"}

    def to_json(self) -> dict:
        return {"type": "help", "commands": self.commands}


@dataclass
class AgentMessage:
    """A markdown-formatted message produced by the agent via ``message()``."""

    content: str
    # False for 2nd+ message() calls in the same turn — suppresses the OO ── rule
    show_rule: bool = True

    def to_json(self) -> dict:
        return {"type": "agent_message", "content": self.content, "show_rule": self.show_rule}


@dataclass
class CodeExecution:
    """One Python execution turn: code + outputs.

    ``start_line`` / ``highlight_line`` let a *snippet* (e.g. the ±N lines
    around an agent suspend point shown by ``/activity``) number from its real
    file offset and tint the one line that matters. ``start_line`` defaults to
    1 (whole-cell rendering); ``highlight_line`` is absolute (file line number),
    not relative to the snippet.
    """

    tool_call_id: str
    code: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    error: str | None = None
    value: str | None = None
    start_line: int = 1
    highlight_line: int | None = None

    def to_json(self) -> dict:
        v = self.value
        if v is not None and not isinstance(v, str):
            v = repr(v)
        return {
            "type": "code_execution",
            "tool_call_id": self.tool_call_id,
            "code": self.code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "value": v,
            "start_line": self.start_line,
            "highlight_line": self.highlight_line,
        }


@dataclass
class StartupInfo:
    """Structured startup banner data."""

    model: str  # full model id
    short_model: str  # display name
    working_dir: str
    vi_mode: bool
    history_policy: str | None = None
    history_limit: int | None = None
    tracing_enabled: bool = False
    trace_dir: str | None = None
    custom_agent: str | None = None
    llm_ready: bool = True

    def to_json(self) -> dict:
        return {
            "type": "startup",
            "model": self.model,
            "short_model": self.short_model,
            "working_dir": self.working_dir,
            "vi_mode": self.vi_mode,
            "history_policy": self.history_policy,
            "history_limit": self.history_limit,
            "tracing_enabled": self.tracing_enabled,
            "trace_dir": self.trace_dir,
            "custom_agent": self.custom_agent,
            "llm_ready": self.llm_ready,
        }


@dataclass
class ClearScreen:
    """Clear the visible output area."""

    def to_json(self) -> dict:
        return {"type": "clear"}


@dataclass
class Thinking:
    """Show or hide the loading/thinking indicator."""

    active: bool
    message: str = "thinking..."

    def to_json(self) -> dict:
        return {
            "type": "thinking_start" if self.active else "thinking_stop",
            "message": self.message,
        }


@dataclass
class StopReasonOutput:
    """Why the agent ended the turn or what it is waiting for."""

    kind: str
    explanation: str = ""

    @property
    @hidden
    def label(self) -> str:
        if str(self.kind) == "WAIT":
            return "waiting"
        if str(self.kind) == "DONE":
            return "done"
        if str(self.kind) == "KEEP_GOING":
            return "keep going"
        if str(self.kind) in {"NEED_INPUT", "GET_USER_INPUT"}:
            return "need input"
        return "paused"

    @hidden
    def display_text(self) -> str:
        explanation = _sanitize_stop_reason_text(self.explanation)
        if explanation:
            return f"∴ {self.label}: {explanation}"
        return f"∴ {self.label}"

    @hidden
    def to_json(self) -> dict:
        explanation = _sanitize_stop_reason_text(self.explanation)
        return {
            "type": "stop_reason",
            "kind": str(self.kind),
            "label": self.label,
            "explanation": explanation,
            "text": self.display_text(),
        }


@dataclass
class BashOutput:
    """Result from a ``!bash`` command."""

    stdout: str
    stderr: str
    return_code: int

    def to_json(self) -> dict:
        return {
            "type": "bash_output",
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
        }


@dataclass
class CommandStatus:
    """Lifecycle status for one TUI command invocation.

    This is UI-local command state, not an agent background job. Slash commands
    and bang commands use it for consistent queued/running/finished feedback.
    """

    id: int
    kind: Literal["slash", "bang"]
    state: Literal["queued", "running", "done", "failed", "cancelled"]
    text: str
    error: str = ""

    def to_json(self) -> dict:
        return {
            "type": "command_status",
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "text": self.text,
            "error": self.error,
        }


@dataclass
class HistoryTurn:
    """One turn in a replayed session history."""

    role: Literal["user", "agent"]
    content: str

    def to_json(self) -> dict:
        return {"type": "history_turn", "role": self.role, "content": self.content}


@dataclass
class HistoryReplay:
    """A block of past conversation turns rendered above the live prompt.

    Used when resuming or continuing a session so the user can see what was
    said before.  Frontends should render this in a visually dimmed / muted
    style to distinguish it from the live conversation.

    When rich content is interleaved with history (session resume inside
    ``nooa-term``), one session's history may be split into several
    ``HistoryReplay`` chunks with ``_RichReplayPayload`` objects between them.
    ``show_header`` / ``show_footer`` control which chunk renders the enclosing
    rule bars so they appear exactly once around the whole block.
    """

    turns: list[HistoryTurn]
    session_id: str  # short (8-char) id for the header label
    show_header: bool = True
    show_footer: bool = True
    omitted_count: int = 0  # number of older turns that were truncated

    def to_json(self) -> dict:
        return {
            "type": "history_replay",
            "session_id": self.session_id,
            "turns": [t.to_json() for t in self.turns],
        }


@dataclass
class DiffOutput:
    """A unified diff between two versions of a file, shown after /edit saves."""

    diff: str
    filename: str = ""

    def to_json(self) -> dict:
        return {"type": "diff", "diff": self.diff, "filename": self.filename}


@dataclass
class RichOutput:
    """Generic rich visual output for frontends that support it.

    The ``kind`` field identifies the rendering type; ``data`` is the
    kind-specific payload.  The terminal falls back to ``fallback_text``;
    the web terminal (``nooa-term``) renders the real thing.

    Built-in kinds recognised by the web terminal:

    ``"plotly"``
        ``data["figure_json"]`` — result of ``plotly.Figure.to_json()``.
        Rendered via Plotly.js.

    ``"html"``
        ``data["html"]`` — arbitrary safe HTML fragment.

    ``"image"``
        ``data["src"]`` — a ``data:`` URI or URL.
        ``data["alt"]`` — optional alt-text.

    ``"vega"``
        ``data["spec"]`` — a Vega-Lite or Vega spec dict.
        Rendered via the Vega CDN.

    ``"dataframe"``
        ``data["columns"]``, ``data["rows"]`` — rendered as a sortable table.

    Any other ``kind`` value is forwarded as-is to the frontend; unknown
    kinds are displayed as a JSON code block.
    """

    kind: str
    data: dict
    title: str = ""
    fallback_text: str = ""  # shown in terminal / non-graphical frontends

    def to_json(self) -> dict:
        return {
            "type": "rich",
            "kind": self.kind,
            "data": self.data,
            "title": self.title,
            "fallback_text": self.fallback_text,
        }


# ---------------------------------------------------------------------------
# _RichReplayPayload — internal sentinel for interleaved rich-content replay
# ---------------------------------------------------------------------------


@dataclass
class _RichReplayPayload:
    """Carry a raw WebPublisher payload through an output list.

    Not part of the public ``Output`` union — intercepted by ``CommandHandler``
    and the ``--continue`` startup path before reaching any frontend renderer.
    Each instance is POSTed to ``NEMO_OO_RICH_URL`` in sequence so that plots
    appear at their correct inline positions between history turns.
    """

    payload: dict


# ---------------------------------------------------------------------------
# Union alias used in type annotations
# ---------------------------------------------------------------------------

Output = (
    TextOutput
    | TableOutput
    | HelpOutput
    | AgentMessage
    | CodeExecution
    | StartupInfo
    | ClearScreen
    | Thinking
    | StopReasonOutput
    | BashOutput
    | CommandStatus
    | DiffOutput
    | RichOutput
    | HistoryReplay
)
