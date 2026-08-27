# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event explorer model, renderer, and in-app subview."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from typing import Any

from .explorer_base import (
    ExplorerChecklistOption,
    ExplorerConfig,
    ExplorerOption,
    ExplorerView,
    highlight_style_code,
    highlight_terms,
    search_terms,
    wrap_plain_line,
)
from .subapp import SubviewKeyResult
from .theme import ThemeSyntax, create_theme, get_syntax_theme


def highlight_terms_with_current(
    text: str, terms: list[str], current_occurrence: int | None = None
) -> str:
    """ANSI-highlight query terms, using a distinct color for the selected occurrence."""
    if not terms:
        return text
    import re

    pattern = re.compile(
        "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)), re.IGNORECASE
    )
    seen = -1

    def replace(match: re.Match[str]) -> str:
        nonlocal seen
        seen += 1
        style = highlight_style_code(
            current=current_occurrence is not None and seen == current_occurrence
        )
        return f"{style}{match.group(0)}\x1b[0m"

    return pattern.sub(replace, text)


@dataclass
class EventExplorerRow:
    tag: str
    event_type: str
    summary: str
    search_text: str
    detail: str
    code: str | None = None
    code_language: str = "python"
    markdown: str | None = None


class EventExplorerModel:
    """Searchable, keyboard-navigable event list."""

    def __init__(self, rows: list[EventExplorerRow]) -> None:
        self.rows = rows
        self.query = ""
        self.matches = list(range(len(rows)))
        self.cursor = max(len(self.matches) - 1, 0)
        self.detail_offset = 0
        self.focus = "list"
        self.search_active = False
        self.search_line_cursor = 0
        self._last_detail_line_count = 0
        self._last_detail_match_lines: list[int] = []
        self._last_detail_match_occurrences: list[tuple[int, int]] = []
        self._last_detail_visible_lines = 0
        self.enabled_types = {row.event_type for row in rows}
        self.sort_mode = "recorded"

    @property
    def empty(self) -> bool:
        return not self.rows

    @property
    def current_index(self) -> int | None:
        if not self.matches:
            return None
        self.cursor = min(max(self.cursor, 0), len(self.matches) - 1)
        return self.matches[self.cursor]

    @property
    def current(self) -> EventExplorerRow | None:
        idx = self.current_index
        return None if idx is None else self.rows[idx]

    def set_query(self, query: str) -> None:
        self.query = query
        words = [w.lower() for w in query.split() if w.strip()]
        self.matches = [
            i
            for i, row in enumerate(self.rows)
            if row.event_type in self.enabled_types
            and (not words or all(word in row.search_text.lower() for word in words))
        ]
        if self.sort_mode == "type":
            self.matches.sort(key=lambda index: (self.rows[index].event_type, index))
        self.cursor = 0
        self.detail_offset = 0
        self.search_line_cursor = 0

    def set_enabled_types(self, enabled: set[str]) -> None:
        self.enabled_types = set(enabled)
        self.set_query(self.query)

    def set_sort_mode(self, mode: str) -> None:
        self.sort_mode = mode
        self.set_query(self.query)

    def edit_query(self, text: str) -> None:
        self.set_query(text)
        self.search_active = True

    def move(self, delta: int) -> None:
        if not self.matches:
            return
        self.cursor = min(max(self.cursor + delta, 0), len(self.matches) - 1)
        self.detail_offset = 0
        self.search_line_cursor = 0

    def current_search_line(self) -> int | None:
        if self._last_detail_match_occurrences:
            match_count = len(self._last_detail_match_occurrences)
            self.search_line_cursor = min(max(self.search_line_cursor, 0), match_count - 1)
            return self._last_detail_match_occurrences[self.search_line_cursor][0]
        if self._last_detail_match_lines:
            match_count = len(self._last_detail_match_lines)
            self.search_line_cursor = min(max(self.search_line_cursor, 0), match_count - 1)
            return self._last_detail_match_lines[self.search_line_cursor]
        return None

    def center_detail_on_line(self, line: int, visible_lines: int | None = None) -> None:
        visible = max(
            visible_lines if visible_lines is not None else self._last_detail_visible_lines, 1
        )
        self.detail_offset = max(line - visible // 2, 0)
        self.clamp_detail_offset(visible)

    def move_search_occurrence(self, delta: int) -> None:
        if not self.matches:
            return
        match_count = len(self._last_detail_match_occurrences) or len(self._last_detail_match_lines)
        if not match_count:
            self.move(delta)
            return
        next_cursor = self.search_line_cursor + delta
        if 0 <= next_cursor < match_count:
            self.search_line_cursor = next_cursor
            target = self.current_search_line()
            if target is not None:
                self.center_detail_on_line(target)
            return
        old_cursor = self.cursor
        self.move(delta)
        if self.cursor != old_cursor:
            self.search_line_cursor = 0 if delta > 0 else 10**9

    def move_or_scroll(self, delta: int) -> None:
        if self.search_active and self.focus == "list":
            self.move_search_occurrence(delta)
        elif self.focus == "list":
            self.move(delta)
        else:
            self.scroll_detail(delta)

    def jump_home(self) -> None:
        self.cursor = 0
        self.detail_offset = 0
        self.search_line_cursor = 0

    def jump_end(self) -> None:
        if self.matches:
            self.cursor = len(self.matches) - 1
            self.detail_offset = 0
            self.search_line_cursor = 0

    def toggle_focus(self) -> None:
        self.focus = "detail" if self.focus == "list" else "list"

    def scroll_detail(self, delta: int) -> None:
        max_offset = max(self._last_detail_line_count - 1, 0)
        self.detail_offset = min(max(self.detail_offset + delta, 0), max_offset)

    def page_detail(self, delta: int) -> None:
        self.scroll_detail(delta)

    def clamp_detail_offset(self, visible_lines: int) -> None:
        max_offset = max(self._last_detail_line_count - max(visible_lines, 1), 0)
        self.detail_offset = min(max(self.detail_offset, 0), max_offset)


class EventExplorerView(ExplorerView):
    """Event-specific configuration for the shared full-screen browser."""

    item_name = "event"
    list_heading = "  id       event type              summary"

    def __init__(self, event_manager: Any) -> None:
        super().__init__(
            EventExplorerModel(build_event_rows(event_manager)),
            ExplorerConfig(
                title="Event Explorer",
                detail_pane_name="event detail",
                empty_message="No events recorded.",
                no_match_message="No events matching {query!r}.",
            ),
        )
        event_types = tuple(sorted(self.model.enabled_types, key=str.casefold))

        self.configure_options(
            ExplorerChecklistOption(
                "event-types",
                "Event types",
                (("__all__", "All"), *((event_type, event_type) for event_type in event_types)),
                set(event_types),
                self.model.set_enabled_types,
                all_value="__all__",
            ),
            ExplorerOption(
                "sort",
                "Sort",
                (("recorded", "Recorded"), ("type", "Event type")),
                "recorded",
                self.model.set_sort_mode,
            ),
        )

    def format_row(self, row: EventExplorerRow, width: int) -> str:
        return f"{row.tag:<8} {row.event_type:<22} {row.summary}"[:width]

    def detail_lines(self, row: EventExplorerRow, width: int) -> list[str]:
        return highlighted_detail_lines(row, width, self.model.query)

    def handle_action(self, action: str, row: Any) -> SubviewKeyResult:
        return "ignored"

    def handle_key(self, action: str, value: str = "") -> SubviewKeyResult:
        model = self.model
        interaction = self.handle_interaction_action(action)
        if interaction != "ignored":
            return interaction
        if action == "quit":
            if model.search_active:
                model.edit_query(model.query + "q")
                return "handled"
            return "close"
        if action == "resume":
            if model.search_active:
                model.edit_query(model.query + "r")
                return "handled"
            return "ignored"
        if action == "escape":
            if model.search_active:
                model.search_active = False
            elif model.query:
                model.set_query("")
            else:
                return "ignored"
        elif action == "enter":
            model.search_active = False
        elif action == "slash":
            model.search_active = True
        elif action == "backspace":
            if model.search_active:
                model.edit_query(model.query[:-1])
        elif action == "tab":
            model.toggle_focus()
        elif action == "down":
            model.move_or_scroll(+1)
        elif action == "j":
            if model.search_active:
                model.edit_query(model.query + "j")
            else:
                model.move_or_scroll(+1)
        elif action == "up":
            model.move_or_scroll(-1)
        elif action == "k":
            if model.search_active:
                model.edit_query(model.query + "k")
            else:
                model.move_or_scroll(-1)
        elif action == "page_down":
            if not model.search_active:
                model.page_detail(+10)
        elif action == "page_up":
            if not model.search_active:
                model.page_detail(-10)
        elif action == "home":
            if not model.search_active:
                model.jump_home()
        elif action == "end":
            if not model.search_active:
                model.jump_end()
        elif action == "scroll_down":
            model.focus = "detail"
            model.scroll_detail(+3)
        elif action == "scroll_up":
            model.focus = "detail"
            model.scroll_detail(-3)
        elif action == "text":
            if model.search_active and value and value.isprintable():
                model.edit_query(model.query + value)
            else:
                return "ignored"
        else:
            return "ignored"
        return "handled"

    def on_open(self) -> None:
        pass

    def on_close(self) -> None:
        pass


def _event_to_mapping(event: Any) -> dict[str, Any]:
    if hasattr(event, "model_dump"):
        try:
            return event.model_dump()
        except Exception:
            pass
    if hasattr(event, "dict"):
        try:
            return event.dict()
        except Exception:
            pass
    return {"repr": repr(event)}


def _event_summary(event: Any, event_type: str) -> str:
    if event_type == "ToolCallEvent":
        name = str(getattr(event, "name", "?"))
        args = getattr(event, "arguments", {})
        if name == "execute_python" and isinstance(args, dict):
            code = str(args.get("code", ""))
            for line in code.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    return f"{name} — {stripped[:90]}"
        return name
    if event_type == "PythonOutput":
        status = str(getattr(event, "execution_status", "?")).rsplit(".", 1)[-1]
        stdout = str(getattr(event, "stdout", "") or "").strip()
        error = str(getattr(event, "error", "") or "").strip()
        if error:
            return f"{status} — {error.splitlines()[-1][:90]}"
        if stdout:
            return f"{status} — {stdout.splitlines()[0][:90]}"
        return status
    if event_type in {"SessionUserMessage", "TUIUserInput"}:
        field = "content" if event_type == "SessionUserMessage" else "text"
        return str(getattr(event, field, "") or "")[:100]
    if event_type == "Task":
        prompt = str(getattr(event, "prompt", "") or "").strip()
        return prompt.splitlines()[0][:100] if prompt else ""
    data = _event_to_mapping(event)
    for key in ("content", "text", "message", "name", "summary"):
        value = data.get(key)
        if value:
            return str(value).strip().splitlines()[0][:100]
    return repr(event)[:100]


def _extract_fenced_code(text: str) -> tuple[str, str] | None:
    import re

    match = re.search(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", text, re.DOTALL)
    if not match:
        return None
    language = match.group(1).strip() or "python"
    return match.group(2).rstrip("\n"), language


def _event_code(event: Any, event_type: str) -> tuple[str | None, str]:
    if event_type == "ToolCallEvent" and getattr(event, "name", None) == "execute_python":
        args = getattr(event, "arguments", {})
        if isinstance(args, dict) and args.get("code"):
            return str(args["code"]), "python"
    for value in _event_to_mapping(event).values():
        if isinstance(value, str):
            extracted = _extract_fenced_code(value)
            if extracted is not None:
                return extracted
    return None, "python"


def _format_detail(tag: str, event: Any) -> str:
    event_type = str(getattr(event, "event_type", type(event).__name__))
    data = _event_to_mapping(event)
    fields = [(k, v) for k, v in data.items() if k != "event_type"]
    if fields:
        body = f"{event_type}(" + ", ".join(f"{k}={v!r}" for k, v in fields) + ")"
    else:
        body = repr(event)
    return f"tag = {tag!r}\ntype = {event_type!r}\n\nevent = {body}"


_NOISE_FIELDS = {
    "event_type",
    "id",
    "event_id",
    "uuid",
    "timestamp",
    "created_at",
    "updated_at",
    "metadata",
    "children_tags",
    "images",
    "execution_count",
    "explicit_return",
    "parent_call_id",
    "call_id",
    "generation_id",
    "parent_generation_id",
    "trace_id",
    "span_id",
    "tool_call_id",
}

_RENDERED_FIELD_ORDER = {
    "AgentMessage": ("content",),
    "TUIAgentMessage": ("content",),
    "Message": ("content",),
    "SessionUserMessage": ("content",),
    "TUIUserInput": ("text",),
    "Task": ("prompt",),
    "Reasoning": ("content",),
    "Error": ("content",),
    "Feedback": ("content",),
    "DebugTrace": ("content",),
    "TextOnlyReply": ("content", "finish_reason", "route"),
    "LLMOutput": ("content",),
    "PythonOutput": ("stdout", "stderr", "error", "value"),
    "ToolCallEvent": ("name", "arguments", "result"),
    "Summary": ("summary_text", "summary_tag", "replaced_range"),
    "SessionResumed": ("session_id", "restored"),
    "TuiSessionResumed": ("session_id", "restored"),
    "SessionCleared": ("session_id",),
    "TuiSessionCleared": ("session_id",),
    "LLMComplete": ("model", "finish_reason", "usage", "content"),
    "SessionStarted": ("host", "model", "agent", "working_directory"),
    "TUISessionStart": ("model", "agent_cls", "working_dir"),
    "SessionTitleUpdated": ("title", "user_set"),
    "TUISessionRename": ("name", "user_named"),
}

_MARKDOWN_FIELDS = {"content", "text", "prompt", "summary_text"}
_CODE_EVENT_FIELDS = {
    "PythonOutput": {"stdout": "text", "stderr": "text", "error": "pytb"},
}


def _escape_terminal_controls(text: str) -> str:
    safe: list[str] = []
    for char in text:
        codepoint = ord(char)
        if char in {"\n", "\t"}:
            safe.append(char)
        elif char == "\r":
            safe.append("\\r")
        elif codepoint < 32 or codepoint == 127 or 0x80 <= codepoint <= 0x9F:
            safe.append(f"\\x{codepoint:02x}")
        else:
            safe.append(char)
    return "".join(safe)


def _safe_inline(value: Any, max_len: int = 160) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = repr(value)
    text = _escape_terminal_controls(" ".join(str(text).split()))
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def _short_id(value: Any, max_len: int = 8) -> str:
    text = _safe_inline(value, 80)
    if len(text) <= max_len:
        return text
    return text[:max_len]


def _pretty_timestamp(value: Any) -> str:
    if isinstance(value, datetime.datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, datetime.date):
        return value.isoformat()
    text = _safe_inline(value, 80)
    if not text:
        return ""
    if "T" in text and len(text) >= 19:
        return text[:19].replace("T", " ")
    if text.startswith("datetime.datetime("):
        parts = text.removeprefix("datetime.datetime(").split(")", 1)[0].split(",")
        try:
            nums = [int(part.strip()) for part in parts[:6]]
        except ValueError:
            return text
        while len(nums) < 6:
            nums.append(0)
        return (
            f"{nums[0]:04d}-{nums[1]:02d}-{nums[2]:02d} {nums[3]:02d}:{nums[4]:02d}:{nums[5]:02d}"
        )
    return text


def _event_header_line(tag: str, event_type: str, data: dict[str, Any]) -> str:
    parts = [f"**[{tag}]** *{event_type}*"]
    timestamp = data.get("timestamp") or data.get("created_at") or data.get("updated_at")
    if not _is_empty_event_field(timestamp):
        parts.append(_pretty_timestamp(timestamp))
    event_id = data.get("id") or data.get("event_id") or data.get("uuid")
    if not _is_empty_event_field(event_id):
        parts.append(f"id={_short_id(event_id)}")
    tool_call_id = data.get("tool_call_id")
    if not _is_empty_event_field(tool_call_id):
        parts.append(f"tool={_short_id(tool_call_id, 12)}")
    call_id = data.get("call_id") or (
        data.get("metadata", {}).get("call_id") if isinstance(data.get("metadata"), dict) else None
    )
    if not _is_empty_event_field(call_id):
        parts.append(f"call={_short_id(call_id)}")
    return " · ".join(parts)


def _metadata_footer_line(data: dict[str, Any]) -> str:
    parts = []
    for field, value in data.items():
        if field == "event_type":
            continue
        if field in _NOISE_FIELDS and not _is_empty_event_field(value):
            parts.append(f"{field}={_safe_inline(value, 96)}")
    if not parts:
        return ""
    return "_metadata: " + " · ".join(parts) + "_"


def _append_metadata_footer(lines: list[str], data: dict[str, Any]) -> None:
    footer = _metadata_footer_line(data)
    if footer:
        lines.extend(["", "---", "", footer])


def _json_block(value: Any, language: str = "json") -> str:
    try:
        text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = repr(value)
        language = "python"
    return _markdown_code_block(text, language)


def _markdown_code_block(value: Any, language: str = "") -> str:
    text = value if isinstance(value, str) else repr(value)
    text = _escape_terminal_controls(text)
    fence = "```"
    while fence in text:
        fence += "`"
    lang = language.strip()
    opener = f"{fence}{lang}" if lang else fence
    return f"{opener}\n{text.rstrip()}\n{fence}"


def _is_empty_event_field(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _field_title(field: str) -> str:
    return field.replace("_", " ").title()


def _event_field_markdown(value: Any, *, language: str | None = None) -> str:
    if isinstance(value, str):
        extracted = _extract_fenced_code(value)
        if extracted is not None:
            return _escape_terminal_controls(value.rstrip())
        if language is not None:
            return _markdown_code_block(value, language)
        return _escape_terminal_controls(value.rstrip())
    if isinstance(value, dict | list | tuple):
        return _json_block(value)
    return _markdown_code_block(value, "python")


def _ordered_content_fields(data: dict[str, Any], event_type: str) -> list[str]:
    preferred = [field for field in _RENDERED_FIELD_ORDER.get(event_type, ()) if field in data]
    remaining = [
        field
        for field in data
        if field not in preferred
        and field not in _NOISE_FIELDS
        and not _is_empty_event_field(data[field])
    ]
    return preferred + remaining


def _append_section(lines: list[str], title: str, body: str) -> None:
    body = body.rstrip()
    if not body:
        return
    lines.extend(["", f"## {title}", "", body])


def _append_compact_field(lines: list[str], title: str, body: str) -> None:
    """Render short metadata on one line instead of as a spacious section."""
    body = body.strip()
    if not body:
        return
    lines.extend(["", f"**{title}:** {body}"])


def _is_compact_event_field(field: str, value: Any) -> bool:
    """Return whether a field is short metadata rather than substantive content."""
    if field in _MARKDOWN_FIELDS or isinstance(value, dict | list | tuple):
        return False
    return "\n" not in str(value) and len(_safe_inline(value)) <= 80


def _append_event_field(
    lines: list[str], field: str, value: Any, *, language: str | None = None
) -> None:
    title = _field_title(field)
    if language is None and _is_compact_event_field(field, value):
        _append_compact_field(lines, title, f"`{_safe_inline(value)}`")
    else:
        _append_section(lines, title, _event_field_markdown(value, language=language))


def _event_markdown(tag: str, event: Any, event_type: str) -> str | None:
    data = _event_to_mapping(event)
    lines = [_event_header_line(tag, event_type, data)]

    if event_type in {"AgentMessage", "TUIAgentMessage", "Message"}:
        content = data.get("content")
        if not _is_empty_event_field(content):
            lines.extend(["", _escape_terminal_controls(str(content).rstrip())])
        _append_metadata_footer(lines, data)
        return "\n".join(lines)

    if event_type in {
        "SessionUserMessage",
        "TUIUserInput",
        "Task",
        "Reasoning",
        "Error",
        "Feedback",
        "DebugTrace",
    }:
        field = (
            "prompt"
            if event_type == "Task"
            else "content"
            if event_type == "SessionUserMessage"
            else "text"
            if event_type == "TUIUserInput"
            else "content"
        )
        value = data.get(field)
        if not _is_empty_event_field(value):
            title = (
                "User input"
                if event_type in {"SessionUserMessage", "TUIUserInput"}
                else _field_title(field)
            )
            _append_section(lines, title, _event_field_markdown(value))

    elif event_type == "ToolCallEvent":
        name = str(data.get("name", "tool"))
        args = data.get("arguments", {})
        _append_compact_field(lines, "Tool", f"`{_safe_inline(name)}`")
        if name == "execute_python" and isinstance(args, dict) and args.get("code"):
            _append_section(lines, "Python", _markdown_code_block(str(args["code"]), "python"))
            extras = {k: v for k, v in args.items() if k != "code" and not _is_empty_event_field(v)}
            if extras:
                _append_section(lines, "Arguments", _json_block(extras))
        elif not _is_empty_event_field(args):
            _append_section(lines, "Arguments", _json_block(args))
        result = data.get("result")
        if not _is_empty_event_field(result):
            _append_event_field(lines, "result", result)

    elif event_type == "PythonOutput":
        status = _safe_inline(data.get("execution_status", ""))
        if status:
            _append_compact_field(lines, "Status", f"`{status}`")
        for field, language in _CODE_EVENT_FIELDS["PythonOutput"].items():
            value = data.get(field)
            if not _is_empty_event_field(value):
                _append_section(
                    lines, _field_title(field), _event_field_markdown(value, language=language)
                )
        value = data.get("value")
        if not _is_empty_event_field(value):
            _append_event_field(lines, "value", value)

    elif event_type == "LLMOutput":
        content = data.get("content")
        if not _is_empty_event_field(content):
            language = "json" if str(content).lstrip().startswith(("{", "[")) else "python"
            _append_section(lines, "LLM output", _markdown_code_block(str(content), language))

    elif event_type == "Summary":
        summary = data.get("summary_text")
        if not _is_empty_event_field(summary):
            lines.extend(["", _escape_terminal_controls(str(summary).rstrip())])
        summary_bits = []
        for field in ("summary_tag", "replaced_range", "children_tags"):
            value = data.get(field)
            if not _is_empty_event_field(value):
                summary_bits.append(f"**{_field_title(field)}:** `{_safe_inline(value)}`")
        if summary_bits:
            lines.extend(["", " ".join(summary_bits)])

    else:
        for field in _ordered_content_fields(data, event_type):
            value = data[field]
            if _is_empty_event_field(value):
                continue
            language = _CODE_EVENT_FIELDS.get(event_type, {}).get(field)
            if field in _MARKDOWN_FIELDS and language is None and isinstance(value, str):
                _append_section(
                    lines, _field_title(field), _escape_terminal_controls(value.rstrip())
                )
            else:
                _append_event_field(lines, field, value, language=language)

    rendered_fields = "\n".join(lines)
    # Fallback: if no content sections were added (only the header line exists),
    # render all non-noise fields so unknown event types still show something useful.
    if rendered_fields.strip() == _event_header_line(tag, event_type, data).strip():
        for field, value in data.items():
            if field == "event_type" or field in _NOISE_FIELDS or _is_empty_event_field(value):
                continue
            _append_event_field(lines, field, value)
    _append_metadata_footer(lines, data)
    return "\n".join(lines)


def build_event_rows(event_manager: Any) -> list[EventExplorerRow]:
    """Build explorer rows from an EventManager-like object."""
    rows: list[EventExplorerRow] = []
    if hasattr(event_manager, "items"):
        items = list(event_manager.items())
    else:
        items = [(tag, event_manager.get(tag)) for tag in event_manager.keys()]
    for tag, event in items:
        if event is None:
            continue
        event_type = str(getattr(event, "event_type", type(event).__name__))
        summary = _event_summary(event, event_type)
        detail = _format_detail(str(tag), event)
        code, code_language = _event_code(event, event_type)
        markdown = _event_markdown(str(tag), event, event_type)
        search_text = f"{tag} {event_type} {summary} {detail} {code or ''} {markdown or ''}"
        rows.append(
            EventExplorerRow(
                tag=str(tag),
                event_type=event_type,
                summary=summary,
                search_text=search_text,
                detail=detail,
                code=code,
                code_language=code_language,
                markdown=markdown,
            )
        )
    return rows


def detail_match_lines(row: EventExplorerRow, width: int, query: str) -> list[int]:
    terms = search_terms(query)
    if not terms:
        return []
    lower_terms = [term.lower() for term in terms]
    lines = wrapped_detail_lines(row, width)
    matches: list[int] = []
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(term in lower for term in lower_terms):
            matches.append(i)
    return matches


def detail_match_occurrences(
    row: EventExplorerRow, width: int, query: str
) -> list[tuple[int, int]]:
    terms = search_terms(query)
    if not terms:
        return []
    import re

    pattern = re.compile(
        "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)), re.IGNORECASE
    )
    matches: list[tuple[int, int]] = []
    for line_no, line in enumerate(wrapped_detail_lines(row, width)):
        for occurrence_no, _match in enumerate(pattern.finditer(line)):
            matches.append((line_no, occurrence_no))
    return matches


def wrapped_detail_lines(row: EventExplorerRow, width: int) -> list[str]:
    """Return width-wrapped detail lines, with extracted code at the top."""
    width = max(int(width), 20)
    lines: list[str] = []
    if row.markdown is not None:
        for line in row.markdown.splitlines() or [""]:
            lines.extend(wrap_plain_line(line, width))
        return lines
    if row.code:
        lines.append(f"code ({row.code_language}):")
        for line in row.code.rstrip("\n").splitlines() or [""]:
            lines.extend(wrap_plain_line(line, width))
        lines.append("")
    lines.append("event:")
    for line in row.detail.splitlines():
        lines.extend(wrap_plain_line(line, width))
    return lines


def _highlight_syntax(text: str, width: int, language: str = "python") -> str:
    try:
        import io

        from rich.console import Console as RichConsole

        render_width = max(int(width), 20)
        buf = io.StringIO()
        console = RichConsole(
            file=buf,
            force_terminal=True,
            color_system="256",
            theme=create_theme(),
            width=render_width,
            _environ={"COLUMNS": str(render_width), "LINES": "25"},
        )
        console.print(
            ThemeSyntax(
                text,
                language or "python",
                word_wrap=True,
                line_numbers=False,
                background_color="default",
            )
        )
        return buf.getvalue()
    except Exception:
        wrapped: list[str] = []
        for line in text.splitlines():
            wrapped.extend(wrap_plain_line(line, width))
        return "\n".join(wrapped)


def _render_markdown(markdown: str, width: int) -> str:
    try:
        import io

        from rich.console import Console as RichConsole
        from rich.markdown import Markdown

        render_width = max(int(width), 20)
        buf = io.StringIO()
        console = RichConsole(
            file=buf,
            force_terminal=True,
            color_system="256",
            theme=create_theme(),
            width=render_width,
            _environ={"COLUMNS": str(render_width), "LINES": "25"},
        )
        console.print(Markdown(markdown, code_theme=get_syntax_theme()))
        return buf.getvalue()
    except Exception:
        wrapped: list[str] = []
        for line in markdown.splitlines():
            wrapped.extend(wrap_plain_line(line, width))
        return "\n".join(wrapped)


_EVENT_HEADER_STYLE = "\x1b[1;38;5;230;48;5;238m"


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _style_event_header_line(lines: list[str], width: int) -> list[str]:
    if not lines:
        return lines
    styled = list(lines)
    for i, line in enumerate(styled):
        if not line.strip():
            continue
        plain = _strip_ansi(line).strip()
        # Header lines render as "[tag] EventType ..." after Rich strips markdown emphasis.
        if not plain.startswith("["):
            return styled
        banner = f" {plain} "[: max(width, 1)]
        styled[i] = f"{_EVENT_HEADER_STYLE}{banner.ljust(width)}\x1b[0m"
        return styled
    return styled


def highlighted_detail_lines(
    row: EventExplorerRow,
    width: int,
    query: str = "",
    current_match_line: int | None = None,
    current_match_occurrence: int | None = None,
) -> list[str]:
    """Return detail lines with code and Python-repr event syntax highlighted."""
    width = max(int(width), 20)
    lines: list[str] = []
    terms = search_terms(query)

    def mark(line: str) -> str:
        line_no = len(lines)
        if current_match_line is not None and line_no == current_match_line:
            return highlight_terms_with_current(line, terms, current_match_occurrence)
        return highlight_terms(line, terms)

    if terms:
        # Search navigation is computed from wrapped plain text. Use the same
        # representation for search rendering so the selected occurrence index
        # stays correct even when many matches share one line.
        for line in wrapped_detail_lines(row, width):
            lines.append(mark(line))
        return lines

    if row.markdown is not None:
        rendered_lines = _render_markdown(row.markdown, width).splitlines() or [""]
        return _style_event_header_line(rendered_lines, width)
    if row.code:
        lines.append(f"code ({row.code_language}):")
        code_lines = _highlight_syntax(
            row.code.rstrip("\n"), width, row.code_language
        ).splitlines() or [""]
        lines.extend(code_lines)
        lines.append("")
    lines.append("event:")
    rendered = _highlight_syntax(row.detail, width, "python")
    lines.extend(rendered.splitlines() or [""])
    return lines
