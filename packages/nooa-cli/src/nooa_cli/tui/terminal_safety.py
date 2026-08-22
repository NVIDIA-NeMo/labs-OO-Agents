# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal-output normalization shared by the native-scrollback TUI.

The transcript accepts ANSI produced by Rich, but it must never accept terminal
*commands* from user, model, subprocess, exception, or server text.  Styling
(SGR) and hyperlinks (OSC 8) are harmless presentation metadata; cursor moves,
screen erases, scroll-region changes, device queries, BEL, CR, and every other
control can invalidate prompt_toolkit's idea of the live-region origin.
"""

from __future__ import annotations

import re
import shutil
from urllib.parse import urlsplit

from rich.cells import split_graphemes

_ESC = "\x1b"

_MAX_SAFE_HTTP_URL_LENGTH = 2_048

# This expression is used only after ``sanitize_transcript_ansi`` has reduced
# the language to SGR and OSC-8.  It deliberately recognizes both OSC
# terminators because Rich versions differ between BEL and ST.
_SAFE_ANSI_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*m|\x1b\]8;[^\x07\x1b\r\n]*;(?:[^\x07\x1b\r\n]*)(?:\x07|\x1b\\)"
)


def _visible_control(codepoint: int) -> str:
    width = 2 if codepoint <= 0xFF else 4
    prefix = "x" if width == 2 else "u"
    return f"\\{prefix}{codepoint:0{width}x}"


def _find_csi_end(value: str, start: int) -> int | None:
    """Return the index of a CSI final byte, or ``None`` when incomplete."""
    for index in range(start, len(value)):
        codepoint = ord(value[index])
        if 0x40 <= codepoint <= 0x7E:
            return index
        if not 0x20 <= codepoint <= 0x3F:
            return None
    return None


def _find_osc_end(value: str, start: int) -> tuple[int, int] | None:
    """Return ``(payload_end, sequence_end)`` for BEL/ST-terminated OSC."""
    index = start
    while index < len(value):
        character = value[index]
        if character == "\x07":
            return index, index + 1
        if character == _ESC and index + 1 < len(value) and value[index + 1] == "\\":
            return index, index + 2
        codepoint = ord(character)
        if (
            character in "\r\n"
            or codepoint < 0x20
            or codepoint == 0x7F
            or 0x80 <= codepoint <= 0x9F
        ):
            return None
        index += 1
    return None


def sanitize_transcript_ansi(value: str) -> str:
    """Keep printable text, newlines, SGR, and OSC-8; expose other controls.

    Unsupported escape sequences are rendered visibly (for example,
    ``ESC[2J`` becomes ``\\x1b[2J``), so diagnostic/code output remains useful
    without being allowed to clear or reposition the real terminal.
    """
    value = str(value)
    output: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        codepoint = ord(character)

        if character == _ESC:
            # CSI: retain only Select Graphic Rendition's numeric parameter
            # grammar. Some private xterm commands also end in ``m``.
            if index + 1 < len(value) and value[index + 1] == "[":
                end = _find_csi_end(value, index + 2)
                if end is not None:
                    sequence = value[index : end + 1]
                    sgr_parameters = value[index + 2 : end]
                    if value[end] == "m" and all(
                        character.isdigit() or character in ":;" for character in sgr_parameters
                    ):
                        output.append(sequence)
                    else:
                        output.append("\\x1b" + sequence[1:])
                    index = end + 1
                    continue

            # OSC: retain only well-formed hyperlinks.  Clipboard writes and
            # title changes intentionally remain outside the transcript path.
            if index + 1 < len(value) and value[index + 1] == "]":
                end_info = _find_osc_end(value, index + 2)
                if end_info is not None:
                    payload_end, sequence_end = end_info
                    payload = value[index + 2 : payload_end]
                    sequence = value[index:sequence_end]
                    if payload.startswith("8;") and payload.count(";") >= 2:
                        output.append(sequence)
                    else:
                        terminator = "\\x07" if value[payload_end] == "\x07" else "\\x1b\\\\"
                        output.append("\\x1b]" + payload + terminator)
                    index = sequence_end
                    continue

            output.append("\\x1b")
            index += 1
            continue

        if character == "\r":
            if index + 1 < len(value) and value[index + 1] == "\n":
                output.append("\n")
                index += 2
            else:
                output.append("\\r")
                index += 1
            continue
        if character == "\n":
            output.append(character)
        elif character == "\t":
            output.append("    ")
        elif codepoint < 0x20 or codepoint == 0x7F or 0x80 <= codepoint <= 0x9F:
            output.append(_visible_control(codepoint))
        else:
            output.append(character)
        index += 1
    return "".join(output)


def safe_http_url(value: str | None) -> str | None:
    """Return a control-free HTTP(S) URL suitable for an explicit browser launch."""
    if (
        not value
        or len(value) > _MAX_SAFE_HTTP_URL_LENGTH
        or any(
            character.isspace() or ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F
            for character in value
        )
    ):
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return value


def hyperlink_at_plain_offset(value: str, offset: int) -> str | None:
    """Return the active safe OSC-8 target at one ANSI-stripped character offset."""
    for start, stop, target in safe_hyperlink_spans(value):
        if start <= offset < stop:
            return target
        if start > offset:
            break
    return None


def safe_hyperlink_spans(value: str) -> tuple[tuple[int, int, str], ...]:
    """Return safe OSC-8 targets as spans in ANSI-stripped character offsets."""
    safe = sanitize_transcript_ansi(value)
    spans: list[tuple[int, int, str]] = []
    active: tuple[int, str] | None = None
    plain_offset = 0
    index = 0
    while index < len(safe):
        match = _SAFE_ANSI_RE.match(safe, index)
        if match is not None:
            sequence = match.group(0)
            if sequence.startswith(f"{_ESC}]8;"):
                if active is not None and active[0] < plain_offset:
                    spans.append((active[0], plain_offset, active[1]))
                payload = sequence[4:-1] if sequence.endswith("\x07") else sequence[4:-2]
                _parameters, _separator, target = payload.partition(";")
                url = safe_http_url(target)
                active = (plain_offset, url) if url is not None else None
            index = match.end()
            continue
        plain_offset += 1
        index += 1
    if active is not None and active[0] < plain_offset:
        spans.append((active[0], plain_offset, active[1]))
    return tuple(spans)


def strip_safe_ansi(value: str) -> str:
    """Strip the presentation escapes accepted by ``sanitize_transcript_ansi``."""
    return _SAFE_ANSI_RE.sub("", value)


def project_prompt_toolkit_ansi(value: str) -> str:
    """Project safe transcript ANSI onto prompt_toolkit's smaller ANSI grammar.

    prompt_toolkit's ``ANSI`` parser handles conventional semicolon SGR, but
    not OSC-8 hyperlinks or colon-delimited SGR. Strip only that unsupported
    presentation metadata so its parameter bytes can never become visible text.
    Input is sanitized here as a defense-in-depth boundary for direct callers.
    """
    safe = sanitize_transcript_ansi(value)

    def supported_sgr(match: re.Match[str]) -> str:
        return "" if ":" in match.group(1) else match.group(0)

    without_unsupported_sgr = re.sub(
        r"\x1b\[([0-9:;]*)m",
        supported_sgr,
        safe,
    )
    return re.sub(
        r"\x1b\]8;[^\x07\x1b\r\n]*;[^\x07\x1b\r\n]*(?:\x07|\x1b\\)",
        "",
        without_unsupported_sgr,
    )


def fallback_transcript_columns(default: int = 120) -> int:
    """Safe native-scrollback width when no prompt_toolkit Output exists."""
    try:
        physical = shutil.get_terminal_size((default, 24)).columns
    except Exception:
        physical = default
    return max(int(physical) - 1, 1)


def _wrap_safe_ansi(value: str, columns: int) -> str:
    """Hard-wrap sanitized ANSI without splitting styles or graphemes.

    ``value`` must already contain only printable text, newlines, SGR, and
    OSC-8.  Explicit newlines keep every physical line at or below ``columns``
    cells, avoiding the terminal's delayed-autowrap state at the last column.
    """
    columns = max(int(columns), 1)
    output: list[str] = []
    line_cells = 0
    index = 0
    while index < len(value):
        if value[index] == _ESC:
            match = _SAFE_ANSI_RE.match(value, index)
            if match is not None:
                output.append(match.group(0))
                index = match.end()
                continue
            # The sanitizer should make this unreachable.  Keep the fallback
            # visibly inert if this helper is ever called independently.
            output.append(r"\x1b")
            index += 1
            continue
        if value[index] == "\n":
            output.append("\n")
            line_cells = 0
            index += 1
            continue

        next_escape = value.find(_ESC, index)
        next_newline = value.find("\n", index)
        ends = [end for end in (next_escape, next_newline) if end >= 0]
        end = min(ends) if ends else len(value)
        text = value[index:end]
        spans, _cell_count = split_graphemes(text)
        for start, stop, cell_size in spans:
            grapheme = text[start:stop]
            if cell_size > columns:
                # A two-cell grapheme cannot be rendered without touching the
                # reserved last column of a two-column terminal.
                grapheme = "�"
                cell_size = 1
            if line_cells and line_cells + cell_size > columns:
                output.append("\n")
                line_cells = 0
            output.append(grapheme)
            line_cells += max(cell_size, 0)
        index = end
    return "".join(output)


def normalize_transcript_block(value: str, *, columns: int | None = None) -> str:
    """Return one safe, style-contained block ending at a line boundary.

    When ``columns`` is provided, this is also the sole native-scrollback
    width boundary: every printable line is explicitly cell-wrapped before it
    can reach the terminal.
    """
    normalized = sanitize_transcript_ansi(value)
    visible = strip_safe_ansi(normalized)
    if visible and not visible.endswith("\n"):
        normalized += "\n"
    # Replay concatenates retained blocks without a prompt_toolkit reset in
    # between.  Contain every producer's styling even when it forgot SGR 0.
    if "\x1b]8;" in normalized:
        # A producer may provide a valid opening hyperlink without its closing
        # OSC-8.  Contain that presentation state to this block.
        normalized += "\x1b]8;;\x1b\\"
    if "\x1b[" in normalized and not normalized.endswith("\x1b[0m"):
        normalized += "\x1b[0m"
    if columns is not None:
        normalized = _wrap_safe_ansi(normalized, columns)
    return normalized


def sanitize_live_text(value: str) -> str:
    """Make plain prompt_toolkit text safe without interpreting ANSI styles."""
    # Run the strict scanner first, then expose even the two presentation
    # sequence families: FormattedTextControl receives styles separately.
    return strip_safe_ansi(sanitize_transcript_ansi(value))
