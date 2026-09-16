# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for wrap-safe code blocks in terminal markdown.

Two rendering defects are pinned here:

1. ``Console.print(..., soft_wrap=True)`` (used for agent messages) sets
   ``no_wrap`` on nested renderables, which *cropped* fenced code at the
   terminal width — a long single-line URL silently lost its suffix.
2. ``_SemanticBlockQuote`` rendered children at the full console width and
   then prepended its ``▌ `` prefix, pushing nested code-block Copy headers
   past the right edge (``Copy`` split across rows as ``Cop`` / ``y``).
"""

from __future__ import annotations

import io
import re

from nooa_cli.tui.copyable_markdown import CopyableMarkdown, TerminalMarkdown
from rich.console import Console

_LONG_URL = (
    "mesh://pmgb300ws-0375.ipp9c1.colossus.nvidia.com:8765/enroll/-a9qqtkoEII4"
    "#pin=0f91600d527801e582d761f834b4e979214c602bf6d81559804ad887735c52a6"
    "&secret=bm-TvbJ_DGmz6rB1WIAJx73X-DCZe9fUis_UsWAJljo&name=Wren"
)


def _render(markup: str, markdown_cls: type, width: int = 131) -> str:
    # height is explicit so Rich honors `width` on a StringIO console — the
    # same trick production uses in session._render_to_ansi.
    buf = io.StringIO()
    console = Console(file=buf, width=width, height=1, force_terminal=True, color_system=None)
    console.print(markdown_cls(markup), soft_wrap=True)
    return re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())


def test_terminal_markdown_wraps_long_code_lines_under_soft_wrap() -> None:
    """soft_wrap=True must fold a long code line, not crop its suffix."""
    markup = f"```\n{_LONG_URL}\n```\n"
    rendered = _render(markup, TerminalMarkdown)

    # Every visual row fits the console width...
    for line in rendered.splitlines():
        assert len(line.rstrip()) <= 131, line

    # ...and the concatenated code content still contains the full URL.
    joined = "".join(rendered.split())
    assert _LONG_URL in joined, rendered


def test_copyable_markdown_copy_payload_is_the_exact_source() -> None:
    markup = f"```\n{_LONG_URL}\n```\n"
    markdown = CopyableMarkdown(markup)
    assert list(markdown.copy_actions.values()) == [_LONG_URL]


def test_copy_header_stays_inside_a_blockquote() -> None:
    """The quote prefix must be reserved so the Copy header cannot overflow."""
    markup = f"> ```\n> {_LONG_URL}\n> ```\n"
    rendered = _render(markup, CopyableMarkdown, width=100)

    for line in rendered.splitlines():
        assert len(line.rstrip()) <= 100, line
    # A header squeezed by the prefix previously split "Copy" mid-word.
    assert "Cop\n" not in rendered
    assert "Copy" in rendered


def test_copy_header_stays_inside_nested_quote_and_list() -> None:
    for markup in (
        f"> - ```\n>   {_LONG_URL}\n>   ```\n",
        f"- > ```\n  > {_LONG_URL}\n  > ```\n",
    ):
        rendered = _render(markup, CopyableMarkdown, width=100)
        for line in rendered.splitlines():
            assert len(line.rstrip()) <= 100, (markup, line)
        assert "Copy" in rendered
