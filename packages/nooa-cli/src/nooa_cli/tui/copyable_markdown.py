# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Markdown rendering with semantic copy affordances for fenced code blocks."""

from __future__ import annotations

import secrets
from typing import Any

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, ListItem, Markdown
from rich.segment import Segment
from rich.syntax import Syntax
from rich.text import Text

_COPY_URI_PREFIX = "nooa-copy://"
_CODE_SOURCE_URI_PREFIX = "nooa-code-source://"


class _CopyableCodeBlock(CodeBlock):
    """Rich code block with a clickable, source-backed Copy label."""

    @classmethod
    def create(cls, markdown: Markdown, token: Any) -> _CopyableCodeBlock:
        node_info = token.info or ""
        lexer_name = node_info.partition(" ")[0] or "text"
        copyable = markdown
        assert isinstance(copyable, CopyableMarkdown)
        action_id = copyable._action_for_token(token)
        return cls(lexer_name, markdown.code_theme, action_id)

    def __init__(self, lexer_name: str, theme: str, action_id: str | None) -> None:
        super().__init__(lexer_name, theme)
        self.action_id = action_id

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # Drop only Markdown's structural newline. Source trailing spaces and
        # blank lines are real clipboard content and must remain addressable.
        code = str(self.text).removesuffix("\n")
        source_lines = code.split("\n")
        # Rich has no character on which to attach OSC-8 metadata for an empty
        # line. A single styled space occupies an already-painted code cell and
        # gives semantic selection a stable anchor for that source newline.
        rendered_code = (
            "\n".join(line or " " for line in source_lines) if self.action_id is not None else code
        )
        syntax = Syntax(
            rendered_code,
            self.lexer_name,
            theme=self.theme,
            word_wrap=True,
            # The footer below occupies the existing bottom padding row.
            padding=(1, 1, 0, 1) if self.action_id is not None else 1,
        )
        if self.action_id is not None:
            for line_number, line in enumerate(source_lines, 1):
                rendered_length = len(line.expandtabs(4)) or 1
                syntax.stylize_range(
                    f"link {_CODE_SOURCE_URI_PREFIX}{self.action_id}/{line_number - 1}",
                    (line_number, 0),
                    (line_number, rendered_length),
                )
        # ``Console.print(..., soft_wrap=True)`` is required to keep Markdown
        # prose semantic, but it sets ``no_wrap`` on every nested renderable.
        # Override that option locally so long source lines still wrap rather
        # than being cropped from the transcript.
        yield from console.render(syntax, options.update(no_wrap=False, overflow="fold"))
        if self.action_id is not None:
            # Keep the language and action inside the code panel by painting the
            # former bottom padding row with Syntax's own background style.
            footer = Text(style=Syntax.get_theme(self.theme).get_background_style())
            if self.lexer_name != "text":
                footer.append(f"{self.lexer_name} · ", style="dim")
            footer.append(
                "Copy",
                style=f"bold underline link {_COPY_URI_PREFIX}{self.action_id}",
            )
            footer.append(" ")
            footer.align("right", options.max_width)
            yield footer


class _SemanticListItem(ListItem):
    """Render complete list-item text for renderer-owned wrapping.

    Rich's normal list renderer calls ``render_lines`` even under
    ``soft_wrap=True``. That clips each list item to the Console width before
    the fullscreen transcript can reflow it, permanently losing the suffix.
    Emit the prefix and semantic child stream instead; the transcript model
    owns visual wrapping at the current viewport width.
    """

    def _semantic_lines(self, console: Console, options: ConsoleOptions) -> list[list[Segment]]:
        """Render logical child lines without Rich's width-cropping pass."""
        rendered = console.render(self.elements, options)
        styled = Segment.apply_style(rendered, self.style)
        return list(Segment.split_lines(styled)) or [[]]

    def render_bullet(self, console: Console, options: ConsoleOptions) -> RenderResult:
        bullet_style = console.get_style("markdown.item.bullet", default="none")
        for index, line in enumerate(self._semantic_lines(console, options)):
            yield Segment(" • " if index == 0 else "   ", bullet_style)
            yield from line
            yield Segment.line()

    def render_number(
        self,
        console: Console,
        options: ConsoleOptions,
        number: int,
        last_number: int,
    ) -> RenderResult:
        number_width = len(str(last_number)) + 2
        number_style = console.get_style("markdown.item.number", default="none")
        for index, line in enumerate(self._semantic_lines(console, options)):
            prefix = f"{number}".rjust(number_width - 1) + " " if index == 0 else " " * number_width
            yield Segment(prefix, number_style)
            yield from line
            yield Segment.line()


class CopyableMarkdown(Markdown):
    """Rich Markdown that exposes exact fenced-code payloads by stable action ID."""

    elements = {
        **Markdown.elements,
        "fence": _CopyableCodeBlock,
        "code_block": _CopyableCodeBlock,
        "list_item_open": _SemanticListItem,
    }

    def __init__(self, markup: str, **kwargs: Any) -> None:
        super().__init__(markup, **kwargs)
        self.copy_actions: dict[str, str] = {}
        self._token_actions: dict[int, str] = {}
        # Allocate metadata eagerly. This keeps source provenance available even
        # before Rich performs a render and makes replay reuse the same IDs.
        for token in self.parsed:
            if token.type in {"fence", "code_block"}:
                self._register_token(token)

    def _register_token(self, token: Any) -> str | None:
        payload = str(token.content)
        if payload.endswith("\n"):
            payload = payload[:-1]
        if not payload:
            return None
        # The identifier appears in ANSI presentation metadata, but authority
        # remains in the separately retained mapping. A random ID prevents
        # model-authored Markdown links from impersonating generated controls.
        action_id = secrets.token_urlsafe(18)
        self._token_actions[id(token)] = action_id
        # Markdown parsers include the structural newline before the closing
        # fence. Clipboard UX conventionally excludes that one delimiter.
        self.copy_actions[action_id] = payload
        return action_id

    def _action_for_token(self, token: Any) -> str | None:
        if id(token) in self._token_actions:
            return self._token_actions[id(token)]
        return self._register_token(token)
