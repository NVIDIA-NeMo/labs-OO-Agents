# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Markdown rendering with semantic copy affordances for fenced code blocks."""

from __future__ import annotations

import secrets
from typing import Any

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.text import Text

_COPY_URI_PREFIX = "nooa-copy://"


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
        if self.action_id is not None:
            header = Text(justify="right")
            if self.lexer_name != "text":
                header.append(f"{self.lexer_name} · ", style="dim")
            header.append(
                "Copy",
                style=f"bold underline link {_COPY_URI_PREFIX}{self.action_id}",
            )
            yield header
        yield Syntax(
            str(self.text).rstrip(),
            self.lexer_name,
            theme=self.theme,
            word_wrap=True,
            padding=1,
        )


class CopyableMarkdown(Markdown):
    """Rich Markdown that exposes exact fenced-code payloads by stable action ID."""

    elements = {
        **Markdown.elements,
        "fence": _CopyableCodeBlock,
        "code_block": _CopyableCodeBlock,
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
