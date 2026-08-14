# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO

from nooa_cli.tui.commands import _mcp_oauth_markdown_link
from nooa_cli.tui.console import TUIConsole
from rich.console import Console


def test_agent_message_soft_wrap_preserves_long_url_as_one_logical_line():
    url = "https://login.example.test/authorize?state=" + "a" * 500
    stream = StringIO()
    console = TUIConsole()
    console.replace_console(Console(file=stream, width=60, color_system=None))

    console.print_agent(f"[{url}](<{url}>)", show_rule=False, soft_wrap=True)

    assert stream.getvalue() == f"{url}\n"


def test_oauth_link_rejects_malformed_bracketed_host():
    assert _mcp_oauth_markdown_link("https://[not-ipv6]/oauth") is None


def test_oauth_link_rejects_terminal_controls():
    assert _mcp_oauth_markdown_link("https://example.test/a\x00b") is None
    assert _mcp_oauth_markdown_link("https://example.test/a\x1bb") is None
    assert _mcp_oauth_markdown_link("https://example.test/a\x85b") is None
