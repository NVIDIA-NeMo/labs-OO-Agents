# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO

from nooa_cli.tui.commands import _mcp_oauth_markdown_link
from nooa_cli.tui.console import TUIConsole
from nooa_cli.tui.output import AgentMessage
from nooa_cli.tui.terminal_safety import strip_safe_ansi
from rich.console import Console


def test_agent_message_defaults_to_terminal_managed_wrapping():
    message = (
        "I left a brief closing note that the interaction design and implementation "
        "need a more fundamental rethink. The local diagnostic mouse fix remains "
        "uncommitted and was not pushed."
    )
    stream = StringIO()
    console = TUIConsole()
    console.replace_console(Console(file=stream, width=60, color_system=None))

    output = AgentMessage(message)
    console.print_agent(output.content, show_rule=False, soft_wrap=output.soft_wrap)

    assert output.soft_wrap is True
    assert stream.getvalue() == f"{message}\n"


def test_agent_code_block_retains_visual_highlight_and_indent():
    stream = StringIO()
    console = TUIConsole()
    console.replace_console(
        Console(file=stream, width=40, force_terminal=True, color_system="truecolor")
    )

    console.print_agent(
        'Before\n\n```python\nif ready:\n    print("one")\n```\n\nAfter',
        show_rule=False,
        soft_wrap=True,
    )

    rendered = stream.getvalue()
    plain = strip_safe_ansi(rendered)
    assert "\x1b[" in rendered
    assert " " * 40 in plain
    assert " if ready:" in plain
    assert '     print("one")' in plain


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
