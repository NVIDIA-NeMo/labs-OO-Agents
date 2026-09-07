# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from io import StringIO
from types import SimpleNamespace

import pytest
from nooa_cli.tui.commands import CommandRegistry
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


@pytest.mark.asyncio
async def test_oauth_bridge_emits_no_scrollback_url_render():
    """The overlay is the single URL surface: no scrollback render before it.

    The duplicate scrollback write was the cursor-corruption root cause; this
    pins that the bridge does not reintroduce it.
    """
    rendered: list[object] = []

    class FakeMCP:
        def _bind_oauth_code_prompt(self, callback):
            rendered.append(callback)

    class RecordingFrontend:
        async def render(self, output):
            rendered.append(output)
            raise AssertionError("bridge must not render to scrollback before the overlay")

        async def prompt_sensitive(self, title, message, *, link_url=None):
            return "http://localhost:8090/callback?code=ok&state=expected"

    registry = CommandRegistry.__new__(CommandRegistry)
    registry.agent = SimpleNamespace(mcp=FakeMCP())
    registry.frontend = RecordingFrontend()
    registry._bind_mcp_oauth_prompt()
    callback = rendered[0]

    result = await asyncio.to_thread(
        lambda: asyncio.run(callback("https://login.example.test/authorize?state=expected"))
    )

    assert result == "http://localhost:8090/callback?code=ok&state=expected"
    # Only the bound callback remains; no AgentMessage render ever happened.
    assert len(rendered) == 1
