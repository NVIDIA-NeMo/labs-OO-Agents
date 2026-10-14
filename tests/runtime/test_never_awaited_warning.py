# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cell's "coroutine ... was never awaited" warning becomes a one-line hint (#447)."""

from nooa.runtime.actor import _rewrite_never_awaited_warnings


class ShellTools:
    async def read(self, path):
        return path


class MyAgent:
    def __init__(self):
        self.shell = ShellTools()

    async def delegate(self, prompt):
        return prompt


STOCK = (
    "Cell In[3]:3: RuntimeWarning: coroutine 'ShellTools.read' was never awaited\n"
    "RuntimeWarning: Enable tracemalloc to get the object allocation traceback\n"
)


def test_names_the_agent_call_and_drops_the_tracemalloc_line():
    assert _rewrite_never_awaited_warnings(STOCK, MyAgent()) == (
        "`ShellTools.read(...)` was called without await and never ran; "
        "use `await self.shell.read(...)`\n"
    )


def test_source_line_and_agent_methods():
    text = (
        "Cell In[4]:1: RuntimeWarning: coroutine 'MyAgent.delegate' was never awaited\n"
        '  self.delegate("x")\n'
        "RuntimeWarning: Enable tracemalloc to get the object allocation traceback\n"
    )
    assert _rewrite_never_awaited_warnings(text, MyAgent()) == (
        "`MyAgent.delegate(...)` was called without await and never ran; "
        "use `await self.delegate(...)`\n"
    )


def test_unknown_owner_still_gets_the_fix():
    text = "x.py:1: RuntimeWarning: coroutine 'fetch' was never awaited\n"
    assert _rewrite_never_awaited_warnings(text, MyAgent()) == (
        "`fetch(...)` was called without await and never ran; put `await` before the call\n"
    )


def test_other_warnings_and_text_are_unchanged():
    other = (
        "before\n"
        "Cell In[2]:1: SyntaxWarning: invalid escape sequence '\\s'\n"
        "Cell In[2]:2: RuntimeWarning: divide by zero encountered\n"
        "  x = 1 / 0\n"
    )
    assert _rewrite_never_awaited_warnings(other, MyAgent()) == other
    assert _rewrite_never_awaited_warnings(other + STOCK, MyAgent()) == other + (
        "`ShellTools.read(...)` was called without await and never ran; "
        "use `await self.shell.read(...)`\n"
    )
