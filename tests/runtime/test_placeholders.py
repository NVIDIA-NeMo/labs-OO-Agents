# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Docstring placeholders are bounded and go through nooa.snapshot, in both places that expand them.

``ActorRuntime.expand_variables`` (method docstrings, Task prompts) and
``Agent._resolve_system_prompt`` (the class docstring) share
``nooa.runtime.placeholders.render_placeholder``.
"""

import pytest

from nooa import Agent
from nooa.runtime.placeholders import PLACEHOLDER_TEXT_CHARS, render_placeholder
from nooa.unifiedllm import FakeLLMClient

BOUND = PLACEHOLDER_TEXT_CHARS + 200  # the cut line adds a few dozen chars
HUGE = "a" * 5_000_000 + "b" * 5_000_000  # 10 MB


class _Boom:
    def __init__(self, size):
        self.size = size

    def explode(self):
        raise ValueError("x" * self.size)


class _Holder(Agent):
    """Huge {self.huge}. Items {self.items}. Repr {self.items!r}. Spec {self.huge:>10}. Fixed {self.pi:.2f}. Err {self.boom.explode()}. End."""

    async def work(self) -> str: ...


def _holder():
    agent = _Holder(llm=FakeLLMClient([]))
    agent.huge = HUGE
    agent.items = list(range(1_000_000))
    agent.pi = 3.14159
    agent.boom = _Boom(10_000_000)
    return agent


SMALL = [
    ("text", "text"),
    (42, "42"),
    (2.5, "2.5"),
    (None, "None"),
    (True, "True"),
    ([1, "a", 2.5], "[1, 'a', 2.5]"),
    ((1,), "(1,)"),
    ({"k": [1, 2]}, "{'k': [1, 2]}"),
]


@pytest.mark.parametrize(("value", "expected"), SMALL)
def test_small_values_render_as_str_did(value, expected):
    assert render_placeholder(value) == expected == str(value)


def test_a_huge_string_keeps_its_ends_around_the_cut_line():
    text = render_placeholder(HUGE)
    assert len(text) < BOUND
    assert text.startswith("a" * 1000) and text.endswith("b" * 1000)
    assert "chars omitted; full output not kept]" in text


def test_a_huge_list_is_a_snapshot_not_its_str():
    text = render_placeholder(list(range(1_000_000)))
    assert text.startswith("list(len=1000000, [:")
    assert len(text) < 1_000


def test_conversion_and_format_spec_are_honoured_then_cut():
    assert render_placeholder("hi\n", "r") == repr("hi\n")
    assert render_placeholder(3.14159, None, ".2f") == "3.14"
    assert render_placeholder("x", None, ">3") == "  x"
    assert len(render_placeholder(list(range(1_000_000)), "r")) < BOUND
    assert len(render_placeholder(HUGE, None, ">10")) < BOUND
    with pytest.raises(ValueError):
        render_placeholder("abc", None, ".2f")


@pytest.mark.asyncio
async def test_expand_variables_bounds_every_placeholder():
    agent = _holder()
    context = {"huge": HUGE, "items": agent.items}
    runtime = agent.runtime
    assert len(await runtime.expand_variables("{huge}", context)) < BOUND
    items = await runtime.expand_variables("{items}", context)
    assert items.startswith("list(len=1000000, [:") and len(items) < 1_000
    assert len(await runtime.expand_variables("{items!r}", context)) < BOUND
    assert len(await runtime.expand_variables("{huge:>10}", context)) < BOUND
    assert await runtime.expand_variables("{3.14159:.2f} {len(items)}", context) == "3.14 1000000"
    error = await runtime.expand_variables("{self.boom.explode()}")
    assert error.startswith("{ERROR: xxx") and error.endswith("xxx}") and len(error) < BOUND


@pytest.mark.asyncio
async def test_expand_variables_keeps_small_values_byte_identical():
    agent = _holder()
    for value, expected in SMALL:
        assert await agent.runtime.expand_variables("[{v}]", {"v": value}) == f"[{expected}]"


def test_system_prompt_bounds_every_placeholder():
    prompt = _holder()._resolve_system_prompt()
    assert len(prompt) < 4 * BOUND
    huge, rest = prompt.split(" Items ", 1)
    assert len(huge) < BOUND and huge.startswith("Huge aaa")
    items, rest = rest.split(" Repr ", 1)
    assert items.startswith("list(len=1000000, [:") and len(items) < 1_000
    assert "Fixed 3.14." in rest
    # A failing placeholder stays as written (today's behaviour), so its huge message never enters.
    assert rest.endswith("Err {self.boom.explode()}. End.")


def test_system_prompt_small_values_are_unchanged():
    class Small(Agent):
        """You are {type(self).__name__} with {len(self.tools)} tools, {self.ratio:.1%} {self.name!r}."""

    agent = Small(llm=FakeLLMClient([]))
    agent.tools = [1, 2, 3]
    agent.ratio = 0.5
    agent.name = "n"
    assert agent._resolve_system_prompt() == "You are Small with 3 tools, 50.0% 'n'."
