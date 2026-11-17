# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Docstring placeholders are bounded and go through nooa.snapshot, in both places that expand them.

``ActorRuntime.expand_variables`` (method docstrings, Task prompts) and
``Agent._resolve_system_prompt`` (the class docstring) share
``nooa.runtime.placeholders.render_placeholder``.
"""

from typing import Any

import pytest

from nooa import Agent
from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
from nooa.runtime.placeholders import render_placeholder as _render_placeholder
from nooa.unifiedllm import FakeLLMClient

LIMIT = DEFAULT_TRUNCATION_CONFIG.placeholder_chars
assert LIMIT == 50_000
BOUND = LIMIT + 200  # the cut line adds a few dozen chars
HUGE = "a" * 5_000_000 + "b" * 5_000_000  # 10 MB


def render_placeholder(value, conversion=None, format_spec=""):
    """``render_placeholder`` under the default ``TruncationConfig.placeholder_chars``."""
    return _render_placeholder(value, conversion, format_spec, max_chars=LIMIT)


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


class _BadConversion:
    def __str__(self):
        raise ValueError("bad-str")

    def __repr__(self):
        raise ValueError("bad-repr")

    def __format__(self, spec):
        raise ValueError("bad-format")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["show", "silent", "raise"])
@pytest.mark.parametrize("template", ["{v!s}", "{v!s:>3}", "{v!r:>3}"])
async def test_a_failing_conversion_raises_with_or_without_a_format_spec(mode, template):
    """As before #580: a conversion runs outside the format-spec handling, so its error raises."""
    agent = _holder()
    with pytest.raises(ValueError, match="bad-(str|repr)"):
        await agent.runtime.expand_variables(template, {"v": _BadConversion()}, mode)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("show", "{v:.2f | FORMAT ERROR}"),
        ("silent", "{v:.2f}"),
        ("raise", "{v:.2f | FORMAT ERROR}"),
    ],
)
async def test_a_failing_format_spec_is_handled_as_before(mode, expected):
    agent = _holder()
    assert (
        await agent.runtime.expand_variables("{v:.2f}", {"v": _BadConversion()}, mode) == expected
    )


@pytest.mark.asyncio
async def test_a_conversion_then_a_format_spec_applies_both():
    agent = _holder()
    assert await agent.runtime.expand_variables("[{v!r:>6}]", {"v": "ab"}) == "[  'ab']"
    assert await agent.runtime.expand_variables("[{v!s:.2f}]", {"v": 1.5}, "silent") == (
        "[{v!s:.2f}]"  # str(1.5) rejects .2f: the format error is handled as before
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", [".30000", ".50050", ">70000"])
async def test_a_conversion_then_a_format_spec_is_cut_once_at_the_end(spec):
    # The spec applies to the whole converted text, never to an already-cut one:
    # {v!s:.30000} inserts the first 30,000 characters, as format(str(v), spec) does.
    value = "A" * 30_000 + "B" * 30_000
    agent = _holder()
    actual = await agent.runtime.expand_variables("{v!s:" + spec + "}", {"v": value})
    assert actual == render_placeholder(value, "s", spec)
    if spec == ".30000":
        assert actual == "A" * 30_000


def test_a_converted_spec_placeholder_reads_the_same_in_the_system_prompt():
    class Spec(Agent):
        """{self.value!s:.30000}"""

    agent = Spec(llm=FakeLLMClient([]))
    agent.value = "A" * 30_000 + "B" * 30_000
    assert agent._resolve_system_prompt() == "A" * 30_000


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


class _Configured(Agent):
    """Value {self.value}."""

    async def work(self) -> str: ...


def _configured(**limits):
    from nooa.config.truncation_config import FormatConfig, TruncationConfig

    agent = _Configured(
        llm=FakeLLMClient([]), truncation=TruncationConfig(prefill_format=FormatConfig(**limits))
    )
    return agent


async def test_placeholders_honour_tighter_prefill_limits():
    # correctness 4 (placeholder_config.py): both callers used the global defaults.
    agent = _configured(max_string=4, max_length=1, max_depth=1)
    agent.value = ["abcdefghij", "second"]
    expected = "list(len=2, [:1]=[str(len=10, [:2]='ab', [-2:]='ij')])"
    assert agent._resolve_system_prompt() == f"Value {expected}."
    assert await agent.runtime.expand_variables("{self.value}") == expected


async def test_placeholders_honour_looser_prefill_limits():
    agent = _configured(max_string=10_000, max_length=500, max_depth=6)
    agent.value = list(range(300))  # the default prefill keeps 25 items
    whole = str(list(range(300)))
    assert agent._resolve_system_prompt() == f"Value {whole}."
    assert await agent.runtime.expand_variables("{self.value}") == whole


async def test_placeholder_strings_keep_the_50k_bound_whatever_the_limits():
    agent = _configured(max_string=10_000_000)
    agent.value = "a" * 200_000
    assert len(await agent.runtime.expand_variables("{self.value}")) < BOUND


# TruncationConfig.placeholder_chars: the bound is the agent's config, None = whole.


def _with_placeholder_chars(limit) -> Any:
    from nooa.config.truncation_config import TruncationConfig

    agent = _Configured(llm=FakeLLMClient([]), truncation=TruncationConfig(placeholder_chars=limit))
    agent.value = "A" * 30_000 + "B" * 30_000
    return agent


def test_placeholder_chars_bounds_the_system_prompt():
    prompt = _with_placeholder_chars(1_000)._resolve_system_prompt()
    assert len(prompt) < 1_200
    assert prompt.startswith("Value AAA") and prompt.endswith("BBB.")


def test_placeholder_chars_none_keeps_the_system_prompt_whole():
    agent = _with_placeholder_chars(None)
    assert agent._resolve_system_prompt() == f"Value {agent.value}."


async def test_placeholder_chars_bounds_task_prompts():
    agent = _with_placeholder_chars(1_000)
    for template in ("{self.value}", "{self.value!s}", "{self.value:>10}"):
        text = await agent.runtime.expand_variables(template)
        assert len(text) < 1_200 and text.startswith("AAA") and text.endswith("BBB")
    error = await agent.runtime.expand_variables("{self.boom.explode()}", {"self": _holder()})
    assert error.startswith("{ERROR: xxx") and len(error) < 1_200


async def test_placeholder_chars_none_keeps_task_prompts_whole():
    agent = _with_placeholder_chars(None)
    assert await agent.runtime.expand_variables("{self.value}") == agent.value
    assert await agent.runtime.expand_variables("{self.value!s}") == agent.value


async def test_expand_variables_placeholder_chars_overrides_the_config():
    agent = _with_placeholder_chars(1_000)
    text = await agent.runtime.expand_variables("{self.value}", placeholder_chars=None)
    assert text == agent.value


def test_placeholder_chars_must_be_positive():
    from nooa.config.truncation_config import TruncationConfig

    with pytest.raises(ValueError, match="placeholder_chars"):
        TruncationConfig(placeholder_chars=0)
