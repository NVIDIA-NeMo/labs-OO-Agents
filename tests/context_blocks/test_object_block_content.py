# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialized objects remain intact until formatting or budget eviction."""

from copy import deepcopy
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from nooa import Block, CacheBoundary, apply_context_budget, context_text
from nooa.agentdoc import pformat
from nooa.config.truncation_config import FormatConfig
from nooa.context_blocks import OpenAIProviderFormatter, XMLBlockFormatter, render_context
from nooa.strategies.current_call import CurrentCall


@dataclass
class Account:
    balance: int
    eligible: bool


class AccountModel(BaseModel):
    balance: int
    eligible: bool


def render(block):
    return render_context(
        (block,),
        block_formatter=XMLBlockFormatter(),
        provider_formatter=OpenAIProviderFormatter(),
    ).output


def test_string_block_content_is_not_reformatted():
    value = '  {"balance": 120}\n\n literal "quotes"  '
    block = Block(key="account", content=value)

    assert value in render(block)[0]["content"]
    assert block.content is value


def test_structured_budget_and_rendering_use_same_format_limits():
    value = {"payload": "x" * 10_000}
    call = CurrentCall(
        id="bounded",
        method_name="run",
        decorator="plan",
        context_budget=500,
        _context_format=FormatConfig(max_string=30),
        _context_token_counter=len,
    )
    block = Block(key="state", content=value)
    items = apply_context_budget((block,), call=call, evictable=(block,))
    actual = render_context(
        items,
        block_formatter=XMLBlockFormatter(),
        provider_formatter=OpenAIProviderFormatter(),
        context_format=call.context_format,
    ).output
    expected = render(Block(key="state", content=context_text(value, call=call)))
    assert items[0] is block
    assert actual == expected
    assert len(actual[0]["content"]) < 500
    assert block.content == value


@pytest.mark.parametrize(
    "value",
    [
        {"balance": 120, "eligible": True},
        [{"balance": 120}, False, None],
        Account(balance=120, eligible=True),
        AccountModel(balance=120, eligible=True),
    ],
    ids=["dictionary", "list", "dataclass", "pydantic"],
)
def test_object_block_rendering_matches_explicit_formatting_without_mutation(value):
    original = deepcopy(value)
    block = Block(key="account", content=value)
    formatted = Block(key="account", content=pformat(value, unquote_strings=True))

    assert render(block) == render(formatted)
    assert block.content is value
    assert value == original


@pytest.mark.parametrize("budget", [0, 10_000])
def test_budget_counts_formatted_object_without_changing_original(budget):
    value = {"entries": [{"balance": 120, "eligible": True}]}
    original = deepcopy(value)
    counted = []

    def counter(text):
        assert isinstance(text, str)
        counted.append(text)
        return len(text)

    call = CurrentCall(
        id="budget",
        method_name="run",
        decorator="plan",
        context_budget=budget,
        _context_token_counter=counter,
    )
    block = Block(key="account", content=value)
    items = (block, CacheBoundary())

    result = apply_context_budget(items, call=call, evictable=(block,))

    assert context_text(value, call=call) in counted
    assert result[1] == CacheBoundary()
    if budget == 0:
        assert result[0].content.startswith("EVICTED:")
        assert result[0] is not block
    else:
        assert result[0] is block
    assert block.content is value
    assert value == original
