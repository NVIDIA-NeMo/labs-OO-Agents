# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill string cuts go through nooa.snapshot's capture + render_python (PR 2b)."""

import sys
from unittest import mock

import pytest

from nooa.agentdoc import pformat
from nooa.config.truncation_config import TruncationConfig


def test_prefill_limits_are_unchanged():
    prefill = TruncationConfig().prefill_format
    assert (prefill.max_string, prefill.max_length, prefill.max_depth) == (2_000, 25, 4)


def test_a_cut_string_is_render_python_with_an_even_split():
    text = "".join(chr(97 + i % 26) for i in range(5_000))
    assert pformat(text, max_string=2_000) == (
        f"str(len=5000, [:1000]={text[:1000]!r}, [-1000:]={text[-1000:]!r})"
    )
    assert pformat("abcdefg", max_string=3) == "str(len=7, [:2]='ab', [-1:]='g')"
    assert pformat("abcdefg", max_string=1) == "str(len=7, [:1]='a')"
    assert pformat("ab'\"\ncd", max_string=4) == "str(len=7, [:2]='ab', [-2:]='cd')"


def test_the_cut_runs_through_capture_and_render_python():
    module = sys.modules["nooa.agentdoc._pformat"]  # nooa.agentdoc._pformat is also a function
    with mock.patch.object(module, "_render_cut_string", wraps=module._render_cut_string) as cut:
        assert "'k': str(len=3000, [:1000]=" in pformat({"k": "x" * 3_000}, max_string=2_000)
    cut.assert_called_with("x" * 3_000, 2_000)  # pformat may lay a value out twice


@pytest.mark.parametrize("max_string", [0, 1, 2, 7, 2_000])
def test_uncut_strings_render_as_before(max_string):
    assert pformat("", max_string=max_string) == "''"
    assert pformat("a\nb", max_string=None) == "'''a\nb'''"
