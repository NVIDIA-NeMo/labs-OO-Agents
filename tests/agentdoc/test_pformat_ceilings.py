# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""pformat's documented safety ceilings: None is no caller limit, not unbounded."""

from nooa.agentdoc import pformat


def _nested(depth: int) -> list:
    value: list = ["end"]
    for _ in range(depth):
        value = [value]
    return value


def test_max_depth_none_shows_64_levels():
    assert "'end'" in pformat(_nested(60))
    assert "'end'" not in pformat(_nested(100))


def test_a_tuple_key_has_a_fixed_budget():
    text = pformat({tuple(range(40)): "value"})
    assert text == "{tuple(len=40, [:8]=(0, 1, 2, 3, 4, 5, 6, 7)): 'value'}"
