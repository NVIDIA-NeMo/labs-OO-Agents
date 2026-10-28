# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for frozen native JSON: lone surrogates are detected, never rewritten."""

from nooa._immutable_json import freeze, json_containers, utf8_encodable


def test_freeze_keeps_lone_surrogates_exact():
    value = {"k\ud800": ["x\udc80", {"text": "ok \ud83d"}], "n": 1}

    assert json_containers(freeze(value)) == value


def test_utf8_encodable_finds_lone_surrogates_in_keys_and_values():
    assert utf8_encodable(freeze({"a": ["Héllo 🎉", {"b": None}], "n": 1.5}))
    assert utf8_encodable(None)
    assert not utf8_encodable(freeze({"a": ("ok", {"b": "x\ud800"})}))
    assert not utf8_encodable(freeze({"k\udcff": "v"}))
