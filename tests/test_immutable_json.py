# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for frozen native JSON: lone surrogates are escaped once, on capture."""

import json

from nooa._immutable_json import freeze, json_containers, utf8_safe_text
from nooa.llm_types import AssistantReasoning


def test_utf8_safe_text_escapes_only_lone_surrogates():
    assert utf8_safe_text("a\ud800b") == "a\\ud800b"
    assert utf8_safe_text("\udcff") == "\\udcff"
    text = "Héllo 🎉 — ok"
    assert utf8_safe_text(text) is text


def test_freeze_escapes_nested_values_and_keys():
    frozen = freeze({"k\ud800": ["x\udc80", {"text": "ok \ud83d"}], "n": 1})

    wire = json_containers(frozen)

    assert wire == {"k\\ud800": ["x\\udc80", {"text": "ok \\ud83d"}], "n": 1}
    json.dumps(wire, ensure_ascii=False).encode("utf-8")


def test_native_replay_payload_is_encodable_on_the_wire():
    """A provider payload holding a lone surrogate can still be replayed."""
    part = AssistantReasoning(
        text="thought",
        native={"reasoning_content": {"type": "reasoning", "summary": ["s\ud800"]}},
    )

    wire = json_containers(part.native)

    assert wire["reasoning_content"]["summary"] == ["s\\ud800"]
    json.dumps(wire, ensure_ascii=False).encode("utf-8")


def test_freeze_keeps_encodable_strings_as_the_same_objects():
    blob = "A" * 100_000
    frozen = freeze({"encrypted_content": blob, "note": "naïve"})

    assert frozen["encrypted_content"] is blob
