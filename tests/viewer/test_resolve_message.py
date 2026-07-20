# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case unit tests for ``_resolve_message`` -- the inverse of
``_journal_builder.build_journal_payload`` / ``_skeleton_dict_message``.

The integration round-trip (`tests/runtime/test_journal_payload.py`,
`tests/integration/test_file_viewer_equivalence.py`) covers the happy
paths.  These tests pin the awkward shapes the builder doesn't produce
today but the resolver must still handle gracefully:

- mixed text-and-block parts (block-formatted system messages with
  inline literal text between blocks),
- empty ``parts=[]`` (a formatter that produced no output),
- ``content`` and ``parts`` both present (a future writer might do
  this; the resolver picks ``content`` -- see
  ``test_content_wins_when_both_present``).
"""

from __future__ import annotations

from nooa.viewer.otlp_store import _resolve_message


def test_mixed_text_and_block_parts_concatenate_in_order():
    """Block-formatted system messages emit interleaved text + block
    parts; the resolver must concatenate them *in order* into ``content``."""
    blocks = {
        "sha256:b1": "<notes>\nHello.\n</notes>",
        "sha256:b2": "<status>\nIdle.\n</status>",
    }
    msg = {
        "role": "system",
        "parts": [
            {"text": "Preamble: "},
            {"block_hash": "sha256:b1", "key": "notes"},
            {"text": "\n---\n"},
            {"block_hash": "sha256:b2", "key": "status"},
            {"text": " end."},
        ],
    }
    out = _resolve_message(msg, blocks)
    assert out == {
        "role": "system",
        "content": ("Preamble: <notes>\nHello.\n</notes>\n---\n<status>\nIdle.\n</status> end."),
    }


def test_empty_parts_produces_none_content():
    """Empty ``parts=[]`` -- the formatter ran but emitted nothing.
    Resolver returns ``content=None``, not ``""``, so the downstream
    flattener can skip the ``llm.input_messages.N.message.content``
    attribute entirely (mirroring the OpenInference convention of
    omitting empty content)."""
    msg = {"role": "user", "parts": []}
    out = _resolve_message(msg, {})
    assert out["content"] is None


def test_missing_block_hash_inserts_placeholder():
    """A hash the receiver doesn't have lands as a clearly-marked
    placeholder so the gap is visible in the rendered span."""
    msg = {
        "role": "user",
        "parts": [{"block_hash": "sha256:nope", "key": "k"}],
    }
    out = _resolve_message(msg, {})
    assert out["content"] == "<missing block: sha256:nope>"


def test_content_preserved_when_parts_absent():
    """v3 also carries plain ``content`` (no parts) for messages where
    the builder didn't content-address.  Pass-through unchanged."""
    msg = {"role": "user", "content": "plain text"}
    out = _resolve_message(msg, {})
    assert out == {"role": "user", "content": "plain text"}


def test_content_wins_when_both_present():
    """Defensive: if a future writer happens to emit both ``content``
    and ``parts``, the existing ``content`` takes precedence.  The
    builder doesn't produce this shape today; this test pins the
    behaviour so a downstream change to the resolver is intentional."""
    blocks = {"sha256:b1": "from-block"}
    msg = {
        "role": "user",
        "content": "from-content",
        "parts": [{"block_hash": "sha256:b1"}],
    }
    out = _resolve_message(msg, blocks)
    assert out["content"] == "from-content"
    assert "parts" not in out  # parts get consumed/popped from the output


def test_tool_call_arguments_hash_resolved_to_arguments():
    """The output side replaces ``arguments`` with ``arguments_hash``;
    the resolver reverses it.  Resolver must put ``arguments`` back on
    the function dict and drop ``arguments_hash``."""
    blocks = {"sha256:args": '{"code": "print(1)"}'}
    msg = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "tc1",
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "arguments_hash": "sha256:args",
                },
            }
        ],
    }
    out = _resolve_message(msg, blocks)
    fn = out["tool_calls"][0]["function"]
    assert fn["name"] == "execute_python"
    assert fn["arguments"] == '{"code": "print(1)"}'
    assert "arguments_hash" not in fn


def test_tool_call_with_missing_arguments_hash_uses_placeholder():
    msg = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "tc1",
                "type": "function",
                "function": {"name": "x", "arguments_hash": "sha256:nope"},
            }
        ],
    }
    out = _resolve_message(msg, {})
    fn = out["tool_calls"][0]["function"]
    assert fn["arguments"] == "<missing block: sha256:nope>"


def test_image_hashes_round_trip_to_images_list():
    blocks = {
        "sha256:i1": '{"type":"image_url","image_url":{"url":"data:..."}}',
        "sha256:i2": "https://cdn.example/photo.png",
    }
    msg = {
        "role": "user",
        "parts": [{"text": "look"}],
        "image_hashes": ["sha256:i1", "sha256:i2"],
    }
    out = _resolve_message(msg, blocks)
    assert out["images"] == [
        {"type": "image_url", "image_url": {"url": "data:..."}},  # decoded
        "https://cdn.example/photo.png",  # passed through (URL string)
    ]
    assert "image_hashes" not in out
